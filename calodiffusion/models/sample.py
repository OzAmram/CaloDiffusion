"""
methods for sampling on inference

"""

from abc import ABC
from typing import Any
import torch
import numpy as np

from calodiffusion.utils import sampling


class Sample:
    def __init__(self, config) -> None:
        self.config = config
        self.sample_config = self.config.get("SAMPLER_OPTIONS", {})

    def __call__(
        self, model, start, energy, layers, num_steps, sample_offset, debug
    ) -> Any:
        raise NotImplementedError


class DDim(Sample):
    def __init__(self, config):
        """
        _summary_

        Args:
            config: _description_
        """
        super().__init__(config)
        self.ddim_eta = 0.0

    @torch.no_grad()
    def __call__(
        self, model, start, energy, layers, num_steps, sample_offset, debug
    ) -> Any:
        betas = sampling.cosine_beta_schedule(num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        alphas_cumprod_prev = torch.nn.functional.pad(
            alphas_cumprod[:-1], (1, 0), value=1.0
        )

        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        gen_size = start.shape[0]
        time_steps = torch.arange(num_steps)
        time_steps = torch.flip(time_steps, [0])

        if sample_offset > 0:
            time_steps = time_steps[sample_offset:]

        sigma_start = (
            sqrt_one_minus_alphas_cumprod[time_steps[0]]
            / sqrt_alphas_cumprod[time_steps[0]]
        )
        x = start * sigma_start

        xs = []
        x0s = []

        for t in time_steps:
            t = torch.full((gen_size,), t, device=x.device, dtype=torch.long)

            sqrt_one_minus_alphas_cumprod_t = sampling.extract(
                sqrt_one_minus_alphas_cumprod, t, x.shape
            )
            sqrt_alphas_cumprod_t = sampling.extract(sqrt_alphas_cumprod, t, x.shape)

            alpha = sampling.extract(alphas_cumprod, t, x.shape)
            alpha_prev = sampling.extract(alphas_cumprod_prev, t, x.shape)
            denom = sampling.extract(
                sqrt_alphas_cumprod, torch.maximum(t - 1, torch.zeros_like(t)), x.shape
            )

            sigma = sqrt_one_minus_alphas_cumprod_t / sqrt_alphas_cumprod_t

            x0_pred = model(x, sigma=sigma, E=energy, layers=layers)

            noise_pred = (x - x0_pred) / sigma

            noise = torch.randn(x.shape, device=x.device)

            ddim_sigma = (
                self.ddim_eta
                * (((1 - alpha_prev) / (1 - alpha)) * (1 - alpha / alpha_prev)) ** 0.5
            )
            num = (1.0 - alpha_prev - ddim_sigma**2).sqrt()
            sigma_prev = num / denom

            # don't step for t= 0
            mask = (t > 0).reshape(-1, *((1,) * (len(x.shape) - 1)))

            x = x0_pred + mask * sigma_prev * noise_pred + ddim_sigma * noise / denom

            x0s.append(x0_pred)
            xs.append(x)

        return x, xs, x0s


class DDPM(DDim):
    def __init__(self, config):
        """
        _summary_

        Args:
            config: _description_
            sample_algorithm: _description_. Defaults to "None".
        """
        super().__init__(config)
        self.ddim_eta = 1.0


class DPM(Sample):
    def __init__(self, config):
        """
        DPM-Solver-Fast (fixed step size). See https://arxiv.org/abs/2206.00927.


        Args:
            config: _description_
            sample_algorithm: _description_. Defaults to "None".

        Config:
            "ETA"
            "S_NOISE"
        """
        super().__init__(config)
        self.eta = self.sample_config.get("ETA", 0)
        self.s_noise =self.sample_config.get("S_NOISE", 1.0)

    def create_sigmas(self, model, num_steps):
        return torch.tensor(
            [
                model.loss_function.sqrt_one_minus_alphas_cumprod[num_steps - t - 1]
                / model.loss_function.sqrt_alphas_cumprod[num_steps - t - 1]
                for t in torch.arange(num_steps)
            ]
        )

    def setup(self, model, start, num_steps):
        if model.nsteps != num_steps:
            model.loss_function.update_step(num_steps)

        # scale starting point to appropriate noise level
        self.sigmas = self.create_sigmas(model, num_steps)
        x = start * self.sigmas[0]
        return x

    def sample(self, model, x, num_steps, energy, layers):
        sigma_min, sigma_max = self.sigmas[-1], self.sigmas[0]

        """DPM-Solver-Fast (fixed step size). See https://arxiv.org/abs/2206.00927."""
        if sigma_min <= 0 or sigma_max <= 0:
            raise ValueError("sigma_min and sigma_max must not be 0")
        dpm_solver = sampling.DPMSolver(model, E=energy, layers=layers)
        return dpm_solver.dpm_solver_fast(
            x,
            dpm_solver.t(torch.tensor(sigma_max)),
            dpm_solver.t(torch.tensor(sigma_min)),
            num_steps,
            self.eta,
            self.s_noise,
            None,
        )

    @torch.no_grad()
    def __call__(
        self, model, start, energy, layers, num_steps, sample_offset, debug
    ) -> Any:
        x = self.setup(model, start, num_steps)
        x = self.sample(model, x, num_steps, energy, layers)
        return x, None, None


class DPMAdaptive(DPM):
    def __init__(self, config):
        super().__init__(config)
        self.order = self.sample_config.get("ORDER", 3)
        self.r_tol = self.sample_config.get("R_TOL", 0.05)
        self.a_tol = self.sample_config.get("A_TOL", 0.0078)
        self.h_init = self.sample_config.get("H_INIT", 0.05)
        self.p_coeff = self.sample_config.get("P_COEFF", 0)
        self.i_coeff = self.sample_config.get("I_COEFF", 1.)
        self.d_coeff = self.sample_config.get("d_COEFF", 0)
        self.accept_safety = self.sample_config.get("ACCEPT_SAFETY", 0.81)

    def sample(self, model, x, num_steps, energy, layers):
        """DPM-Solver-12 and 23 (adaptive step size). See https://arxiv.org/abs/2206.00927."""
        sigma_min, sigma_max = self.sigmas[-1], self.sigmas[0]

        if sigma_min <= 0 or sigma_max <= 0:
            raise ValueError("sigma_min and sigma_max must not be 0")
        dpm_solver = sampling.DPMSolver(model, E=energy, layers=layers)
        x, _ = dpm_solver.dpm_solver_adaptive(
            x,
            dpm_solver.t(torch.tensor(sigma_max)),
            dpm_solver.t(torch.tensor(sigma_min)),
            self.order,
            self.r_tol,
            self.a_tol,
            self.h_init,
            self.p_coeff,
            self.i_coeff,
            self.d_coeff,
            self.accept_safety,
            self.eta,
            self.s_noise,
            None,
        )
        return x


class DPMPP2S(DPM):
    @staticmethod
    def sigma_fn(t):
        return t.neg().exp()

    @staticmethod
    def time_fn(t):
        return t.log().neg()

    def sample(self, model, x, num_steps, energy, layers):
        noise_sampler = sampling.default_noise_sampler(x)
        s_in = x.new_ones([x.shape[0]])

        for i in range(len(self.sigmas) - 1):
            denoised = model.denoise(x, sigma=self.sigmas[i] * s_in, E=energy, layers=layers)
            sigma_down, sigma_up = sampling.get_ancestral_step(
                self.sigmas[i], self.sigmas[i + 1], eta=self.eta
            )

            # DPM-Solver++(2S)
            t, t_next = DPMPP2S.time_fn(self.sigmas[i]), DPMPP2S.time_fn(sigma_down)
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (DPMPP2S.sigma_fn(s) / DPMPP2S.sigma_fn(t)) * x - (
                -h * r
            ).expm1() * denoised
            denoised_2 = model.denoise(
                x_2, sigma=DPMPP2S.sigma_fn(s) * s_in, E=energy, layers=layers
            )
            x = (DPMPP2S.sigma_fn(t_next) / DPMPP2S.sigma_fn(t)) * x - (
                -h
            ).expm1() * denoised_2
        # Noise addition
        if self.sigmas[i + 1] > 0:
            x = (
                x
                + noise_sampler(self.sigmas[i], self.sigmas[i + 1])
                * self.s_noise
                * sigma_up
            )
        return x


class DPMPPSDE(DPM):
    def __init__(self, config):
        super().__init__(config)
        self.r = self.sample_config.get("R", 0.5)

    @staticmethod
    def sigma_fn(t):
        return t.neg().exp()

    @staticmethod
    def time_fn(t):
        return t.log().neg()

    def sample(self, model, x, num_steps, energy, layers):
        sigma_min, sigma_max = self.sigmas[self.sigmas > 0].min(), self.sigmas.max()
        noise_sampler = sampling.BrownianTreeNoiseSampler(x, sigma_min, sigma_max)
        s_in = x.new_ones([x.shape[0]])

        for i in range(len(self.sigmas) - 1):
            denoised = model(x, sigma=self.sigmas[i] * s_in, E=energy, layers=layers)

            # DPM-Solver++
            t, t_next = (
                DPMPPSDE.time_fn(self.sigmas[i]),
                DPMPPSDE.time_fn(self.sigmas[i + 1]),
            )
            h = t_next - t
            s = t + h * self.r
            fac = 1 / (2 * self.r)

            # Step 1
            sd, su = sampling.get_ancestral_step(
                DPMPPSDE.sigma_fn(t), DPMPPSDE.sigma_fn(s), self.eta
            )
            s_ = DPMPPSDE.time_fn(sd)
            x_2 = (DPMPPSDE.sigma_fn(s_) / DPMPPSDE.sigma_fn(t)) * x - (
                t - s_
            ).expm1() * denoised
            x_2 = (
                x_2
                + noise_sampler(DPMPPSDE.sigma_fn(t), DPMPPSDE.sigma_fn(s))
                * self.s_noise
                * su
            )
            denoised_2 = model(
                x_2, sigma=DPMPPSDE.sigma_fn(s) * s_in, E=energy, layers=layers
            )

            # Step 2
            sd, su = sampling.get_ancestral_step(
                DPMPPSDE.sigma_fn(t), DPMPPSDE.sigma_fn(t_next), self.eta
            )
            t_next_ = DPMPPSDE.time_fn(sd)
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            x = (DPMPPSDE.sigma_fn(t_next_) / DPMPPSDE.sigma_fn(t)) * x - (
                t - t_next_
            ).expm1() * denoised_d
            x = (
                x
                + noise_sampler(DPMPPSDE.sigma_fn(t), DPMPPSDE.sigma_fn(t_next))
                * self.s_noise
                * su
            )
        return x


class DPMPP2M(DPM):
    @staticmethod
    def sigma_fn(t):
        return t.neg().exp()

    @staticmethod
    def time_fn(t):
        return t.log().neg()

    def sample(self, model, x, num_steps, energy, layers):
        """DPM-Solver++(2M)."""
        s_in = x.new_ones([x.shape[0]])
        old_denoised = None

        for i in range(len(self.sigmas) - 1):
            denoised = model(x, sigma=self.sigmas[i] * s_in, E=energy, layers=layers)

            t, t_next = (
                DPMPP2M.time_fn(self.sigmas[i]),
                DPMPP2M.time_fn(self.sigmas[i + 1]),
            )
            h = t_next - t
            if old_denoised is None or self.sigmas[i + 1] == 0:
                x = (DPMPP2M.sigma_fn(t_next) / DPMPP2M.sigma_fn(t)) * x - (
                    -h
                ).expm1() * denoised
            else:
                h_last = t - DPMPP2M.time_fn(self.sigmas[i - 1])
                r = h_last / h
                denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
                x = (DPMPP2M.sigma_fn(t_next) / DPMPP2M.sigma_fn(t)) * x - (
                    -h
                ).expm1() * denoised_d
            old_denoised = denoised
        return x


class DPMPP2MSDE(DPM):
    def __init__(self, config):
        super().__init__(config)
        self.solver_type = self.sample_config.get("SOLVER", "heun")
        if self.solver_type not in {"heun", "midpoint"}:
            raise ValueError("solver_type must be 'heun' or 'midpoint'")

    def sample(self, model, x, num_steps, energy, layers):
        """DPM-Solver++(2M) SDE."""
        s_in = x.new_ones([x.shape[0]])

        sigma_min, sigma_max = self.sigmas[self.sigmas > 0].min(), self.sigmas.max()
        noise_sampler = sampling.BrownianTreeNoiseSampler(x, sigma_min, sigma_max)

        old_denoised = None
        h_last = None

        for i in range(len(self.sigmas) - 1):
            denoised = model(x, sigma=self.sigmas[i] * s_in, E=energy, layers=layers)

            if self.sigmas[i + 1] == 0:
                # Denoising step
                x = denoised
            else:
                # DPM-Solver++(2M) SDE
                t, s = -self.sigmas[i].log(), -self.sigmas[i + 1].log()
                h = s - t
                eta_h = self.eta * h

                x = (
                    self.sigmas[i + 1] / self.sigmas[i] * (-eta_h).exp() * x
                    + (-h - eta_h).expm1().neg() * denoised
                )

                if old_denoised is not None:
                    r = h_last / h
                    if self.solver_type == "heun":
                        x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (
                            1 / r
                        ) * (denoised - old_denoised)
                    elif self.solver_type == "midpoint":
                        x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (
                            denoised - old_denoised
                        )

                if self.eta:
                    x = (
                        x
                        + noise_sampler(self.sigmas[i], self.sigmas[i + 1])
                        * self.sigmas[i + 1]
                        * (-2 * eta_h).expm1().neg().sqrt()
                        * self.s_noise
                    )

            old_denoised = denoised
            h_last = h
        return x


class DPMPP3MSDE(DPM):
    def sample(self, model, x, num_steps, energy, layers):
        """DPM-Solver++(3M) SDE."""

        sigma_min, sigma_max = self.sigmas[self.sigmas > 0].min(), self.sigmas.max()
        noise_sampler = sampling.BrownianTreeNoiseSampler(x, sigma_min, sigma_max)
        s_in = x.new_ones([x.shape[0]])

        denoised_1, denoised_2 = None, None
        h_1, h_2 = None, None

        for i in range(len(self.sigmas) - 1):
            denoised = model(x, sigma=self.sigmas[i] * s_in, E=energy, layers=layers)
            if self.sigmas[i + 1] == 0:
                # Denoising step
                x = denoised
            else:
                t, s = -self.sigmas[i].log(), -self.sigmas[i + 1].log()
                h = s - t
                h_eta = h * (self.eta + 1)

                x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised

                if h_2 is not None:
                    r0 = h_1 / h
                    r1 = h_2 / h
                    d1_0 = (denoised - denoised_1) / r0
                    d1_1 = (denoised_1 - denoised_2) / r1
                    d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                    d2 = (d1_0 - d1_1) / (r0 + r1)
                    phi_2 = h_eta.neg().expm1() / h_eta + 1
                    phi_3 = phi_2 / h_eta - 0.5
                    x = x + phi_2 * d1 - phi_3 * d2
                elif h_1 is not None:
                    r = h_1 / h
                    d = (denoised - denoised_1) / r
                    phi_2 = h_eta.neg().expm1() / h_eta + 1
                    x = x + phi_2 * d

                x = (
                    x
                    + noise_sampler(self.sigmas[i], self.sigmas[i + 1])
                    * self.sigmas[i + 1]
                    * (-2 * h * self.eta).expm1().neg().sqrt()
                    * self.s_noise
                )

            denoised_1, denoised_2 = denoised, denoised_1
            h_1, h_2 = h, h_1
        return x


class EDMAbstract(ABC, Sample):
    def __init__(self, config) -> None:
        super().__init__(config)
        noisy = self.config.get("NOISY_SAMPLE", False)

        self.S_churn = (
            40 if noisy else 0
        )  # Number of steps to 'reverse' to add back noise
        self.S_min =self.sample_config.get("S_MIN", 0.01)
        self.S_max = 50 if noisy else 1
        self.S_noise = self.sample_config.get("S_NOISE", 1.003)
        self.sigma_min = self.sample_config.get("SIGMA_MIN", 0.002)
        self.sigma_max = self.sample_config.get("SIGMA_MAX", 80.0)
        self.orig_schedule = self.sample_config.get("ORG_SCHEDULE", False)
        self.rho = self.sample_config.get("RHO", 7)
        self.order = self.sample_config.get("ORDER", 4)
        self.restart_gamma = self.sample_config.get("RESTART_GAMMA", 0.05)
        self.C_2 = self.sample_config.get("C2", 0.0008)
        self.C_1 = self.sample_config.get("C1", 0.001)

        self.loop_sampling = self.loop_sample()

        # Reset at setup 
        self.device = None
        self.model = None 
        self.energy = None 
        self.layers = None 
        self.x = None

        self.x_hat = None 
        self.t_hat = None 
        self.x_next = None 
        self.t_next = None 
        self.denoised = None


    def generator_size(self):
        return (self.x.shape[0], *((1,) * (len(self.x.shape) - 1)))

    def loop_sample(self) -> bool:
        loop_sample = True
        try:
            self.in_loop_sampler(**{})
        except NotImplementedError:
            loop_sample = False
        except Exception:
            pass  # Other generalized error from the x_prime, t_prime = None
        return loop_sample

    def in_loop_sampler(self) -> torch.Tensor:
        """
        Make the next sample in the loop. Returns x_next.
        """
        raise NotImplementedError

    def for_loop(
        self, num_steps, t_steps
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        self.x_next = self.x.to(torch.float32) * t_steps[0]
        xs = []
        x0s = []

        # define the vars as class vars: 
            # x_hat
            # t_hat 
            # x_next
            # t_next 
            # denoised


        for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
            x_cur = self.x_next
            self.t_next = t_next
            # Increase noise temporarily.
            gamma = (
                min(self.S_churn / num_steps, np.sqrt(2) - 1)
                if self.S_min <= t_cur <= self.S_max
                else 0
            )
            self.t_hat = torch.as_tensor(t_cur + gamma * t_cur)
            self.x_hat = x_cur + (
                self.t_hat**2 - t_cur**2
            ).sqrt() * self.S_noise * torch.randn_like(x_cur)

            t_hat_full = torch.full(self.generator_size(), self.t_hat, device=self.device)
            self.denoised = self.model.denoise(
                self.x_hat, sigma=t_hat_full, E=self.energy, layers=self.layers
            ).to(torch.float32)

            self.x_next = self.in_loop_sampler()
            xs.append(x_cur)
            x0s.append(self.denoised)

        return self.x_next, xs, x0s

    def alpha_bar(self, j, num_steps):
        return (0.5 * np.pi * j / num_steps / (self.C_2 + 1)).sin() ** 2

    def setup(self, num_steps, sample_offset):
        # Time step discretization from edm paper
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=self.device)

        ###
        # t_steps = get_karras/lu/vp_step()
        ###

        t_steps = (
            self.sigma_max ** (1 / self.rho)
            + step_indices
            / (num_steps - 1)
            * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho

        t_steps = torch.cat(
            [torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]
        )  # t_N = 0
        t_steps = t_steps[sample_offset:]

        if self.orig_schedule:  # use steps from original iDDPM schedule
            M = num_steps  # num steps trained with
            u = torch.zeros(M + 1, dtype=torch.float64, device=self.device)
            for j in torch.arange(M, 0, -1, device=self.device):  # M, ..., 1
                u[j - 1] = (
                    (u[j] ** 2 + 1)
                    / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=self.C_1)
                    - 1
                ).sqrt()
            u_filtered = u[torch.logical_and(u >= self.sigma_min, u <= self.sigma_max)]
            t_steps = u_filtered[
                ((len(u_filtered) - 1) / (num_steps - 1) * step_indices)
                .round()
                .to(torch.int64)
            ]

        return t_steps

    def sampler(self, x, model, t_steps, energy, layers):
        raise NotImplementedError

    @torch.no_grad()
    def __call__(self, model, start, energy, layers, num_steps, sample_offset, debug):
        self.model = model 
        self.energy = energy
        self.layers = layers 

        self.num_steps = num_steps
        self.x = start
        self.device = start.device
        
        time_steps = self.setup(num_steps, sample_offset)
        if self.loop_sampling:
            x, xs, x0s = self.for_loop(
                num_steps, time_steps
            )
        else:
            x, xs, x0s = self.sampler(start, model, time_steps, energy, layers)

        return x, xs, x0s


class LMS(EDMAbstract):
    def sampler(self, x, model, t_steps, energy, layers):
        xs = []
        x0s = []
        t_steps_cpu = t_steps
        ds = []
        x_next = x.to(torch.float32) * t_steps[0]
        
        for i, t_cur in enumerate(t_steps[:-1]):
            x_hat = x_next
            t_hat = torch.as_tensor(t_cur)
            t_hat_full = torch.full(self.generator_size(), t_hat, device=self.device)
            denoised = model.denoise(
                x_hat, sigma=t_hat_full, E=energy, layers=layers
            ).to(torch.float32)
            d_cur = (x_hat - denoised) / t_hat
            ds.append(d_cur)
            if len(ds) > self.order:
                ds.pop(0)
            cur_order = min(i + 1, self.order)
            coeffs = [
                sampling.linear_multistep_coeff(cur_order, t_steps_cpu, i, j)
                for j in range(cur_order)
            ]
            x_next = x_hat + sum(
                coeff * d_cur for coeff, d_cur in zip(coeffs, reversed(ds))
            )

        return x_next, xs, x0s


class Euler(EDMAbstract):
    def in_loop_sampler(
        self,
    ):
        d_cur = (self.x_hat - self.denoised) / self.t_hat
        h = self.t_next - self.t_hat
        return self.x_hat + h * d_cur


class Heun(EDMAbstract):
    def in_loop_sampler(
        self,
    ):
        d_cur = (self.x_hat - self.denoised)/self.t_hat
        h = self.t_next - self.t_hat
        x_prime = self.x_hat +  h * d_cur
        t_prime = self.t_hat +  h
        d_cur = (self.x_hat -self.denoised) / self.t_hat

        t_prime_full = torch.full(
            self.generator_size(), t_prime, device=self.device
        )
        denoised = self.model.denoise(
            x_prime, sigma=t_prime_full, E=self.energy, layers=self.layers
        ).to(torch.float32)
        d_prime = (self.x_next - denoised) / self.t_next
        return self.x_hat + h * (0.5 * d_cur + 0.5 * d_prime)


class DPM2(EDMAbstract):
    def in_loop_sampler(
        self,
    ):
        d_cur = (self.x_hat - self.denoised)/self.t_hat
        h = self.t_next - self.t_hat

        t_mid = self.t_hat.log().lerp(self.t_next.log(), 0.5).exp()
        dt_1 = t_mid - self.t_hat
        x_2 = self.x_hat + d_cur * dt_1
        t_mid_full = torch.full(self.generator_size(), t_mid, device=self.device)
        denoised_2 = self.model.denoise(x_2, sigma=t_mid_full, E=self.energy, layers=self.layers).to(
            torch.float32
        )
        d_2 = (x_2 - denoised_2) / t_mid
        return self.x_hat + h * d_2


class Restart(EDMAbstract):
    def __init__(self, config):
        super().__init__(config)
        default_restart = {"0": [4, 1, 19.35, 40.79], "1": [4, 1, 1.09, 1.92], "2": [4, 4, 0.59, 1.09], "3": [4, 1, 0.30, 0.59], "4": [4, 4, 0.06, 0.30]}
        self.restart_list = self.sample_config.get("RESTART_LIST", default_restart)

    def restart_loop(self, index, x_hat, denoised, t_next, t_steps, t_hat): 

        d_cur = (x_hat - denoised) / t_hat
        h = t_next - t_hat
        # Don't use the originally assigned x_next, this one is derived from x_hat 
        x_next = x_hat + h * d_cur

        # restart sampling, from https://github.com/Newbeeer/diffusion_restart_sampling
        # ================= restart ================== #

        if index + 1 in self.restart_list.keys():
            restart_idx = index + 1

            for _ in range(self.restart_list[restart_idx][1]):
                new_t_steps = sampling.get_karras_step(
                    x=self.x,
                    min_t=t_steps[restart_idx],
                    max_t=self.restart_list[restart_idx][3],
                    num_step=self.restart_list[restart_idx][0],
                    rho=self.rho,
                )
                new_total_step = len(new_t_steps)

                x_next = (
                    x_next
                    + torch.randn_like(x_next)
                    * (new_t_steps[0] ** 2 - new_t_steps[-1] ** 2).sqrt()
                    * self.S_noise
                )

                for j, (t_cur, t_next) in enumerate(
                    zip(new_t_steps[:-1], new_t_steps[1:])
                ):  # 0, ..., N_restart -1
                    x_cur = x_next
                    gamma = self.restart_gamma if self.S_min <= t_cur <= self.S_max else 0
                    t_hat = torch.as_tensor(t_cur + gamma * t_cur)

                    x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * self.S_noise * torch.randn_like(
                        x_cur
                    )

                    t_hat_full = torch.full(self.generator_size(), t_hat, device=self.device)

                    denoised = self.model.denoise(x_hat, sigma=t_hat_full, E=self.energy, layers=self.layers).to(
                        torch.float32
                    )
                    d_cur = (x_hat - denoised) / (t_hat)
                    x_next = x_hat + (t_next - t_hat) * d_cur

                    # Apply 2nd order correction.
                    if (
                        j < new_total_step - 2 or new_t_steps[-1] != 0
                    ):
                        t_next_full = torch.full(self.generator_size(), t_next, device=self.device)
                        denoised = self.model.denoise(
                            x_next, sigma=t_next_full, E=self.energy, layers=self.layers
                        ).to(torch.float32)
                        d_prime = (x_next - denoised) / t_next
                        x_next = x_hat + (t_next - t_hat) * (
                            0.5 * d_cur + 0.5 * d_prime
                        )
        return x_next

    def sampler(self, x, model, t_steps, energy, layers):
        xs = []
        x0s = []
        x_next = self.x.to(torch.float32) * t_steps[0]

        for index, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = torch.as_tensor(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            t_hat_full = torch.full(self.generator_size(), t_hat, device=x.device)
            denoised = model.denoise(x_hat, sigma=t_hat_full, E=energy, layers=layers).to(torch.float32) 

            x_next = self.restart_loop(index, x_hat, denoised, t_next, t_steps, t_hat)
            xs.append(next)
            x0s.append(denoised)

        return x_next, xs, x0s


class Consistency(Sample):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.consis_nsteps = self.config.get("CONSIS_NSTEPS", 100)

    def __call__(
        self, model, start, energy, layers, num_steps, sample_offset, debug
    ) -> Any:
        orig_num_steps = model.nsteps

        model.loss_function.update_step(self.consis_nsteps)
        print("Consis steps %i" % self.consis_nsteps)

        # hardcoded for now...
        sample_idxs = [
            0,
            int(round(self.consis_nsteps * 0.5)),
            int(round(self.consis_nsteps * 0.7)),
            int(round(self.consis_nsteps * 0.9)),
            int(round(self.consis_nsteps * 0.95)),
        ]

        t_all_steps = [
            model.loss_function.sqrt_one_minus_alphas_cumprod[
                self.consis_nsteps - t - 1
            ]
            / model.loss_function.sqrt_alphas_cumprod[self.consis_nsteps - t - 1]
            for t in torch.arange(self.consis_nsteps)
        ]

        if num_steps > 1:
            t_steps = torch.tensor([t_all_steps[i] for i in sample_idxs[:num_steps]])
        else:
            t_steps = torch.tensor([t_all_steps[0]])
        sigmas = torch.cat(
            [torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]
        )  # end point is zero noise

        x, xs, x0s = sampling.sample_consis(
            model,
            start,
            sigmas,
            extra_args={
                "E": energy,
                "layers": layers,
            },
        )
        model.loss_function.update_step(orig_num_steps)
        return x, xs, x0s
