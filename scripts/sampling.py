import numpy as np
import torch
import torch.nn as nn
import math
import torchsde


from scipy import integrate

def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)

def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up

def get_karras_step(num_step, min_t, max_t, rho=7):

    step_indices = torch.arange(num_step, dtype=torch.float32)
    t_steps = (max_t ** (1 / rho) + step_indices / (num_step - 1) * (min_t ** (1 / rho) - max_t ** (1 / rho))) ** rho
    return t_steps

def get_lu_step(num_step, min_t, max_t, rho=1):

    step_indices = torch.arange(num_step, dtype=torch.float32)
    lambda_min=np.log(min_t)
    lambda_max=np.log(max_t)
    t_steps = (lambda_max ** (1 / rho) + step_indices / (num_step - 1) * (lambda_min ** (1 / rho) - lambda_max ** (1 / rho))) ** rho
    return t_steps

def get_vp_step(num_step, eps_s=1e-3, beta_d = 19.9, beta_min = 0.1)
    t1 = torch.linspace(1, eps_s, num_step)
    t_steps = torch.sqrt(torch.exp(beta_d * t1 ** 2 / 2 + beta_min * t1) - 1)
    return t_steps

#util function for LMS, order default = 4         
            
def linear_multistep_coeff(order, t, i, j):
    if order - 1 > i:
        raise ValueError(f'Order {order} too high for step {i}')
    def fn(tau):
        prod = 1.
        for k in range(order):
            if j == k:
                continue
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod
    return integrate.quad(fn, t[i], t[i + 1], epsrel=1e-4)[0]
    
##########

def edm_sampler( model, x, E, layers = None, sample_algo = 'euler', randn_like=torch.randn_like, num_steps=400, sigma_min=0.002, sigma_max=1, rho=7,
    S_churn=0, S_min=0, S_max=1.0, S_noise=1,sample_offset = 0, order=4, 
    restart_info='{"0": [4, 1, 19.35, 40.79], "1": [4, 1, 1.09, 1.92], "2": [4, 4, 0.59, 1.09], "3": [4, 1, 0.30, 0.59], "4": [4, 4, 0.06, 0.30]}', restart_gamma=0.05, 
    orig_schedule = False, layer_sample = False):
    #EDM sampler (and variations), adapted from  https://github.com/NVlabs/edm


    gen_size = x.shape[0]

    # Time step discretization from edm paper
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=x.device)
    
    ###
    #t_steps = get_karras/lu/vp_step()
    ###
    
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    t_steps = t_steps[sample_offset:]


    if(orig_schedule): #use steps from original iDDPM schedule 
        M = num_steps # num steps trained with
        C_2 = 0.0008
        C_1 = 0.001
        u = torch.zeros(M + 1, dtype=torch.float64, device=x.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=x.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        t_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)] 


    #print(t_steps)

    # Main sampling loop.
    x_next = x.to(torch.float32) * t_steps[0]
    t_next = t_steps[0]
    
    if (sample_algo == 'lms'):
        t_steps_cpu = t_steps
        ds = []
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_hat = x_next
            t_hat = torch.as_tensor(t_cur)
            t_hat_full = torch.full((gen_size,), t_hat, device=x.device)
            denoised = model.denoise(x_hat, E, t_hat_full, layers = layers, layer_pred = layer_sample).to(torch.float32)
            d_cur = (x_hat - denoised) / t_hat
            ds.append(d_cur)
            if len(ds) > order:
                ds.pop(0)
            cur_order = min(i + 1, order)
            coeffs = [linear_multistep_coeff(cur_order, t_steps_cpu, i, j) for j in range(cur_order)]
            x_next = x_hat + sum(coeff * d_cur for coeff, d_cur in zip(coeffs, reversed(ds)))
    
    else:
        xs = []
        x0s = []
        # {[num_steps, number of restart iteration (K), t_min, t_max], ... }
        #some option
        #multi level for imagenet {"0": [3, 1, 19.35, 40.79], "1": [4, 1, 1.09, 1.92], "2": [4, 4, 0.59, 1.09], "3": [4, 1, 0.30, 0.59], "4": [4, 4, 0.06, 0.30]}
        #single level for cifar-10 {"0": [3, 2, 0.14, 0.30]}
        import json
        print(restart_info)
        restart_list = json.loads(restart_info) if restart_info != '' else {}
        # cast t_min to the index of nearest value in t_steps
        restart_list = {int(torch.argmin(abs(t_steps - v[2]), dim=0)): v for k, v in restart_list.items()}

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = torch.as_tensor(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

            t_hat_full = torch.full((gen_size,), t_hat, device=x.device)
            denoised = model.denoise(x_hat, E, t_hat_full, layers = layers, layer_pred = layer_sample).to(torch.float32) 
        
            d_cur = (x_hat - denoised)/t_hat
            h = t_next - t_hat
            x_prime = x_hat +  h * d_cur
            t_prime = t_hat +  h

            #euler step
            if 'euler' in sample_algo or i == num_steps - sample_offset - 1:
                x_next = x_hat + h * d_cur
            elif 'dpm2' in sample_algo and (i < num_steps - 1):
                t_mid = t_hat.log().lerp(t_next.log(), 0.5).exp()
                dt_1 = t_mid - t_hat
                x_2 = x_hat + d_cur * dt_1
                t_mid_full = torch.full((gen_size,), t_mid, device=x.device)
                denoised_2 = model.denoise(x_2, E, t_mid_full, layers = layers, layer_pred = layer_sample).to(torch.float32)
                d_2 = (x_2 - denoised_2) / t_mid
                x_next = x_hat + h * d_2
            elif (sample_algo == 'restart'):

                # restart sampling, from https://github.com/Newbeeer/diffusion_restart_sampling

                # ================= restart ================== #
                if i + 1 in restart_list.keys():
                    restart_idx = i + 1

                    for restart_iter in range(restart_list[restart_idx][1]):

                        new_t_steps = get_karras_step(min_t=t_steps[restart_idx], max_t=restart_list[restart_idx][3], num_step=restart_list[restart_idx][0], rho=rho, x=x)
                        #print(f"restart at {restart_idx} with {new_t_steps}")
                        new_total_step = len(new_t_steps)

                        x_next = x_next + randn_like(x_next) * (new_t_steps[0] ** 2 - new_t_steps[-1] ** 2).sqrt() * S_noise


                        for j, (t_cur, t_next) in enumerate(zip(new_t_steps[:-1], new_t_steps[1:])):  # 0, ..., N_restart -1

                            x_cur = x_next
                            gamma = restart_gamma if S_min <= t_cur <= S_max else 0
                            t_hat = torch.as_tensor(t_cur + gamma * t_cur)

                            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)


                            t_hat_full = torch.full((gen_size,), t_hat, device=x.device)


                            denoised = model.denoise(x_hat, E, t_hat_full, layers = layers, layer_pred = layer_sample).to(torch.float32)
                            d_cur = (x_hat - denoised) / (t_hat)
                            x_next = x_hat + (t_next - t_hat) * d_cur

                            # Apply 2nd order correction.
                            if (sample_algo == 'restart') and (j < new_total_step - 2 or new_t_steps[-1] != 0):
                                t_next_full = torch.full((gen_size,), t_next, device=x.device)
                                denoised = model.denoise(x_next, E, t_next_full, layers = layers, layer_pred = layer_sample).to(torch.float32)
                                d_prime = (x_next - denoised) / t_next
                                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            else:
                # 2nd order correction.
                assert ('heun' in sample_algo or 'edm' in sample_algo)
                t_prime_full = torch.full((gen_size,), t_prime, device = x.device)
                denoised = model.denoise(x_prime, E, t_prime_full,layers = layers, layer_pred = layer_sample).to(torch.float32)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + h * (0.5 * d_cur + 0.5 * d_prime)

            #print(i, torch.mean(x_cur), torch.mean(denoised))
            xs.append(x_cur)
            x0s.append(denoised)


    return x_next, xs, x0s



#copied from k-diffusion repo 
#https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py

class BatchedBrownianTree:
    """A wrapper around torchsde.BrownianTree that enables batches of entropy."""

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get('w0', torch.zeros_like(x))
        if seed is None:
            seed = torch.randint(0, 2 ** 63 - 1, []).item()
        self.batched = True
        try:
            assert len(seed) == x.shape[0]
            w0 = w0[0]
        except TypeError:
            seed = [seed]
            self.batched = False
        self.trees = [torchsde.BrownianTree(t0, w0, t1, entropy=s, **kwargs) for s in seed]

    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self.sort(t0, t1)
        w = torch.stack([tree(t0, t1) for tree in self.trees]) * (self.sign * sign)
        return w if self.batched else w[0]


class BrownianTreeNoiseSampler:
    """A noise sampler backed by a torchsde.BrownianTree.

    Args:
        x (Tensor): The tensor whose shape, device and dtype to use to generate
            random samples.
        sigma_min (float): The low end of the valid interval.
        sigma_max (float): The high end of the valid interval.
        seed (int or List[int]): The random seed. If a list of seeds is
            supplied instead of a single integer, then the noise sampler will
            use one BrownianTree per batch item, each with its own seed.
        transform (callable): A function that maps sigma to the sampler's
            internal timestep.
    """

    def __init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x):
        self.transform = transform
        t0, t1 = self.transform(torch.as_tensor(sigma_min)), self.transform(torch.as_tensor(sigma_max))
        self.tree = BatchedBrownianTree(x, t0, t1, seed)

    def __call__(self, sigma, sigma_next):
        t0, t1 = self.transform(torch.as_tensor(sigma)), self.transform(torch.as_tensor(sigma_next))
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()

class DPMSolver(nn.Module):
    """DPM-Solver. See https://arxiv.org/abs/2206.00927."""

    def __init__(self, model, extra_args=None, eps_callback=None, info_callback=None):
        super().__init__()
        self.model = model
        self.extra_args = {} if extra_args is None else extra_args
        self.eps_callback = eps_callback
        self.info_callback = info_callback

    def t(self, sigma):
        return -sigma.log()

    def sigma(self, t):
        return t.neg().exp()

    def eps(self, eps_cache, key, x, t, *args, **kwargs):
        if key in eps_cache:
            return eps_cache[key], eps_cache
        sigma = self.sigma(t) * x.new_ones([x.shape[0]])
        eps = (x - self.model.denoise(x, sigma=sigma, *args, **self.extra_args, **kwargs)) / self.sigma(t)
        if self.eps_callback is not None:
            self.eps_callback()
        return eps, {key: eps, **eps_cache}

    def dpm_solver_1_step(self, x, t, t_next, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        x_1 = x - self.sigma(t_next) * h.expm1() * eps
        return x_1, eps_cache

    def dpm_solver_2_step(self, x, t, t_next, r1=1 / 2, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        s1 = t + r1 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1)
        x_2 = x - self.sigma(t_next) * h.expm1() * eps - self.sigma(t_next) / (2 * r1) * h.expm1() * (eps_r1 - eps)
        return x_2, eps_cache

    def dpm_solver_3_step(self, x, t, t_next, r1=1 / 3, r2=2 / 3, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        s1 = t + r1 * h
        s2 = t + r2 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1)
        u2 = x - self.sigma(s2) * (r2 * h).expm1() * eps - self.sigma(s2) * (r2 / r1) * ((r2 * h).expm1() / (r2 * h) - 1) * (eps_r1 - eps)
        eps_r2, eps_cache = self.eps(eps_cache, 'eps_r2', u2, s2)
        x_3 = x - self.sigma(t_next) * h.expm1() * eps - self.sigma(t_next) / r2 * (h.expm1() / h - 1) * (eps_r2 - eps)
        return x_3, eps_cache

    def dpm_solver_fast(self, x, t_start, t_end, nfe, eta=0., s_noise=1., noise_sampler=None):
        noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
        if not t_end > t_start and eta:
            raise ValueError('eta must be 0 for reverse sampling')

        m = math.floor(nfe / 3) + 1
        ts = torch.linspace(t_start, t_end, m + 1, device=x.device)

        if nfe % 3 == 0:
            orders = [3] * (m - 2) + [2, 1]
        else:
            orders = [3] * (m - 1) + [nfe % 3]

        for i in range(len(orders)):
            eps_cache = {}
            t, t_next = ts[i], ts[i + 1]
            if eta:
                sd, su = get_ancestral_step(self.sigma(t), self.sigma(t_next), eta)
                t_next_ = torch.minimum(t_end, self.t(sd))
                su = (self.sigma(t_next) ** 2 - self.sigma(t_next_) ** 2) ** 0.5
            else:
                t_next_, su = t_next, 0.

            eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
            denoised = x - self.sigma(t) * eps
            if self.info_callback is not None:
                self.info_callback({'x': x, 'i': i, 't': ts[i], 't_up': t, 'denoised': denoised})

            if orders[i] == 1:
                x, eps_cache = self.dpm_solver_1_step(x, t, t_next_, eps_cache=eps_cache)
            elif orders[i] == 2:
                x, eps_cache = self.dpm_solver_2_step(x, t, t_next_, eps_cache=eps_cache)
            else:
                x, eps_cache = self.dpm_solver_3_step(x, t, t_next_, eps_cache=eps_cache)

            x = x + su * s_noise * noise_sampler(self.sigma(t), self.sigma(t_next))

        return x

    def dpm_solver_adaptive(self, x, t_start, t_end, order=3, rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81, eta=0., s_noise=1., noise_sampler=None):
        noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
        if order not in {2, 3}:
            raise ValueError('order should be 2 or 3')
        forward = t_end > t_start
        if not forward and eta:
            raise ValueError('eta must be 0 for reverse sampling')
        h_init = abs(h_init) * (1 if forward else -1)
        atol = torch.tensor(atol)
        rtol = torch.tensor(rtol)
        s = t_start
        x_prev = x
        accept = True
        pid = PIDStepSizeController(h_init, pcoeff, icoeff, dcoeff, 1.5 if eta else order, accept_safety)
        info = {'steps': 0, 'nfe': 0, 'n_accept': 0, 'n_reject': 0}

        while s < t_end - 1e-5 if forward else s > t_end + 1e-5:
            eps_cache = {}
            t = torch.minimum(t_end, s + pid.h) if forward else torch.maximum(t_end, s + pid.h)
            if eta:
                sd, su = get_ancestral_step(self.sigma(s), self.sigma(t), eta)
                t_ = torch.minimum(t_end, self.t(sd))
                su = (self.sigma(t) ** 2 - self.sigma(t_) ** 2) ** 0.5
            else:
                t_, su = t, 0.

            eps, eps_cache = self.eps(eps_cache, 'eps', x, s)
            denoised = x - self.sigma(s) * eps

            if order == 2:
                x_low, eps_cache = self.dpm_solver_1_step(x, s, t_, eps_cache=eps_cache)
                x_high, eps_cache = self.dpm_solver_2_step(x, s, t_, eps_cache=eps_cache)
            else:
                x_low, eps_cache = self.dpm_solver_2_step(x, s, t_, r1=1 / 3, eps_cache=eps_cache)
                x_high, eps_cache = self.dpm_solver_3_step(x, s, t_, eps_cache=eps_cache)
            delta = torch.maximum(atol, rtol * torch.maximum(x_low.abs(), x_prev.abs()))
            error = torch.linalg.norm((x_low - x_high) / delta) / x.numel() ** 0.5
            accept = pid.propose_step(error)
            if accept:
                x_prev = x_low
                x = x_high + su * s_noise * noise_sampler(self.sigma(s), self.sigma(t))
                s = t
                info['n_accept'] += 1
            else:
                info['n_reject'] += 1
            info['nfe'] += order
            info['steps'] += 1

            if self.info_callback is not None:
                self.info_callback({'x': x, 'i': info['steps'] - 1, 't': s, 't_up': s, 'denoised': denoised, 'error': error, 'h': pid.h, **info})

        return x, info


@torch.no_grad()
def sample_dpm_fast(model, x, sigma_min, sigma_max, n, extra_args=None, eta=0., s_noise=1., noise_sampler=None):
    """DPM-Solver-Fast (fixed step size). See https://arxiv.org/abs/2206.00927."""
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError('sigma_min and sigma_max must not be 0')
    dpm_solver = DPMSolver(model, extra_args)
    return dpm_solver.dpm_solver_fast(x, dpm_solver.t(torch.tensor(sigma_max)), dpm_solver.t(torch.tensor(sigma_min)), n, eta, s_noise, noise_sampler)

@torch.no_grad()
def sample_dpm_adaptive(model, x, sigma_min, sigma_max, extra_args=None, callback=None, disable=None, order=3, rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81, eta=0., s_noise=1., noise_sampler=None, return_info=False):
    """DPM-Solver-12 and 23 (adaptive step size). See https://arxiv.org/abs/2206.00927."""
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError('sigma_min and sigma_max must not be 0')
    dpm_solver = DPMSolver(model, extra_args)
    x, info = dpm_solver.dpm_solver_adaptive(x, dpm_solver.t(torch.tensor(sigma_max)), dpm_solver.t(torch.tensor(sigma_min)), order, rtol, atol, h_init, pcoeff, icoeff, dcoeff, accept_safety, eta, s_noise, noise_sampler)
    if return_info:
        return x, info
    return x


@torch.no_grad()
def sample_dpmpp_2s_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """Ancestral sampling with DPM-Solver++(2S) second-order steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in range(len(sigmas) - 1):
        denoised = model(x, sigma=sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigma_down == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised
            denoised_2 = model(x_2, sigma=sigma_fn(s) * s_in, **extra_args)
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2
        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


@torch.no_grad()
def sample_dpmpp_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, r=1 / 2):
    """DPM-Solver++ (stochastic)."""
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in range(len(sigmas) - 1):
        denoised = model(x, sigma=sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigmas[i + 1] - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            s = t + h * r
            fac = 1 / (2 * r)

            # Step 1
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(s), eta)
            s_ = t_fn(sd)
            x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - (t - s_).expm1() * denoised
            x_2 = x_2 + noise_sampler(sigma_fn(t), sigma_fn(s)) * s_noise * su
            denoised_2 = model(x_2, sigma=sigma_fn(s) * s_in, **extra_args)

            # Step 2
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
            t_next_ = t_fn(sd)
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            x = (sigma_fn(t_next_) / sigma_fn(t)) * x - (t - t_next_).expm1() * denoised_d
            x = x + noise_sampler(sigma_fn(t), sigma_fn(t_next)) * s_noise * su
    return x


@torch.no_grad()
def sample_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM-Solver++(2M)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in range(len(sigmas) - 1):
        denoised = model(x, sigma=sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x


@torch.no_grad()
def sample_dpmpp_2m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, solver_type='midpoint'):
    """DPM-Solver++(2M) SDE."""

    if solver_type not in {'heun', 'midpoint'}:
        raise ValueError('solver_type must be \'heun\' or \'midpoint\'')

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    old_denoised = None
    h_last = None

    for i in range(len(sigmas) - 1):
        denoised = model(x, sigma=sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++(2M) SDE
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h

            x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x + (-h - eta_h).expm1().neg() * denoised

            if old_denoised is not None:
                r = h_last / h
                if solver_type == 'heun':
                    x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - old_denoised)
                elif solver_type == 'midpoint':
                    x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (denoised - old_denoised)

            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise

        old_denoised = denoised
        h_last = h
    return x


@torch.no_grad()
def sample_dpmpp_3m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """DPM-Solver++(3M) SDE."""

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2 = None, None
    h_1, h_2 = None, None

    for i in range(len(sigmas) - 1):
        denoised = model(x, sigma=sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)

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

            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x


@torch.no_grad()
def sample_unipc(model, x, sigmas, use_corrector = False, x_t=None, variants = 'bh', order = 1, extra_args=None, callback=None, disable=None ):
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None
    h_last = None

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    for i in range(len(sigmas) - 1):
        denoised = model(x, sigma=sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])

        h = t_next - t
                
        rks = []
        D1s = []
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
                    
            rks.append(r)
            D1s.append((denoised - old_denoised) / r)
            rks.append(1.)
            rks = torch.tensor(rks, device=x.device)

            if variants == 'vc':
                K = len(rks)
                # build C matrix
                C = []

                col = torch.ones_like(rks)
                for k in range(1, K + 1):
                    C.append(col)
                    col = col * rks / (k + 1) 
                C = torch.stack(C, dim=1)


                if len(D1s) > 0:
                    D1s = torch.stack(D1s, dim=1) # (B, K)
                    C_inv_p = torch.linalg.inv(C[:-1, :-1])
                    A_p = C_inv_p

                if use_corrector:
                    C_inv = torch.linalg.inv(C)
                    A_c = C_inv

                hh = -h
                h_phi_1 = torch.expm1(hh)
                h_phi_ks = []
                factorial_k = 1
                h_phi_k = h_phi_1
                for k in range(1, K + 2):
                    h_phi_ks.append(h_phi_k)
                    h_phi_k = h_phi_k / hh - 1 / factorial_k
                    factorial_k *= (k + 1)




                denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
                x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d

                # now predictor
                if len(D1s) > 0:
                    # compute the residuals for predictor

                    for k in range(K - 1):
                        x = x - h_phi_ks[k + 1] * torch.einsum('bkcthw,k->bcthw', D1s, A_p[k])
                # now corrector
                if use_corrector:
                    denoised_t = model.denoise(x, sigma=sigmas[i] * s_in, **extra_args)
                    D1_t = (denoised_t - old_denoised)
                    k = 0
                    for k in range(K - 1):
                        x_t = x - h_phi_ks[k + 1] * torch.einsum('bkcthw,k->bcthw', D1s, A_c[k][:-1])
                    x = x_t - h_phi_ks[K] * (D1_t * A_c[k][-1])
            if variants == 'bh':
                R = []
                b = []

                hh = -h
                h_phi_1 = torch.expm1(hh) # h\phi_1(h) = e^h - 1
                h_phi_k = h_phi_1 / hh - 1

                factorial_i = 1


                B_h = torch.expm1(hh)
                
                for i in range(1, 3):
                    R.append(torch.pow(rks, i - 1))
                    b.append(h_phi_k * factorial_i / B_h)
                    factorial_i *= (i + 1)
                    h_phi_k = h_phi_k / hh - 1 / factorial_i 

                R = torch.stack(R)
                #b = torch.cat(b)

                # now predictor
                #use_predictor = len(D1s) > 0 and x_t is None
                if len(D1s) > 0:
                    D1s = torch.stack(D1s, dim=1) # (B, K)
                    rhos_p = torch.tensor([0.5], device=x.device)
                        
                else:
                    D1s = None
                    
                if use_corrector:
                    print('using corrector')
                    # for order 1, we use a simplified version
                    if order == 1:
                        rhos_c = torch.tensor([0.5], device=b.device)
                    else:
                        rhos_c = torch.linalg.solve(R, b)
                        
                        
                denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
                x  = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
                
                
                if len(D1s) > 0:
                    pred_res = torch.einsum('k,bkcthw->bcthw', rhos_p, D1s)
                else:
                    pred_res = 0

                x = x - B_h * pred_res

        old_denoised = denoised
        h_last = h
    return x

            




def sample_consis(model, x, sigmas = None, extra_args = None, sigma_min = 0.002):

    gen_size = x_start.shape[0]
    x = x_start * t_steps[0]
    
    x0s = []
    xs = []


    for i, (sigma_cur, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])): # 0, ..., N-1

        x0 = model.denoise(x, sigma_cur, **extra_args).to(torch.float32) 

        sigma_next = torch.clip(sigma_next, sigma_min, None)
        if(sigma_min.item() > sigma_min):
            noise = torch.randn_like(x)
            x = x0 + noise * torch.sqrt(sigma_next**2 - sigma_min**2)
        else: x = x0

        x0s.append(x0)
        xs.append(x)

    return x,xs,x0



