"""
methods for sampling on inference

"""

from typing import Any
import torch

from calodiffusion.utils import sampling

class Sample:
    def __init__(self, config, sample_algorithm) -> None:
        self.sample_algorithm = sample_algorithm
        self.config = config 

    def __call__(self, model, start, energy, layers, num_steps, sample_offset, debug) -> Any:
        raise NotImplementedError
    
class DDim(Sample): 
    def __call__(self, model, start, energy, layers, num_steps, sample_offset, debug) -> Any:
        return sampling.sample_dd(
            model,
            start,
            num_steps,
            sample_offset=sample_offset,
            sample_algo= self.sample_algorithm,
            debug=debug,
            extra_args= {
                "E": energy,
                "layers": layers}
        )

class DPM(Sample): 
    def __call__(self, model, start, energy, layers, num_steps, sample_offset, debug) -> Any:
        if model.nsteps != num_steps:
            model.loss_function.update_step(num_steps)

        time_steps = list(range(0, num_steps))
        time_steps.reverse()

        # scale starting point to appropriate noise level
        sigmas = torch.tensor(
            [
                model.loss_function.sqrt_one_minus_alphas_cumprod[num_steps - t - 1]
                / model.loss_function.sqrt_alphas_cumprod[num_steps - t - 1]
                for t in torch.arange(num_steps)
            ]
        )
        sigma_min = sigmas[-1]
        sigma_max = sigmas[0]

        x = start * sigmas[0]
        x = sampling.sample_dpm_fast(
            model,
            x,
            sigma_min,
            sigma_max,
            num_steps,
            extra_args={"E": energy, "layers": layers},
        )

        return x, None, None

class EDM(Sample): 
    def __init__(self, config) -> None:
        super().__init__(config)
        # TODO Define constants from config
        noisy = self.config.get('NOISY_SAMPLE', False)

        self.S_churn = 40 if noisy else 0 # Number of steps to 'reverse' to add back noise
        self.S_min = 0.01
        self.S_max = 50 if noisy else 1
        self.S_noise = 1.003
        self.sigma_min = 0.002
        self.sigma_max = 80.0
        self.orig_schedule = False

    def __call__(self, model, start, energy, layers, num_steps, sample_offset, debug) -> Any:
        x, xs, x0s = sampling.edm_sampler(
                model,
                start,
                energy,
                layers=layers,
                num_steps=num_steps,
                sample_algo= self.sample_algorithm,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                S_churn=self.S_churn,
                S_min=self.S_min,
                S_max=self.S_max,
                S_noise=self.S_noise,
                sample_offset=sample_offset,
                orig_schedule=self.orig_schedule,
                extra_args={
                    "E": energy,
                    "layers": layers,
                    "layer_sample": model,
                    "model": model,
                },
            )
        return x, xs, x0s

class Consistency(Sample): 
    def __init__(self, config) -> None:
        super().__init__(config)
        self.consis_nsteps = self.config.get("CONSIS_NSTEPS", 100)

    def __call__(self, model, start, energy, layers, num_steps, sample_offset, debug) -> Any:
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
            model.loss.sqrt_one_minus_alphas_cumprod[self.consis_nsteps - t - 1]
            / model.loss.sqrt_alphas_cumprod[self.consis_nsteps - t - 1]
            for t in torch.arange(self.consis_nsteps)
        ]

        if num_steps > 1:
            t_steps = torch.tensor(
                [t_all_steps[i] for i in sample_idxs[:num_steps]]
            )
        else:
            t_steps = torch.tensor([t_all_steps[0]])
        sigmas = torch.cat(
            [torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]
        )  # end point is zero noise

        x, xs, x0s = sampling.sample_consis(
            model, start, sigmas, extra_args={
                    "E": energy,
                    "layers": layers,
                    "layer_sample": model,
                    "model": model,
                }
        )
        model.loss_function.update_step(orig_num_steps)    
        return x, xs, x0s