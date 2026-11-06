"""Trajectory Branch - branch a flow sampling run mid-flight.

Two nodes for Anima-family rectified flow:

- Traj Branch Sampler runs a plain flow-Euler generation and, at a
  chosen step, snapshots the latent (the branch point). It outputs
  the finished latent, the branch latent, and the branch sigma.
- Traj Resume Sampler continues from a branch latent. It takes its
  own model / positive / negative, so the second half of the
  trajectory can change checkpoint, prompt or CFG at the branch.

Correctness contract: capture at step k, then resume with the
remaining number of steps and the same shift, reproduces the
uninterrupted run exactly - the resume grid is the tail of the
original linear-shift grid (same formula, inverted at the branch
sigma). Any change of settings at the branch is a deliberate fork.

Euler only for now: momentum / AB2 carry velocity history, which a
branch latent alone does not hold. Resuming those exactly would
need the history exported too.
"""

import torch

import comfy.sample
import comfy.samplers


def flow_linear_shift_sigmas(steps, shift, denoise=1.0):
    """Linear-shift sigma grid, descending, ending at exactly 0."""
    t = torch.linspace(float(denoise), 0.0, int(steps) + 1,
                       dtype=torch.float32)
    s = float(shift)
    sigmas = s * t / (1.0 + (s - 1.0) * t)
    sigmas[-1] = 0.0
    return sigmas


def tail_sigmas(start_sigma, steps, shift):
    """The tail of the linear-shift grid starting at start_sigma.

    Inverts sigma = shift*t / (1 + (shift-1)*t) at start_sigma and
    walks the same parameterisation to 0, so with the original
    remaining step count this reproduces the original grid tail.
    """
    s = float(shift)
    sig = float(start_sigma)
    t_start = sig / (s - (s - 1.0) * sig)
    t = torch.linspace(t_start, 0.0, int(steps) + 1,
                       dtype=torch.float32)
    sigmas = s * t / (1.0 + (s - 1.0) * t)
    sigmas[0] = sig
    sigmas[-1] = 0.0
    return sigmas


def _model_wants_temporal(model):
    dm = getattr(getattr(model, "model", None), "diffusion_model", None)
    mod = type(dm).__module__ if dm is not None else ""
    return "cosmos" in mod or "anima" in mod


def _fix_latent(model, latent_image):
    """Channel-fix the samples tensor; returns (samples, expanded)."""
    samples = comfy.sample.fix_empty_latent_channels(
        model, latent_image["samples"])
    expanded = _model_wants_temporal(model) and samples.ndim == 4
    if expanded:
        samples = samples.unsqueeze(2)
    return samples, expanded


def _euler_loop(model, x, sigmas, extra_args, callback, capture=None):
    """Plain flow Euler. If capture is a dict with 'step', stores a
    clone of the state at that step index (state at sigmas[step],
    before that step runs; step == len(sigmas)-1 stores the final
    state) under capture['x']."""
    total = len(sigmas) - 1
    for i in range(total):
        if capture is not None and capture.get("step") == i:
            capture["x"] = x.clone()
        sigma = sigmas[i]
        denoised = model(x, sigma * x.new_ones([x.shape[0]]),
                         **extra_args)
        if float(sigmas[i + 1]) == 0.0:
            x = denoised
        else:
            v = (x - denoised) / sigma
            x = x + v * (sigmas[i + 1] - sigma)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma,
                      "denoised": denoised})
    if capture is not None and capture.get("step") == total:
        capture["x"] = x.clone()
    return x


class TrajBranchSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0,
                                 "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 0.0,
                                  "max": 30.0, "step": 0.1}),
                "capture_step": ("INT", {"default": 15, "min": 0,
                                         "max": 200}),
                "flow_shift": ("FLOAT", {"default": 3.0, "min": 0.5,
                                         "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "sigmas": ("SIGMAS",),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "FLOAT")
    RETURN_NAMES = ("latent", "branch_latent", "branch_sigma")
    FUNCTION = "sample"
    CATEGORY = "sampling/trajbranch"

    def sample(self, model, positive, negative, latent_image, seed,
               steps, cfg, capture_step, flow_shift, sigmas=None):
        latent_samples, expanded = _fix_latent(model, latent_image)
        noise = comfy.sample.prepare_noise(latent_samples, seed)
        if expanded:
            noise = noise.unsqueeze(2)
        if sigmas is None:
            sigmas = flow_linear_shift_sigmas(steps, flow_shift)
        total = len(sigmas) - 1
        step = max(0, min(int(capture_step), total))
        capture = {"step": step}

        def sampler_function(model_, x, sigs, extra_args=None,
                             callback=None, disable=None, **kwargs):
            return _euler_loop(model_, x, sigs, extra_args or {},
                               callback, capture)

        sampler = comfy.samplers.KSAMPLER(sampler_function)
        print(f"[TrajBranch] capture_step={step}/{total}, "
              f"shift={flow_shift}, steps={steps}, cfg={cfg}")
        samples = comfy.sample.sample_custom(
            model, noise, cfg, sampler, sigmas, positive, negative,
            latent_samples,
            noise_mask=latent_image.get("noise_mask"), callback=None,
            disable_pbar=False, seed=seed)
        branch = capture.get("x")
        if branch is None:  # sampler ended early; fall back to final
            branch = samples
        out = dict(latent_image)
        branch_out = dict(latent_image)
        if expanded:
            samples = samples.squeeze(2)
            branch = branch.squeeze(2)
        out["samples"] = samples
        branch_out["samples"] = branch
        return (out, branch_out, float(sigmas[step]))


class TrajResumeSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_sigma": ("FLOAT", {"default": 0.5, "min": 0.0,
                                          "max": 1.0, "step": 0.01}),
                "steps": ("INT", {"default": 15, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 0.0,
                                  "max": 30.0, "step": 0.1}),
                "flow_shift": ("FLOAT", {"default": 3.0, "min": 0.5,
                                         "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "sigmas": ("SIGMAS",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/trajbranch"

    def sample(self, model, positive, negative, latent_image,
               start_sigma, steps, cfg, flow_shift, sigmas=None):
        if float(start_sigma) <= 0.0:
            return (latent_image,)
        latent_samples, expanded = _fix_latent(model, latent_image)
        start = latent_samples.clone()

        if sigmas is None:
            sigmas = tail_sigmas(start_sigma, steps, flow_shift)

        def sampler_function(model_, x, sigs, extra_args=None,
                             callback=None, disable=None, **kwargs):
            # Ignore the x ComfyUI prepared (fresh noise / blended
            # latent): a resume starts from the branch latent itself.
            x = start.to(x.device, x.dtype)
            return _euler_loop(model_, x, sigs, extra_args or {},
                               callback)

        sampler = comfy.samplers.KSAMPLER(sampler_function)
        print(f"[TrajResume] start_sigma={float(start_sigma):.4f}, "
              f"shift={flow_shift}, steps={steps}, cfg={cfg}")
        noise = comfy.sample.prepare_noise(latent_samples, 0)
        if expanded:
            noise = noise.unsqueeze(2)
        samples = comfy.sample.sample_custom(
            model, noise, cfg, sampler, sigmas, positive, negative,
            latent_samples,
            noise_mask=latent_image.get("noise_mask"), callback=None,
            disable_pbar=False, seed=0)
        out = dict(latent_image)
        if expanded:
            samples = samples.squeeze(2)
        out["samples"] = samples
        return (out,)


NODE_CLASS_MAPPINGS = {
    "TrajBranchSampler": TrajBranchSampler,
    "TrajResumeSampler": TrajResumeSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrajBranchSampler": "Traj Branch Sampler (capture)",
    "TrajResumeSampler": "Traj Resume Sampler (branch)",
}
