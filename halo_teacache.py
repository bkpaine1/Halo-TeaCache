"""
Halo-TeaCache: AMD Unified Memory TeaCache for LTX2 (LTXAV)

Created by: bkpaine1 & Claude Code (Anthropic Claude Opus 4.5)
License: MIT
Version: 1.0.0

Lean, mean, AMD-only TeaCache for LTX2 audio-video generation.
No CPU offload, no device toggling - unified memory means the cache
stays on GPU where it belongs.

For the AMD community.
"""

import torch
import math
import comfy.ldm.common_dit
import comfy.model_management as mm
from unittest.mock import patch as mock_patch

# ============================================================================
# CORE TEACACHE LOGIC
# ============================================================================

# LTXV polynomial coefficients (baseline for LTXAV - same transformer architecture)
LTXAV_COEFFICIENTS = [2.14700694e+01, -1.28016453e+01, 2.31279151e+00, 7.92487521e-01, 9.69274326e-03]


def poly1d(coefficients, x):
    """Evaluate polynomial using Horner's method"""
    result = 0
    for coeff in coefficients:
        result = result * x + coeff
    return result


def halo_teacache_process_blocks(
    self, x, context, attention_mask, timestep, pe, transformer_options={}, **kwargs
):
    """
    Cached _process_transformer_blocks for LTXAV.

    Strategy: Use video timestep embedding to detect when consecutive denoising
    steps produce similar outputs. When they do, skip all 48 transformer layers
    and reuse the cached residual. Both video and audio streams are cached together
    since they're coupled through cross-attention.
    """
    rel_l1_thresh = transformer_options.get("halo_rel_l1_thresh")
    coefficients = transformer_options.get("halo_coefficients")
    cond_or_uncond = transformer_options.get("cond_or_uncond")
    enable_teacache = transformer_options.get("halo_enable_teacache", True)

    # If TeaCache isn't configured, run normally
    if rel_l1_thresh is None or coefficients is None or cond_or_uncond is None:
        return self._original_process_transformer_blocks(
            x, context, attention_mask, timestep, pe, transformer_options, **kwargs
        )

    vx = x[0]
    ax = x[1]
    v_timestep = timestep[0]
    batch_size = vx.shape[0]

    # Compute modulated input from video stream for cache comparison
    # Uses first transformer block's scale_shift_table + video timestep
    num_ada_params = self.transformer_blocks[0].scale_shift_table.shape[0]
    ada_values = (
        self.transformer_blocks[0].scale_shift_table[None, None].to(
            device=v_timestep.device, dtype=v_timestep.dtype
        )
        + v_timestep.reshape(batch_size, v_timestep.size(1), num_ada_params, -1)
    )
    shift_msa, scale_msa = ada_values.unbind(dim=2)[:2]
    modulated_inp = comfy.ldm.common_dit.rms_norm(vx) * (1 + scale_msa) + shift_msa

    # Initialize cache state if needed
    if not hasattr(self, 'halo_cache_state'):
        self.halo_cache_state = {
            0: {
                'should_calc': True,
                'accumulated_rel_l1_distance': 0,
                'previous_modulated_input': None,
                'previous_v_residual': None,
                'previous_a_residual': None,
            },
            1: {
                'should_calc': True,
                'accumulated_rel_l1_distance': 0,
                'previous_modulated_input': None,
                'previous_v_residual': None,
                'previous_a_residual': None,
            },
        }

    # Update cache state per cond/uncond slice
    b = int(len(vx) / len(cond_or_uncond))

    for i, k in enumerate(cond_or_uncond):
        cache = self.halo_cache_state[k]
        chunk = modulated_inp[i * b : (i + 1) * b]

        if cache['previous_modulated_input'] is not None:
            try:
                rel_diff = (
                    (chunk - cache['previous_modulated_input']).abs().mean()
                    / cache['previous_modulated_input'].abs().mean()
                )
                cache['accumulated_rel_l1_distance'] += poly1d(coefficients, rel_diff)
                if cache['accumulated_rel_l1_distance'] < rel_l1_thresh:
                    cache['should_calc'] = False
                else:
                    cache['should_calc'] = True
                    cache['accumulated_rel_l1_distance'] = 0
            except Exception:
                cache['should_calc'] = True
                cache['accumulated_rel_l1_distance'] = 0

        cache['previous_modulated_input'] = chunk

    # Determine if we should compute or use cache
    if enable_teacache:
        should_calc = any(self.halo_cache_state[k]['should_calc'] for k in cond_or_uncond)
    else:
        should_calc = True

    if not should_calc:
        # Skip all 48 transformer layers - use cached residuals
        for i, k in enumerate(cond_or_uncond):
            cache = self.halo_cache_state[k]
            if cache['previous_v_residual'] is not None:
                vx[i * b : (i + 1) * b] = (
                    vx[i * b : (i + 1) * b] + cache['previous_v_residual']
                )
            if cache['previous_a_residual'] is not None:
                ax[i * b : (i + 1) * b] = (
                    ax[i * b : (i + 1) * b] + cache['previous_a_residual']
                )
        return [vx, ax]
    else:
        # Full computation - run all transformer blocks
        orig_vx = vx.clone()
        orig_ax = ax.clone()

        result = self._original_process_transformer_blocks(
            x, context, attention_mask, timestep, pe, transformer_options, **kwargs
        )

        # Cache the residuals (output - input) for both streams
        for i, k in enumerate(cond_or_uncond):
            self.halo_cache_state[k]['previous_v_residual'] = (
                result[0][i * b : (i + 1) * b] - orig_vx[i * b : (i + 1) * b]
            )
            self.halo_cache_state[k]['previous_a_residual'] = (
                result[1][i * b : (i + 1) * b] - orig_ax[i * b : (i + 1) * b]
            )

        return result


# ============================================================================
# COMFYUI NODE
# ============================================================================

class HaloTeaCache:
    """
    Halo-TeaCache: AMD Unified Memory TeaCache for LTX2

    Caches transformer block outputs when consecutive denoising steps
    produce similar results. Skips all 48 transformer layers on cache hits.
    Both video and audio streams cached together (coupled via cross-attention).

    No CPU offload. Unified memory. For the AMD community.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The LTX2 diffusion model to accelerate."}),
                "rel_l1_thresh": ("FLOAT", {
                    "default": 0.20,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Cache threshold - higher = more skipping (faster but lower quality). 0 = disabled. Try 0.15-0.30.",
                }),
                "start_percent": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Start caching after this % of steps (early steps need full compute).",
                }),
                "end_percent": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Stop caching after this % of steps.",
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_teacache"
    CATEGORY = "Halo-TeaCache"
    TITLE = "Halo-TeaCache (AMD LTX2)"

    def apply_teacache(self, model, rel_l1_thresh: float, start_percent: float, end_percent: float):
        if rel_l1_thresh == 0:
            return (model,)

        new_model = model.clone()
        diffusion_model = new_model.get_model_object("diffusion_model")

        # Verify this is an LTXAV model (LTX2)
        model_class = diffusion_model.__class__.__name__
        if model_class not in ("LTXAVModel", "LTXVModel"):
            print(f"[Halo-TeaCache] WARNING: Expected LTXAVModel or LTXVModel, got {model_class}")
            print(f"[Halo-TeaCache] Proceeding anyway - may not work correctly")

        # Store original _process_transformer_blocks
        if not hasattr(diffusion_model, '_original_process_transformer_blocks'):
            diffusion_model._original_process_transformer_blocks = (
                diffusion_model._process_transformer_blocks
            )

        # Create the patched method bound to this instance
        patched_fn = halo_teacache_process_blocks.__get__(diffusion_model, diffusion_model.__class__)

        # Use mock.patch to cleanly swap the method during inference
        context = mock_patch.multiple(
            diffusion_model,
            _process_transformer_blocks=patched_fn,
        )

        coefficients = LTXAV_COEFFICIENTS

        def unet_wrapper_function(model_function, kwargs):
            input = kwargs["input"]
            timestep = kwargs["timestep"]
            c = kwargs["c"]

            # Step counting from sigmas
            sigmas = c["transformer_options"]["sample_sigmas"]
            matched_step_index = (sigmas == timestep[0]).nonzero()
            if len(matched_step_index) > 0:
                current_step_index = matched_step_index.item()
            else:
                current_step_index = 0
                for i in range(len(sigmas) - 1):
                    if (sigmas[i] - timestep[0]) * (sigmas[i + 1] - timestep[0]) <= 0:
                        current_step_index = i
                        break

            # Reset cache state at step 0
            if current_step_index == 0:
                if hasattr(diffusion_model, 'halo_cache_state'):
                    if (diffusion_model.halo_cache_state[0]['previous_modulated_input'] is not None and
                        diffusion_model.halo_cache_state[1]['previous_modulated_input'] is not None):
                        delattr(diffusion_model, 'halo_cache_state')

            # Determine if caching is active for this step
            current_percent = current_step_index / max(1, (len(sigmas) - 1))
            if start_percent <= current_percent <= end_percent:
                c["transformer_options"]["halo_enable_teacache"] = True
            else:
                c["transformer_options"]["halo_enable_teacache"] = False

            # Pass config through transformer_options
            c["transformer_options"]["halo_rel_l1_thresh"] = rel_l1_thresh
            c["transformer_options"]["halo_coefficients"] = coefficients

            with context:
                return model_function(input, timestep, **c)

        new_model.set_model_unet_function_wrapper(unet_wrapper_function)

        print(f"[Halo-TeaCache] v{__version__} Active: thresh={rel_l1_thresh:.2f}, "
              f"range={start_percent:.0%}-{end_percent:.0%}")
        print(f"[Halo-TeaCache] AMD unified memory - no CPU offload, cache stays on GPU")

        return (new_model,)


# ============================================================================
# NODE REGISTRATION
# ============================================================================

__version__ = "1.0.0"
__author__ = "bkpaine1 & Claude Code"

NODE_CLASS_MAPPINGS = {
    "HaloTeaCache": HaloTeaCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HaloTeaCache": f"Halo-TeaCache v{__version__} (AMD LTX2)",
}

print(f"[Halo-TeaCache] v{__version__} Loaded - AMD unified memory TeaCache for LTX2 by bkpaine1 & Claude Code")
