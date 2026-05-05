"""
Unified attention modulation for Vision-Language Models.

Implements inference-time attention interventions that rebalance the contribution
of visual tokens during decoding, supporting amplification, dampening, and masking
of target/background token groups across configurable transformer layers.
"""

import functools
import math
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
import inflect
from PIL import Image
from transformers import (
    Qwen3VLForConditionalGeneration,
    Gemma3ForConditionalGeneration,
    AutoProcessor,
)
from transformers.models.qwen3_vl import modeling_qwen3_vl
from transformers.models.gemma3 import modeling_gemma3
from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv as qwen3_repeat_kv
from transformers.models.gemma3.modeling_gemma3 import repeat_kv as gemma3_repeat_kv

from evaluation_utils import (
    find_target_positions_in_prompt,
    evaluate_counting,
    print_results_from_file,
)
from masking_utils import (
    get_mask_and_bb_positions,
    mask_image_pixels,
    mask_image_pixels_bb,
    mask_image_pixels_bb_full,
)

p = inflect.engine()


# ---------------------------------------------------------------------------
# Configuration expansion utilities
# ---------------------------------------------------------------------------

def expand_configs_with_scales(configs, amplify_scales=None, dampen_factors=None):
    """Expand config templates by enumerating amplify/dampen scale combinations.

    Args:
        configs: List of config dicts with ``name`` and ``config`` keys.
        amplify_scales: Values of α to sweep (default: [1.25, 1.5, 1.75, 2.0, 2.5, 3.0]).
        dampen_factors: Values of β to sweep (default: [0.25, 0.5, 0.75]).

    Returns:
        Expanded list of configs covering all requested (α, β) combinations.
    """
    if amplify_scales is None:
        amplify_scales = [1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    if dampen_factors is None:
        dampen_factors = [0.25, 0.5, 0.75]

    expanded = []
    for config_spec in configs:
        operation_mode = config_spec["config"]["operation_mode"]
        base_name = config_spec["name"]
        base_config = config_spec["config"]

        needs_amplify = "amplify" in operation_mode
        needs_dampen = "dampen" in operation_mode

        if needs_amplify and needs_dampen:
            for amp in amplify_scales:
                for damp in dampen_factors:
                    expanded.append({
                        "name": f"{base_name}_{amp}x_{damp}x",
                        "config": {**base_config, "amplify_scale": amp, "dampen_factor": damp},
                    })
        elif needs_amplify:
            for amp in amplify_scales:
                expanded.append({
                    "name": f"{base_name}_{amp}x",
                    "config": {**base_config, "amplify_scale": amp},
                })
        elif needs_dampen:
            for damp in dampen_factors:
                expanded.append({
                    "name": f"{base_name}_{damp}x",
                    "config": {**base_config, "dampen_factor": damp},
                })
        else:
            expanded.append(config_spec)

    return expanded


# ---------------------------------------------------------------------------
# Custom eager attention forwards
# ---------------------------------------------------------------------------

def _resolve_positions(all_positions, target_positions, dampen_positions,
                       suppress_positions, operation_mode):
    """Determine which token positions to amplify / dampen / suppress."""
    pos_amplify, pos_dampen, pos_suppress = [], [], []

    if all_positions is None:
        return pos_amplify, pos_dampen, pos_suppress

    mode_map = {
        "amplify_all": lambda: (all_positions, [], []),
        "amplify_target": lambda: (target_positions or [], [], []),
        "amplify_target_dampen_rest": lambda: (
            target_positions or [],
            [p for p in all_positions if p not in (target_positions or [])],
            [],
        ),
        "amplify_target_dampen_specific": lambda: (
            target_positions or [], dampen_positions or [], [],
        ),
        "amplify_target_suppress_rest": lambda: (
            target_positions or [],
            [],
            [p for p in all_positions if p not in (target_positions or [])],
        ),
        "amplify_target_suppress_specific": lambda: (
            target_positions or [], [], suppress_positions or [],
        ),
        "dampen_all": lambda: ([], all_positions, []),
        "dampen_rest": lambda: (
            [],
            [p for p in all_positions if p not in (target_positions or [])],
            [],
        ),
        "dampen_specific": lambda: ([], dampen_positions or [], []),
        "suppress_rest": lambda: (
            [],
            [],
            [p for p in all_positions if p not in (target_positions or [])],
        ),
        "suppress_specific": lambda: ([], [], suppress_positions or []),
        "suppress_target": lambda: ([], [], target_positions or []),
    }

    if operation_mode not in mode_map:
        raise ValueError(
            f"Unknown operation_mode: '{operation_mode}'. "
            f"Valid modes: {', '.join(['baseline'] + list(mode_map.keys()))}"
        )

    return mode_map[operation_mode]()


def _should_modify_layer(current_layer, total_layers, modify_layers):
    """Return True if *current_layer* falls within the requested layer group."""
    if modify_layers is None:
        return True
    third = total_layers // 3
    if modify_layers == "early":
        return current_layer < third
    if modify_layers == "middle":
        return third <= current_layer < 2 * third
    if modify_layers == "late":
        return current_layer >= 2 * third
    return True


def _apply_logit_modulation(attn_weights, query_idx, positions_to_amplify,
                            positions_to_dampen, positions_to_suppress,
                            amplify_scale, dampen_factor, should_modify,
                            current_layer, current_token, log_file):
    """Apply additive logit shifts (amplify / dampen) and masking (suppress)."""
    if not should_modify:
        return attn_weights

    do_log = (current_layer == 0 and current_token == 0 and log_file is not None)

    if do_log:
        with open(log_file, "a") as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"LAYER {current_layer} | AMPLIFY: {amplify_scale} | DAMPEN: {dampen_factor}\n")
            f.write(f"#AMPLIFY: {len(positions_to_amplify)} | #DAMPEN: {len(positions_to_dampen)} | #SUPPRESS: {len(positions_to_suppress)}\n")

    if positions_to_amplify:
        attn_weights[:, :, query_idx, positions_to_amplify] += math.log(amplify_scale)

    if positions_to_dampen and dampen_factor is not None:
        attn_weights[:, :, query_idx, positions_to_dampen] += math.log(dampen_factor)

    if positions_to_suppress:
        attn_weights[:, :, query_idx, positions_to_suppress] = float("-inf")

    return attn_weights


# ---- Qwen3-VL ----

def qwen3vl_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    all_positions: list = None,
    target_positions: list = None,
    dampen_positions: list = None,
    suppress_positions: list = None,
    amplify_scale: float = 2.0,
    dampen_factor: float = None,
    operation_mode: str = "baseline",
    modify_layers: str = None,
    total_layers: int = 36,
    log_file: str = None,
    **kwargs,
):
    """Custom eager attention forward for Qwen3-VL with logit modulation."""
    repeat_kv = qwen3_repeat_kv

    if not isinstance(module, modeling_qwen3_vl.Qwen3VLTextAttention) or operation_mode == "baseline":
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    # Layer / token tracking
    if not hasattr(qwen3vl_eager_attention_forward, "layer_counter"):
        qwen3vl_eager_attention_forward.layer_counter = 0
    current_call = qwen3vl_eager_attention_forward.layer_counter
    current_layer = current_call % total_layers
    current_token = current_call // total_layers
    qwen3vl_eager_attention_forward.layer_counter += 1

    should_modify = _should_modify_layer(current_layer, total_layers, modify_layers)
    query_idx = slice(None)

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    pos_amp, pos_damp, pos_sup = _resolve_positions(
        all_positions, target_positions, dampen_positions,
        suppress_positions, operation_mode,
    )

    attn_weights = _apply_logit_modulation(
        attn_weights, query_idx, pos_amp, pos_damp, pos_sup,
        amplify_scale, dampen_factor, should_modify,
        current_layer, current_token, log_file,
    )

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


# ---- Gemma3 ----

def gemma3_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    all_positions: list = None,
    target_positions: list = None,
    dampen_positions: list = None,
    suppress_positions: list = None,
    amplify_scale: float = 2.0,
    dampen_factor: float = None,
    operation_mode: str = "baseline",
    modify_layers: str = None,
    total_layers: int = 48,
    log_file: str = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Custom eager attention forward for Gemma3 with logit modulation."""
    repeat_kv = gemma3_repeat_kv

    if scaling is None:
        scaling = module.head_dim ** -0.5
    softcap = getattr(module, "attn_logit_softcapping", None)

    if operation_mode == "baseline":
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if softcap is not None:
            attn_weights = torch.tanh(attn_weights / softcap) * softcap
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    if not hasattr(gemma3_eager_attention_forward, "layer_counter"):
        gemma3_eager_attention_forward.layer_counter = 0
    current_call = gemma3_eager_attention_forward.layer_counter
    current_layer = current_call % total_layers
    current_token = current_call // total_layers
    gemma3_eager_attention_forward.layer_counter += 1

    should_modify = _should_modify_layer(current_layer, total_layers, modify_layers)
    query_idx = slice(None)

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = torch.tanh(attn_weights / softcap) * softcap

    pos_amp, pos_damp, pos_sup = _resolve_positions(
        all_positions, target_positions, dampen_positions,
        suppress_positions, operation_mode,
    )

    attn_weights = _apply_logit_modulation(
        attn_weights, query_idx, pos_amp, pos_damp, pos_sup,
        amplify_scale, dampen_factor, should_modify,
        current_layer, current_token, log_file,
    )

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

class AttentionExperimentRunner:
    """Run attention-modulation experiments across images and configurations."""

    def __init__(self, model, processor, metadata, prompts, config,
                 bbox_json=None, masks_dir=None, experiment_type="image"):
        self.model = model
        self.processor = processor
        self.metadata = metadata
        self.prompts = prompts
        self.config = config
        self.bbox_json = bbox_json
        self.masks_dir = masks_dir
        self.experiment_type = experiment_type
        self.image_type = config.get("image_type", "anomaly")
        self.current_image_index = 0

        # Determine grid size from model architecture
        if isinstance(self.model, Qwen3VLForConditionalGeneration):
            self.grid_size = 15
        elif isinstance(self.model, Gemma3ForConditionalGeneration):
            self.grid_size = 16
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

        # Count attention layers
        if isinstance(self.model, Qwen3VLForConditionalGeneration):
            self.total_layers = sum(
                1 for m in self.model.modules()
                if isinstance(m, modeling_qwen3_vl.Qwen3VLTextAttention)
            )
        elif isinstance(self.model, Gemma3ForConditionalGeneration):
            self.total_layers = sum(
                1 for m in self.model.modules()
                if isinstance(m, modeling_gemma3.Gemma3Attention)
            )
        print(f"Total attention layers: {self.total_layers}")

        Path(config["results_output_dir"]).mkdir(parents=True, exist_ok=True)

        if self.experiment_type == "image":
            self.config["attention_configs"] = self._expand_image_configs(
                self.config["attention_configs"]
            )

    # ---- Config expansion ----

    def _expand_image_configs(self, configs):
        """Duplicate region-targeted configs into mask / bb / bb_mask variants."""
        no_target_modes = {"baseline", "amplify_all", "dampen_all"}
        expanded = []
        for spec in configs:
            op = spec["config"]["operation_mode"]
            if spec["config"].get("_pixel_mask", False) or op in no_target_modes:
                expanded.append(spec)
            else:
                regions = ["bb"] if self.image_type == "ordinary" else ["mask", "bb", "bb_mask"]
                for r in regions:
                    expanded.append({
                        "name": f"{spec['name']}_{r}",
                        "config": {**spec["config"], "_region_type": r},
                    })
        return expanded

    # ---- Position extraction ----

    def get_all_positions(self, inputs, metadata_info):
        """Extract image and text token positions from tokenized input."""
        input_ids = inputs["input_ids"][0].cpu().tolist()
        vocab = self.processor.tokenizer.get_vocab()

        image_positions, text_positions = [], []

        if isinstance(self.model, Qwen3VLForConditionalGeneration):
            vs = vocab.get("<|vision_start|>", vocab.get("<|image_pad|>"))
            ve = vocab.get("<|vision_end|>")
            ie = vocab.get("<|im_end|>")
            if vs and vs in input_ids:
                si = input_ids.index(vs)
                if ve and ve in input_ids:
                    ei = input_ids.index(ve)
                    image_positions = list(range(si + 1, ei))
                    assert len(image_positions) == self.grid_size ** 2
                    if ie and ie in input_ids[ei:]:
                        text_positions = list(range(ei + 1, input_ids.index(ie, ei)))
                    else:
                        text_positions = list(range(ei + 1, len(input_ids)))

        elif isinstance(self.model, Gemma3ForConditionalGeneration):
            vs = vocab.get("<start_of_image>")
            ve = vocab.get("<end_of_image>")
            ie = vocab.get("<end_of_turn>")
            if vs in input_ids and ve in input_ids:
                si = input_ids.index(vs)
                ei = input_ids.index(ve)
                image_positions = list(range(si + 1, ei))
                assert len(image_positions) == self.grid_size ** 2
                if ie and ie in input_ids[ei:]:
                    text_positions = list(range(ei + 1, input_ids.index(ie, ei)))
                else:
                    text_positions = list(range(ei + 1, len(input_ids)))

        positions = {
            "image_positions": image_positions,
            "text_positions": text_positions,
            "target_positions": [],
            "suppress_positions": [],
            "dampen_positions": [],
        }

        if self.experiment_type == "text":
            amplify_words = {
                metadata_info["anomaly"]: [
                    metadata_info["anomaly"],
                    p.plural(metadata_info["anomaly"]),
                ]
            }
            amp_dict = find_target_positions_in_prompt(
                input_ids, self.processor.tokenizer, amplify_words, text_positions,
            )
            positions["target_positions"] = [
                pos for pl in amp_dict.values() for pos in pl
            ]

            suppress_words = {metadata_info["name"]: [metadata_info["name"]]}
            sup_dict = find_target_positions_in_prompt(
                input_ids, self.processor.tokenizer, suppress_words, text_positions,
            )
            name_pos = [pos for pl in sup_dict.values() for pos in pl]
            positions["suppress_positions"] = name_pos
            positions["dampen_positions"] = name_pos

        return positions

    def get_image_target_positions(self, image_id, image_positions):
        """Get mask, BB, and Mask-BB target positions for an image."""
        json_key = None
        for key, entry in self.bbox_json.items():
            if entry["img_name"] == image_id:
                json_key = key
                break
        if json_key is None:
            raise ValueError(f"No bbox entry for image_id='{image_id}'")

        import os
        mask_path = os.path.join(
            self.masks_dir, f"{self.bbox_json[json_key]['img_name']}_mask.png"
        )
        return get_mask_and_bb_positions(
            mask_path, self.bbox_json[json_key], image_positions,
            grid_size=self.grid_size, threshold=0.1,
        )

    # ---- Logging ----

    def should_enable_logging(self):
        if not self.config.get("enable_attention_logging", False):
            return False
        if self.config.get("log_first_image_only", True):
            return self.current_image_index == 0
        return True

    # ---- Inference ----

    def run_inference(self, inputs, attention_config, positions,
                      mask_positions=None, bb_positions=None,
                      bb_mask_positions=None):
        """Run a single inference pass with the given attention configuration."""
        start_time = time.time()
        cfg = attention_config.copy()

        if self.experiment_type == "text":
            cfg["all_positions"] = positions["text_positions"]
            cfg["target_positions"] = positions["target_positions"]
            cfg["suppress_positions"] = positions["suppress_positions"]
            cfg["dampen_positions"] = positions["dampen_positions"]
        elif self.experiment_type == "image":
            cfg["all_positions"] = positions["image_positions"]
            region = attention_config.get("_region_type")
            if region == "mask":
                cfg["target_positions"] = mask_positions
            elif region == "bb":
                cfg["target_positions"] = bb_positions
            elif region == "bb_mask":
                cfg["target_positions"] = bb_mask_positions

        cfg.pop("_region_type", None)
        cfg.pop("_pixel_mask", None)
        cfg["total_layers"] = self.total_layers
        cfg["log_file"] = (
            self.config.get("log_file") if self.should_enable_logging() else None
        )

        if isinstance(self.model, Qwen3VLForConditionalGeneration):
            modeling_qwen3_vl.eager_attention_forward = functools.partial(
                qwen3vl_eager_attention_forward, **cfg,
            )
            qwen3vl_eager_attention_forward.layer_counter = 0
        elif isinstance(self.model, Gemma3ForConditionalGeneration):
            modeling_gemma3.eager_attention_forward = functools.partial(
                gemma3_eager_attention_forward, **cfg,
            )
            gemma3_eager_attention_forward.layer_counter = 0

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config["max_new_tokens"],
                do_sample=self.config.get("do_sample", False),
                temperature=self.config.get("temperature", 0.7),
                output_attentions=False,
                return_dict_in_generate=True,
            )

        generated_ids = outputs.sequences if hasattr(outputs, "sequences") else outputs
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, generated_ids)]
        text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        return text[0], time.time() - start_time

    # ---- Prompt processing ----

    def process_counting_prompts(self, image_id, image_path, img_metadata,
                                 attention_config, results,
                                 mask_positions=None, bb_positions=None,
                                 bb_mask_positions=None):
        """Evaluate counting accuracy for an image under a given config."""
        import os

        ordinary_num = img_metadata["ordinary_number"]
        anomaly_num = img_metadata["anomaly_number"]
        expected = ordinary_num if self.image_type == "ordinary" else anomaly_num

        results["counting"] = {}
        base_prompt = self.prompts[image_id]["prompt"]

        if self.config["prompt_type"] == "mcq":
            options_str = ", ".join(self.prompts[image_id]["options"])
            prompt = (
                f"{base_prompt} "
                f"{self.prompts[image_id]['instruction']['mcq']} {options_str}. "
                f"Reply with only one word from the given options and nothing else."
            )
        else:
            prompt = f"{base_prompt} {self.prompts[image_id]['instruction']['open_ended']}"

        use_pixel_mask = (
            self.experiment_type == "image"
            and attention_config.get("_pixel_mask", False)
        )

        if use_pixel_mask:
            variants = ["bb"] if self.image_type == "ordinary" else ["mask", "bb", "bb_mask"]
            json_key = next(
                k for k, v in self.bbox_json.items() if v["img_name"] == image_id
            )
            all_bbox_coords = [
                b["scaled_selection"]
                for b in self.bbox_json[json_key][f"grid_{self.grid_size}x{self.grid_size}"]
            ]
            if self.image_type != "ordinary":
                mask_path = os.path.join(
                    self.masks_dir,
                    f"{self.bbox_json[json_key]['img_name']}_mask.png",
                )
        else:
            variants = [None]

        for variant in variants:
            vkey = "counting" if variant is None else f"counting_{variant}"

            if variant == "mask":
                input_image = mask_image_pixels(image_path, mask_path, fill_value=0)
            elif variant == "bb":
                input_image = image_path
                for bc in all_bbox_coords:
                    input_image = mask_image_pixels_bb_full(input_image, bc, fill_value=0)
            elif variant == "bb_mask":
                input_image = image_path
                for bc in all_bbox_coords:
                    input_image = mask_image_pixels_bb(input_image, mask_path, bc, fill_value=0)
            else:
                input_image = image_path

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": input_image},
                    {"type": "text", "text": prompt},
                ],
            }]
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(self.model.device)

            positions = self.get_all_positions(inputs, img_metadata)
            if self.experiment_type == "image" and mask_positions is None:
                mask_positions, bb_positions, bb_mask_positions = (
                    self.get_image_target_positions(image_id, positions["image_positions"])
                )

            results["counting"][vkey] = {
                "prompt": prompt,
                "expected_answer": expected,
                "runs": [],
            }
            if variant is not None:
                results["counting"][vkey]["variant"] = variant

            for run_id in range(self.config["num_runs_per_prompt"]):
                try:
                    response, dt = self.run_inference(
                        inputs, attention_config, positions,
                        mask_positions, bb_positions, bb_mask_positions,
                    )
                    is_correct, is_biased, parsed = evaluate_counting(
                        response=response,
                        question=prompt,
                        ordinary_number=ordinary_num,
                        anomaly_number=anomaly_num,
                        correct_answer=expected,
                        model=self.model,
                        processor=self.processor,
                        qwen3vl_attn_forward=qwen3vl_eager_attention_forward,
                        gemma3_attn_forward=gemma3_eager_attention_forward,
                    )
                    if self.image_type == "ordinary":
                        is_correct = is_biased
                        is_biased = False
                    run_result = {
                        "run_id": run_id,
                        "response": response,
                        "parsed_answer": parsed,
                        "is_correct": is_correct,
                        "is_biased": is_biased,
                        "inference_time": dt,
                        "error": None,
                    }
                except Exception as e:
                    run_result = {
                        "run_id": run_id,
                        "response": "FAILED",
                        "parsed_answer": None,
                        "is_correct": False,
                        "is_biased": False,
                        "inference_time": None,
                        "error": str(e),
                    }
                results["counting"][vkey]["runs"].append(run_result)

            runs = results["counting"][vkey]["runs"]
            valid = [r for r in runs if r["response"] != "FAILED"]
            if valid:
                accs = [float(r["is_correct"]) for r in valid]
                biases = [float(r["is_biased"]) for r in valid]
                results["counting"][vkey]["accuracy"] = np.mean(accs)
                results["counting"][vkey]["bias_rate"] = np.mean(biases)
                results["counting"][vkey]["accuracy_std"] = np.std(accs) if len(accs) > 1 else 0.0
                results["counting"][vkey]["bias_std"] = np.std(biases) if len(biases) > 1 else 0.0
            else:
                results["counting"][vkey].update(
                    accuracy=0.0, bias_rate=0.0, accuracy_std=0.0, bias_std=0.0,
                )

        return results

    # ---- Single image ----

    def process_single_image(self, image_id, attention_config, config_name):
        img_meta = self.metadata[image_id]
        image_path = str(Path(self.config["images_dir"]) / f"{image_id}.png")
        print(f"  Processing {image_id}: {img_meta['name']} | config: {config_name}")

        results = {
            "image_id": image_id,
            "config_name": config_name,
            "image_metadata": img_meta,
            "results": {},
        }
        results["results"] = self.process_counting_prompts(
            image_id, image_path, img_meta, attention_config, results["results"],
        )
        return results

    # ---- Experiment loop ----

    def run_experiment(self, image_ids=None):
        if image_ids is None:
            image_ids = list(self.metadata.keys())

        print(f"\n{'=' * 80}")
        print(f"RUNNING ATTENTION EXPERIMENTS ({self.experiment_type.upper()})")
        print(f"Images: {len(image_ids)} | Configs: {len(self.config['attention_configs'])} "
              f"| Runs/prompt: {self.config['num_runs_per_prompt']}")
        print(f"{'=' * 80}\n")

        if self.config.get("enable_attention_logging") and self.config.get("log_file"):
            with open(self.config["log_file"], "w") as f:
                f.write(f"ATTENTION LOG | {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")

        per_image = {}
        for idx, img_id in tqdm(enumerate(image_ids), total=len(image_ids), desc="Images"):
            self.current_image_index = idx
            img_dir = Path(self.config["results_output_dir"]) / img_id
            img_dir.mkdir(parents=True, exist_ok=True)
            per_image[img_id] = {}

            for ci, spec in enumerate(self.config["attention_configs"], 1):
                cname = spec["name"]
                res = self.process_single_image(img_id, spec["config"], cname)
                per_image[img_id][cname] = res
                with open(img_dir / f"{cname}.json", "w") as f:
                    json.dump(res, f, indent=2)

        # Aggregate
        agg = {}
        for spec in self.config["attention_configs"]:
            cn = spec["name"]
            all_res = [per_image[iid][cn] for iid in image_ids]
            agg[cn] = {
                "aggregated_metrics": self.aggregate_results(all_res, cn),
                "per_image_results": all_res,
            }

        out_path = Path(self.config["results_output_dir"]) / "all_configs_aggregated.json"
        with open(out_path, "w") as f:
            json.dump(agg, f, indent=2)

        print(f"\nResults saved to: {out_path}")
        return agg

    # ---- Aggregation ----

    def aggregate_results(self, all_results, config_name):
        ct_types = list(all_results[0]["results"].get("counting", {}).keys())
        metrics = {ct: {"acc": [], "bias": [], "acc_std": [], "bias_std": []} for ct in ct_types}

        for r in all_results:
            for ct in ct_types:
                if ct in r["results"].get("counting", {}):
                    d = r["results"]["counting"][ct]
                    metrics[ct]["acc"].append(d["accuracy"])
                    metrics[ct]["bias"].append(d["bias_rate"])
                    metrics[ct]["acc_std"].append(d["accuracy_std"])
                    metrics[ct]["bias_std"].append(d["bias_std"])

        summary = {
            "config_name": config_name,
            "num_images": len(all_results),
            "aggregated_accuracy": {"counting": {}},
            "aggregated_bias_rate": {"counting": {}},
            "aggregated_std": {"counting": {"accuracy": {}, "bias": {}}},
        }
        for ct in ct_types:
            if metrics[ct]["acc"]:
                summary["aggregated_accuracy"]["counting"][ct] = np.mean(metrics[ct]["acc"])
                summary["aggregated_bias_rate"]["counting"][ct] = np.mean(metrics[ct]["bias"])
                summary["aggregated_std"]["counting"]["accuracy"][ct] = np.mean(metrics[ct]["acc_std"])
                summary["aggregated_std"]["counting"]["bias"][ct] = np.mean(metrics[ct]["bias_std"])
        return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="CounterCount: Attention modulation for VLM counting bias",
    )
    parser.add_argument("--image_type", type=str, default="anomaly",
                        choices=["anomaly", "ordinary"])
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--results_output_dir", type=str, required=True)
    parser.add_argument("--masks_dir", type=str, required=True)
    parser.add_argument("--bbox_json_path", type=str, required=True)
    parser.add_argument("--prompts_path", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--num_runs_per_prompt", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--enable_attention_logging", action="store_true", default=True)
    parser.add_argument("--log_first_image_only", action="store_true", default=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="./model_cache")
    parser.add_argument("--prompt_type", type=str, default="open_ended",
                        choices=["open_ended", "mcq"])
    parser.add_argument("--configs_path", type=str,
                        default="configs/image_configurations.json",
                        help="Path to attention configuration JSON file")
    return parser.parse_args()


def load_model(model_name, cache_dir):
    """Load model and processor based on model name."""
    if "qwen" in model_name.lower():
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, dtype="auto", device_map="auto",
            cache_dir=cache_dir, attn_implementation="eager",
        ).eval()
    elif "gemma" in model_name.lower():
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name, device_map="auto", dtype=torch.bfloat16,
            cache_dir=cache_dir, attn_implementation="eager",
        ).eval()
    else:
        raise ValueError("Model name must contain 'Qwen' or 'Gemma'.")

    processor = AutoProcessor.from_pretrained(
        model_name, cache_dir=cache_dir,
        **({} if "qwen" in model_name.lower() else {"use_fast": True}),
    )
    return model, processor


def main():
    args = parse_args()

    model, processor = load_model(args.model_name_or_path, args.cache_dir)

    with open(args.prompts_path) as f:
        prompts_data = json.load(f)
    with open(args.metadata_path) as f:
        metadata_data = json.load(f)
    with open(args.configs_path) as f:
        configs = json.load(f)
    with open(args.bbox_json_path) as f:
        bbox_json = json.load(f)

    expanded_configs = expand_configs_with_scales(configs)

    results_dir = str(
        Path(args.results_output_dir)
        / Path(args.model_name_or_path).name
        / args.image_type
        / args.prompt_type
    )

    experiment_config = {
        "image_type": args.image_type,
        "images_dir": args.images_dir,
        "num_runs_per_prompt": args.num_runs_per_prompt,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "results_output_dir": results_dir,
        "log_file": str(Path(results_dir) / "experiment_debug.log"),
        "enable_attention_logging": args.enable_attention_logging,
        "log_first_image_only": args.log_first_image_only,
        "attention_configs": expanded_configs,
        "prompt_type": args.prompt_type,
    }

    runner = AttentionExperimentRunner(
        model=model,
        processor=processor,
        metadata=metadata_data,
        prompts=prompts_data,
        config=experiment_config,
        bbox_json=bbox_json,
        masks_dir=args.masks_dir,
        experiment_type="image",
    )

    runner.run_experiment()

    print_results_from_file(
        json_path=str(Path(results_dir) / "all_configs_aggregated.json"),
        output_file=str(Path(results_dir) / "summarized_results.txt"),
    )


if __name__ == "__main__":
    main()
