# CounterCount: Probing and Mitigating Counting Bias in Vision-Language Models

> **Anonymous submission — do not distribute.**

This repository provides the code for **CounterCount**, a counterfactual counting benchmark and attention-modulation framework for diagnosing and mitigating counting bias in Vision-Language Models (VLMs).

## Overview

VLMs frequently default to canonical world knowledge (e.g., "a rabbit has two ears") instead of grounding predictions in the image. CounterCount exposes this failure mode through paired factual and counterfactual images and provides a training-free attention-modulation strategy that improves counterfactual counting accuracy by up to **8%** across multiple VLMs.

### Key contributions

1. **CounterCount benchmark** — 168 factual–counterfactual image pairs across 10 semantic categories with region-level mask and bounding-box annotations.
2. **Systematic bias analysis** — Evaluation of Qwen3-VL (4B/8B/32B), Gemma3 (4B/12B/27B), and Claude-haiku-4.5 revealing consistent accuracy drops under counterfactual edits.
3. **Unified attention modulation** — A single-parameter logit-shift mechanism that amplifies target visual tokens, dampens background tokens, or masks them entirely during inference.

## Repository structure

```
countercount/
├── src/
│   ├── attention_modulation.py   # Main experiment runner & attention hooks
│   ├── evaluation_utils.py       # Answer extraction, VLM judge, result tables
│   └── masking_utils.py          # Mask/BB position mapping & pixel masking
├── configs/
│   └── image_configurations.json # Attention modulation configurations
├── scripts/
│   ├── run_experiment.sh         # Run single category experiment
│   └── run_all_categories.sh     # Run all 10 categories
├── data/
│   └── README.md                 # Dataset placement instructions
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

### Requirements

- Python ≥ 3.10
- CUDA-capable GPU (tested on NVIDIA A100 80GB)
- ~80GB GPU memory for 32B models; ~24GB for 4B models

### Installation

```bash
git clone <this-repo-url>
cd countercount
pip install -r requirements.txt
```

### Data

Download the CounterCount benchmark from the link provided in the supplementary material and place it under `data/CounterCount/` following the structure described in [`data/README.md`](data/README.md).

## Usage

### Single category experiment

```bash
python src/attention_modulation.py \
    --image_type anomaly \
    --images_dir data/CounterCount/Birds/Anomaly \
    --results_output_dir results/Birds \
    --masks_dir data/CounterCount/Birds/masks \
    --bbox_json_path data/CounterCount/Birds/masks/birds_bbox.json \
    --prompts_path data/CounterCount/Birds/birds_prompts.json \
    --metadata_path data/CounterCount/Birds/birds_metadata.json \
    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
    --prompt_type open_ended
```

Or using the convenience script:

```bash
bash scripts/run_experiment.sh Qwen/Qwen3-VL-8B-Instruct Birds anomaly open_ended
```

### All categories

```bash
bash scripts/run_all_categories.sh Qwen/Qwen3-VL-8B-Instruct anomaly mcq
```

### Key arguments

| Argument | Description | Options |
|---|---|---|
| `--image_type` | Evaluate on factual or counterfactual images | `ordinary`, `anomaly` |
| `--prompt_type` | Question format | `open_ended`, `mcq` |
| `--model_name_or_path` | HuggingFace model identifier | Any Qwen3-VL or Gemma3 model |
| `--configs_path` | Attention configuration file | Default: `configs/image_configurations.json` |

### Supported models

| Family | Tested variants |
|---|---|
| Qwen3-VL | `Qwen/Qwen3-VL-4B-Instruct`, `Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen3-VL-32B-Instruct` |
| Gemma3 | `google/gemma-3-4b-it`, `google/gemma-3-12b-it`, `google/gemma-3-27b-it` |

## Attention modulation

The method modifies pre-softmax attention logits with a single additive correction:

$$\tilde{z}_{h,i,j} = z_{h,i,j} + \log(\alpha)$$

where α > 1 amplifies, 0 < α < 1 dampens, and α = 0 masks the position entirely.

### Configurations

The config file defines operation modes that are automatically expanded with scale sweeps:

- **Target amplification** (`amplify_target`): Boost attention to mask/BB tokens
- **Background dampening** (`amplify_target_dampen_rest`): Boost target + dampen non-target
- **Background masking** (`amplify_target_suppress_rest`): Boost target + zero-out non-target
- **Layer selection**: Apply modulation to `early`, `middle`, `late`, or `all` layers

Default scale sweeps: α ∈ {1.25, 1.5, 1.75, 2.0, 2.5, 3.0}, β ∈ {0.25, 0.5, 0.75}.

## Output

Results are saved per-image and aggregated:

```
results/<Category>/<ModelName>/<image_type>/<prompt_type>/
├── img0/
│   ├── baseline.json
│   ├── amplify_target_bb_1.5x.json
│   └── ...
├── all_configs_aggregated.json    # Full aggregated results
└── summarized_results.txt         # Human-readable summary tables
```

## License

This project is released under the [MIT License](LICENSE).
