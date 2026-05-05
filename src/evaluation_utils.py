import json
import inflect
import re
from typing import Dict, List, Tuple, Optional
from prettytable import PrettyTable
import numpy as np
from word2number import w2n
from transformers import Qwen3VLForConditionalGeneration, Gemma3ForConditionalGeneration
import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl
from transformers.models.gemma3 import modeling_gemma3
import functools

p = inflect.engine()


def find_target_positions_in_prompt(
    input_ids: list,
    tokenizer,
    target_words: dict,
    text_positions: list
) -> dict:

    def clean_token(token: str) -> str:
        cleaned = token
        if cleaned.startswith('Ġ'):
            cleaned = cleaned[1:]
        if cleaned.startswith('▁'):
            cleaned = cleaned[1:]
        if cleaned.startswith('##'):
            cleaned = cleaned[2:]
        return cleaned.lower().strip()

    # Decode text tokens to get their string representations
    text_tokens = []
    for pos in text_positions:
        token_str = tokenizer.decode([input_ids[pos]])
        text_tokens.append((pos, token_str))

    word_positions = {}

    for word_name, variants in target_words.items():
        positions = []
        clean_variants = [v.lower().strip() for v in variants]

        if '_' in word_name:
            expected_words = word_name.split('_')

            for i in range(len(text_tokens) - len(expected_words) + 1):
                matched_positions = []
                current_idx = i
                word_idx = 0

                while word_idx < len(expected_words) and current_idx < len(text_tokens):
                    pos, token_str = text_tokens[current_idx]
                    token_clean = clean_token(token_str)
                    expected_word = expected_words[word_idx].lower()

                    if token_clean == expected_word or token_clean.startswith(expected_word):
                        matched_positions.append(pos)
                        word_idx += 1
                        current_idx += 1
                    elif token_clean == 's' and len(matched_positions) > 0:
                        matched_positions.append(pos)
                        current_idx += 1
                    elif word_idx == 0:
                        break
                    else:
                        current_idx += 1
                        if current_idx - i > 3:
                            break

                if word_idx == len(expected_words):
                    positions.extend(matched_positions)
                    matched_tokens = [tokenizer.decode([input_ids[p]]) for p in matched_positions]
                    print(f"Found phrase '{word_name}' at positions: {matched_positions} → tokens: {matched_tokens}")
                    break
        else:
            # First pass: try single-token match (original logic)
            for pos, token_str in text_tokens:
                token_clean = clean_token(token_str)
                if any(token_clean == v or token_clean.startswith(v) for v in clean_variants):
                    positions.append(pos)

            # Second pass: if single-token match failed, try concatenating adjacent tokens
            if not positions:
                for variant in clean_variants:
                    for i in range(len(text_tokens)):
                        concatenated = ""
                        matched_positions = []

                        for j in range(i, len(text_tokens)):
                            pos, token_str = text_tokens[j]
                            token_clean = clean_token(token_str)
                            concatenated += token_clean
                            matched_positions.append(pos)

                            # Exact match found
                            if concatenated == variant:
                                positions.extend(matched_positions)
                                matched_tokens = [tokenizer.decode([input_ids[p]]) for p in matched_positions]
                                print(f"Found '{word_name}' (multi-token) at positions: {matched_positions} → tokens: {matched_tokens}")
                                break

                            # Concatenation already exceeds the variant, no point continuing
                            if len(concatenated) >= len(variant):
                                break

                    # Stop trying other variants once we found a match
                    if positions:
                        break

        if positions:
            word_positions[word_name] = positions
            if '_' not in word_name:
                sample_tokens = [tokenizer.decode([input_ids[p]]) for p in positions[:5]]
                print(f"Found '{word_name}' at {len(positions)} position(s): {positions[:5]} → tokens: {sample_tokens}")
        else:
            print(f"WARNING: '{word_name}' not found in prompt text")

    return word_positions


def extract_all_numbers(response: str) -> list:
    """Extract all numbers (digits or words) from response with their positions."""
    response_lower = response.lower().strip()

    number_words = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
        'eighteen': '18', 'nineteen': '19', 'twenty': '20'
    }

    matches = []

    # Find digit matches
    for match in re.finditer(r'\b(\d+)\b', response_lower):
        digit = match.group(1)
        if digit in {v for v in number_words.values()}:  # only care about small numbers
            matches.append((match.start(), digit))

    # Find word matches
    for word, digit in number_words.items():
        for match in re.finditer(r'\b' + word + r'\b', response_lower):
            matches.append((match.start(), digit))

    # Sort by position and deduplicate (same position = same match)
    matches = sorted(matches, key=lambda x: x[0])

    # Remove duplicates at same position (e.g. "two" matching both word and digit)
    seen_positions = set()
    unique_matches = []
    for pos, num in matches:
        if pos not in seen_positions:
            seen_positions.add(pos)
            unique_matches.append(num)

    return unique_matches

def extract_yes_no_from_response(response: str) -> Optional[str]:
    """
    Extract yes/no from response.

    Returns:
        "yes" or "no" or None if neither found
    """
    response_lower = response.lower().strip()

    # Look for "yes" or "no" as whole words
    if re.search(r'\byes\b', response_lower):
        return "yes"
    elif re.search(r'\bno\b', response_lower):
        return "no"

    return None

def vlm_judge_extract(
    response: str,
    question: str,
    correct_answer: str,
    model,
    processor,
    qwen3vl_attn_forward=None,
    gemma3_attn_forward=None
) -> Optional[str]:
    """Use model as judge to extract the answer number when regex is ambiguous."""

    # Reset attention to baseline for judge call
    if isinstance(model, Qwen3VLForConditionalGeneration) and qwen3vl_attn_forward is not None:
        modeling_qwen3_vl.eager_attention_forward = functools.partial(
            qwen3vl_attn_forward, operation_mode="baseline"
        )
    elif isinstance(model, Gemma3ForConditionalGeneration) and gemma3_attn_forward is not None:
        modeling_gemma3.eager_attention_forward = functools.partial(
            gemma3_attn_forward, operation_mode="baseline"
        )

    judge_prompt = (
        f"Question: {question}\n"
        f"Model response: '{response}'\n"
        f"The correct answer to this question is {correct_answer}. "
        f"What single number did the model give as its final answer to the question? "
        f"Reply with only a single digit, nothing else."
    )

    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": judge_prompt}]
    }]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            repetition_penalty=1.3
        )

    input_len = inputs['input_ids'].shape[1]
    generated = output[0][input_len:]
    result = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()

    digit_match = re.search(r'\b(\d+)\b', result)
    if digit_match:
        return digit_match.group(1)

    word_to_num = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
        'eighteen': '18', 'nineteen': '19', 'twenty': '20'
    }

    result_lower = result.lower()
    for word, num in word_to_num.items():
        if re.search(rf'\b{word}\b', result_lower):
            return num

    return None

def evaluate_counting(
    response: str,
    question: str,
    ordinary_number: str,
    anomaly_number: str,
    correct_answer: str,
    model,
    processor,
    qwen3vl_attn_forward=None,
    gemma3_attn_forward=None
) -> Tuple[bool, bool, Optional[str]]:
    """
    Evaluate counting prompt response.

    Args:
        response: Model's response
        question: The original question asked (for LLM judge context)
        ordinary_number: Expected ordinary count (e.g., "2")
        anomaly_number: Expected anomaly count (e.g., "4")
        model: The loaded model (used as judge if needed)
        processor: The loaded processor

    Returns:
        (is_correct, is_biased, parsed_answer)
        - is_correct: True if answer matches anomaly_number
        - is_biased: True if answer matches ordinary_number
        - parsed_answer: The extracted number or None
    """
    numbers_found = extract_all_numbers(response)

    if len(numbers_found) == 0:
        # No number found - neither correct nor biased
        return False, False, None

    elif len(numbers_found) == 1:
        # Unambiguous - use regex result directly
        parsed = numbers_found[0]

    else:
        # Multiple numbers - use LLM judge to resolve
        print(f"  [Judge] Multiple numbers found {numbers_found}, calling LLM judge...")
        parsed = vlm_judge_extract(
            response, question, correct_answer, model, processor,
            qwen3vl_attn_forward, gemma3_attn_forward
        )
        if parsed is None:
            return False, False, None

    def normalize_to_digit(value: str) -> str:
        try:
            return str(w2n.word_to_num(str(value).lower()))
        except ValueError:
            return str(value)

    parsed_str = str(parsed)
    anomaly_digit = normalize_to_digit(str(anomaly_number))
    ordinary_digit = normalize_to_digit(str(ordinary_number))

    is_correct = (parsed_str == anomaly_digit)
    is_biased = (parsed_str == ordinary_digit)

    return is_correct, is_biased, parsed_str


def evaluate_binary(
        response: str,
        question_type: str,
        ordinary_number: str,
        anomaly_number: str
) -> Tuple[bool, bool, Optional[str]]:
    """
    Evaluate binary prompt response.
    Args:
        response: Model's response
        question_type: "ordinary", "anomaly", or "hard_negative"
        ordinary_number: The ordinary count
        anomaly_number: The anomaly count (actual count in image)
    Returns:
        (is_correct, is_biased, parsed_answer)
    Logic:
        - For "ordinary" question: expects "no" (object has anomaly, not ordinary)
          - Correct: says "no"
          - Biased: says "yes" (biased toward ordinary)
        - For "anomaly" question: expects "yes" (object has anomaly)
          - Correct: says "yes"
          - Biased: says "no" (biased toward ordinary, rejecting anomaly)
        - For "hard_negative" question: expects "no" (neither ordinary nor anomaly)
          - Correct: says "no"
          - Biased: never (just incorrect if "yes")
    """
    parsed = extract_yes_no_from_response(response)
    if parsed is None:
        return False, False, None

    if question_type == "ordinary":
        # Asking about ordinary number when object has anomaly
        # Expected: "no"
        is_correct = (parsed == "no")
        is_biased = (parsed == "yes")  # Biased toward ordinary
    elif question_type == "anomaly":
        # Asking about anomaly number when object has anomaly
        # Expected: "yes"
        is_correct = (parsed == "yes")
        is_biased = (parsed == "no")  # Biased toward ordinary (rejecting anomaly)
    elif question_type == "hard_negative":
        # Asking about incorrect number (neither ordinary nor anomaly)
        # Expected: "no"
        is_correct = (parsed == "no")
        is_biased = False  # No bias, just wrong if answered "yes"
    else:
        return False, False, parsed

    return is_correct, is_biased, parsed


def print_results_from_file(json_path: str, output_file: str = None):
    """
    Loads aggregated results from all_configs_aggregated.json and prints them.
    Args:
        json_path: Path to all_configs_aggregated.json
        output_file: Optional path to save results as .txt file
    """
    with open(json_path, 'r') as f:
        aggregated_results = json.load(f)
    print_results(aggregated_results, output_file=output_file)


def print_results(aggregated_results: dict, output_file: str = None):
    """
    Prints accuracy and bias rate for each config in table format.
    Args:
        aggregated_results: The full aggregated results dict from run_experiment
        output_file: Optional path to save results as .txt file
    """
    # Extract baseline metrics first
    baseline = aggregated_results["baseline"]["aggregated_metrics"]

    # Group configs by base name and layer
    grouped_configs = group_configs_by_layers(aggregated_results)

    # Collect all output strings
    output_lines = []

    # Counting results
    output_lines.append("\n" + "=" * 80)
    output_lines.append("COUNTING RESULTS")
    output_lines.append("=" * 80)
    counting_tables = get_tables_string(grouped_configs, baseline, "counting")
    output_lines.append(counting_tables)

    # # Binary results
    # output_lines.append("\n" + "=" * 80)
    # output_lines.append("BINARY RESULTS")
    # output_lines.append("=" * 80)
    # binary_tables = get_tables_string(grouped_configs, baseline, "binary")
    # output_lines.append(binary_tables)
    #
    # # Binary average results
    # output_lines.append("\n" + "=" * 80)
    # output_lines.append("BINARY AVERAGE RESULTS")
    # output_lines.append("=" * 80)
    # binary_avg_table = get_binary_average_table_string(grouped_configs, baseline, aggregated_results)  # <-- pass aggregated_results
    # output_lines.append(binary_avg_table)

    # Join all output
    full_output = "\n".join(output_lines)

    # Print to console
    print(full_output)

    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(full_output)
        print(f"\nResults saved to: {output_file}")


def group_configs_by_layers(aggregated_results: dict) -> dict:
    """
    Group configurations by base name and layer suffix.

    Returns:
        {
            'baseline': {
                'all': config_data,  # or whichever layer exists
            },
            'amplify_target_pre_softmax_1.5x': {
                'all': config_data,
                'early': config_data,
                'middle': config_data,
                'late': config_data,
            },
            ...
        }
    """
    grouped = {}

    for config_name, config_data in aggregated_results.items():
        # Parse config name to extract base and layer
        base_name, layer = parse_config_name(config_name)

        if base_name not in grouped:
            grouped[base_name] = {}

        grouped[base_name][layer] = config_data

    return grouped


def parse_config_name(config_name: str) -> tuple:
    if config_name == "baseline":
        return ("baseline", "all")

    layer_names = ['early', 'middle', 'late']  # Remove 'all' — handle separately
    variant_suffixes = ['_bb_mask', '_bb', '_mask']

    # 1. Layer + variant at end: ..._early_bb, ..._middle_mask
    for layer_name in layer_names:
        for variant_suffix in variant_suffixes:
            combined = f"_{layer_name}{variant_suffix}"
            if config_name.endswith(combined):
                base_name = config_name[:-len(combined)] + variant_suffix
                return (base_name, layer_name)

    # 2. Layer anywhere, only if followed by a multiplier — use LAST occurrence
    for layer_name in layer_names:
        pattern = f"_{layer_name}_"
        idx = config_name.rfind(pattern)  # rfind = last occurrence
        if idx != -1:
            after = config_name[idx + len(pattern):]
            if re.match(r'[\d.]+x', after):
                base_name = config_name[:idx] + "_" + after
                return (base_name, layer_name)

    # 3. Layer at end (no variant, no multiplier): ..._early, ..._middle
    for layer_name in layer_names:
        if config_name.endswith(f"_{layer_name}"):
            base_name = config_name[:-len(f"_{layer_name}")]
            return (base_name, layer_name)

    # 4. Variant suffix only at end (no layer)
    for variant_suffix in variant_suffixes:
        if config_name.endswith(variant_suffix):
            return (config_name, "all")

    # 5. Default
    return (config_name, "all")


def format_cell(value: float, std: float, baseline_value: float = None) -> str:
    """
    Format a cell value with percentage, delta, and std.

    Args:
        value: The metric value (0-1)
        std: Standard deviation
        baseline_value: Baseline value for computing delta (optional)

    Returns:
        Formatted string like "85.00 (+35.00)% ± 0.000" or "50.00% ± 0.000"
    """
    percentage = value * 100

    if baseline_value is not None:
        delta = (value - baseline_value) * 100
        # Add tolerance to avoid showing (+0.00) or (-0.00) due to floating point errors
        if delta > 0.005:
            delta_str = f" (+{delta:.2f})"
        elif delta < -0.005:
            delta_str = f" ({delta:.2f})"
        else:
            delta_str = ""
    else:
        delta_str = ""

    return f"{percentage:.2f}{delta_str}% ± {std:.3f}"


def get_tables_string(
    grouped_configs: dict,
    baseline: dict,
    task_type: str,
    top_n: int = None,
    ordinary_grouped_configs: dict = None
) -> str:
    """
    Generate table strings for either counting or binary task.

    Args:
        grouped_configs: Grouped configurations by base name and layer (anomaly)
        baseline: Baseline metrics
        task_type: "counting" or "binary"
        top_n: If set, only show top N configs by max accuracy delta per prompt key.
               If None, show all configs (original behavior).
        ordinary_grouped_configs: If provided and top_n is set, also show ordinary
                                  results for the selected configs.
    Returns:
        String containing all tables for this task type
    """
    output_lines = []

    layer_columns = ["All", "Early", "Middle", "Late"]
    layer_keys = ["all", "early", "middle", "late"]

    all_prompt_keys = set()
    for config_data in grouped_configs.values():
        for layer_data in config_data.values():
            metrics = layer_data["aggregated_metrics"]
            if task_type in metrics["aggregated_accuracy"]:
                all_prompt_keys.update(metrics["aggregated_accuracy"][task_type].keys())

    variant_suffixes = ['_mask', '_bb_mask', '_bb']
    base_prompts = set()
    variant_prompts = {}

    for key in all_prompt_keys:
        is_variant = False
        for suffix in variant_suffixes:
            if key.endswith(suffix):
                base = key[:-len(suffix)]
                if base not in variant_prompts:
                    variant_prompts[base] = []
                variant_prompts[base].append(key)
                is_variant = True
                break
        if not is_variant:
            base_prompts.add(key)

    sorted_base_prompts = sorted(base_prompts)

    all_sorted_configs = sorted([k for k in grouped_configs.keys() if k != "baseline"])

    # ------------------------------------------------------------------ #
    # Base prompt tables                                                   #
    # ------------------------------------------------------------------ #
    for prompt_key in sorted_base_prompts:
        baseline_acc = baseline["aggregated_accuracy"][task_type].get(prompt_key)
        baseline_bias = baseline["aggregated_bias_rate"][task_type].get(prompt_key)
        baseline_acc_std = baseline["aggregated_std"][task_type]["accuracy"].get(prompt_key)
        baseline_bias_std = baseline["aggregated_std"][task_type]["bias"].get(prompt_key)

        # Determine which configs to show
        if top_n is not None:
            configs_to_show = _get_top_n_configs(
                grouped_configs, all_sorted_configs, prompt_key,
                task_type, baseline_acc, layer_keys, top_n
            )
        else:
            configs_to_show = all_sorted_configs

        # Anomaly table
        table = PrettyTable()
        table.field_names = ["Configuration", "Metrics"] + layer_columns
        table.align["Configuration"] = "l"
        table.align["Metrics"] = "l"
        table.padding_width = 0
        table.horizontal_char = '-'
        table.junction_char = '+'

        if baseline_acc is not None:
            table.add_row([
                "baseline", "Accuracy",
                format_cell(baseline_acc, baseline_acc_std),
                "-", "-", "-"
            ])
            table.add_row([
                "", "Bias",
                format_cell(baseline_bias, baseline_bias_std),
                "-", "-", "-"
            ])

        for base_name in configs_to_show:
            layer_data = grouped_configs[base_name]
            acc_row = [base_name, "Accuracy"]
            bias_row = ["", "Bias"]

            for layer_key in layer_keys:
                if layer_key in layer_data:
                    metrics = layer_data[layer_key]["aggregated_metrics"]
                    acc = metrics["aggregated_accuracy"][task_type].get(prompt_key)
                    acc_std = metrics["aggregated_std"][task_type]["accuracy"].get(prompt_key)
                    bias = metrics["aggregated_bias_rate"][task_type].get(prompt_key)
                    bias_std = metrics["aggregated_std"][task_type]["bias"].get(prompt_key)

                    if acc is not None and acc_std is not None:
                        acc_row.append(format_cell(acc, acc_std, baseline_acc))
                    else:
                        acc_row.append("-")

                    if bias is not None and bias_std is not None:
                        bias_row.append(format_cell(bias, bias_std, baseline_bias))
                    else:
                        bias_row.append("-")
                else:
                    acc_row.append("-")
                    bias_row.append("-")

            table.add_row(acc_row)
            table.add_row(bias_row)

        output_lines.append(f"\n{task_type.upper()} - {prompt_key}")
        output_lines.append(str(table))

        # Ordinary counterpart table (only in top_n mode)
        if top_n is not None and ordinary_grouped_configs is not None:
            ordinary_baseline_config = list(ordinary_grouped_configs.get("baseline", {}).values())
            if ordinary_baseline_config:
                ord_baseline_metrics = ordinary_baseline_config[0]["aggregated_metrics"]
            else:
                ord_baseline_metrics = baseline  # fallback

            ord_baseline_acc = ord_baseline_metrics["aggregated_accuracy"][task_type].get(prompt_key)
            ord_baseline_bias = ord_baseline_metrics["aggregated_bias_rate"][task_type].get(prompt_key)
            ord_baseline_acc_std = ord_baseline_metrics["aggregated_std"][task_type]["accuracy"].get(prompt_key)
            ord_baseline_bias_std = ord_baseline_metrics["aggregated_std"][task_type]["bias"].get(prompt_key)

            ord_table = PrettyTable()
            ord_table.field_names = ["Configuration", "Metrics"] + layer_columns
            ord_table.align["Configuration"] = "l"
            ord_table.align["Metrics"] = "l"
            ord_table.padding_width = 0
            ord_table.horizontal_char = '-'
            ord_table.junction_char = '+'

            if ord_baseline_acc is not None:
                ord_table.add_row([
                    "baseline [ordinary]", "Accuracy",
                    format_cell(ord_baseline_acc, ord_baseline_acc_std),
                    "-", "-", "-"
                ])
                ord_table.add_row([
                    "", "Bias",
                    format_cell(ord_baseline_bias, ord_baseline_bias_std),
                    "-", "-", "-"
                ])

            for base_name in configs_to_show:
                if _has_mask_variant(base_name):
                    ord_table.add_row([f"{base_name} [ordinary]", "Accuracy", "N/A (mask)", "-", "-", "-"])
                    ord_table.add_row(["", "Bias", "N/A (mask)", "-", "-", "-"])
                    continue

                if base_name not in ordinary_grouped_configs:
                    ord_table.add_row([f"{base_name} [ordinary]", "Accuracy", "-", "-", "-", "-"])
                    ord_table.add_row(["", "Bias", "-", "-", "-", "-"])
                    continue

                layer_data = ordinary_grouped_configs[base_name]
                acc_row = [f"{base_name} [ordinary]", "Accuracy"]
                bias_row = ["", "Bias"]

                for layer_key in layer_keys:
                    if layer_key in layer_data:
                        metrics = layer_data[layer_key]["aggregated_metrics"]
                        acc = metrics["aggregated_accuracy"][task_type].get(prompt_key)
                        acc_std = metrics["aggregated_std"][task_type]["accuracy"].get(prompt_key)
                        bias = metrics["aggregated_bias_rate"][task_type].get(prompt_key)
                        bias_std = metrics["aggregated_std"][task_type]["bias"].get(prompt_key)

                        if acc is not None and acc_std is not None:
                            acc_row.append(format_cell(acc, acc_std, ord_baseline_acc))
                        else:
                            acc_row.append("-")

                        if bias is not None and bias_std is not None:
                            bias_row.append(format_cell(bias, bias_std, ord_baseline_bias))
                        else:
                            bias_row.append("-")
                    else:
                        acc_row.append("-")
                        bias_row.append("-")

                ord_table.add_row(acc_row)
                ord_table.add_row(bias_row)

            output_lines.append(f"\n{task_type.upper()} - {prompt_key} [ORDINARY COUNTERPART]")
            output_lines.append(str(ord_table))

    # ------------------------------------------------------------------ #
    # Variant prompt tables (mask, bb, bb_mask)                           #
    # ------------------------------------------------------------------ #
    for base_prompt in sorted(variant_prompts.keys()):
        variants = sorted(variant_prompts[base_prompt])

        for variant_key in variants:
            table = PrettyTable()
            table.field_names = ["Configuration", "Metrics"] + layer_columns
            table.align["Configuration"] = "l"
            table.align["Metrics"] = "l"
            table.padding_width = 0
            table.horizontal_char = '-'
            table.junction_char = '+'

            if top_n is not None:
                # No baseline for variant prompts, rank by max raw accuracy
                all_accs = []
                for base_name in all_sorted_configs:
                    if base_name not in grouped_configs:
                        continue
                    layer_data = grouped_configs[base_name]
                    for layer_key in layer_keys:
                        if layer_key in layer_data:
                            metrics = layer_data[layer_key]["aggregated_metrics"]
                            acc = metrics["aggregated_accuracy"][task_type].get(variant_key)
                            if acc is not None:
                                all_accs.append((acc, base_name))
                if all_accs:
                    all_accs = [(round(a, 2), name) for a, name in all_accs]
                    all_accs.sort(key=lambda x: x[0], reverse=True)
                    unique_values = sorted(set(a for a, _ in all_accs), reverse=True)
                    threshold = unique_values[min(top_n, len(unique_values)) - 1]
                    top_configs = set(name for acc, name in all_accs if acc >= threshold)
                    configs_to_show = [c for c in all_sorted_configs if c in top_configs]
                else:
                    configs_to_show = []
            else:
                configs_to_show = all_sorted_configs

            for base_name in configs_to_show:
                if base_name not in grouped_configs:
                    continue
                layer_data = grouped_configs[base_name]

                acc_row = [base_name, "Accuracy"]
                bias_row = ["", "Bias"]

                has_data = False
                for layer_key in layer_keys:
                    if layer_key in layer_data:
                        metrics = layer_data[layer_key]["aggregated_metrics"]
                        acc = metrics["aggregated_accuracy"][task_type].get(variant_key)
                        acc_std = metrics["aggregated_std"][task_type]["accuracy"].get(variant_key)
                        bias = metrics["aggregated_bias_rate"][task_type].get(variant_key)
                        bias_std = metrics["aggregated_std"][task_type]["bias"].get(variant_key)

                        if acc is not None and acc_std is not None:
                            acc_row.append(format_cell(acc, acc_std))
                            has_data = True
                        else:
                            acc_row.append("-")

                        if bias is not None and bias_std is not None:
                            bias_row.append(format_cell(bias, bias_std))
                        else:
                            bias_row.append("-")
                    else:
                        acc_row.append("-")
                        bias_row.append("-")

                if has_data:
                    table.add_row(acc_row)
                    table.add_row(bias_row)

            output_lines.append(f"\n{task_type.upper()} - {variant_key}")
            output_lines.append(str(table))

    return "\n".join(output_lines)


def get_binary_average_table_string(grouped_configs: dict, baseline: dict, aggregated_results: dict) -> str:
    """
    Generate average table string for binary task, computed per-image.
    For each image, accuracy = correct_question_types / total_question_types.
    Args:
        grouped_configs: Grouped configurations by base name and layer
        baseline: Baseline aggregated_metrics
        aggregated_results: Full aggregated results dict (to access per_image_results)
    Returns:
        String containing the average table
    """
    layer_columns = ["All", "Early", "Middle", "Late"]
    layer_keys = ["all", "early", "middle", "late"]

    prompt_keys = list(baseline["aggregated_accuracy"]["binary"].keys())
    if len(prompt_keys) == 0:
        return ""

    def compute_per_image_binary_avg(config_data: dict):
        per_image_results = config_data.get("per_image_results", [])
        if not per_image_results:
            return None, None, None, None

        image_acc_scores = []
        image_bias_scores = []
        image_acc_stds = []
        image_bias_stds = []

        for img_result in per_image_results:
            binary_results = img_result.get("results", {}).get("binary", {})
            acc_values = []
            bias_values = []
            acc_std_values = []
            bias_std_values = []
            for pk in prompt_keys:
                if pk in binary_results:
                    acc_values.append(binary_results[pk]["accuracy"])
                    bias_values.append(binary_results[pk]["bias_rate"])
                    acc_std_values.append(binary_results[pk]["accuracy_std"])
                    bias_std_values.append(binary_results[pk]["bias_std"])
            if acc_values:
                image_acc_scores.append(np.mean(acc_values))
                image_bias_scores.append(np.mean(bias_values))
                image_acc_stds.append(np.mean(acc_std_values))
                image_bias_stds.append(np.mean(bias_std_values))

        if not image_acc_scores:
            return None, None, None, None

        return (
            np.mean(image_acc_scores),
            np.mean(image_acc_stds),  # avg of per-image run stds → always 0.000 with greedy
            np.mean(image_bias_scores),
            np.mean(image_bias_stds),  # avg of per-image run stds → always 0.000 with greedy
        )

    table = PrettyTable()
    table.field_names = ["Configuration", "Metrics"] + layer_columns
    table.align["Configuration"] = "l"
    table.align["Metrics"] = "l"
    table.padding_width = 0
    table.horizontal_char = '-'
    table.junction_char = '+'

    # Baseline row
    baseline_acc, baseline_acc_std, baseline_bias, baseline_bias_std = compute_per_image_binary_avg(
        aggregated_results["baseline"]
    )
    if baseline_acc is not None:
        table.add_row([
            "baseline", "Accuracy",
            format_cell(baseline_acc, baseline_acc_std),
            "-", "-", "-"
        ])
        table.add_row([
            "", "Bias",
            format_cell(baseline_bias, baseline_bias_std),
            "-", "-", "-"
        ])

    sorted_configs = sorted([k for k in grouped_configs.keys() if k != "baseline"])

    for base_name in sorted_configs:
        layer_data = grouped_configs[base_name]
        acc_row = [base_name, "Accuracy"]
        bias_row = ["", "Bias"]

        for layer_key in layer_keys:
            if layer_key in layer_data:
                # Find the original config name to look up per_image_results
                # grouped_configs stores config_data directly, so use it
                config_data = layer_data[layer_key]
                acc, acc_std, bias, bias_std = compute_per_image_binary_avg(config_data)

                if acc is not None:
                    acc_row.append(format_cell(acc, acc_std, baseline_acc))
                else:
                    acc_row.append("-")

                if bias is not None:
                    bias_row.append(format_cell(bias, bias_std, baseline_bias))
                else:
                    bias_row.append("-")
            else:
                acc_row.append("-")
                bias_row.append("-")

        table.add_row(acc_row)
        table.add_row(bias_row)

    return f"\nBINARY - average (per-image)\n{str(table)}"

def _get_top_n_configs(
    grouped_configs: dict,
    all_sorted_configs: list,
    prompt_key: str,
    task_type: str,
    baseline_acc: float,
    layer_keys: list,
    top_n: int
) -> list:
    """
    Collect all (delta, config_name) pairs across all configs and all layers.
    Find the Nth largest delta value (threshold), then return all configs
    that have at least one layer with delta >= threshold.
    """
    all_deltas = []  # list of (delta_value, config_name)

    for base_name in all_sorted_configs:
        layer_data = grouped_configs[base_name]
        for layer_key in layer_keys:
            if layer_key in layer_data:
                metrics = layer_data[layer_key]["aggregated_metrics"]
                acc = metrics["aggregated_accuracy"][task_type].get(prompt_key)
                if acc is not None and baseline_acc is not None:
                    delta = (acc - baseline_acc) * 100
                    all_deltas.append((delta, base_name))

    if not all_deltas:
        return []

    # Sort descending and find the Nth delta value as threshold
    all_deltas = [(round(d, 2), name) for d, name in all_deltas]
    all_deltas.sort(key=lambda x: x[0], reverse=True)
    unique_values = sorted(set(d for d, _ in all_deltas), reverse=True)
    threshold = unique_values[min(top_n, len(unique_values)) - 1]
    top_configs = set(name for delta, name in all_deltas if delta >= threshold)

    # Return in original sorted order
    return [c for c in all_sorted_configs if c in top_configs]

def _has_mask_variant(base_name: str) -> bool:
    """Check if a config name ends with _mask or _bb_mask (ordinary doesn't have these)."""
    return base_name.endswith('_mask') or base_name.endswith('_bb_mask')


def _average_results_across_categories(all_category_results: dict, categories: List[str]) -> dict:
    """
    Average counting metrics across categories.
    Returns a dict in the same format as aggregated_results.
    """
    config_names = list(all_category_results[categories[0]].keys())
    averaged_results = {}

    for config_name in config_names:
        first = all_category_results[categories[0]][config_name]["aggregated_metrics"]
        counting_keys = list(first["aggregated_accuracy"]["counting"].keys())

        acc_sums = {k: 0.0 for k in counting_keys}
        bias_sums = {k: 0.0 for k in counting_keys}
        acc_std_sums = {k: 0.0 for k in counting_keys}
        bias_std_sums = {k: 0.0 for k in counting_keys}

        for category in categories:
            metrics = all_category_results[category][config_name]["aggregated_metrics"]
            for k in counting_keys:
                acc_sums[k] += metrics["aggregated_accuracy"]["counting"].get(k, 0.0)
                bias_sums[k] += metrics["aggregated_bias_rate"]["counting"].get(k, 0.0)
                acc_std_sums[k] += metrics["aggregated_std"]["counting"]["accuracy"].get(k, 0.0)
                bias_std_sums[k] += metrics["aggregated_std"]["counting"]["bias"].get(k, 0.0)

        n = len(categories)
        averaged_results[config_name] = {
            "aggregated_metrics": {
                "config_name": config_name,
                "num_images": first["num_images"],
                "aggregated_accuracy": {
                    "counting": {k: acc_sums[k] / n for k in counting_keys},
                    "binary": {}
                },
                "aggregated_bias_rate": {
                    "counting": {k: bias_sums[k] / n for k in counting_keys},
                    "binary": {}
                },
                "aggregated_std": {
                    "counting": {
                        "accuracy": {k: acc_std_sums[k] / n for k in counting_keys},
                        "bias": {k: bias_std_sums[k] / n for k in counting_keys}
                    },
                    "binary": {"accuracy": {}, "bias": {}}
                }
            },
            "per_image_results": []
        }

    return averaged_results

def _recompute_metrics_from_per_image(data: dict, num_images: int = None, exclude_ids: set = None) -> dict:
    if num_images is None and not exclude_ids:
        return data

    valid_ids = {f"img{i}" for i in range(num_images)} if num_images is not None else None
    exclude_ids = exclude_ids or set()

    for config_name, config_data in data.items():
        filtered = [
            r for r in config_data["per_image_results"]
            if (valid_ids is None or r["image_id"] in valid_ids)
            and r["image_id"] not in exclude_ids
        ]
        config_data["per_image_results"] = filtered

        n = len(filtered)
        if n == 0:
            continue

        # Collect counting keys from first result
        first = filtered[0]["results"]["counting"]
        counting_keys = list(first.keys())

        acc_sums = {k: 0.0 for k in counting_keys}
        bias_sums = {k: 0.0 for k in counting_keys}

        for img_result in filtered:
            for k in counting_keys:
                acc_sums[k] += img_result["results"]["counting"][k]["accuracy"]
                bias_sums[k] += img_result["results"]["counting"][k]["bias_rate"]

        metrics = config_data["aggregated_metrics"]
        metrics["num_images"] = n
        metrics["aggregated_accuracy"]["counting"] = {k: acc_sums[k] / n for k in counting_keys}
        metrics["aggregated_bias_rate"]["counting"] = {k: bias_sums[k] / n for k in counting_keys}
        # Zero out std since we're not recomputing it (or recompute if needed)
        metrics["aggregated_std"]["counting"]["accuracy"] = {k: 0.0 for k in counting_keys}
        metrics["aggregated_std"]["counting"]["bias"] = {k: 0.0 for k in counting_keys}

    return data

def print_results_multi_category(
    base_dir: str,
    categories: List[str],
    model_path: str,
    question_type: str,
    image_type: str = 'anomaly',
    output_file: str = None,
    top_n: int = None,
    ordinary_base_dir: str = None,
    num_images: int = None,
    exclude_ids: Dict[str, List[int]] = None  # e.g. {"birds": [5, 6], "mammals": [2]}
):
    """
    Load results from multiple categories, average counting metrics, and print.
    Args:
        base_dir: Base directory path
        categories: List of category names
        model_path: Model path folder name (e.g. "qwen3-vl-7b")
        question_type: "mcq" or "open_ended"
        image_type: 'anomaly' or 'ordinary'
        output_file: Optional path to save results as .txt file
        top_n: If set, only show top N configs by max accuracy delta.
               Also displays ordinary counterpart tables alongside anomaly results.
        ordinary_base_dir: Base dir for ordinary results when top_n is set.
                           Defaults to base_dir if not provided.
        num_images: If set, only include results from the first num_images (e.g. 100) based on image_id.
        exclude_ids: If set, a dict mapping category to list of image indices to exclude (e.g. {"birds": [5, 6]}).
                     Image IDs are expected to be in the format "img{index}".
    """
    all_category_results = {}
    for category in categories:
        json_path = f"{base_dir}/{category.capitalize()}/{model_path}/{image_type}/{question_type}/all_configs_aggregated.json"
        with open(json_path, 'r') as f:
            data = json.load(f)
        cat_exclude = {f"img{i}" for i in (exclude_ids or {}).get(category, [])}
        all_category_results[category] = _recompute_metrics_from_per_image(data, num_images=num_images,
                                                                           exclude_ids=cat_exclude)

    averaged_results = _average_results_across_categories(all_category_results, categories)

    # Load ordinary counterparts if top_n is set and we're looking at anomaly results
    ordinary_grouped_configs = None
    all_ordinary_results = {}
    if top_n is not None and image_type == 'anomaly':
        ord_base_dir = ordinary_base_dir if ordinary_base_dir else base_dir
        for category in categories:
            json_path = f"{ord_base_dir}/{category.capitalize()}/{model_path}/ordinary/{question_type}/all_configs_aggregated.json"
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                cat_exclude = {f"img{i}" for i in (exclude_ids or {}).get(category, [])}
                all_ordinary_results[category] = _recompute_metrics_from_per_image(data, num_images=num_images,
                                                                                   exclude_ids=cat_exclude)
            except FileNotFoundError:
                pass  # ordinary results not available, ordinary tables will be skipped
        if all_ordinary_results:
            averaged_ordinary = _average_results_across_categories(
                all_ordinary_results, list(all_ordinary_results.keys())
            )
            ordinary_grouped_configs = group_configs_by_layers(averaged_ordinary)

    baseline = averaged_results["baseline"]["aggregated_metrics"]
    grouped_configs = group_configs_by_layers(averaged_results)
    output_lines = []
    output_lines.append("\n" + "=" * 80)
    output_lines.append("COUNTING RESULTS")
    if top_n is not None:
        output_lines.append(f"(Showing top {top_n} configs by max accuracy delta)")
    output_lines.append("=" * 80)

    if top_n is not None:
        all_sorted_configs = sorted([k for k in grouped_configs.keys() if k != "baseline"])
        layer_keys = ["all", "early", "middle", "late"]
        all_deltas = []
        for prompt_key in baseline["aggregated_accuracy"]["counting"]:
            baseline_acc = baseline["aggregated_accuracy"]["counting"].get(prompt_key)
            if baseline_acc is None:
                continue
            for base_name in all_sorted_configs:
                for layer_key in layer_keys:
                    if layer_key in grouped_configs[base_name]:
                        metrics = grouped_configs[base_name][layer_key]["aggregated_metrics"]
                        acc = metrics["aggregated_accuracy"]["counting"].get(prompt_key)
                        if acc is not None:
                            all_deltas.append(round((acc - baseline_acc) * 100, 2))
        all_deltas.sort(reverse=True)
        unique_deltas = sorted(set(all_deltas), reverse=True)[:top_n]
        delta_str = ", ".join(f"{d:+.2f}%" for d in unique_deltas)
        output_lines.append(f"Top deltas: {delta_str}")

    counting_tables = get_tables_string(
        grouped_configs,
        baseline,
        "counting",
        top_n=top_n,
        ordinary_grouped_configs=ordinary_grouped_configs
    )
    output_lines.append(counting_tables)
    # Per-category breakdown for top N configs
    if top_n is not None:
        # Get the top N configs (same logic as in get_tables_string)
        prompt_key = list(baseline["aggregated_accuracy"]["counting"].keys())[0]
        layer_keys = ["all", "early", "middle", "late"]
        top_configs = _get_top_n_configs(
            grouped_configs,
            sorted([k for k in grouped_configs.keys() if k != "baseline"]),
            prompt_key,
            "counting",
            baseline["aggregated_accuracy"]["counting"].get(prompt_key),
            layer_keys,
            top_n
        )

        ordinary_category_results = None
        if ordinary_grouped_configs is not None:
            # Reload per-category ordinary results — already in all_ordinary_results
            ordinary_category_results = all_ordinary_results if all_ordinary_results else None

        breakdown_str = _get_per_category_breakdown_string(
            top_configs=top_configs,
            grouped_configs=grouped_configs,
            all_category_results=all_category_results,
            categories=categories,
            prompt_key=prompt_key,
            image_type=image_type,
            ordinary_category_results=ordinary_category_results
        )
        output_lines.append(breakdown_str)

    full_output = "\n".join(output_lines)
    print(full_output)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(full_output)
        print(f"\nResults saved to: {output_file}")
def _get_per_category_breakdown_string(
    top_configs: list,
    grouped_configs: dict,
    all_category_results: dict,
    categories: list,
    prompt_key: str = "counting",
    image_type: str = "anomaly",
    ordinary_category_results: dict = None
) -> str:
    """
    For each top N config, print a per-category breakdown table with baseline and config performance.
    """
    layer_columns = ["All", "Early", "Middle", "Late"]
    layer_keys = ["all", "early", "middle", "late"]
    output_lines = []

    for base_name in top_configs:
        for current_image_type, category_results in [
            (image_type, all_category_results),
            ("ordinary", ordinary_category_results)
        ]:
            if category_results is None:
                continue

            output_lines.append(f"\nPER-CATEGORY BREAKDOWN - Config: {base_name} [{current_image_type.upper()}]")

            table = PrettyTable()
            table.field_names = ["Category", "Metrics"] + layer_columns
            table.align["Category"] = "l"
            table.align["Metrics"] = "l"
            table.padding_width = 0
            table.horizontal_char = '-'
            table.junction_char = '+'

            # For computing average rows at the bottom
            avg_bas_acc = {lk: [] for lk in layer_keys}
            avg_bas_bias = {lk: [] for lk in layer_keys}
            avg_acc = {lk: [] for lk in layer_keys}
            avg_bias = {lk: [] for lk in layer_keys}

            for category in categories:
                if category not in category_results:
                    continue

                cat_data = category_results[category]

                # Baseline for this category
                baseline_metrics = cat_data.get("baseline", {}).get("aggregated_metrics", {})
                bas_acc = baseline_metrics.get("aggregated_accuracy", {}).get("counting", {}).get(prompt_key)
                bas_bias = baseline_metrics.get("aggregated_bias_rate", {}).get("counting", {}).get(prompt_key)
                bas_acc_std = baseline_metrics.get("aggregated_std", {}).get("counting", {}).get("accuracy", {}).get(prompt_key)
                bas_bias_std = baseline_metrics.get("aggregated_std", {}).get("counting", {}).get("bias", {}).get(prompt_key)

                bas_acc_row = [category.capitalize(), "Bas. Acc"]
                bas_bias_row = ["", "Bas. Bias"]
                acc_row = ["", "Accuracy"]
                bias_row = ["", "Bias"]

                for layer_key in layer_keys:
                    # Baseline only has 'all' layer
                    if layer_key == "all":
                        if bas_acc is not None:
                            bas_acc_row.append(format_cell(bas_acc, bas_acc_std))
                            avg_bas_acc[layer_key].append(bas_acc)
                        else:
                            bas_acc_row.append("-")
                        if bas_bias is not None:
                            bas_bias_row.append(format_cell(bas_bias, bas_bias_std))
                            avg_bas_bias[layer_key].append(bas_bias)
                        else:
                            bas_bias_row.append("-")
                    else:
                        bas_acc_row.append("-")
                        bas_bias_row.append("-")

                    # Config performance for this category
                    # Find the config data for this category
                    config_layer_data = grouped_configs.get(base_name, {}).get(layer_key)
                    if config_layer_data is None:
                        acc_row.append("-")
                        bias_row.append("-")
                        continue

                    # We need per-category data, not averaged — look it up from category_results
                    # Find matching config name in cat_data
                    matched_config = None
                    for config_name, config_data in cat_data.items():
                        parsed_base, parsed_layer = parse_config_name(config_name)
                        if parsed_base == base_name and parsed_layer == layer_key:
                            matched_config = config_data
                            break

                    if matched_config is None:
                        acc_row.append("-")
                        bias_row.append("-")
                        continue

                    metrics = matched_config["aggregated_metrics"]
                    acc = metrics["aggregated_accuracy"]["counting"].get(prompt_key)
                    acc_std = metrics["aggregated_std"]["counting"]["accuracy"].get(prompt_key)
                    bias = metrics["aggregated_bias_rate"]["counting"].get(prompt_key)
                    bias_std = metrics["aggregated_std"]["counting"]["bias"].get(prompt_key)

                    if acc is not None and acc_std is not None:
                        acc_row.append(format_cell(acc, acc_std, bas_acc))
                        avg_acc[layer_key].append(acc)
                    else:
                        acc_row.append("-")

                    if bias is not None and bias_std is not None:
                        bias_row.append(format_cell(bias, bias_std, bas_bias))
                        avg_bias[layer_key].append(bias)
                    else:
                        bias_row.append("-")

                table.add_row(bas_acc_row)
                table.add_row(bas_bias_row)
                table.add_row(acc_row)
                table.add_row(bias_row)

            # Average row at the bottom
            avg_bas_acc_row = ["Average", "Bas. Acc"]
            avg_bas_bias_row = ["", "Bas. Bias"]
            avg_acc_row = ["", "Accuracy"]
            avg_bias_row = ["", "Bias"]

            for layer_key in layer_keys:
                avg_bas_acc_row.append(
                    format_cell(np.mean(avg_bas_acc[layer_key]), 0.0)
                    if avg_bas_acc[layer_key] else "-"
                )
                avg_bas_bias_row.append(
                    format_cell(np.mean(avg_bas_bias[layer_key]), 0.0)
                    if avg_bas_bias[layer_key] else "-"
                )
                avg_bas_val = np.mean(avg_bas_acc["all"]) if avg_bas_acc["all"] else None
                avg_acc_row.append(
                    format_cell(np.mean(avg_acc[layer_key]), 0.0, avg_bas_val)
                    if avg_acc[layer_key] else "-"
                )
                avg_bas_bias_val = np.mean(avg_bas_bias["all"]) if avg_bas_bias["all"] else None
                avg_bias_row.append(
                    format_cell(np.mean(avg_bias[layer_key]), 0.0, avg_bas_bias_val)
                    if avg_bias[layer_key] else "-"
                )

            table.add_row(avg_bas_acc_row)
            table.add_row(avg_bas_bias_row)
            table.add_row(avg_acc_row)
            table.add_row(avg_bias_row)

            output_lines.append(str(table))

    return "\n".join(output_lines)

def print_category_breakdown(
    base_dir: str,
    categories: List[str],
    model_paths: List[str],
    question_type: str,
    image_type: str = 'anomaly',
    config_filter: List[str] = None,
    output_file: str = None,
    num_images: int = None,
    exclude_ids: Dict[str, List[int]] = None  # e.g. {"birds": [5, 6], "mammals": [2]}
):
    """
    Print per-category accuracy breakdown comparing two models side by side.
    Only shows the 'all' layer for each config.
    Args:
        base_dir: Base directory path
        categories: List of category names
        model_paths: List of two model path folder names
        question_type: "mcq" or "open_ended"
        image_type: 'anomaly' or 'ordinary'
        config_filter: List of config base names to include (e.g. ["amplify_all"]).
                       If None, show all configs.
        output_file: Optional path to save results as .txt file
        num_images: If set, only include results from the first num_images (e.g. 100) based on image_id.
        exclude_ids: If set, a dict mapping category to list of image indices to exclude (e.g. {"birds": [5, 6]}).
                     Image IDs are expected to be in the format "img{index}".
    """
    # Load per-category results for each model
    # { model_path: { category: aggregated_results } }
    all_data = {}
    for model_path in model_paths:
        all_data[model_path] = {}
        for category in categories:
            json_path = f"{base_dir}/{category.capitalize()}/{model_path}/{image_type}/{question_type}/all_configs_aggregated.json"
            with open(json_path, 'r') as f:
                data = json.load(f)
            cat_exclude = {f"img{i}" for i in (exclude_ids or {}).get(category, [])}
            all_data[model_path][category] = _recompute_metrics_from_per_image(data, num_images=num_images,
                                                                               exclude_ids=cat_exclude)

    # Collect all config names from first model/category, filter if needed
    first_model = model_paths[0]
    first_cat = categories[0]
    all_config_names = list(all_data[first_model][first_cat].keys())

    # Group by base name and keep only 'all' layer
    # { base_name: original_config_name } for configs that have an 'all' layer
    config_map = {}  # base_name -> config_name in JSON
    for config_name in all_config_names:
        base_name, layer = parse_config_name(config_name)
        if layer == 'all':
            config_map[base_name] = config_name

    # Apply config_filter if provided
    if config_filter is not None:
        filtered_map = {}
        for base_name, config_name in config_map.items():
            if base_name == 'baseline':
                filtered_map[base_name] = config_name
                continue
            for f in config_filter:
                if base_name.startswith(f):
                    filtered_map[base_name] = config_name
                    break
        config_map = filtered_map

    # Sort: baseline first, then alphabetically
    sorted_base_names = ['baseline'] + sorted(
        [k for k in config_map.keys() if k != 'baseline']
    )

    # Build table
    prompt_key = "counting"  # only counting for now

    output_lines = []
    output_lines.append("\n" + "=" * 80)
    output_lines.append(f"CATEGORY BREAKDOWN - {question_type} - {image_type}")
    output_lines.append("=" * 80)

    table = PrettyTable()
    table.field_names = ["Category", "Config", "Metrics"] + model_paths
    table.align["Category"] = "l"
    table.align["Config"] = "l"
    table.align["Metrics"] = "l"
    table.padding_width = 0
    table.horizontal_char = '-'
    table.junction_char = '+'

    for category in categories:
        # Get baseline acc per model for delta computation
        baseline_accs = {}
        baseline_biases = {}
        for model_path in model_paths:
            baseline_config = all_data[model_path][category].get("baseline", {})
            baseline_accs[model_path] = baseline_config.get("aggregated_metrics", {}) \
                .get("aggregated_accuracy", {}).get("counting", {}).get(prompt_key)
            baseline_biases[model_path] = baseline_config.get("aggregated_metrics", {}) \
                .get("aggregated_bias_rate", {}).get("counting", {}).get(prompt_key)

        first_row_in_category = True
        for base_name in sorted_base_names:
            if base_name not in config_map:
                continue
            config_name = config_map[base_name]
            acc_row = [category.capitalize() if first_row_in_category else "", base_name, "Accuracy"]
            bias_row = ["", "", "Bias"]
            first_row_in_category = False
            for model_path in model_paths:
                config_data = all_data[model_path][category].get(config_name)
                if config_data is None:
                    acc_row.append("-")
                    bias_row.append("-")
                    continue
                metrics = config_data["aggregated_metrics"]
                acc = metrics["aggregated_accuracy"]["counting"].get(prompt_key)
                acc_std = metrics["aggregated_std"]["counting"]["accuracy"].get(prompt_key)
                bias = metrics["aggregated_bias_rate"]["counting"].get(prompt_key)
                bias_std = metrics["aggregated_std"]["counting"]["bias"].get(prompt_key)
                baseline_acc = baseline_accs[model_path] if base_name != 'baseline' else None
                baseline_bias = baseline_biases[model_path] if base_name != 'baseline' else None
                acc_row.append(
                    format_cell(acc, acc_std, baseline_acc) if acc is not None and acc_std is not None else "-")
                bias_row.append(
                    format_cell(bias, bias_std, baseline_bias) if bias is not None and bias_std is not None else "-")
            table.add_row(acc_row)
            table.add_row(bias_row)

        # Divider between categories
        table.add_row(["", "", ""] + [""] * len(model_paths))

    output_lines.append(str(table))
    full_output = "\n".join(output_lines)
    print(full_output)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(full_output)
        print(f"\nResults saved to: {output_file}")