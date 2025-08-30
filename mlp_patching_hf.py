import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
import logging
import os

def get_mlp_layers(model):
    base_model = model.language_model if hasattr(model, 'language_model') else model

    if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
        return [layer.mlp for layer in base_model.model.layers]
    elif hasattr(base_model, 'transformer') and hasattr(base_model.transformer, 'h'):
        return [layer.mlp for layer in base_model.transformer.h]
    elif hasattr(base_model, 'model') and hasattr(base_model.model, 'blocks'):
        return [block.mlp for block in base_model.model.blocks]
    else:
        raise ValueError(f"Unsupported model architecture: {model.__class__.__name__}. Please add a new case to get_mlp_layers.")

def plot_contributions_to_logit_diff(contribution_data, error_data, output_filename, title_text, y_labels):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, max(8, len(y_labels) / 2)))
    ax.barh(y_labels, contribution_data, xerr=error_data, color='skyblue', capsize=5)
    ax.set_title(title_text, fontsize=14, pad=20)
    ax.set_xlabel("Contribution to Logit Difference (Normalized)", fontsize=12)
    ax.set_ylabel("MLP Layer", fontsize=12)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Contributions plot saved to {output_filename}")

def print_tokenization_info(tokenizer, clean_prompt, corrupted_prompt):
    print("\n--- Tokenization Info (First Prompt Pair) ---")
    clean_ids = tokenizer.encode(clean_prompt)
    corrupted_ids = tokenizer.encode(corrupted_prompt)
    clean_tokens = tokenizer.convert_ids_to_tokens(clean_ids)
    corrupted_tokens = tokenizer.convert_ids_to_tokens(corrupted_ids)
    max_len = max(len(clean_tokens), len(corrupted_tokens))
    print(f"{'POS':<5}{'CLEAN':<20}{'CORRUPTED':<20}")
    print("-" * 45)
    for i in range(max_len):
        clean_tok = clean_tokens[i] if i < len(clean_tokens) else ""
        corrupt_tok = corrupted_tokens[i] if i < len(corrupted_tokens) else ""
        print(f"{i:<5}{repr(clean_tok):<20}{repr(corrupt_tok):<20}")
    print("-" * 45)
    print("Use the 'POS' column to determine the `pos_to_patch` value.\n")

def run_activation_patching_and_plot(
    model_name,
    clean_prompts,
    corrupted_prompts,
    clean_continuations,
    corrupted_continuations,
    pos_to_patch,
    layers_to_patch,
    output_filename_base="mlp_patching",
    logger=None,
    task_name=""
):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    print_tokenization_info(tokenizer, clean_prompts[0], corrupted_prompts[0])
    
    mlp_layers = get_mlp_layers(model)
    n_layers = len(mlp_layers)

    if layers_to_patch[0] < 0:
        layers_to_patch[0] = n_layers + layers_to_patch[0]
    if layers_to_patch[1] < 0:
        layers_to_patch[1] = n_layers + layers_to_patch[1]
    layers_to_patch_range = list(range(layers_to_patch[0], layers_to_patch[1] + 1))

    num_prompts = len(clean_prompts)
    num_layers_to_patch = len(layers_to_patch_range)
    all_prompts_patching_results = torch.zeros(num_prompts, num_layers_to_patch, device=model.device)
    valid_prompt_indices = []
    
    clean_correct_count = 0
    corrupted_correct_count = 0

    def logits_to_logit_diff(logits, clean_token_id, corrupted_token_id):
        last_token_logits = logits[0, -1, :]
        return last_token_logits[clean_token_id] - last_token_logits[corrupted_token_id]

    print(f"Running Iterative MLP Patching for {num_prompts} prompt pair(s) at position {pos_to_patch}...")
    for i in tqdm(range(num_prompts), desc="Processing Prompts"):
        clean_tokens = tokenizer(clean_prompts[i], return_tensors="pt").input_ids.to(device)
        corrupted_tokens = tokenizer(corrupted_prompts[i], return_tensors="pt").input_ids.to(device)
        clean_token_id = tokenizer.encode(clean_continuations[i], add_special_tokens=False)[0]
        corrupted_token_id = tokenizer.encode(corrupted_continuations[i], add_special_tokens=False)[0]

        with torch.no_grad():
            corrupted_logits = model(corrupted_tokens).logits
            clean_logits = model(clean_tokens).logits
            
            clean_pred_token_id = clean_logits[0, -1].argmax()
            corrupted_pred_token_id = corrupted_logits[0, -1].argmax()
            
            if clean_pred_token_id == clean_token_id:
                clean_correct_count += 1
            if corrupted_pred_token_id == corrupted_token_id:
                corrupted_correct_count += 1

            if clean_pred_token_id != clean_token_id or corrupted_pred_token_id != corrupted_token_id:
                print(f"\nWarning: Skipping prompt pair {i} for patching due to incorrect prediction.")
                if clean_pred_token_id != clean_token_id:
                    print(f"  Clean expected: {repr(clean_continuations[i])}, got: {repr(tokenizer.decode(clean_pred_token_id))}")
                if corrupted_pred_token_id != corrupted_token_id:
                    print(f"  Corrupted expected: {repr(corrupted_continuations[i])}, got: {repr(tokenizer.decode(corrupted_pred_token_id))}")
                continue
            
            clean_logit_diff = logits_to_logit_diff(clean_logits, clean_token_id, corrupted_token_id)
            corrupted_logit_diff = logits_to_logit_diff(corrupted_logits, clean_token_id, corrupted_token_id)
        
        if torch.isclose(clean_logit_diff, corrupted_logit_diff):
            print(f"Skipping prompt {i} for patching due to identical logit differences.")
            continue

        clean_cache = {}
        def cache_hook_fn(module, args, output, layer_idx):
            activation = output[0] if isinstance(output, tuple) else output
            clean_cache[layer_idx] = activation.detach()

        hooks = []
        for layer_idx in layers_to_patch_range:
            hook = mlp_layers[layer_idx].register_forward_hook(partial(cache_hook_fn, layer_idx=layer_idx))
            hooks.append(hook)
        
        with torch.no_grad():
            model(clean_tokens) 
        
        for hook in hooks:
            hook.remove()

        prompt_patching_results = torch.zeros(num_layers_to_patch, device=model.device)

        def patching_hook_fn(module, args, output, position_to_patch, cached_activation):
            activation = output[0] if isinstance(output, tuple) else output
            activation[:, position_to_patch, :] = cached_activation[:, position_to_patch, :]
            if isinstance(output, tuple):
                return (activation,) + output[1:]
            return activation

        with torch.no_grad():
            for layer_idx, layer in enumerate(layers_to_patch_range):
                target_mlp_layer = mlp_layers[layer]
                hook = target_mlp_layer.register_forward_hook(
                    partial(patching_hook_fn, position_to_patch=pos_to_patch, cached_activation=clean_cache[layer])
                )
                patched_logits = model(corrupted_tokens).logits
                hook.remove()

                patched_logit_diff = logits_to_logit_diff(patched_logits, clean_token_id, corrupted_token_id)
                normalized_diff = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
                prompt_patching_results[layer_idx] = normalized_diff
        
        all_prompts_patching_results[i] = prompt_patching_results
        valid_prompt_indices.append(i)
    
    if logger:
        logger.info(f"{model_name},{task_name},{clean_correct_count},{corrupted_correct_count},{num_prompts}")
        print(f"Logged performance for {model_name} on task {task_name}.")

    if not valid_prompt_indices:
        print("\nError: No valid prompt pairs were processed for patching. Halting analysis for this model.")
        return

    valid_prompts_count = len(valid_prompt_indices)
    valid_results = all_prompts_patching_results[valid_prompt_indices]
    
    averaged_patching_results = valid_results.mean(dim=0)
    std_err_results = valid_results.std(dim=0) / torch.sqrt(torch.tensor(valid_prompts_count))

    print(f"\nPatching Complete. Averaged results over {valid_prompts_count} valid prompt pair(s).")
    print("Generating contribution plot...")

    contribution_data_np = averaged_patching_results.cpu().numpy()
    error_data_np = std_err_results.cpu().numpy()
    y_labels = [f"L{i}" for i in layers_to_patch_range]

    plot_contributions_to_logit_diff(
        contribution_data=contribution_data_np,
        error_data=error_data_np,
        output_filename=f"{output_filename_base}_contributions_pos{pos_to_patch}.png",
        title_text=f"Individual MLP Layer Contributions for {model_name}\n(Position: {pos_to_patch}, N={valid_prompts_count})",
        y_labels=y_labels
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MLP activation patching experiments and plot contributions.")
    parser.add_argument("--model_names", type=str, nargs='+', required=True, help="One or more model names to test.")
    parser.add_argument("--prompts_file", type=str, required=True, help="Path to the JSON file with prompts.")
    parser.add_argument("--pos_to_patch", type=int, required=True, help="The single token position to patch the MLP outputs.")
    parser.add_argument("--layers_to_patch", type=int, nargs=2, required=True, help="Start and end layer indices to patch.")
    parser.add_argument("--output_filename_base", type=str, default="mlp_patching_results")
    parser.add_argument("--log_file", type=str, default="performance_log.csv", help="Path to the CSV log file for performance.")
    args = parser.parse_args()

    log_file_path = args.log_file
    file_exists = os.path.exists(log_file_path)
    
    logging.basicConfig(level=logging.INFO, format='%(message)s', filename=log_file_path, filemode='a')
    logger = logging.getLogger()
    
    if not file_exists or os.path.getsize(log_file_path) == 0:
        logger.info("model_name,task_name,clean_correct,corrupted_correct,total_prompts")

    with open(args.prompts_file, 'r') as f:
        data = json.load(f)

    CLEAN_PROMPTS = [item['clean_prompt'] for item in data['prompt_pairs']]
    CORRUPTED_PROMPTS = [item['corrupted_prompt'] for item in data['prompt_pairs']]
    CLEAN_CONTINUATIONS = [item['clean_continuation'] for item in data['prompt_pairs']]
    CORRUPTED_CONTINUATIONS = [item['corrupted_continuation'] for item in data['prompt_pairs']]
    
    task_name = os.path.basename(args.prompts_file)

    for model_name in args.model_names:
        print(f"\n{'='*25} Processing Model: {model_name} {'='*25}")
        sanitized_model_name = model_name.replace("/", "_")
        model_specific_output_base = f"{args.output_filename_base}_{sanitized_model_name}"
        run_activation_patching_and_plot(
            model_name=model_name,
            clean_prompts=CLEAN_PROMPTS,
            corrupted_prompts=CORRUPTED_PROMPTS,
            clean_continuations=CLEAN_CONTINUATIONS,
            corrupted_continuations=CORRUPTED_CONTINUATIONS,
            pos_to_patch=args.pos_to_patch,
            layers_to_patch=[args.layers_to_patch[0], args.layers_to_patch[1]],
            output_filename_base=model_specific_output_base,
            logger=logger,
            task_name=task_name
        )