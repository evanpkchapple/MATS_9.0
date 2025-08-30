import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformer_lens import HookedTransformer
from tqdm.auto import tqdm
import argparse
import json
from functools import partial

def run_patching_with_ablation(
    model_name,
    prompts_file,
    neurons_csv,
    layer_to_intervene,
    pos_to_intervene=-1,
    num_ablations_to_test=4
):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"--- Using device: {device} ---")

    model = HookedTransformer.from_pretrained(model_name, trust_remote_code=True, device=device)
    model.eval()
    
    if layer_to_intervene < 0:
        layer_to_intervene = model.cfg.n_layers + layer_to_intervene

    print(f"--- Intervening on MLP Layer {layer_to_intervene} at Position {pos_to_intervene} ---")

    with open(prompts_file, 'r') as f:
        prompts = json.load(f)['prompt_pairs']

    neuron_df = pd.read_csv(neurons_csv)
    # Sort by descending effect to find suppressive neurons (most positive effect)
    top_suppressive_neurons = neuron_df.sort_values(
        by='avg_effect_on_logit_diff', 
        ascending=False
    )['neuron_index'].head(num_ablations_to_test).tolist()
    
    print(f"--- Top {num_ablations_to_test} suppressive neurons identified for ablation: {top_suppressive_neurons} ---")

    def logits_to_logit_diff(logits, clean_token_id, corrupted_token_id):
        last_token_logits = logits[0, -1, :]
        return last_token_logits[clean_token_id] - last_token_logits[corrupted_token_id]

    def combined_patch_ablate_hook(mlp_post, hook, clean_cache, neurons_to_ablate_list, position):
        # Step 1: Patch the full activation from the clean run
        mlp_post[0, position, :] = clean_cache[hook.name][0, position, :]
        # Step 2: If any neurons are specified, ablate them (set to 0)
        if neurons_to_ablate_list:
            mlp_post[0, position, neurons_to_ablate_list] = 0
        return mlp_post

    hook_name = f"blocks.{layer_to_intervene}.mlp.hook_post"
    
    final_results = []

    for num_ablated in range(num_ablations_to_test + 1):
        
        neurons_to_ablate_this_run = top_suppressive_neurons[:num_ablated]
        
        run_logit_diffs = []
        desc = f"Testing with {num_ablated} ablations"
        
        for prompt_pair in tqdm(prompts, desc=desc):
            clean_prompt = prompt_pair['clean_prompt']
            corrupted_prompt = prompt_pair['corrupted_prompt']
            clean_continuation = prompt_pair['clean_continuation']
            corrupted_continuation = prompt_pair['corrupted_continuation']

            clean_tokens = model.to_tokens(clean_prompt)
            corrupted_tokens = model.to_tokens(corrupted_prompt)
            clean_token_id = model.to_single_token(clean_continuation)
            corrupted_token_id = model.to_single_token(corrupted_continuation)

            with torch.no_grad():
                _, clean_cache = model.run_with_cache(clean_tokens)
                
                hook_fn = partial(
                    combined_patch_ablate_hook, 
                    clean_cache=clean_cache, 
                    neurons_to_ablate_list=neurons_to_ablate_this_run, 
                    position=pos_to_intervene
                )
                
                patched_logits = model.run_with_hooks(
                    corrupted_tokens,
                    fwd_hooks=[(hook_name, hook_fn)]
                )
                
                patched_logit_diff = logits_to_logit_diff(patched_logits, clean_token_id, corrupted_token_id)
                run_logit_diffs.append(patched_logit_diff.item())
        
        avg_logit_diff = sum(run_logit_diffs) / len(run_logit_diffs)
        final_results.append(avg_logit_diff)

    print("\n--- Experiment Complete ---")
    print(f"Average Logit Differences: {final_results}")
    
    # --- Plotting the results ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_labels = [f'{i} Neurons' for i in range(num_ablations_to_test + 1)]
    x_labels[0] = '0 Neurons\n(Patch Only)'
    
    colors = sns.color_palette("coolwarm", n_colors=len(x_labels))
    
    bars = ax.bar(x_labels, final_results, color=colors)
    
    ax.set_title('Effect of Ablating Suppressive Neurons on Final Layer Patching', fontsize=16, pad=20)
    ax.set_xlabel('Number of Top Suppressive Neurons Ablated', fontsize=12)
    ax.set_ylabel('Average Logit Difference', fontsize=12)
    ax.axhline(0, color='grey', linewidth=0.8)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom' if yval >=0 else 'top', ha='center')

    output_filename = "patch_plus_ablation_effect.png"
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"--- Graph saved to {output_filename} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run combined patching and ablation experiments."
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--neurons_csv", type=str, required=True)
    parser.add_argument("--layer_to_intervene", type=int, required=True)
    parser.add_argument("--pos_to_intervene", type=int, default=-1)
    parser.add_argument("--num_ablations_to_test", type=int, default=4)
    args = parser.parse_args()

    run_patching_with_ablation(
        model_name=args.model_name,
        prompts_file=args.prompts_file,
        neurons_csv=args.neurons_csv,
        layer_to_intervene=args.layer_to_intervene,
        pos_to_intervene=args.pos_to_intervene,
        num_ablations_to_test=args.num_ablations_to_test
    )