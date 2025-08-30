import torch
import pandas as pd
from transformer_lens import HookedTransformer
from tqdm.auto import tqdm
import argparse
import json
from functools import partial

def run_neuron_ablation(
    model_name,
    prompts_file,
    neurons_csv,
    layer_to_ablate,
    pos_to_ablate=-1,
    top_n_neurons=20,
    output_csv="ablation_results.csv"
):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"--- Using device: {device} ---")

    model = HookedTransformer.from_pretrained(model_name, trust_remote_code=True, device=device)
    model.eval()
    
    print(f"--- Performing ablation on MLP Layer {layer_to_ablate} at Position {pos_to_ablate} ---")

    with open(prompts_file, 'r') as f:
        data = json.load(f)

    prompts = data['prompt_pairs']

    neuron_df = pd.read_csv(neurons_csv)
    neurons_to_ablate = neuron_df['neuron_index'].head(top_n_neurons).tolist()
    print(f"--- Loaded Top {len(neurons_to_ablate)} neurons to ablate from {neurons_csv} ---")

    def logits_to_logit_diff(logits, clean_token_id, corrupted_token_id):
        last_token_logits = logits[0, -1, :]
        return last_token_logits[clean_token_id] - last_token_logits[corrupted_token_id]

    def ablation_hook(mlp_post, hook, neuron_idx, position):
        mlp_post[0, position, neuron_idx] = 0
        return mlp_post

    hook_name = f"blocks.{layer_to_ablate}.mlp.hook_post"
    
    ablation_results = {}

    for neuron_idx in tqdm(neurons_to_ablate, desc="Ablating Neurons"):
        effects = []
        for prompt_pair in prompts:
            clean_prompt = prompt_pair['clean_prompt']
            clean_continuation = prompt_pair['clean_continuation']
            corrupted_continuation = prompt_pair['corrupted_continuation']
            
            clean_tokens = model.to_tokens(clean_prompt)
            clean_token_id = model.to_single_token(clean_continuation)
            corrupted_token_id = model.to_single_token(corrupted_continuation)

            with torch.no_grad():
                original_logits = model(clean_tokens)
                original_logit_diff = logits_to_logit_diff(original_logits, clean_token_id, corrupted_token_id)

                hook_fn = partial(ablation_hook, neuron_idx=neuron_idx, position=pos_to_ablate)
                
                ablated_logits = model.run_with_hooks(
                    clean_tokens,
                    fwd_hooks=[(hook_name, hook_fn)]
                )
                ablated_logit_diff = logits_to_logit_diff(ablated_logits, clean_token_id, corrupted_token_id)
                
                effect = ablated_logit_diff - original_logit_diff
                effects.append(effect.item())

        ablation_results[neuron_idx] = sum(effects) / len(effects)

    print("\n--- Ablation Complete ---")
    
    results_df = pd.DataFrame(
        list(ablation_results.items()), 
        columns=['neuron_index', 'avg_effect_on_logit_diff']
    )
    
    results_df_sorted = results_df.sort_values(
        by='avg_effect_on_logit_diff', 
        ascending=True
    ).reset_index(drop=True)

    results_df_sorted.to_csv(output_csv, index=False)
    print(f"--- Saved ablation results to {output_csv} ---")

    print(f"\n--- Top {top_n_neurons} Neurons by Impact on Logit Difference (Lower is more impactful) ---")
    print(results_df_sorted.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run neuron ablation experiments to measure their effect on logit difference."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name from Hugging Face."
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        required=True,
        help="Path to the JSON file with prompt pairs."
    )
    parser.add_argument(
        "--neurons_csv",
        type=str,
        required=True,
        help="Path to the CSV file containing the neuron indices to ablate."
    )
    parser.add_argument(
        "--layer_to_ablate",
        type=int,
        required=True,
        help="The MLP layer index to perform ablation on."
    )
    parser.add_argument(
        "--pos_to_ablate",
        type=int,
        default=-1,
        help="The token position to ablate (default: -1)."
    )
    parser.add_argument(
        "--top_n_neurons",
        type=int,
        default=20,
        help="Number of top neurons from the CSV to test."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="ablation_results.csv",
        help="Name of the output CSV file for ablation results."
    )
    args = parser.parse_args()

    run_neuron_ablation(
        model_name=args.model_name,
        prompts_file=args.prompts_file,
        neurons_csv=args.neurons_csv,
        layer_to_ablate=args.layer_to_ablate,
        pos_to_ablate=args.pos_to_ablate,
        top_n_neurons=args.top_n_neurons,
        output_csv=args.output_csv
    )