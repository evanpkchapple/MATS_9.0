import torch
import pandas as pd
from transformer_lens import HookedTransformer
from tqdm.auto import tqdm
import argparse
import json

def find_key_neurons(
    model_name,
    prompts_file,
    layer_to_analyze,
    pos_to_analyze,
    output_csv,
    top_k=10
):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"--- Using device: {device} ---")

    model = HookedTransformer.from_pretrained(model_name, trust_remote_code=True, device=device)
    model.eval()

    if layer_to_analyze < 0:
        layer_to_analyze = model.cfg.n_layers + layer_to_analyze
    
    print(f"--- Analyzing MLP Layer {layer_to_analyze} at Position {pos_to_analyze} ---")

    with open(prompts_file, 'r') as f:
        data = json.load(f)

    clean_prompts = [item['clean_prompt'] for item in data['prompt_pairs']]
    corrupted_prompts = [item['corrupted_prompt'] for item in data['prompt_pairs']]
    clean_continuations = [item['clean_continuation'] for item in data['prompt_pairs']]
    corrupted_continuations = [item['corrupted_continuation'] for item in data['prompt_pairs']]

    captured_activations = {}
    
    def activation_capture_hook(mlp_post, hook):
        captured_activations[hook.name] = mlp_post[0, pos_to_analyze, :].detach()
        return mlp_post

    hook_name = f"blocks.{layer_to_analyze}.mlp.hook_post"
    
    clean_activations_list = []
    corrupted_activations_list = []
    abs_diff_activations_list = []
    
    print(f"--- Processing {len(clean_prompts)} prompt pairs ---")
    for i in tqdm(range(len(clean_prompts)), desc="Analyzing Prompts"):
        clean_prompt, corrupted_prompt = clean_prompts[i], corrupted_prompts[i]
        clean_token_id = model.to_single_token(clean_continuations[i])
        corrupted_token_id = model.to_single_token(corrupted_continuations[i])
        
        clean_tokens = model.to_tokens(clean_prompt)
        corrupted_tokens = model.to_tokens(corrupted_prompt)

        with torch.no_grad():
            model.run_with_hooks(
                clean_tokens,
                fwd_hooks=[(hook_name, activation_capture_hook)]
            )
            clean_mlp_activations = captured_activations[hook_name].clone()
            
            model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(hook_name, activation_capture_hook)]
            )
            corrupted_mlp_activations = captured_activations[hook_name].clone()

            # We don't need to check predictions as per the original script's logic,
            # but if needed, that validation would go here.
            
            clean_activations_list.append(clean_mlp_activations)
            corrupted_activations_list.append(corrupted_mlp_activations)
            
            diff = clean_mlp_activations - corrupted_mlp_activations
            abs_diff_activations_list.append(torch.abs(diff))

    if not abs_diff_activations_list:
        print("\nError: No valid prompt pairs were processed. Cannot generate CSV.")
        return

    print(f"\n--- Analysis complete. Found {len(abs_diff_activations_list)} valid prompt pairs. ---")
    
    avg_clean_activations = torch.stack(clean_activations_list).mean(dim=0)
    avg_corrupted_activations = torch.stack(corrupted_activations_list).mean(dim=0)
    mean_abs_difference = torch.stack(abs_diff_activations_list).mean(dim=0)

    print(f"--- Saving neuron activation data to {output_csv} ---")
    df = pd.DataFrame({
        'neuron_index': range(model.cfg.d_mlp),
        'avg_clean_activation': avg_clean_activations.cpu().numpy(),
        'avg_corrupted_activation': avg_corrupted_activations.cpu().numpy(),
        'mean_absolute_difference': mean_abs_difference.cpu().numpy()
    })

    df_sorted = df.sort_values(by='mean_absolute_difference', ascending=False).reset_index(drop=True)
    df_sorted.to_csv(output_csv, index=False)

    print(f"Successfully saved data for {len(df_sorted)} neurons.")

    print(f"\n--- Top {top_k} Neurons with Largest Mean Absolute Difference ---")
    print(df_sorted.head(top_k).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find key neurons by analyzing MLP activation differences."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name from Hugging Face (e.g., 'Qwen/Qwen3-4B')."
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        required=True,
        help="Path to the JSON file with prompt pairs."
    )
    parser.add_argument(
        "--layer_to_analyze",
        type=int,
        required=True,
        help="The MLP layer index to analyze (e.g., 35, or -1 for the last layer)."
    )
    parser.add_argument(
        "--pos_to_analyze",
        type=int,
        default=-1,
        help="The token position to analyze (default: -1 for the last token)."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="neuron_activations.csv",
        help="Name of the output CSV file."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top differing neurons to display after running."
    )
    
    args = parser.parse_args()

    find_key_neurons(
        model_name=args.model_name,
        prompts_file=args.prompts_file,
        layer_to_analyze=args.layer_to_analyze,
        pos_to_analyze=args.pos_to_analyze,
        output_csv=args.output_csv,
        top_k=args.top_k
    )