import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformer_lens import HookedTransformer
from tqdm.auto import tqdm
import argparse
import json
from functools import partial

def test_token_group_specificity(
    model_name,
    prompts_file,
    neurons_csv,
    layer_to_ablate,
    top_n_neurons=5,
    pos_to_ablate=-1
):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"--- Using device: {device} ---")

    model = HookedTransformer.from_pretrained(model_name, trust_remote_code=True, device=device)
    model.eval()
    
    if layer_to_ablate < 0:
        layer_to_ablate = model.cfg.n_layers + layer_to_ablate

    print(f"--- Testing token group specificity by ablating layer {layer_to_ablate} ---")

    with open(prompts_file, 'r') as f:
        prompts = json.load(f)['prompt_pairs']

    neuron_df = pd.read_csv(neurons_csv)
    top_suppressive_neurons = neuron_df.sort_values(
        by='avg_effect_on_logit_diff', 
        ascending=False
    )['neuron_index'].head(top_n_neurons).tolist()
    
    print(f"--- Ablating Top {top_n_neurons} suppressive neurons: {top_suppressive_neurons} ---")

    all_choice_tokens_str = [' A', ' B', ' C', ' D']
    unrelated_token_str = ' Dog'
    unrelated_token_id = model.to_single_token(unrelated_token_str)
    all_choice_token_ids = [model.to_single_token(t) for t in all_choice_tokens_str]

    def ablation_hook(mlp_post, hook, neurons_to_ablate_list, position):
        if neurons_to_ablate_list:
            mlp_post[0, position, neurons_to_ablate_list] = 0
        return mlp_post

    hook_name = f"blocks.{layer_to_ablate}.mlp.hook_post"
    
    changes_correct = []
    changes_incorrect_choices = []
    changes_unrelated = []

    for prompt_pair in tqdm(prompts, desc="Processing Prompts"):
        clean_prompt = prompt_pair['clean_prompt']
        correct_token_str = prompt_pair['clean_continuation']
        
        tokens = model.to_tokens(clean_prompt)
        
        correct_token_id = model.to_single_token(correct_token_str)
        incorrect_choice_token_ids = [t_id for t_id in all_choice_token_ids if t_id != correct_token_id]

        with torch.no_grad():
            baseline_logits = model(tokens)
            
            hook_fn = partial(ablation_hook, neurons_to_ablate_list=top_suppressive_neurons, position=pos_to_ablate)
            intervened_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, hook_fn)]
            )
            
            change_correct = (intervened_logits[0, -1, correct_token_id] - baseline_logits[0, -1, correct_token_id]).item()
            changes_correct.append(change_correct)

            change_unrelated = (intervened_logits[0, -1, unrelated_token_id] - baseline_logits[0, -1, unrelated_token_id]).item()
            changes_unrelated.append(change_unrelated)

            incorrect_changes_for_prompt = []
            for token_id in incorrect_choice_token_ids:
                change = (intervened_logits[0, -1, token_id] - baseline_logits[0, -1, token_id]).item()
                incorrect_changes_for_prompt.append(change)
            
            if incorrect_changes_for_prompt:
                avg_incorrect_change = sum(incorrect_changes_for_prompt) / len(incorrect_changes_for_prompt)
                changes_incorrect_choices.append(avg_incorrect_change)

    avg_results = {
        "Correct Token": sum(changes_correct) / len(changes_correct),
        "Avg. Incorrect Choices": sum(changes_incorrect_choices) / len(changes_incorrect_choices),
        f"Unrelated Token ('{unrelated_token_str}')": sum(changes_unrelated) / len(changes_unrelated)
    }

    print("\n--- Experiment Complete ---")
    print("Average change in logit for each token group after ablation:")
    for group, change in avg_results.items():
        print(f"  - {group}: {change:.4f}")
        
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    groups = list(avg_results.keys())
    values = list(avg_results.values())
    
    colors = sns.color_palette("plasma", n_colors=len(groups))
    
    bars = ax.bar(groups, values, color=colors)
    
    ax.set_title(f'Effect of Ablating Top {top_n_neurons} Suppressive Neurons on Token Groups', fontsize=16, pad=20)
    ax.set_xlabel('Token Group', fontsize=12)
    ax.set_ylabel('Average Change in Logit', fontsize=12)
    ax.axhline(0, color='grey', linewidth=0.8)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom' if yval >=0 else 'top', ha='center')

    output_filename = "token_group_specificity.png"
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"--- Graph saved to {output_filename} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the specificity of neuron ablation on different token groups."
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--neurons_csv", type=str, required=True)
    parser.add_argument("--layer_to_ablate", type=int, required=True)
    parser.add_argument("--top_n_neurons", type=int, default=5)
    args = parser.parse_args()

    test_token_group_specificity(
        model_name=args.model_name,
        prompts_file=args.prompts_file,
        neurons_csv=args.neurons_csv,
        layer_to_ablate=args.layer_to_ablate,
        top_n_neurons=args.top_n_neurons
    )