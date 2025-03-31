import argparse
import json
import os
from dotenv import load_dotenv
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from openai import OpenAI
from pydantic import BaseModel
import constants

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Define a structured output model for emotion detection tasks.
class EmotionDetectionResult(BaseModel):
    emotion: str
    reasoning: str

def call_gpt4o_mini(sample, system_prompt):
    """
    Constructs a user message from the sample details and calls the gpt-4o-mini API.
    Returns a tuple (predicted emotion label, reasoning text) by parsing the structured JSON output.
    """
    smp_reason_caption = sample.get("smp_reason_caption", "")
    user_message = f"{smp_reason_caption}"
    
    try:
        # Instruct the model to output a JSON object with two keys: 'emotion' and 'reasoning'.
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        system_prompt
                    )
                },
                {"role": "user", "content": user_message}
            ],
            response_format=EmotionDetectionResult,
        )
        result = completion.choices[0].message.parsed
        predicted_label = result.emotion.strip().lower()
        reasoning = result.reasoning.strip()
    except Exception as e:
        print(f"Error processing sample: {e}")
        predicted_label = "error"
        reasoning = ""
    
    return predicted_label, reasoning

def main(mode):
    # Choose the appropriate system prompt and file name based on mode.
    if mode == "tree_of_thought":
        system_prompt = constants.SYSTEM_PROMPT_TREE_OF_THOUGHT
        file_name = "gpt4o_mini_results_tree_of_thought.txt"
    elif mode == "chain_of_thought":
        system_prompt = constants.SYSTEM_PROMPT_CHAIN_OF_THOUGHT
        file_name = "gpt4o_mini_results_chain_of_thought.txt"
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    # Load the JSON data from file.
    with open("MERR_fine_grained.json", "r") as f:
        data = json.load(f)
        print("data loaded")
    
    predictions = []
    ground_truths = []
    result_details = []
    
    # Process each sample in the JSON.
    for i, (sample_id, sample) in enumerate(data.items()):
        predicted, reasoning = call_gpt4o_mini(sample, system_prompt)
        predictions.append(predicted)
        # The ground truth emotion label is stored in 'pseu_emotion'
        ground_truth = sample.get("pseu_emotion", "").strip().lower()
        ground_truths.append(ground_truth)
        detail = (
            f"Sample {sample_id}:\n"
            f"  Ground Truth: {ground_truth}\n"
            f"  Predicted: {predicted}\n"
            f"  Reasoning: {reasoning}\n"
        )
        print(detail)
        result_details.append(detail)
        if i >= 1:
            break
    
    # Define the set of possible labels.
    labels = ["happy", "sad", "neutral", "angry", "worried", "surprise", "fear", "contempt", "doubt"]
    
    # Compute precision, recall, and F1 score for each label.
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truths, predictions, labels=labels, zero_division=0
    )
    overall_accuracy = accuracy_score(ground_truths, predictions)
    
    print("\nEvaluation Metrics:")
    print("Overall Accuracy: {:.2f}".format(overall_accuracy))
    
    # Create metrics summary
    metrics_summary = [f"Overall Accuracy: {overall_accuracy:.2f}"]
    for i, label in enumerate(labels):
        metric_line = (
            f"Label: {label} -> Precision: {precision[i]:.2f}, "
            f"Recall: {recall[i]:.2f}, F1 Score: {f1[i]:.2f}, Support: {support[i]}"
        )
        print(metric_line)
        metrics_summary.append(metric_line)
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Save results to file
    with open(f"results/{file_name}", "w") as f:
        f.write("PREDICTION RESULTS\n")
        f.write("=================\n\n")
        f.write("\n".join(result_details))
        f.write("\n\nEVALUATION METRICS\n")
        f.write("=================\n\n")
        f.write("\n".join(metrics_summary))
    
    print(f"\nResults saved to results/{file_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gpt-4o-mini evaluation with different prompting modes.")
    parser.add_argument(
        "--mode",
        type=str,
        default="chain_of_thought",
        choices=["chain_of_thought", "tree_of_thought"],
        help="Select prompt type to use: 'chain_of_thought' or 'tree_of_thought'."
    )
    args = parser.parse_args()
    main(args.mode)
