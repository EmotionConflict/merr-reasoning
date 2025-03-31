from openai import OpenAI
import json
import os
import argparse
from dotenv import load_dotenv
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import constants

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = constants.SYSTEM_PROMPT

def call_llm(sample, model):
    """
    Constructs a user message from the sample details and calls the specified LLM API.
    Returns the predicted emotion label in lower-case.
    """
    smp_reason_caption = sample.get("smp_reason_caption", "")
    user_message = f"{smp_reason_caption}"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        )
        predicted_label = response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"Error processing sample: {e}")
        predicted_label = "error"
    
    return predicted_label

def main():
    # Parse the model flag from command line arguments.
    parser = argparse.ArgumentParser(description="Run LLM model predictions")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model to use")
    args = parser.parse_args()
    
    selected_model = args.model
    
    # Load the JSON data from file.
    with open("MERR_fine_grained.json", "r") as f:
        data = json.load(f)
        print("Data loaded")
    
    predictions = []
    ground_truths = []
    result_details = []
    
    # Process each sample in the JSON.
    for sample_id, sample in data.items():
        predicted = call_llm(sample, selected_model)
        predictions.append(predicted)
        ground_truth = sample.get("pseu_emotion", "").strip().lower()
        ground_truths.append(ground_truth)
        print(f"Sample {sample_id}: Ground Truth: {ground_truth}, Predicted: {predicted}")
        result_details.append(f"Sample {sample_id}: Ground Truth: {ground_truth}, Predicted: {predicted}")
    
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
            f"Label: {label} -> Precision: {precision[i]:.2f}, Recall: {recall[i]:.2f}, "
            f"F1 Score: {f1[i]:.2f}, Support: {support[i]}"
        )
        print(metric_line)
        metrics_summary.append(metric_line)
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Use the selected model name as part of the output file name.
    file_name = f"{selected_model}_results.txt"
    with open(os.path.join("results", file_name), "w") as f:
        f.write("PREDICTION RESULTS\n")
        f.write("=================\n\n")
        f.write("\n".join(result_details))
        f.write("\n\nEVALUATION METRICS\n")
        f.write("=================\n\n")
        f.write("\n".join(metrics_summary))
    
    print(f"\nResults saved to results/{file_name}")

if __name__ == "__main__":
    main()
