from openai import OpenAI
import json
import os
import argparse
from dotenv import load_dotenv
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import constants
from pydantic import BaseModel
import importlib

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

# Define a Pydantic model for our structured LLM output
class LLMResponse(BaseModel):
    conversation: str
    label: str

def call_llm(sample, model, comb):
    """
    Constructs a user message from the sample details based on the combination flag and calls the specified LLM API.
    Returns a tuple containing:
      - predicted_label: the final emotion label in lower-case.
      - reasoning_conversation: the complete chain-of-thought from the LLM.
    Uses Structured Outputs with response_format.
    """
    message_parts = []
    # Add transcript if 'T' is in comb
    if "T" in comb:
        message_parts.append(f"The person in the video says: {sample.get('transcript', '')}")
    # Add audio cues if 'A' is in comb
    if "A" in comb:
        message_parts.append(f"Audio cues: {sample.get('audio_description', '')}")
    # Add visual cues if 'V' is in comb
    if "V" in comb:
        visual_cues = sample.get("visual_objective_description", "")
        if isinstance(visual_cues, list):
            visual_cues = ", ".join(visual_cues)
        message_parts.append(f"Visual cues: {visual_cues}")
    # Add reasoning caption if 'R' is in comb
    if "R" in comb:
        message_parts.append(f"Reasoning caption: {sample.get('smp_reason_caption', '')}")

    user_message = "\n".join(message_parts)
    try:
        # Call the beta endpoint using structured outputs via response_format.
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            response_format=LLMResponse,
        )
        # Extract the parsed values from the response.
        predicted_label = completion.choices[0].message.parsed.label.strip().lower()
        reasoning_conversation = completion.choices[0].message.parsed.conversation
    except Exception as e:
        print(f"Error processing sample: {e}")
        predicted_label = "error"
        reasoning_conversation = ""
    
    return predicted_label, reasoning_conversation

def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description="Run LLM model predictions")
    parser.add_argument("--input", type=str, default="MERR_fine_grained.json", help="Input JSON file path")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--output", type=str, default=None, help="Output file name for the results")
    parser.add_argument("--comb", type=str, choices=["T", "TV", "TA", "AV", "TAV", "RTAV"], default="T",
                        help="Specify the combination of modalities to use: T (text), TV (text and visual), TA (text and audio), AV (audio and visual), or TAV (all three)")
    parser.add_argument("--prompt", type=str, choices=["", "TOT", "COT", "TOT-3-EXPERT-UNI", "TOT-4-EXPERT-UNI", 
                                                       "TOT-3-EXPERT-BI", "TOT-4-EXPERT-BI",
                                                       "TOT-3-EXPERT-DEBATE-UNI", "TOT-4-EXPERT-DEBATE-UNI",
                                                       "TOT-3-EXPERT-DEBATE-BI", "TOT-4-EXPERT-DEBATE-BI"],
                        default="", help="Select prompt type from the available options")
    parser.add_argument("--dataset", type=str, choices=["MER", "MELD"], default="MER", help="Dataset to use: MER or MELD")
    args = parser.parse_args()
    
    # Dynamically import the correct constants module
    if args.dataset == "MELD":
        constants_mod = importlib.import_module("MELD_constants")
        print("MELD constants imported")
    elif args.dataset == "MER":
        constants_mod = importlib.import_module("MER_constants")
        print("MER constants imported")
    else:
        constants_mod = importlib.import_module("constants")
        print("Constants imported")

    # Update the global SYSTEM_PROMPT and PROMPT_MAPPING based on the dataset
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = constants_mod.SYSTEM_PROMPT
    PROMPT_MAPPING = {
        "COT": constants_mod.SYSTEM_PROMPT_CHAIN_OF_THOUGHT,
        "TOT": constants_mod.SYSTEM_PROMPT_TREE_OF_THOUGHT,
        "TOT-3-EXPERT-UNI": constants_mod.SYSTEM_PROMPT_TREE_OF_THOUGHT_UNIMODAL_3_EXPERT,
        "TOT-4-EXPERT-UNI": constants_mod.SYSTEM_PROMPT_TREE_OF_THOUGHT_UNIMODAL_4_EXPERT,
        "TOT-3-EXPERT-BI": constants_mod.SYSTEM_PROMPT_TREE_OF_THOUGHT_BIMODAL_3_EXPERT,
        "TOT-4-EXPERT-BI": constants_mod.SYSTEM_PROMPT_TREE_OF_THOUGHT_BIMODAL_4_EXPERT,
        "TOT-3-EXPERT-DEBATE-UNI": constants_mod.SYSTEM_PROMPT_TREE_OF_THOUGHT_UNIMODAL_3_EXPERT_DEBATE,
        "TOT-4-EXPERT-DEBATE-UNI": constants_mod.SYSTEM_PROMPT_TREE_OF_THOUGHT_UNIMODAL_4_EXPERT_DEBATE,
        "TOT-3-EXPERT-DEBATE-BI": constants_mod.SYSTEM_PROMPT_TREE_OF_THOUGHT_BIMODAL_3_EXPERT_DEBATE,
        "TOT-4-EXPERT-DEBATE-BI": constants_mod.SYSTEM_PROMPT_TREE_OF_THOUGHT_BIMODAL_4_EXPERT_DEBATE,
    }
    if args.prompt in PROMPT_MAPPING:
        SYSTEM_PROMPT = PROMPT_MAPPING[args.prompt]
    else:
        SYSTEM_PROMPT = constants_mod.SYSTEM_PROMPT

    input_file = args.input  # Get the input file path
    selected_model = args.model
    comb_flag = args.comb
    # Determine output file name based on provided argument or default to model name.
    output_file_name = args.output if args.output is not None else f"{selected_model}_results.txt"
    
    # Load the JSON data from file.
    with open(input_file, "r") as f:
        data = json.load(f)
        print("Data loaded")
    
    results = []       # List to hold structured results for each sample.
    ground_truths = [] # For evaluation.
    predictions = []   # For evaluation.
    
    # Process each sample in the JSON.
    for sample_id, sample in enumerate(data):
        predicted, reasoning = call_llm(sample, selected_model, comb_flag)
        video_id = sample.get("video_id", f"sample_{sample_id}")
        ground_truth = sample.get("true_label", "").strip().lower()
        ground_truths.append(ground_truth)
        predictions.append(predicted)
        print(f"Video {video_id}: Ground Truth: {ground_truth}, Predicted: {predicted}")
        # Store the structured result.
        result_entry = {
            "video_id": video_id,
            "ground_truth": ground_truth,
            "predicted_label": predicted,
            "reasoning": reasoning
        }
        results.append(result_entry)
    
    # Define the set of possible labels.
    if args.dataset == "MELD":
        labels = ["anger", "disgust", "sadness", "joy", "neutral", "surprise", "fear"]
    else:
        labels = ["happy", "sad", "neutral", "angry", "worried", "surprise"]
    
    # Compute evaluation metrics.
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truths, predictions, labels=labels, zero_division=0
    )
    overall_accuracy = accuracy_score(ground_truths, predictions)
    
    print("\nEvaluation Metrics:")
    print("Overall Accuracy: {:.2f}".format(overall_accuracy))
    
    metrics_summary = [f"Overall Accuracy: {overall_accuracy:.2f}"]
    for i, label in enumerate(labels):
        metric_line = (
            f"Label: {label} -> Precision: {precision[i]:.2f}, Recall: {recall[i]:.2f}, "
            f"F1 Score: {f1[i]:.2f}, Support: {support[i]}"
        )
        print(metric_line)
        metrics_summary.append(metric_line)
    
    # Create results directory if it doesn't exist.
    os.makedirs("results", exist_ok=True)
    
    # Save structured results as a JSON file.
    output_dir = os.path.dirname(output_file_name)
    if output_dir:  # Only create directory if there's a path
        os.makedirs(output_dir, exist_ok=True)
    
    # Save structured results as a JSON file.
    json_output_file = os.path.join(output_file_name.replace(".txt", ".json"))
    with open(json_output_file, "w") as f:
        json.dump(results, f, indent=2)
        
    with open(output_file_name, "w") as f:
        f.write("PREDICTION RESULTS\n")
        f.write("=================\n\n")
        for entry in results:
            f.write(f"Video {entry['video_id']}: Ground Truth: {entry['ground_truth']}, Predicted: {entry['predicted_label']}\n")
        f.write("\nEVALUATION METRICS\n")
        f.write("=================\n\n")
        f.write("\n".join(metrics_summary))
    
    print(f"\nResults saved to {output_file_name}")

if __name__ == "__main__":
    main()
