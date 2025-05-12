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

def call_llm(sample, model, comb, i):
    """
    Constructs a user message from the sample details based on the combination flag and calls the specified LLM API.
    Returns the predicted emotion label in lower-case.
    Also appends the full parsed JSON output to a global list for saving later.
    """
    message_parts = []
    # Add transcript if 'T' is in comb
    if "T" in comb:
        message_parts.append(f"whisper_transcript: {sample.get('transcript', '')}")
    # Add audio_cues if 'A' is in comb
    if "A" in comb:
        message_parts.append(f"audio_description: {sample.get('audio_description', '')}")
    # Add visual_cues if 'V' is in comb
    if "V" in comb:
        visual_cues = sample.get("visual_expression_description", "")
        if isinstance(visual_cues, list):
            visual_cues = ", ".join(visual_cues)
        message_parts.append(f"visual_expression_description: {visual_cues}")
        message_parts.append(f"visual_objective_description: {sample.get('visual_objective_description', '')}")
    # Add reasoning caption if 'R' is in comb
    if "R" in comb:
        message_parts.append(f"Reasoning caption: {sample.get('smp_reason_caption', '')}")
    user_message = "\n\n".join(message_parts)

    user_message = "\n".join(message_parts)
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        )
        predicted_label = response.choices[0].message.content.strip()
        parsed = None
        # Try to parse as JSON directly
        try:
            parsed = json.loads(predicted_label)
        except json.JSONDecodeError:
            # If wrapped in triple backticks, strip them and try again
            if predicted_label.startswith("```json"):
                json_str = predicted_label.strip().removeprefix("```json").removesuffix("```").strip()
                try:
                    parsed = json.loads(json_str)
                except Exception:
                    pass
        if parsed and isinstance(parsed, dict):
            # Build the output entry with all fields from the LLM's JSON output
            prediction_json_entry = dict(parsed)
            # Add video_id and ground_truth from the sample
            prediction_json_entry["video_id"] = sample.get("video_id", f"sample_{i}")
            prediction_json_entry["ground_truth"] = sample.get("true_label", "").strip().lower()
            # Append to a global list for saving later
            if not hasattr(call_llm, "json_results"):
                call_llm.json_results = []
            call_llm.json_results.append(prediction_json_entry)
            predicted_label = parsed.get("first_emotion", "").strip().lower()
        else:
            predicted_label = predicted_label.strip().lower()
    except Exception as e:
        print(f"Error processing sample: {e}")
        predicted_label = "error"
    return predicted_label

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
    parser.add_argument("--dataset", type=str, choices=["MER", "MELD", "IEMOCAP"], default="MER", help="Dataset to use: MER or MELD")
    args = parser.parse_args()
    
    # Dynamically import the correct constants module
    if args.dataset == "MELD":
        constants_mod = importlib.import_module("neurips.MELD_constants")
        print("MELD constants imported")
    elif args.dataset == "MER":
        constants_mod = importlib.import_module("neurips.MER_constants")
        print("MER constants imported")
    elif args.dataset == "IEMOCAP":
        constants_mod = importlib.import_module("neurips.IEMOCAP_constants")
        print("IEMOCAP constants imported")
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    # Update the global SYSTEM_PROMPT and PROMPT_MAPPING based on the dataset
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = constants_mod.SYSTEM_PROMPT
    PROMPT_MAPPING = {
        "COT": constants_mod.SYSTEM_PROMPT_CHAIN_OF_THOUGHT,
        "TOT": constants_mod.SYSTEM_PROMPT_TREE_OF_THOUGHT,
        "TOT-3-EXPERT-UNI": constants_mod.SYSTEM_PROMPT_TREE_OF_THOUGHT_UNIMODAL_3_EXPERT,
        # "TOT-4-EXPERT-UNI": constants_mod.SYSTEM_PROMPT_TREE_OF_THOUGHT_UNIMODAL_4_EXPERT,
        "TOT-3-EXPERT-BI": constants_mod.SYSTEM_PROMPT_TREE_OF_THOUGHT_BIMODAL_3_EXPERT,
        # "TOT-4-EXPERT-BI": constants_mod.SYSTEM_PROMPT_TREE_OF_THOUGHT_BIMODAL_4_EXPERT,
        "TOT-3-EXPERT-DEBATE-UNI": constants_mod.SYSTEM_PROMPT_TREE_OF_THOUGHT_UNIMODAL_3_EXPERT_DEBATE,
        # "TOT-4-EXPERT-DEBATE-UNI": constants_mod.SYSTEM_PROMPT_TREE_OF_THOUGHT_UNIMODAL_4_EXPERT_DEBATE,
        "TOT-3-EXPERT-DEBATE-BI": constants_mod.SYSTEM_PROMPT_TREE_OF_THOUGHT_BIMODAL_3_EXPERT_DEBATE,
        # "TOT-4-EXPERT-DEBATE-BI": constants_mod.SYSTEM_PROMPT_TREE_OF_THOUGHT_BIMODAL_4_EXPERT_DEBATE,
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
        predicted = call_llm(sample, selected_model, comb_flag, sample_id)
        video_id = sample.get("video_id", f"sample_{sample_id}")
        ground_truth = sample.get("true_label", "").strip().lower()
        ground_truths.append(ground_truth)
        predictions.append(predicted)
        print(f"Video {video_id}: Ground Truth: {ground_truth}, Predicted: {predicted}")
        # Store the structured result.
        result_entry = {
            "video_id": video_id,
            "ground_truth": ground_truth,
            "predicted_label": predicted
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
    
    # Save all predictions to a JSON file (from call_llm global list)
    output_file_name_json = output_file_name.replace('.txt', '.json')
    json_results = getattr(call_llm, "json_results", [])
    with open(output_file_name_json, "w") as jf:
        json.dump(json_results, jf, indent=2)
        
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
