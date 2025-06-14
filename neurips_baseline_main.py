from openai import OpenAI
import json
import os
import argparse
from dotenv import load_dotenv
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import importlib
import constants

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

def call_llm(sample, model, comb, i):
    """
    Constructs a user message from the sample details based on the combination flag and calls the specified LLM API.
    The message is structured as:
        The person in the video says: [text]. Audio cues: [audio_prior_list]. Visual cues: [visual_prior_list].
    depending on the --comb flag.
    Returns the predicted emotion label in lower-case.
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
    parser.add_argument("--dataset", type=str, choices=["MELD", "MER", "IEMOCAP"], default="MER", help="Dataset to use: MELD or MER (default: MER)")
    args = parser.parse_args()
    
    selected_model = args.model
    comb_flag = args.comb
    input_file = args.input  # Get the input file path
    # Determine output file name based on provided argument or default to model name.
    output_file_name = args.output if args.output is not None else f"{selected_model}_results.txt"
    
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
    
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = constants_mod.SYSTEM_PROMPT
    
    # Load the JSON data from file.
    with open(input_file, "r") as f:
        data = json.load(f)
        print("Data loaded")
    
    predictions = []
    ground_truths = []
    result_details = []
    
    # Process each sample in the list
    for i, sample in enumerate(data):
        predicted = call_llm(sample, selected_model, comb_flag, i)
        video_id = sample.get("video_id", f"sample_{i}")
        ground_truth = sample.get("true_label", "").strip().lower()
        ground_truths.append(ground_truth)
        predictions.append(predicted)
        print(f"Video {video_id}: Ground Truth: {ground_truth}, Predicted: {predicted}")
        result_details.append(f"Video {video_id}: Ground Truth: {ground_truth}, Predicted: {predicted}")
    
    # Define the set of possible labels.
    if args.dataset == "MELD":
        labels = ['disgust', 'surprise', 'anger', 'joy', 'fear', 'sadness', 'neutral']
    elif args.dataset == "IEMOCAP":
        labels = ['frustrated', 'excited', 'happy', 'fearful', 'neutral', 'sad', 'angry', 'surprised']
    elif args.dataset == "MER":
        labels = ['happy', 'neutral', 'worried', 'surprise', 'angry', 'sad']
    elif args.dataset == "MERR":
        labels = ['neutral', 'sad', 'doubt', 'happy', 'worried', 'contempt', 'angry', 'fear', 'surprise']
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    
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
    output_dir = os.path.dirname(output_file_name)
    if output_dir:  # Only create directory if there's a path
        os.makedirs(output_dir, exist_ok=True)
    
    # Save results to the specified output file.
    with open(output_file_name, "w") as f:
        f.write("PREDICTION RESULTS\n")
        f.write("=================\n\n")
        f.write("\n".join(result_details))
        f.write("\n\nEVALUATION METRICS\n")
        f.write("=================\n\n")
        f.write("\n".join(metrics_summary))
    
    print(f"\nResults saved to {output_file_name}")

    # Save all predictions to a JSON file
    output_file_name_json = output_file_name.replace('.txt', '.json')
    # If using the global list from call_llm
    json_results = getattr(call_llm, "json_results", [])
    with open(output_file_name_json, "w") as jf:
        json.dump(json_results, jf, indent=2)
    print(f"JSON results saved to {output_file_name_json}")

if __name__ == "__main__":
    main()
