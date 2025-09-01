from openai import OpenAI
import json
import os
import argparse
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import importlib
import constants
from collections import defaultdict
from datetime import datetime

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

def get_emotions_for_dataset(dataset):
    """
    Returns the list of emotions for a given dataset.
    """
    if dataset == "MELD":
        return ['disgust', 'surprise', 'anger', 'joy', 'fear', 'sadness', 'neutral']
    elif dataset == "IEMOCAP":
        return ['frustrated', 'excited', 'happy', 'fearful', 'neutral', 'sad', 'angry', 'surprised']
    elif dataset == "MER":
        return ['happy', 'neutral', 'worried', 'surprise', 'angry', 'sad']
    elif dataset == "MERR":
        return ['neutral', 'sad', 'doubt', 'happy', 'worried', 'contempt', 'angry', 'fear', 'surprise']
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def call_llm_with_confidence(sample, model, modality, i, dataset):
    """
    Calls the LLM for a specific modality and returns emotion predictions with confidence scores.
    Returns a dictionary with emotion predictions and confidence scores.
    """
    message_parts = []
    
    # Add content based on modality
    if modality == "T":
        message_parts.append(f"whisper_transcript: {sample.get('transcript', '')}")
    elif modality == "A":
        message_parts.append(f"audio_description: {sample.get('audio_description', '')}")
    elif modality == "V":
        visual_cues = sample.get("visual_expression_description", "")
        if isinstance(visual_cues, list):
            visual_cues = ", ".join(visual_cues)
        message_parts.append(f"visual_expression_description: {visual_cues}")
        message_parts.append(f"visual_objective_description: {sample.get('visual_objective_description', '')}")
    elif modality == "TAV":
        # Add all modalities for combined analysis
        if sample.get('transcript'):
            message_parts.append(f"whisper_transcript: {sample.get('transcript', '')}")
        if sample.get('audio_description'):
            message_parts.append(f"audio_description: {sample.get('audio_description', '')}")
        visual_cues = sample.get("visual_expression_description", "")
        if isinstance(visual_cues, list):
            visual_cues = ", ".join(visual_cues)
        if visual_cues:
            message_parts.append(f"visual_expression_description: {visual_cues}")
        if sample.get('visual_objective_description'):
            message_parts.append(f"visual_objective_description: {sample.get('visual_objective_description', '')}")
    
    user_message = "\n\n".join(message_parts)

    # Get emotions for the specific dataset
    emotions = get_emotions_for_dataset(dataset)
    emotions_str = ", ".join(emotions)

    # System prompt for confidence scoring
    confidence_system_prompt = f"""
You are an expert emotion recognition system analyzing {modality} modality data from a video. 
Your task is to identify all possible emotions present, provide confidence scores for each, and assess the data quality.

Available emotions: [{emotions_str}]

Please respond in JSON format only:
{{
  "emotions": [
    {{
      "emotion": "emotion_name_in_lowercase",
      "confidence": confidence_score_1_to_100
    }},
    {{
      "emotion": "second_emotion_if_present",
      "confidence": confidence_score_1_to_100
    }}
  ],
  "primary_emotion": "most_confident_emotion",
  "primary_confidence": primary_confidence_score,
  "data_quality": {{
    "score": data_quality_score_1_to_100,
    "issues": ["list_of_data_quality_issues_if_any"],
    "reasoning": "explanation_of_data_quality_assessment"
  }},
  "reasoning": "brief_explanation_of_analysis"
}}

IMPORTANT: 
- Only use emotions from the provided list. Sort emotions by confidence (highest first).
- Data quality score: 100 = excellent quality, 50 = moderate quality, 10 = poor quality
- Data quality issues: e.g., ["noise in audio", "poor lighting", "unclear speech", "brief content"]
"""

    try:
        # Build request kwargs
        request_kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": confidence_system_prompt},
                {"role": "user", "content": user_message}
            ]
        }
        if not str(model).startswith("gpt-5"):
            request_kwargs["temperature"] = 0.0

        response = client.chat.completions.create(**request_kwargs)
        predicted_response = response.choices[0].message.content.strip()
        
        # Parse JSON response
        parsed = None
        try:
            parsed = json.loads(predicted_response)
        except json.JSONDecodeError:
            if predicted_response.startswith("```json"):
                json_str = predicted_response.strip().removeprefix("```json").removesuffix("```").strip()
                try:
                    parsed = json.loads(json_str)
                except Exception:
                    pass
        
        if parsed and isinstance(parsed, dict):
            # Handle None values safely
            primary_emotion = parsed.get("primary_emotion")
            if primary_emotion is None:
                primary_emotion = "neutral"  # Default fallback
            else:
                primary_emotion = str(primary_emotion).lower()
            
            # Clean emotions list to handle None values
            emotions_list = []
            for emotion_data in parsed.get("emotions", []):
                if isinstance(emotion_data, dict):
                    emotion = emotion_data.get("emotion")
                    confidence = emotion_data.get("confidence", 0)
                    if emotion is not None:
                        emotions_list.append({
                            "emotion": str(emotion).lower(),
                            "confidence": confidence
                        })
            
            return {
                "modality": modality,
                "emotions": emotions_list,
                "primary_emotion": primary_emotion,
                "primary_confidence": parsed.get("primary_confidence", 0),
                "data_quality": parsed.get("data_quality", {"score": 50, "issues": [], "reasoning": "Default quality"}),
                "reasoning": parsed.get("reasoning", ""),
                "video_id": sample.get("video_id", f"sample_{i}"),
                "ground_truth": sample.get("true_label", "").strip().lower()
            }
        else:
            # Fallback if JSON parsing fails
            fallback_emotion = predicted_response.strip().lower() if predicted_response else "neutral"
            return {
                "modality": modality,
                "emotions": [{"emotion": fallback_emotion, "confidence": 50}],
                "primary_emotion": fallback_emotion,
                "primary_confidence": 50,
                "data_quality": {"score": 30, "issues": ["parsing_error"], "reasoning": "Fallback parsing"},
                "reasoning": "Fallback parsing",
                "video_id": sample.get("video_id", f"sample_{i}"),
                "ground_truth": sample.get("true_label", "").strip().lower()
            }
            
    except Exception as e:
        print(f"Error processing {modality} modality for sample {i}: {e}")
        return {
            "modality": modality,
            "emotions": [{"emotion": "error", "confidence": 0}],
            "primary_emotion": "error",
            "primary_confidence": 0,
            "data_quality": {"score": 0, "issues": ["api_error"], "reasoning": f"Error: {str(e)}"},
            "reasoning": f"Error: {str(e)}",
            "video_id": sample.get("video_id", f"sample_{i}"),
            "ground_truth": sample.get("true_label", "").strip().lower()
        }

def ensemble_predictions(modality_results, use_all_emotions=False, use_data_quality=False, use_accuracy_mask=False, use_sabotage_mask=False, dataset_accuracy_weights=None, dataset_sabotage_weights=None):
    """
    Ensemble predictions from all modalities to produce final emotion prediction.
    Uses weighted voting based on confidence scores and optionally data quality, accuracy, and sabotage masks.
    
    Args:
        modality_results: List of modality prediction results
        use_all_emotions: If True, use all emotions from each modality. If False, use only primary emotion.
        use_data_quality: If True, weight predictions by data quality score in addition to confidence.
        use_accuracy_mask: If True, weight predictions by modality accuracy scores.
        use_sabotage_mask: If True, weight predictions by sabotage penalty (1 - sabotage percentage).
        dataset_accuracy_weights: Dict mapping modality to accuracy score (e.g., {"T": 0.186, "A": 0.250, "V": 0.483, "TAV": 0.333})
        dataset_sabotage_weights: Dict mapping modality to sabotage penalty (e.g., {"T": 0.883, "A": 0.383, "V": 0.783, "TAV": 0.683})
    """
    emotion_scores = defaultdict(float)
    total_weight = 0
    
    # Collect all emotion predictions with their weighted scores
    for result in modality_results:
        if result["primary_emotion"] != "error":
            modality = result["modality"]
            
            # Get data quality weight (1.0 if not using data quality, or quality_score/100 if using)
            data_quality_score = float(result.get("data_quality", {}).get("score", 50))
            quality_weight = data_quality_score / 100.0 if use_data_quality else 1.0
            
            # Get accuracy weight (1.0 if not using accuracy mask, or accuracy_score if using)
            accuracy_weight = 1.0
            if use_accuracy_mask and dataset_accuracy_weights and modality in dataset_accuracy_weights:
                accuracy_weight = dataset_accuracy_weights[modality]
            
            # Get sabotage weight (1.0 if not using sabotage mask, or sabotage_penalty if using)
            sabotage_weight = 1.0
            if use_sabotage_mask and dataset_sabotage_weights and modality in dataset_sabotage_weights:
                sabotage_weight = dataset_sabotage_weights[modality]
            
            # Combine all weights
            combined_weight = quality_weight * accuracy_weight * sabotage_weight
            
            if use_all_emotions:
                # Use all emotions from the modality
                for emotion_data in result.get("emotions", []):
                    confidence = float(emotion_data.get("confidence", 0))
                    emotion = emotion_data.get("emotion", "")
                    if emotion and isinstance(emotion, str):
                        emotion = emotion.lower()
                        if emotion and confidence > 0:
                            # Weight by confidence, data quality, accuracy, and sabotage penalty
                            weighted_score = confidence * combined_weight
                            emotion_scores[emotion] += weighted_score
                            total_weight += weighted_score
            else:
                # Use only the primary emotion
                confidence = float(result["primary_confidence"])
                emotion = result["primary_emotion"]
                if emotion and isinstance(emotion, str):
                    emotion = emotion.lower()
                    # Weight by confidence, data quality, accuracy, and sabotage penalty
                    weighted_score = confidence * combined_weight
                    emotion_scores[emotion] += weighted_score
                    total_weight += weighted_score
    
    # If no valid predictions, return neutral
    if total_weight == 0:
        return "neutral", 0, "neutral", 0, {}
    
    # Normalize scores
    for emotion in emotion_scores:
        emotion_scores[emotion] /= total_weight
    
    # Find the top two emotions
    if emotion_scores:
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        primary_emotion, primary_confidence = sorted_emotions[0]
        secondary_emotion, secondary_confidence = sorted_emotions[1] if len(sorted_emotions) > 1 else ("none", 0.0)
        return primary_emotion, primary_confidence, secondary_emotion, secondary_confidence, dict(emotion_scores)
    else:
        return "neutral", 0, "neutral", 0, {}

def process_sample_ensemble(sample, model, i, dataset, use_all_emotions=False, use_data_quality=False, use_accuracy_mask=False, use_sabotage_mask=False, dataset_accuracy_weights=None, dataset_sabotage_weights=None):
    """
    Process a single sample through all modalities and ensemble the results.
    
    Args:
        sample: Video sample data
        model: LLM model to use
        i: Sample index
        dataset: Dataset name (MELD, IEMOCAP, MER, MERR)
        use_all_emotions: If True, use all emotions from each modality. If False, use only primary emotion.
        use_data_quality: If True, weight predictions by data quality score in addition to confidence.
        use_accuracy_mask: If True, weight predictions by modality accuracy scores.
        use_sabotage_mask: If True, weight predictions by sabotage penalty (1 - sabotage percentage).
        dataset_accuracy_weights: Dict mapping modality to accuracy score.
        dataset_sabotage_weights: Dict mapping modality to sabotage penalty.
    """
    modalities = ["T", "A", "V", "TAV"]
    modality_results = []
    
    # Get predictions for each modality
    for modality in modalities:
        result = call_llm_with_confidence(sample, model, modality, i, dataset)
        modality_results.append(result)
    
    # Ensemble the results
    primary_emotion, primary_confidence, secondary_emotion, secondary_confidence, emotion_scores = ensemble_predictions(
        modality_results, use_all_emotions, use_data_quality, use_accuracy_mask, use_sabotage_mask, 
        dataset_accuracy_weights, dataset_sabotage_weights
    )
    
    # Create comprehensive result
    ensemble_result = {
        "video_id": sample.get("video_id", f"sample_{i}"),
        "ground_truth": sample.get("true_label", "").strip().lower(),
        "modality_results": modality_results,
        "ensemble_prediction": primary_emotion,
        "ensemble_confidence": primary_confidence,
        "ensemble_secondary": secondary_emotion,
        "ensemble_secondary_confidence": secondary_confidence,
        "emotion_scores": emotion_scores,
        "ensemble_mode": "all_emotions" if use_all_emotions else "primary_only",
        "data_quality_weighting": "enabled" if use_data_quality else "disabled",
        "accuracy_weighting": "enabled" if use_accuracy_mask else "disabled",
        "sabotage_weighting": "enabled" if use_sabotage_mask else "disabled"
    }
    
    return primary_emotion, ensemble_result

def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description="Run LLM model predictions with ensemble approach")
    
    parser.add_argument("--input", type=str, default="MERR_fine_grained.json", help="Input JSON file path")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--output", type=str, default=None, help="Output file name for the results")
    parser.add_argument("--dataset", type=str, choices=["MELD", "MER", "IEMOCAP", "MERR"], default="MERR", help="Dataset to use")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent API calls (reduced for ensemble)")
    parser.add_argument("--use-all-emotions", action="store_true", help="Use all emotions from each modality instead of just primary emotion")
    parser.add_argument("--use-data-quality", action="store_true", help="Weight predictions by data quality score in addition to confidence")
    parser.add_argument("--use-accuracy-mask", action="store_true", help="Weight predictions by modality accuracy scores")
    parser.add_argument("--use-sabotage-mask", action="store_true", help="Weight predictions by sabotage penalty (1 - sabotage percentage)")
    args = parser.parse_args()
    
    selected_model = args.model
    input_file = args.input
    use_all_emotions = args.use_all_emotions
    use_data_quality = args.use_data_quality
    use_accuracy_mask = args.use_accuracy_mask
    use_sabotage_mask = args.use_sabotage_mask
    
    # Define dataset-specific accuracy and sabotage weights based on the provided data
    dataset_accuracy_weights = {}
    dataset_sabotage_weights = {}
    
    if args.dataset == "MER":
        # MER dataset accuracy weights (from GPT-5-nano results)
        dataset_accuracy_weights = {
            "T": 0.186,    # 18.6%
            "A": 0.250,    # 25.0%
            "V": 0.483,    # 48.3%
            "TAV": 0.333   # 33.3%
        }
        # Sabotage penalty weights (1 - sabotage percentage)
        dataset_sabotage_weights = {
            "T": 0.883,    # 1 - 0.117 (7/60)
            "A": 0.383,    # 1 - 0.617 (37/60)
            "V": 0.783,    # 1 - 0.217 (13/60)
            "TAV": 0.683   # 1 - 0.317 (19/60)
        }
    elif args.dataset == "MELD":
        # MELD dataset accuracy weights
        dataset_accuracy_weights = {
            "T": 0.508,    # 50.8%
            "A": 0.298,    # 29.8%
            "V": 0.188,    # 18.8%
            "TAV": 0.236   # 23.6%
        }
        # Sabotage penalty weights
        dataset_sabotage_weights = {
            "T": 0.885,    # 1 - 0.115 (22/191)
            "A": 0.518,    # 1 - 0.482 (92/191)
            "V": 0.529,    # 1 - 0.471 (90/191)
            "TAV": 0.576   # 1 - 0.424 (81/191)
        }
    elif args.dataset == "IEMOCAP":
        # IEMOCAP dataset accuracy weights
        dataset_accuracy_weights = {
            "T": 0.326,    # 32.6%
            "A": 0.248,    # 24.8%
            "V": 0.171,    # 17.1%
            "TAV": 0.279   # 27.9%
        }
        # Sabotage penalty weights
        dataset_sabotage_weights = {
            "T": 0.837,    # 1 - 0.163 (21/129)
            "A": 0.403,    # 1 - 0.597 (77/129)
            "V": 0.488,    # 1 - 0.512 (66/129)
            "TAV": 0.543   # 1 - 0.457 (59/129)
        }
    elif args.dataset == "MERR":
        # MERR dataset - using default weights (can be updated when data is available)
        dataset_accuracy_weights = {
            "T": 0.5,      # Default 50%
            "A": 0.5,      # Default 50%
            "V": 0.5,      # Default 50%
            "TAV": 0.5     # Default 50%
        }
        dataset_sabotage_weights = {
            "T": 0.8,      # Default 80%
            "A": 0.8,      # Default 80%
            "V": 0.8,      # Default 80%
            "TAV": 0.8     # Default 80%
        }
    
    # Create ensemble suffix based on flags
    ensemble_parts = []
    if use_all_emotions:
        ensemble_parts.append("all_emotions")
    else:
        ensemble_parts.append("primary_only")
    
    if use_data_quality:
        ensemble_parts.append("with_quality")
    
    if use_accuracy_mask:
        ensemble_parts.append("with_accuracy")
    
    if use_sabotage_mask:
        ensemble_parts.append("with_sabotage")
    
    ensemble_suffix = "_".join(ensemble_parts)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.output is not None:
        # If user provided output name, add timestamp before extension
        base_name, ext = os.path.splitext(args.output)
        output_file_name = f"{base_name}_{timestamp}{ext}"
    else:
        # Default filename with timestamp
        output_file_name = f"ensemble_{ensemble_suffix}_{selected_model}_{timestamp}_results.txt"
    
    # Load the JSON data from file.
    with open(input_file, "r") as f:
        data = json.load(f)
        print("Data loaded")
    
    print(f"Ensemble mode: {'All emotions' if use_all_emotions else 'Primary emotion only'}")
    print(f"Data quality weighting: {'Enabled' if use_data_quality else 'Disabled'}")
    print(f"Accuracy weighting: {'Enabled' if use_accuracy_mask else 'Disabled'}")
    print(f"Sabotage weighting: {'Enabled' if use_sabotage_mask else 'Disabled'}")
    print(f"Output file: {output_file_name}")
    
    # Print the weights being used
    if use_accuracy_mask:
        print(f"Accuracy weights: {dataset_accuracy_weights}")
    if use_sabotage_mask:
        print(f"Sabotage penalty weights: {dataset_sabotage_weights}")
    
    predictions = []
    secondary_predictions = []
    third_predictions = []
    fourth_predictions = []
    fifth_predictions = []
    ground_truths = []
    result_details = []
    ensemble_results = []

    # Process each sample with ensemble approach
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results_iter = executor.map(
            lambda args_pair: process_sample_ensemble(
                args_pair[1], selected_model, args_pair[0], args.dataset, 
                use_all_emotions, use_data_quality, use_accuracy_mask, use_sabotage_mask,
                dataset_accuracy_weights, dataset_sabotage_weights
            ),
            enumerate(data)
        )
        for i, (predicted, ensemble_result) in enumerate(results_iter):
            sample = data[i]
            video_id = sample.get("video_id", f"sample_{i}")
            ground_truth = sample.get("true_label", "").strip().lower()
            ground_truths.append(ground_truth)
            predictions.append(predicted)
            
            # Track top 5 predictions
            secondary_pred = ensemble_result.get("ensemble_secondary", "none")
            secondary_predictions.append(secondary_pred)
            
            # Get top 5 predictions from emotion scores
            emotion_scores = ensemble_result.get("emotion_scores", {})
            sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
            
            third_pred = sorted_emotions[2][0] if len(sorted_emotions) > 2 else "none"
            fourth_pred = sorted_emotions[3][0] if len(sorted_emotions) > 3 else "none"
            fifth_pred = sorted_emotions[4][0] if len(sorted_emotions) > 4 else "none"
            
            third_predictions.append(third_pred)
            fourth_predictions.append(fourth_pred)
            fifth_predictions.append(fifth_pred)
            
            # Create detailed result line
            modality_predictions = []
            for mod_result in ensemble_result["modality_results"]:
                if use_all_emotions:
                    # Show all emotions for each modality
                    emotions_str = []
                    for emotion_data in mod_result.get("emotions", []):
                        emotions_str.append(f"{emotion_data['emotion']}({emotion_data['confidence']})")
                    modality_str = f"{mod_result['modality']}:[{', '.join(emotions_str)}]"
                else:
                    # Show only primary emotion
                    modality_str = f"{mod_result['modality']}:{mod_result['primary_emotion']}({mod_result['primary_confidence']})"
                
                # Add data quality info if enabled
                if use_data_quality:
                    quality_score = mod_result.get("data_quality", {}).get("score", 50)
                    modality_str += f"[Q:{quality_score}]"
                
                # Add accuracy info if enabled
                if use_accuracy_mask and dataset_accuracy_weights:
                    accuracy_weight = dataset_accuracy_weights.get(mod_result["modality"], 1.0)
                    modality_str += f"[A:{accuracy_weight:.3f}]"
                
                # Add sabotage penalty info if enabled
                if use_sabotage_mask and dataset_sabotage_weights:
                    sabotage_weight = dataset_sabotage_weights.get(mod_result["modality"], 1.0)
                    modality_str += f"[S:{sabotage_weight:.3f}]"
                
                modality_predictions.append(modality_str)
            
            # Include secondary prediction in output
            secondary_info = f", Secondary:{secondary_pred}({ensemble_result.get('ensemble_secondary_confidence', 0):.1f})" if secondary_pred != "none" else ""
            result_line = f"Video {video_id}: GT:{ground_truth}, Ensemble:{predicted}({ensemble_result['ensemble_confidence']:.1f}){secondary_info}, Modalities:[{', '.join(modality_predictions)}]"
            print(result_line)
            result_details.append(result_line)
            ensemble_results.append(ensemble_result)
    
    # Define the set of possible labels based on dataset
    labels = get_emotions_for_dataset(args.dataset)
    
    # Compute precision, recall, and F1 score for each label.
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truths, predictions, labels=labels, zero_division=0
    )
    overall_accuracy = accuracy_score(ground_truths, predictions)
    
    # Calculate multi-shot accuracies (1-shot through 5-shot)
    one_shot_correct = sum(1 for i in range(len(ground_truths)) if predictions[i] == ground_truths[i])
    one_shot_accuracy = one_shot_correct / len(ground_truths) if ground_truths else 0
    
    two_shot_correct = sum(1 for i in range(len(ground_truths)) 
                          if predictions[i] == ground_truths[i] or secondary_predictions[i] == ground_truths[i])
    two_shot_accuracy = two_shot_correct / len(ground_truths) if ground_truths else 0
    
    three_shot_correct = sum(1 for i in range(len(ground_truths)) 
                            if predictions[i] == ground_truths[i] or secondary_predictions[i] == ground_truths[i] 
                            or third_predictions[i] == ground_truths[i])
    three_shot_accuracy = three_shot_correct / len(ground_truths) if ground_truths else 0
    
    four_shot_correct = sum(1 for i in range(len(ground_truths)) 
                           if predictions[i] == ground_truths[i] or secondary_predictions[i] == ground_truths[i] 
                           or third_predictions[i] == ground_truths[i] or fourth_predictions[i] == ground_truths[i])
    four_shot_accuracy = four_shot_correct / len(ground_truths) if ground_truths else 0
    
    five_shot_correct = sum(1 for i in range(len(ground_truths)) 
                           if predictions[i] == ground_truths[i] or secondary_predictions[i] == ground_truths[i] 
                           or third_predictions[i] == ground_truths[i] or fourth_predictions[i] == ground_truths[i]
                           or fifth_predictions[i] == ground_truths[i])
    five_shot_accuracy = five_shot_correct / len(ground_truths) if ground_truths else 0
    
    print("\nEvaluation Metrics:")
    print("One-Shot Accuracy (Primary Only): {:.2f}".format(one_shot_accuracy))
    print("Two-Shot Accuracy (Primary or Secondary): {:.2f}".format(two_shot_accuracy))
    print("Three-Shot Accuracy (Top 3): {:.2f}".format(three_shot_accuracy))
    print("Four-Shot Accuracy (Top 4): {:.2f}".format(four_shot_accuracy))
    print("Five-Shot Accuracy (Top 5): {:.2f}".format(five_shot_accuracy))
    print("Overall Accuracy: {:.2f}".format(overall_accuracy))
    
    # Create metrics summary
    metrics_summary = [
        f"One-Shot Accuracy (Primary Only): {one_shot_accuracy:.2f}",
        f"Two-Shot Accuracy (Primary or Secondary): {two_shot_accuracy:.2f}",
        f"Three-Shot Accuracy (Top 3): {three_shot_accuracy:.2f}",
        f"Four-Shot Accuracy (Top 4): {four_shot_accuracy:.2f}",
        f"Five-Shot Accuracy (Top 5): {five_shot_accuracy:.2f}",
        f"Overall Accuracy: {overall_accuracy:.2f}"
    ]
    for i, label in enumerate(labels):
        metric_line = (
            f"Label: {label} -> Precision: {precision[i]:.2f}, Recall: {recall[i]:.2f}, "
            f"F1 Score: {f1[i]:.2f}, Support: {support[i]}"
        )
        print(metric_line)
        metrics_summary.append(metric_line)
    
    # Create results directory if it doesn't exist
    output_dir = os.path.dirname(output_file_name)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save results to the specified output file.
    with open(output_file_name, "w") as f:
        f.write("ENSEMBLE PREDICTION RESULTS\n")
        f.write("==========================\n\n")
        
        # Write ensemble configuration
        f.write("ENSEMBLE CONFIGURATION\n")
        f.write("=====================\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model: {selected_model}\n")
        f.write(f"Ensemble mode: {'All emotions' if use_all_emotions else 'Primary emotion only'}\n")
        f.write(f"Data quality weighting: {'Enabled' if use_data_quality else 'Disabled'}\n")
        f.write(f"Accuracy weighting: {'Enabled' if use_accuracy_mask else 'Disabled'}\n")
        f.write(f"Sabotage weighting: {'Enabled' if use_sabotage_mask else 'Disabled'}\n")
        
        if use_accuracy_mask:
            f.write(f"Accuracy weights: {dataset_accuracy_weights}\n")
        if use_sabotage_mask:
            f.write(f"Sabotage penalty weights: {dataset_sabotage_weights}\n")
        
        f.write("\n" + "="*50 + "\n\n")
        f.write("\n".join(result_details))
        f.write("\n\nEVALUATION METRICS\n")
        f.write("=================\n\n")
        f.write("\n".join(metrics_summary))
    
    print(f"\nResults saved to {output_file_name}")

    # Save detailed ensemble results to JSON
    output_file_name_json = output_file_name.replace('.txt', '.json')
    with open(output_file_name_json, "w") as jf:
        json.dump(ensemble_results, jf, indent=2)
    print(f"Detailed ensemble results saved to {output_file_name_json}")

if __name__ == "__main__":
    main()
    
    # Example usage with new flags:
    # python after_knowing_sabotage_experiment.py --dataset MER --use-accuracy-mask --use-sabotage-mask
    # python after_knowing_sabotage_experiment.py --dataset MELD --use-all-emotions --use-data-quality --use-accuracy-mask --use-sabotage-mask
    # python after_knowing_sabotage_experiment.py --dataset IEMOCAP --use-accuracy-mask --use-sabotage-mask --model gpt-4o-mini
