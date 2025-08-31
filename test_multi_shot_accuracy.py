#!/usr/bin/env python3
"""
Test script to demonstrate multi-shot accuracy calculations (1-shot through 5-shot).
"""

import json
from collections import defaultdict

def ensemble_predictions(modality_results, use_all_emotions=False, use_data_quality=False):
    """
    Ensemble predictions from all modalities to produce final emotion prediction.
    Uses weighted voting based on confidence scores and optionally data quality.
    
    Args:
        modality_results: List of modality prediction results
        use_all_emotions: If True, use all emotions from each modality. If False, use only primary emotion.
        use_data_quality: If True, weight predictions by data quality score in addition to confidence.
    """
    emotion_scores = defaultdict(float)
    total_weight = 0
    
    # Collect all emotion predictions with their weighted scores
    for result in modality_results:
        if result["primary_emotion"] != "error":
            # Get data quality weight (1.0 if not using data quality, or quality_score/100 if using)
            data_quality_score = float(result.get("data_quality", {}).get("score", 50))
            quality_weight = data_quality_score / 100.0 if use_data_quality else 1.0
            
            if use_all_emotions:
                # Use all emotions from the modality
                for emotion_data in result.get("emotions", []):
                    confidence = float(emotion_data.get("confidence", 0))
                    emotion = emotion_data.get("emotion", "").lower()
                    if emotion and confidence > 0:
                        # Weight by both confidence and data quality
                        weighted_score = confidence * quality_weight
                        emotion_scores[emotion] += weighted_score
                        total_weight += weighted_score
            else:
                # Use only the primary emotion
                confidence = float(result["primary_confidence"])
                emotion = result["primary_emotion"]
                # Weight by both confidence and data quality
                weighted_score = confidence * quality_weight
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

def calculate_multi_shot_accuracy(ground_truths, predictions, secondary_predictions, third_predictions, fourth_predictions, fifth_predictions):
    """Calculate multi-shot accuracies from 1-shot to 5-shot."""
    
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
    
    return {
        "one_shot": one_shot_accuracy,
        "two_shot": two_shot_accuracy,
        "three_shot": three_shot_accuracy,
        "four_shot": four_shot_accuracy,
        "five_shot": five_shot_accuracy
    }

def test_multi_shot_accuracy():
    """Test multi-shot accuracy calculations with sample data."""
    
    # Sample test cases with ground truth and predictions
    test_cases = [
        {
            "ground_truth": "happy",
            "modality_results": [
                {
                    "modality": "T",
                    "emotions": [{"emotion": "happy", "confidence": 80}, {"emotion": "neutral", "confidence": 20}],
                    "primary_emotion": "happy",
                    "primary_confidence": 80
                },
                {
                    "modality": "A",
                    "emotions": [{"emotion": "happy", "confidence": 70}, {"emotion": "sad", "confidence": 30}],
                    "primary_emotion": "happy",
                    "primary_confidence": 70
                },
                {
                    "modality": "V",
                    "emotions": [{"emotion": "neutral", "confidence": 60}, {"emotion": "happy", "confidence": 40}],
                    "primary_emotion": "neutral",
                    "primary_confidence": 60
                },
                {
                    "modality": "TAV",
                    "emotions": [{"emotion": "happy", "confidence": 85}, {"emotion": "neutral", "confidence": 15}],
                    "primary_emotion": "happy",
                    "primary_confidence": 85
                }
            ]
        },
        {
            "ground_truth": "sad",
            "modality_results": [
                {
                    "modality": "T",
                    "emotions": [{"emotion": "neutral", "confidence": 70}, {"emotion": "sad", "confidence": 30}],
                    "primary_emotion": "neutral",
                    "primary_confidence": 70
                },
                {
                    "modality": "A",
                    "emotions": [{"emotion": "sad", "confidence": 80}, {"emotion": "neutral", "confidence": 20}],
                    "primary_emotion": "sad",
                    "primary_confidence": 80
                },
                {
                    "modality": "V",
                    "emotions": [{"emotion": "neutral", "confidence": 65}, {"emotion": "sad", "confidence": 35}],
                    "primary_emotion": "neutral",
                    "primary_confidence": 65
                },
                {
                    "modality": "TAV",
                    "emotions": [{"emotion": "sad", "confidence": 75}, {"emotion": "neutral", "confidence": 25}],
                    "primary_emotion": "sad",
                    "primary_confidence": 75
                }
            ]
        },
        {
            "ground_truth": "angry",
            "modality_results": [
                {
                    "modality": "T",
                    "emotions": [{"emotion": "angry", "confidence": 90}, {"emotion": "neutral", "confidence": 10}],
                    "primary_emotion": "angry",
                    "primary_confidence": 90
                },
                {
                    "modality": "A",
                    "emotions": [{"emotion": "neutral", "confidence": 60}, {"emotion": "angry", "confidence": 40}],
                    "primary_emotion": "neutral",
                    "primary_confidence": 60
                },
                {
                    "modality": "V",
                    "emotions": [{"emotion": "angry", "confidence": 85}, {"emotion": "neutral", "confidence": 15}],
                    "primary_emotion": "angry",
                    "primary_confidence": 85
                },
                {
                    "modality": "TAV",
                    "emotions": [{"emotion": "angry", "confidence": 88}, {"emotion": "neutral", "confidence": 12}],
                    "primary_emotion": "angry",
                    "primary_confidence": 88
                }
            ]
        },
        {
            "ground_truth": "worried",
            "modality_results": [
                {
                    "modality": "T",
                    "emotions": [{"emotion": "neutral", "confidence": 80}, {"emotion": "worried", "confidence": 20}],
                    "primary_emotion": "neutral",
                    "primary_confidence": 80
                },
                {
                    "modality": "A",
                    "emotions": [{"emotion": "neutral", "confidence": 70}, {"emotion": "worried", "confidence": 30}],
                    "primary_emotion": "neutral",
                    "primary_confidence": 70
                },
                {
                    "modality": "V",
                    "emotions": [{"emotion": "worried", "confidence": 75}, {"emotion": "neutral", "confidence": 25}],
                    "primary_emotion": "worried",
                    "primary_confidence": 75
                },
                {
                    "modality": "TAV",
                    "emotions": [{"emotion": "neutral", "confidence": 65}, {"emotion": "worried", "confidence": 35}],
                    "primary_emotion": "neutral",
                    "primary_confidence": 65
                }
            ]
        },
        {
            "ground_truth": "surprise",
            "modality_results": [
                {
                    "modality": "T",
                    "emotions": [{"emotion": "happy", "confidence": 60}, {"emotion": "surprise", "confidence": 40}],
                    "primary_emotion": "happy",
                    "primary_confidence": 60
                },
                {
                    "modality": "A",
                    "emotions": [{"emotion": "surprise", "confidence": 85}, {"emotion": "happy", "confidence": 15}],
                    "primary_emotion": "surprise",
                    "primary_confidence": 85
                },
                {
                    "modality": "V",
                    "emotions": [{"emotion": "surprise", "confidence": 90}, {"emotion": "happy", "confidence": 10}],
                    "primary_emotion": "surprise",
                    "primary_confidence": 90
                },
                {
                    "modality": "TAV",
                    "emotions": [{"emotion": "surprise", "confidence": 88}, {"emotion": "happy", "confidence": 12}],
                    "primary_emotion": "surprise",
                    "primary_confidence": 88
                }
            ]
        }
    ]
    
    print("=== Testing Multi-Shot Accuracy ===")
    
    ground_truths = []
    primary_predictions = []
    secondary_predictions = []
    third_predictions = []
    fourth_predictions = []
    fifth_predictions = []
    
    for i, test_case in enumerate(test_cases):
        ground_truth = test_case["ground_truth"]
        modality_results = test_case["modality_results"]
        
        # Get ensemble predictions
        primary_emotion, primary_confidence, secondary_emotion, secondary_confidence, emotion_scores = ensemble_predictions(
            modality_results, use_all_emotions=True, use_data_quality=False
        )
        
        ground_truths.append(ground_truth)
        primary_predictions.append(primary_emotion)
        secondary_predictions.append(secondary_emotion)
        
        # Get top 5 predictions from emotion scores
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        
        third_pred = sorted_emotions[2][0] if len(sorted_emotions) > 2 else "none"
        fourth_pred = sorted_emotions[3][0] if len(sorted_emotions) > 3 else "none"
        fifth_pred = sorted_emotions[4][0] if len(sorted_emotions) > 4 else "none"
        
        third_predictions.append(third_pred)
        fourth_predictions.append(fourth_pred)
        fifth_predictions.append(fifth_pred)
        
        print(f"Test Case {i+1}:")
        print(f"  Ground Truth: {ground_truth}")
        print(f"  Top 5 Predictions: {primary_emotion} > {secondary_emotion} > {third_pred} > {fourth_pred} > {fifth_pred}")
        print(f"  All Scores: {emotion_scores}")
        print()
    
    # Calculate multi-shot accuracies
    accuracies = calculate_multi_shot_accuracy(
        ground_truths, primary_predictions, secondary_predictions, 
        third_predictions, fourth_predictions, fifth_predictions
    )
    
    print("=== Multi-Shot Accuracy Results ===")
    print(f"One-Shot Accuracy (Primary Only): {accuracies['one_shot']:.2f}")
    print(f"Two-Shot Accuracy (Primary or Secondary): {accuracies['two_shot']:.2f}")
    print(f"Three-Shot Accuracy (Top 3): {accuracies['three_shot']:.2f}")
    print(f"Four-Shot Accuracy (Top 4): {accuracies['four_shot']:.2f}")
    print(f"Five-Shot Accuracy (Top 5): {accuracies['five_shot']:.2f}")
    
    print("\n=== Detailed Analysis ===")
    for i in range(len(ground_truths)):
        gt = ground_truths[i]
        predictions = [primary_predictions[i], secondary_predictions[i], third_predictions[i], fourth_predictions[i], fifth_predictions[i]]
        
        print(f"Case {i+1}: GT={gt}")
        for j, pred in enumerate(predictions, 1):
            correct = pred == gt
            print(f"  {j}-shot: {pred} ({'✓' if correct else '✗'})")
        print()

if __name__ == "__main__":
    test_multi_shot_accuracy()


