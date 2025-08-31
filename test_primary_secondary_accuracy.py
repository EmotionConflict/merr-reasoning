#!/usr/bin/env python3
"""
Test script to demonstrate primary/secondary prediction tracking and one-shot/two-shot accuracy metrics.
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

def test_primary_secondary_accuracy():
    """Test primary/secondary prediction tracking and accuracy metrics."""
    
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
        }
    ]
    
    print("=== Testing Primary/Secondary Predictions ===")
    
    ground_truths = []
    primary_predictions = []
    secondary_predictions = []
    
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
        
        print(f"Test Case {i+1}:")
        print(f"  Ground Truth: {ground_truth}")
        print(f"  Primary Prediction: {primary_emotion} ({primary_confidence:.2f})")
        print(f"  Secondary Prediction: {secondary_emotion} ({secondary_confidence:.2f})")
        print(f"  All Scores: {emotion_scores}")
        print()
    
    # Calculate accuracy metrics
    one_shot_correct = sum(1 for i in range(len(ground_truths)) if primary_predictions[i] == ground_truths[i])
    one_shot_accuracy = one_shot_correct / len(ground_truths) if ground_truths else 0
    
    two_shot_correct = sum(1 for i in range(len(ground_truths)) 
                          if primary_predictions[i] == ground_truths[i] or secondary_predictions[i] == ground_truths[i])
    two_shot_accuracy = two_shot_correct / len(ground_truths) if ground_truths else 0
    
    print("=== Accuracy Metrics ===")
    print(f"One-Shot Accuracy (Primary Only): {one_shot_accuracy:.2f} ({one_shot_correct}/{len(ground_truths)})")
    print(f"Two-Shot Accuracy (Primary or Secondary): {two_shot_accuracy:.2f} ({two_shot_correct}/{len(ground_truths)})")
    
    print("\n=== Detailed Analysis ===")
    for i in range(len(ground_truths)):
        gt = ground_truths[i]
        primary = primary_predictions[i]
        secondary = secondary_predictions[i]
        
        primary_correct = primary == gt
        secondary_correct = secondary == gt
        two_shot_correct = primary_correct or secondary_correct
        
        print(f"Case {i+1}: GT={gt}, Primary={primary} ({'✓' if primary_correct else '✗'}), "
              f"Secondary={secondary} ({'✓' if secondary_correct else '✗'}), "
              f"Two-shot={'✓' if two_shot_correct else '✗'}")

if __name__ == "__main__":
    test_primary_secondary_accuracy()


