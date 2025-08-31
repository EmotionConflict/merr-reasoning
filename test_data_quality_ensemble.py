#!/usr/bin/env python3
"""
Test script to demonstrate the data quality weighting feature in ensemble predictions.
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
        return "neutral", 0, {}
    
    # Normalize scores
    for emotion in emotion_scores:
        emotion_scores[emotion] /= total_weight
    
    # Find the emotion with highest score
    if emotion_scores:
        best_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        return best_emotion[0], best_emotion[1], dict(emotion_scores)
    else:
        return "neutral", 0, {}

def test_data_quality_impact():
    """Test how data quality affects ensemble predictions."""
    
    # Sample modality results with different data quality scores
    test_results = [
        {
            "modality": "T",
            "emotions": [
                {"emotion": "happy", "confidence": 80},
                {"emotion": "neutral", "confidence": 20}
            ],
            "primary_emotion": "happy",
            "primary_confidence": 80,
            "data_quality": {"score": 90, "issues": [], "reasoning": "Clear speech, good audio quality"},
            "reasoning": "Text indicates positive sentiment"
        },
        {
            "modality": "A", 
            "emotions": [
                {"emotion": "happy", "confidence": 70},
                {"emotion": "sad", "confidence": 30}
            ],
            "primary_emotion": "happy",
            "primary_confidence": 70,
            "data_quality": {"score": 30, "issues": ["background_noise", "low_volume"], "reasoning": "Poor audio quality"},
            "reasoning": "Audio tone suggests happiness"
        },
        {
            "modality": "V",
            "emotions": [
                {"emotion": "neutral", "confidence": 60},
                {"emotion": "happy", "confidence": 40}
            ],
            "primary_emotion": "neutral", 
            "primary_confidence": 60,
            "data_quality": {"score": 75, "issues": ["poor_lighting"], "reasoning": "Moderate lighting conditions"},
            "reasoning": "Facial expression is neutral"
        },
        {
            "modality": "TAV",
            "emotions": [
                {"emotion": "happy", "confidence": 85},
                {"emotion": "neutral", "confidence": 15}
            ],
            "primary_emotion": "happy",
            "primary_confidence": 85,
            "data_quality": {"score": 80, "issues": [], "reasoning": "Good overall quality"},
            "reasoning": "Combined analysis suggests happiness"
        }
    ]
    
    print("=== Testing Primary Emotion Only Mode ===")
    print("Without Data Quality Weighting:")
    final_emotion, final_confidence, emotion_scores = ensemble_predictions(test_results, use_all_emotions=False, use_data_quality=False)
    print(f"Final Emotion: {final_emotion}")
    print(f"Final Confidence: {final_confidence:.2f}")
    print(f"All Emotion Scores: {emotion_scores}")
    print()
    
    print("With Data Quality Weighting:")
    final_emotion_q, final_confidence_q, emotion_scores_q = ensemble_predictions(test_results, use_all_emotions=False, use_data_quality=True)
    print(f"Final Emotion: {final_emotion_q}")
    print(f"Final Confidence: {final_confidence_q:.2f}")
    print(f"All Emotion Scores: {emotion_scores_q}")
    print()
    
    print("=== Testing All Emotions Mode ===")
    print("Without Data Quality Weighting:")
    final_emotion_all, final_confidence_all, emotion_scores_all = ensemble_predictions(test_results, use_all_emotions=True, use_data_quality=False)
    print(f"Final Emotion: {final_emotion_all}")
    print(f"Final Confidence: {final_confidence_all:.2f}")
    print(f"All Emotion Scores: {emotion_scores_all}")
    print()
    
    print("With Data Quality Weighting:")
    final_emotion_all_q, final_confidence_all_q, emotion_scores_all_q = ensemble_predictions(test_results, use_all_emotions=True, use_data_quality=True)
    print(f"Final Emotion: {final_emotion_all_q}")
    print(f"Final Confidence: {final_confidence_all_q:.2f}")
    print(f"All Emotion Scores: {emotion_scores_all_q}")
    print()
    
    print("=== Data Quality Impact Analysis ===")
    print("Data Quality Scores:")
    for result in test_results:
        quality = result["data_quality"]["score"]
        issues = result["data_quality"]["issues"]
        print(f"- {result['modality']}: {quality}/100 {f'(issues: {issues})' if issues else '(no issues)'}")
    
    print("\nKey Observations:")
    print("1. Audio modality has poor quality (30/100) - gets reduced weight")
    print("2. Text modality has excellent quality (90/100) - gets enhanced weight")
    print("3. Visual modality has moderate quality (75/100) - gets moderate weight")
    print("4. Combined modality has good quality (80/100) - gets enhanced weight")

if __name__ == "__main__":
    test_data_quality_impact()


