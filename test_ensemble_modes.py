#!/usr/bin/env python3
"""
Test script to demonstrate both ensemble modes:
1. Primary emotion only
2. All emotions from each modality
"""

import json
from collections import defaultdict

def ensemble_predictions(modality_results, use_all_emotions=False):
    """
    Ensemble predictions from all modalities to produce final emotion prediction.
    Uses weighted voting based on confidence scores.
    
    Args:
        modality_results: List of modality prediction results
        use_all_emotions: If True, use all emotions from each modality. If False, use only primary emotion.
    """
    emotion_scores = defaultdict(float)
    total_confidence = 0
    
    # Collect all emotion predictions with their confidence scores
    for result in modality_results:
        if result["primary_emotion"] != "error":
            if use_all_emotions:
                # Use all emotions from the modality
                for emotion_data in result.get("emotions", []):
                    confidence = float(emotion_data.get("confidence", 0))
                    emotion = emotion_data.get("emotion", "").lower()
                    if emotion and confidence > 0:
                        emotion_scores[emotion] += confidence
                        total_confidence += confidence
            else:
                # Use only the primary emotion
                confidence = float(result["primary_confidence"])
                emotion = result["primary_emotion"]
                emotion_scores[emotion] += confidence
                total_confidence += confidence
    
    # If no valid predictions, return neutral
    if total_confidence == 0:
        return "neutral", 0, {}
    
    # Normalize scores
    for emotion in emotion_scores:
        emotion_scores[emotion] /= total_confidence
    
    # Find the emotion with highest score
    if emotion_scores:
        best_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        return best_emotion[0], best_emotion[1], dict(emotion_scores)
    else:
        return "neutral", 0, {}

def test_both_modes():
    """Test both ensemble modes with sample data."""
    
    # Sample modality results with multiple emotions
    test_results = [
        {
            "modality": "T",
            "emotions": [
                {"emotion": "happy", "confidence": 80},
                {"emotion": "neutral", "confidence": 20}
            ],
            "primary_emotion": "happy",
            "primary_confidence": 80,
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
            "reasoning": "Combined analysis suggests happiness"
        }
    ]
    
    print("=== Testing Primary Emotion Only Mode ===")
    final_emotion, final_confidence, emotion_scores = ensemble_predictions(test_results, use_all_emotions=False)
    print(f"Final Emotion: {final_emotion}")
    print(f"Final Confidence: {final_confidence:.2f}")
    print(f"All Emotion Scores: {emotion_scores}")
    print()
    
    print("=== Testing All Emotions Mode ===")
    final_emotion_all, final_confidence_all, emotion_scores_all = ensemble_predictions(test_results, use_all_emotions=True)
    print(f"Final Emotion: {final_emotion_all}")
    print(f"Final Confidence: {final_confidence_all:.2f}")
    print(f"All Emotion Scores: {emotion_scores_all}")
    print()
    
    print("=== Comparison ===")
    print(f"Primary Only Mode: {final_emotion} ({final_confidence:.2f})")
    print(f"All Emotions Mode: {final_emotion_all} ({final_confidence_all:.2f})")
    
    # Show the difference in calculation
    print("\n=== Calculation Details ===")
    print("Primary Only Mode:")
    print("- happy: 80 + 70 + 85 = 235")
    print("- neutral: 60")
    print("- Total: 295")
    print("- happy: 235/295 = 0.797")
    
    print("\nAll Emotions Mode:")
    print("- happy: 80 + 70 + 40 + 85 = 275")
    print("- neutral: 20 + 60 + 15 = 95")
    print("- sad: 30")
    print("- Total: 400")
    print("- happy: 275/400 = 0.688")
    print("- neutral: 95/400 = 0.238")
    print("- sad: 30/400 = 0.075")

if __name__ == "__main__":
    test_both_modes()


