#!/usr/bin/env python3
"""
Test script for the ensemble emotion recognition system.
"""

import json
from collections import defaultdict

def ensemble_predictions(modality_results):
    """
    Ensemble predictions from all modalities to produce final emotion prediction.
    Uses weighted voting based on confidence scores.
    """
    emotion_scores = defaultdict(float)
    total_confidence = 0
    
    # Collect all emotion predictions with their confidence scores
    for result in modality_results:
        if result["primary_emotion"] != "error":
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

def test_ensemble():
    """Test the ensemble function with sample data."""
    
    # Sample modality results
    test_results = [
        {
            "modality": "T",
            "primary_emotion": "happy",
            "primary_confidence": 80,
            "reasoning": "Text indicates positive sentiment"
        },
        {
            "modality": "A", 
            "primary_emotion": "happy",
            "primary_confidence": 70,
            "reasoning": "Audio tone suggests happiness"
        },
        {
            "modality": "V",
            "primary_emotion": "neutral", 
            "primary_confidence": 60,
            "reasoning": "Facial expression is neutral"
        },
        {
            "modality": "TAV",
            "primary_emotion": "happy",
            "primary_confidence": 85,
            "reasoning": "Combined analysis suggests happiness"
        }
    ]
    
    # Test ensemble
    final_emotion, final_confidence, emotion_scores = ensemble_predictions(test_results)
    
    print("Test Results:")
    print(f"Final Emotion: {final_emotion}")
    print(f"Final Confidence: {final_confidence:.2f}")
    print(f"All Emotion Scores: {emotion_scores}")
    
    # Expected: happy should win due to high confidence scores
    assert final_emotion == "happy", f"Expected 'happy', got '{final_emotion}'"
    print("✓ Test passed!")

if __name__ == "__main__":
    test_ensemble()


