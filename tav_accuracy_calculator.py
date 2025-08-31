import json
from collections import Counter
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_json_data(file_path):
    """Load the JSON data from the file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_tav_results(data):
    """Extract only TAV modality results from the data."""
    tav_results = []
    
    for item in data:
        video_id = item['video_id']
        ground_truth = item['ground_truth']
        
        # Find TAV modality result
        for modality_result in item['modality_results']:
            if modality_result['modality'] == 'TAV':
                tav_prediction = modality_result['primary_emotion']
                tav_confidence = modality_result['primary_confidence']
                tav_emotions = modality_result['emotions']
                
                tav_results.append({
                    'video_id': video_id,
                    'ground_truth': ground_truth,
                    'tav_prediction': tav_prediction,
                    'tav_confidence': tav_confidence,
                    'tav_emotions': tav_emotions
                })
                break
    
    return tav_results

def calculate_multi_shot_accuracy(tav_results):
    """Calculate multi-shot accuracy metrics (1-shot through 5-shot)."""
    shot_accuracies = {}
    
    for shot in range(1, 6):
        correct_count = 0
        total_count = len(tav_results)
        
        for result in tav_results:
            ground_truth = result['ground_truth']
            emotions = result['tav_emotions']
            
            # Get top N emotions by confidence
            top_emotions = [emotion['emotion'] for emotion in emotions[:shot]]
            
            # Check if ground truth is in top N
            if ground_truth in top_emotions:
                correct_count += 1
        
        accuracy = correct_count / total_count
        shot_accuracies[f'{shot}-shot'] = accuracy
    
    return shot_accuracies

def calculate_accuracy(tav_results):
    """Calculate accuracy and other metrics for TAV predictions."""
    ground_truths = [result['ground_truth'] for result in tav_results]
    predictions = [result['tav_prediction'] for result in tav_results]
    
    # Calculate accuracy
    accuracy = accuracy_score(ground_truths, predictions)
    
    # Get unique emotions for analysis
    unique_emotions = sorted(list(set(ground_truths + predictions)))
    
    # Calculate per-emotion accuracy
    emotion_accuracy = {}
    for emotion in unique_emotions:
        emotion_correct = sum(1 for gt, pred in zip(ground_truths, predictions) 
                            if gt == emotion and pred == emotion)
        emotion_total = sum(1 for gt in ground_truths if gt == emotion)
        emotion_accuracy[emotion] = emotion_correct / emotion_total if emotion_total > 0 else 0
    
    return {
        'overall_accuracy': accuracy,
        'emotion_accuracy': emotion_accuracy,
        'ground_truths': ground_truths,
        'predictions': predictions,
        'unique_emotions': unique_emotions
    }

def analyze_confidence_distribution(tav_results):
    """Analyze confidence distribution for correct vs incorrect predictions."""
    correct_confidences = []
    incorrect_confidences = []
    
    for result in tav_results:
        confidence = result['tav_confidence']
        if result['ground_truth'] == result['tav_prediction']:
            correct_confidences.append(confidence)
        else:
            incorrect_confidences.append(confidence)
    
    return {
        'correct_confidences': correct_confidences,
        'incorrect_confidences': incorrect_confidences,
        'avg_correct_confidence': sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0,
        'avg_incorrect_confidence': sum(incorrect_confidences) / len(incorrect_confidences) if incorrect_confidences else 0
    }

def create_confusion_matrix(ground_truths, predictions, unique_emotions):
    """Create and display confusion matrix."""
    cm = confusion_matrix(ground_truths, predictions, labels=unique_emotions)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_emotions, yticklabels=unique_emotions)
    plt.title('TAV Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('tav_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_emotion_coverage(tav_results):
    """Analyze how many emotions are typically predicted per sample."""
    emotion_counts = []
    for result in tav_results:
        emotion_counts.append(len(result['tav_emotions']))
    
    return {
        'avg_emotions_per_sample': sum(emotion_counts) / len(emotion_counts),
        'min_emotions': min(emotion_counts),
        'max_emotions': max(emotion_counts),
        'emotion_count_distribution': Counter(emotion_counts)
    }

def main():
    # File path
    file_path = 'final/result/MER/mar-workshop/all-emos-data-quality_test_ensemble_gpt5-nano_20250829_085624.json'
    
    print("Loading JSON data...")
    data = load_json_data(file_path)
    print(f"Loaded {len(data)} total samples")
    
    print("\nExtracting TAV modality results...")
    tav_results = extract_tav_results(data)
    print(f"Found {len(tav_results)} TAV results")
    
    print("\nCalculating accuracy metrics...")
    metrics = calculate_accuracy(tav_results)
    
    print("\nCalculating multi-shot accuracy metrics...")
    multi_shot_metrics = calculate_multi_shot_accuracy(tav_results)
    
    print(f"\n=== TAV ACCURACY RESULTS ===")
    print(f"Overall Accuracy (Primary Only): {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)")
    
    print(f"\n=== MULTI-SHOT ACCURACY RESULTS ===")
    for shot_name, accuracy in multi_shot_metrics.items():
        print(f"{shot_name} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print(f"\nPer-Emotion Accuracy (Primary Only):")
    for emotion, acc in metrics['emotion_accuracy'].items():
        print(f"  {emotion}: {acc:.4f} ({acc*100:.2f}%)")
    
    # Analyze emotion coverage
    print(f"\nAnalyzing emotion coverage...")
    coverage_analysis = analyze_emotion_coverage(tav_results)
    print(f"Average emotions per sample: {coverage_analysis['avg_emotions_per_sample']:.2f}")
    print(f"Min emotions: {coverage_analysis['min_emotions']}")
    print(f"Max emotions: {coverage_analysis['max_emotions']}")
    print(f"Emotion count distribution: {dict(coverage_analysis['emotion_count_distribution'])}")
    
    # Analyze confidence distribution
    print(f"\nAnalyzing confidence distribution...")
    confidence_analysis = analyze_confidence_distribution(tav_results)
    
    print(f"Average confidence for correct predictions: {confidence_analysis['avg_correct_confidence']:.2f}")
    print(f"Average confidence for incorrect predictions: {confidence_analysis['avg_incorrect_confidence']:.2f}")
    
    # Create detailed classification report
    print(f"\nDetailed Classification Report (Primary Only):")
    report = classification_report(metrics['ground_truths'], metrics['predictions'], 
                                 target_names=metrics['unique_emotions'], digits=4)
    print(report)
    
    # Create confusion matrix
    print(f"\nCreating confusion matrix...")
    create_confusion_matrix(metrics['ground_truths'], metrics['predictions'], metrics['unique_emotions'])
    
    # Save results to CSV for further analysis
    df = pd.DataFrame(tav_results)
    df.to_csv('tav_results.csv', index=False)
    print(f"\nDetailed results saved to 'tav_results.csv'")
    
    # Print some example predictions with multi-shot analysis
    print(f"\n=== SAMPLE PREDICTIONS WITH MULTI-SHOT ANALYSIS ===")
    print("First 10 predictions:")
    for i, result in enumerate(tav_results[:10]):
        status = "✓" if result['ground_truth'] == result['tav_prediction'] else "✗"
        top_emotions = [emotion['emotion'] for emotion in result['tav_emotions'][:5]]
        gt_in_top5 = "✓" if result['ground_truth'] in top_emotions else "✗"
        
        print(f"{i+1:2d}. {result['video_id']}: GT={result['ground_truth']:8s} | Pred={result['tav_prediction']:8s} | Conf={result['tav_confidence']:3.0f} | Primary:{status} | Top5:{gt_in_top5}")
        print(f"    Top 5 emotions: {top_emotions}")
    
    # Count predictions by emotion
    prediction_counts = Counter([result['tav_prediction'] for result in tav_results])
    ground_truth_counts = Counter([result['ground_truth'] for result in tav_results])
    
    print(f"\n=== PREDICTION DISTRIBUTION ===")
    print("Ground Truth Distribution:")
    for emotion in sorted(ground_truth_counts.keys()):
        print(f"  {emotion}: {ground_truth_counts[emotion]}")
    
    print(f"\nTAV Prediction Distribution (Primary):")
    for emotion in sorted(prediction_counts.keys()):
        print(f"  {emotion}: {prediction_counts[emotion]}")
    
    # Save multi-shot results to a separate file
    multi_shot_df = pd.DataFrame([
        {'shot': shot_name, 'accuracy': accuracy, 'accuracy_percent': accuracy*100}
        for shot_name, accuracy in multi_shot_metrics.items()
    ])
    multi_shot_df.to_csv('tav_multi_shot_accuracy.csv', index=False)
    print(f"\nMulti-shot accuracy results saved to 'tav_multi_shot_accuracy.csv'")

if __name__ == "__main__":
    main()
