# Limitations

the data quality is generated based on text only. If we can generate confidence scores when we convert modality to text descrption, it would serve as a better data quality metrics.

# Ensemble Logic Explanation

## Overview

The updated system implements a **weighted voting ensemble** approach that combines predictions from multiple modalities to produce a final emotion prediction with confidence scores.

**Four Ensemble Modes Available:**

1. **Primary Emotion Only** (default): Uses only the most confident emotion from each modality
2. **All Emotions**: Uses all emotions and their confidence scores from each modality
3. **Data Quality Weighting**: Weights predictions by data quality score in addition to confidence
4. **Primary/Secondary Tracking**: Tracks both primary and secondary predictions with one-shot/two-shot accuracy

## Step-by-Step Process

### 1. Individual Modality Analysis

For each video sample, the system analyzes **4 different modalities**:

- **T (Text/Transcript)**: Analyzes only the speech transcript
- **A (Audio)**: Analyzes only audio characteristics (tone, pitch, prosody)
- **V (Visual)**: Analyzes only visual cues (facial expressions, body language)
- **TAV (Combined)**: Analyzes all modalities together

### 2. Confidence Scoring for Each Modality

Each modality returns a structured response with confidence scores and data quality assessment:

```json
{
  "emotions": [
    { "emotion": "happy", "confidence": 85 },
    { "emotion": "neutral", "confidence": 15 }
  ],
  "primary_emotion": "happy",
  "primary_confidence": 85,
  "data_quality": {
    "score": 90,
    "issues": [],
    "reasoning": "Clear speech, good audio quality"
  },
  "reasoning": "Text indicates positive sentiment"
}
```

### 3. Ensemble Voting Algorithm

The `ensemble_predictions()` function implements **weighted voting** with multiple modes:

```python
def ensemble_predictions(modality_results, use_all_emotions=False, use_data_quality=False):
    emotion_scores = defaultdict(float)
    total_weight = 0

    # Step 1: Collect weighted votes
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

    # Step 2: Normalize scores
    for emotion in emotion_scores:
        emotion_scores[emotion] /= total_weight

    # Step 3: Find top two emotions
    if emotion_scores:
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        primary_emotion, primary_confidence = sorted_emotions[0]
        secondary_emotion, secondary_confidence = sorted_emotions[1] if len(sorted_emotions) > 1 else ("none", 0.0)
        return primary_emotion, primary_confidence, secondary_emotion, secondary_confidence, dict(emotion_scores)
    else:
        return "neutral", 0, "neutral", 0, {}
```

### 4. Example Calculation

Let's say we have these modality results:

| Modality       | Primary Emotion | Confidence |
| -------------- | --------------- | ---------- |
| T (Text)       | happy           | 80         |
| A (Audio)      | happy           | 70         |
| V (Visual)     | neutral         | 60         |
| TAV (Combined) | happy           | 85         |

**Step 1: Collect votes**

- happy: 80 + 70 + 85 = 235
- neutral: 60
- Total confidence: 235 + 60 = 295

**Step 2: Normalize scores**

- happy: 235/295 = 0.797 (79.7%)
- neutral: 60/295 = 0.203 (20.3%)

**Step 3: Final result**

- **Primary emotion**: happy
- **Primary confidence**: 79.7%
- **Secondary emotion**: neutral
- **Secondary confidence**: 20.3%
- **All scores**: {"happy": 0.797, "neutral": 0.203}

### 5. Comparison: Primary Only vs All Emotions

**Primary Emotion Only Mode:**

- Only uses the most confident emotion from each modality
- Simpler calculation, focuses on strongest signals
- Example: happy(80) + happy(70) + neutral(60) + happy(85) = happy wins

**All Emotions Mode:**

- Uses all emotions and their confidence scores from each modality
- More nuanced, captures secondary emotions
- Example: happy(80+70+40+85) + neutral(20+60+15) + sad(30) = happy wins but with different distribution

**Data Quality Weighting Mode:**

- Weights predictions by both confidence and data quality score
- Higher quality data gets more influence in the ensemble
- Quality weight = data_quality_score / 100.0
- Helps reduce impact of poor quality modalities

**Example with Data Quality Weighting:**

Consider a video with these modality predictions:

- **Text (T)**: "happy" with 80% confidence, data quality: 90/100
- **Audio (A)**: "happy" with 70% confidence, data quality: 30/100
- **Visual (V)**: "neutral" with 60% confidence, data quality: 75/100
- **Combined (TAV)**: "happy" with 85% confidence, data quality: 80/100

**Without Data Quality Weighting:**

```
Text: happy(80) × 1.0 = 80 points
Audio: happy(70) × 1.0 = 70 points
Visual: neutral(60) × 1.0 = 60 points
Combined: happy(85) × 1.0 = 85 points
Total for "happy": 235 points, "neutral": 60 points → "happy" wins
```

**With Data Quality Weighting:**

```
Text: happy(80) × 0.9 = 72 points (90% quality)
Audio: happy(70) × 0.3 = 21 points (30% quality - poor audio)
Visual: neutral(60) × 0.75 = 45 points (75% quality)
Combined: happy(85) × 0.8 = 68 points (80% quality)
Total for "happy": 161 points, "neutral": 45 points → "happy" wins
```

**Key Impact:** Poor quality audio (30% quality) contributes only 21 points instead of 70 points, reducing its influence on the final prediction.

**Data Quality Assessment:**

The LLM automatically assesses data quality for each modality based on:

- **Audio Quality**: Background noise, speech clarity, audio artifacts
- **Visual Quality**: Lighting conditions, camera angle, facial visibility, image resolution
- **Text Quality**: Speech clarity, transcription accuracy, content completeness
- **Overall Quality**: Completeness and coherence of multimodal information

Quality scores range from 1-100:

- **100**: Excellent quality (clear, complete, reliable)
- **75**: Good quality (minor issues, generally reliable)
- **50**: Moderate quality (some issues, partially reliable)
- **25**: Poor quality (significant issues, limited reliability)
- **10**: Very poor quality (major issues, unreliable)

**Primary/Secondary Prediction Tracking:**

- Tracks both primary (highest confidence) and secondary (second highest) predictions
- Enables one-shot and two-shot accuracy evaluation
- One-shot: Only primary prediction must be correct
- Two-shot: Either primary or secondary prediction can be correct
- Provides more nuanced evaluation of model performance

## Key Features

### 1. Weighted Voting

- Higher confidence predictions have more influence
- No equal weighting - more confident modalities get more say

### 2. Error Handling

- Modalities with "error" predictions are ignored
- If all modalities fail, defaults to "neutral"

### 3. Normalization

- Final confidence scores sum to 1.0 (100%)
- Provides interpretable probability distribution

### 4. Comprehensive Output

The system saves detailed results including:

- Individual modality predictions and confidences
- Data quality scores and issues for each modality
- Primary and secondary ensemble predictions with confidences
- All emotion scores for transparency
- One-shot and two-shot accuracy metrics
- Automatic timestamped filenames for version control

## Advantages of This Approach

1. **Robustness**: Combines strengths of different modalities
2. **Confidence-aware**: Weights predictions by their confidence
3. **Quality-aware**: Considers data quality to reduce impact of poor modalities
4. **Transparent**: Shows contribution of each modality
5. **Fallback-safe**: Handles modality failures gracefully
6. **Multi-prediction**: Tracks primary and secondary predictions for better evaluation

## Example Output Format

```json
{
  "video_id": "sample_1",
  "ground_truth": "happy",
  "modality_results": [
    {
      "modality": "T",
      "primary_emotion": "happy",
      "primary_confidence": 80,
      "data_quality": { "score": 90, "issues": [], "reasoning": "Clear speech" }
    },
    {
      "modality": "A",
      "primary_emotion": "happy",
      "primary_confidence": 70,
      "data_quality": {
        "score": 30,
        "issues": ["background_noise"],
        "reasoning": "Poor audio quality"
      }
    },
    {
      "modality": "V",
      "primary_emotion": "neutral",
      "primary_confidence": 60,
      "data_quality": {
        "score": 75,
        "issues": ["poor_lighting"],
        "reasoning": "Moderate lighting"
      }
    },
    {
      "modality": "TAV",
      "primary_emotion": "happy",
      "primary_confidence": 85,
      "data_quality": {
        "score": 80,
        "issues": [],
        "reasoning": "Good overall quality"
      }
    }
  ],
  "ensemble_prediction": "happy",
  "ensemble_confidence": 0.797,
  "ensemble_secondary": "neutral",
  "ensemble_secondary_confidence": 0.203,
  "emotion_scores": { "happy": 0.797, "neutral": 0.203 },
  "ensemble_mode": "all_emotions",
  "data_quality_weighting": "enabled"
}
```

This ensemble approach leverages the complementary strengths of different modalities while respecting their individual confidence levels, leading to more robust and interpretable emotion recognition results.

## Accuracy Metrics

### One-Shot Accuracy

- **Definition**: Percentage of samples where the primary (highest confidence) prediction matches the ground truth
- **Use Case**: Standard accuracy metric for single prediction evaluation
- **Formula**: `(correct_primary_predictions / total_samples) * 100`

### Two-Shot Accuracy

- **Definition**: Percentage of samples where either the primary OR secondary prediction matches the ground truth
- **Use Case**: More lenient evaluation that considers the model's top two predictions
- **Formula**: `(correct_primary_or_secondary_predictions / total_samples) * 100`
- **Benefits**: Captures cases where the model identifies the correct emotion but ranks it second

## Usage

### Command Line Options

**Primary Emotion Only Mode (default):**

```bash
python prob_experiment_baseline_main.py --input data.json --model gpt-4o-mini --dataset MERR
```

**All Emotions Mode:**

```bash
python prob_experiment_baseline_main.py --input data.json --model gpt-4o-mini --dataset MERR --use-all-emotions
```

**Data Quality Weighting Mode:**

```bash
python prob_experiment_baseline_main.py --input data.json --model gpt-4o-mini --dataset MERR --use-data-quality
```

**Combined Modes:**

```bash
# All emotions + Data quality weighting
python prob_experiment_baseline_main.py --input data.json --model gpt-4o-mini --dataset MERR --use-all-emotions --use-data-quality

# All emotions + Data quality + Primary/Secondary tracking (recommended)
python prob_experiment_baseline_main.py --input data.json --model gpt-4o-mini --dataset MERR --use-all-emotions --use-data-quality
```

### Output Files

The system generates two output files:

1. **Text file** (`.txt`): Summary results with ensemble predictions
2. **JSON file** (`.json`): Detailed results including all modality predictions and ensemble scores

The filename includes the ensemble mode and timestamp:

- `ensemble_primary_only_gpt-4o-mini_20241201_143022_results.txt` (default with timestamp)
- `ensemble_all_emotions_gpt-4o-mini_20241201_143022_results.txt` (with `--use-all-emotions` flag)
- `ensemble_primary_only_with_quality_gpt-4o-mini_20241201_143022_results.txt` (with `--use-data-quality` flag)
- `ensemble_all_emotions_with_quality_gpt-4o-mini_20241201_143022_results.txt` (with both flags)

**Timestamp Format**: `YYYYMMDD_HHMMSS` (e.g., `20241201_143022` for December 1, 2024 at 14:30:22)

**Custom Output Names**: If you provide `--output filename.txt`, it becomes `filename_20241201_143022.txt`

## Example Output

### Console Output

```
Ensemble mode: All emotions
Data quality weighting: Enabled
Output file: ensemble_all_emotions_with_quality_gpt-5-nano_20241201_143022_results.txt

Video sample_00005497.mp4: GT:worried, Ensemble:angry(0.3), Secondary:neutral(0.2), Modalities:[T:[angry(85), doubt(15)][Q:90], A:[neutral(95), happy(5)][Q:30], V:[happy(78), surprise(22)][Q:75], TAV:[angry(60), worried(40)][Q:80]]

One-Shot Accuracy (Primary Only): 0.75
Two-Shot Accuracy (Primary or Secondary): 0.85
Overall Accuracy: 0.75
```

### Key Features in Output

- **Quality scores**: `[Q:90]` shows data quality score for each modality
- **Secondary predictions**: Shows second most confident emotion
- **Multiple accuracy metrics**: One-shot and two-shot evaluation
- **Timestamped files**: Automatic version control
