# Experimental Setup and Scripts Explanation

## Overview

This document explains the two main scripts used in the emotion recognition experiments and provides a complete record of the experimental setup across three datasets (MER, MELD, IEMOCAP) with different models (GPT-5-nano, GPT-4o-mini) and ensemble configurations.

## Script 1: `neurips_baseline_main.py` - Single Modality Baseline

### Purpose

This script implements a **single-modality baseline** approach for emotion recognition. It processes video data using a specified combination of modalities (T, A, V, TAV) and generates predictions using a single LLM call.

### Key Features

1. **Single Modality Processing**: Processes one combination of modalities at a time
2. **JSON Output Support**: Can output structured JSON with confidence scores when available
3. **Multi-dataset Support**: Supports MER, MELD, IEMOCAP datasets
4. **Concurrent Processing**: Uses ThreadPoolExecutor for parallel API calls
5. **Error Handling**: Robust error handling for API failures and parsing issues

### How It Works

#### 1. Modality Combination

The script accepts a `--comb` parameter that specifies which modalities to use:

- `T`: Text/Transcript only
- `A`: Audio only
- `V`: Visual only
- `TA`: Text + Audio
- `TV`: Text + Visual
- `AV`: Audio + Visual
- `TAV`: All three modalities combined
- `RTAV`: All modalities + Reasoning caption

#### 2. Message Construction

```python
def call_llm(sample, model, comb, i):
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
```

#### 3. JSON Response Parsing

The script attempts to parse JSON responses from the LLM to extract structured predictions:

```python
try:
    parsed = json.loads(predicted_label)
    if parsed and isinstance(parsed, dict):
        prediction_json_entry = dict(parsed)
        prediction_json_entry["video_id"] = sample.get("video_id", f"sample_{i}")
        prediction_json_entry["ground_truth"] = sample.get("true_label", "").strip().lower()
        predicted_label = parsed.get("first_emotion", "").strip().lower()
```

### Usage Example

```bash
python neurips_baseline_main.py \
    --input final/data/MER_annotation.json \
    --model gpt-5-nano \
    --output final/result/MER/mar-workshop/baseline_gpt5-nano.txt \
    --comb TAV \
    --dataset MER
```

## Script 2: `prob_experiment_baseline_main.py` - Ensemble Approach

### Purpose

This script implements a **multi-modality ensemble** approach that analyzes each modality separately and then combines their predictions using weighted voting based on confidence scores and data quality.

### Key Features

1. **Multi-Modality Analysis**: Analyzes T, A, V, and TAV modalities separately
2. **Confidence Scoring**: Each modality provides confidence scores for emotions
3. **Data Quality Assessment**: Evaluates data quality for each modality
4. **Ensemble Voting**: Combines predictions using weighted voting
5. **Multiple Ensemble Modes**: Primary-only vs all-emotions, with/without data quality weighting
6. **Multi-shot Accuracy**: Tracks primary and secondary predictions

### How It Works

#### 1. Individual Modality Analysis

For each video sample, the script analyzes 4 modalities separately:

- **T (Text)**: Speech transcript analysis
- **A (Audio)**: Audio characteristics analysis
- **V (Visual)**: Visual cues analysis
- **TAV (Combined)**: All modalities together

#### 2. Confidence and Quality Assessment

Each modality returns structured predictions:

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

#### 3. Ensemble Voting Algorithm

The script implements weighted voting with multiple modes:

**Primary Emotion Only Mode** (default):

- Uses only the most confident emotion from each modality
- Simpler calculation, focuses on strongest signals

**All Emotions Mode** (`--use-all-emotions`):

- Uses all emotions and their confidence scores from each modality
- More nuanced, captures secondary emotions

**Data Quality Weighting** (`--use-data-quality`):

- Weights predictions by both confidence and data quality score
- Higher quality data gets more influence in the ensemble

#### 4. Multi-shot Accuracy Tracking

The script tracks multiple accuracy metrics:

- **One-shot accuracy**: Primary prediction must be correct
- **Two-shot accuracy**: Either primary or secondary prediction can be correct
- **Three-shot accuracy**: Top 3 predictions considered
- **Four-shot accuracy**: Top 4 predictions considered
- **Five-shot accuracy**: Top 5 predictions considered

### Usage Examples

#### Basic Ensemble (Primary Emotions Only)

```bash
python prob_experiment_baseline_main.py \
    --input final/data/MER_annotation.json \
    --model gpt-5-nano \
    --output final/result/MER/mar-workshop/ensemble_gpt5-nano.txt \
    --dataset MER \
    --workers 10
```

#### All Emotions Mode

```bash
python prob_experiment_baseline_main.py \
    --input final/data/MER_annotation.json \
    --model gpt-5-nano \
    --output final/result/MER/mar-workshop/ensemble_gpt5-nano.txt \
    --dataset MER \
    --workers 10 \
    --use-all-emotions
```

#### All Emotions + Data Quality Weighting

```bash
python prob_experiment_baseline_main.py \
    --input final/data/MER_annotation.json \
    --model gpt-5-nano \
    --output final/result/MER/mar-workshop/ensemble_gpt5-nano.txt \
    --dataset MER \
    --workers 10 \
    --use-all-emotions \
    --use-data-quality
```

## Complete Experimental Setup

### Dataset Information

The experiments were conducted on three emotion recognition datasets:

1. **MER (Multimodal Emotion Recognition)**: 6 emotions (happy, neutral, worried, surprise, angry, sad)
2. **MELD (Multimodal EmotionLines Dataset)**: 7 emotions (disgust, surprise, anger, joy, fear, sadness, neutral)
3. **IEMOCAP (Interactive Emotional Dyadic Motion Capture)**: 8 emotions (frustrated, excited, happy, fearful, neutral, sad, angry, surprised)

### Experimental Design

For each dataset, the following experiments were conducted:

1. **Baseline TAV**: Single LLM call with all modalities combined
2. **Ensemble (All Emotions)**: Separate modality analysis with ensemble voting
3. **Ensemble (All Emotions + Data Quality)**: Enhanced ensemble with data quality weighting

### MER Dataset Experiments

#### Baseline Experiments

```bash
# GPT-5-nano baseline
python neurips_baseline_main.py \
    --input final/data/MER_annotation.json \
    --model gpt-5-nano \
    --output final/result/MER/mar-workshop/baseline_gpt5-nano.txt \
    --comb TAV \
    --dataset MER

# GPT-4o-mini baseline
python neurips_baseline_main.py \
    --input final/data/MER_annotation.json \
    --model gpt-4o-mini \
    --output final/result/MER/mar-workshop/baseline_gpt4-mini.txt \
    --comb TAV \
    --dataset MER
```

#### Ensemble Experiments

```bash
# GPT-5-nano ensemble (all emotions)
python prob_experiment_baseline_main.py \
    --input final/data/MER_annotation.json \
    --model gpt-5-nano \
    --output final/result/MER/mar-workshop/ensemble_gpt5-nano.txt \
    --dataset MER \
    --workers 10 \
    --use-all-emotions

# GPT-4o-mini ensemble (all emotions)
python prob_experiment_baseline_main.py \
    --input final/data/MER_annotation.json \
    --model gpt-4o-mini \
    --output final/result/MER/mar-workshop/ensemble_gpt4-mini.txt \
    --dataset MER \
    --workers 10 \
    --use-all-emotions

# GPT-5-nano ensemble (all emotions + data quality)
python prob_experiment_baseline_main.py \
    --input final/data/MER_annotation.json \
    --model gpt-5-nano \
    --output final/result/MER/mar-workshop/ensemble_gpt5-nano.txt \
    --dataset MER \
    --workers 10 \
    --use-all-emotions \
    --use-data-quality

# GPT-4o-mini ensemble (all emotions + data quality)
python prob_experiment_baseline_main.py \
    --input final/data/MER_annotation.json \
    --model gpt-4o-mini \
    --output final/result/MER/mar-workshop/ensemble_gpt4-mini.txt \
    --dataset MER \
    --workers 10 \
    --use-all-emotions \
    --use-data-quality
```

#### Analysis

```bash
python analyze_ensemble_results.py \
    final/result/MER/mar-workshop/ensemble_gpt5-nano_20250830_015314.json \
    --confidence-threshold 70
```

### MELD Dataset Experiments

#### Baseline Experiments

```bash
# GPT-5-nano baseline
python neurips_baseline_main.py \
    --input final/data/MELD_annotation.json \
    --model gpt-5-nano \
    --output final/result/MELD/mar-workshop/baseline_gpt5-nano.txt \
    --comb TAV \
    --dataset MELD

# GPT-4o-mini baseline
python neurips_baseline_main.py \
    --input final/data/MELD_annotation.json \
    --model gpt-4o-mini \
    --output final/result/MELD/mar-workshop/baseline_gpt4-mini.txt \
    --comb TAV \
    --dataset MELD
```

#### Ensemble Experiments

```bash
# GPT-5-nano ensemble (all emotions)
python prob_experiment_baseline_main.py \
    --input final/data/MELD_annotation.json \
    --model gpt-5-nano \
    --output final/result/MELD/mar-workshop/ensemble_gpt5-nano.txt \
    --dataset MELD \
    --workers 10 \
    --use-all-emotions

# GPT-4o-mini ensemble (all emotions)
python prob_experiment_baseline_main.py \
    --input final/data/MELD_annotation.json \
    --model gpt-4o-mini \
    --output final/result/MELD/mar-workshop/ensemble_gpt4-mini.txt \
    --dataset MELD \
    --workers 10 \
    --use-all-emotions

# GPT-5-nano ensemble (all emotions + data quality)
python prob_experiment_baseline_main.py \
    --input final/data/MELD_annotation.json \
    --model gpt-5-nano \
    --output final/result/MELD/mar-workshop/ensemble_gpt5-nano.txt \
    --dataset MELD \
    --workers 10 \
    --use-all-emotions \
    --use-data-quality

# GPT-4o-mini ensemble (all emotions + data quality)
python prob_experiment_baseline_main.py \
    --input final/data/MELD_annotation.json \
    --model gpt-4o-mini \
    --output final/result/MELD/mar-workshop/ensemble_gpt4-mini.txt \
    --dataset MELD \
    --workers 10 \
    --use-all-emotions \
    --use-data-quality
```

#### Analysis

```bash
python analyze_ensemble_results.py \
    final/result/MELD/mar-workshop/ensemble_gpt5-nano_20250830_024815.json \
    --confidence-threshold 70
```

### IEMOCAP Dataset Experiments

#### Baseline Experiments

```bash
# GPT-5-nano baseline
python neurips_baseline_main.py \
    --input final/data/IEMOCAP_annotation.json \
    --model gpt-5-nano \
    --output final/result/IEMOCAP/mar-workshop/baseline_gpt5-nano.txt \
    --comb TAV \
    --dataset IEMOCAP

# GPT-4o-mini baseline
python neurips_baseline_main.py \
    --input final/data/IEMOCAP_annotation.json \
    --model gpt-4o-mini \
    --output final/result/IEMOCAP/mar-workshop/baseline_gpt4-mini.txt \
    --comb TAV \
    --dataset IEMOCAP
```

#### Ensemble Experiments

```bash
# GPT-5-nano ensemble (all emotions)
python prob_experiment_baseline_main.py \
    --input final/data/IEMOCAP_annotation.json \
    --model gpt-5-nano \
    --output final/result/IEMOCAP/mar-workshop/ensemble_gpt5-nano.txt \
    --dataset IEMOCAP \
    --workers 10 \
    --use-all-emotions

# GPT-4o-mini ensemble (all emotions)
python prob_experiment_baseline_main.py \
    --input final/data/IEMOCAP_annotation.json \
    --model gpt-4o-mini \
    --output final/result/IEMOCAP/mar-workshop/ensemble_gpt4-mini.txt \
    --dataset IEMOCAP \
    --workers 10 \
    --use-all-emotions

# GPT-5-nano ensemble (all emotions + data quality)
python prob_experiment_baseline_main.py \
    --input final/data/IEMOCAP_annotation.json \
    --model gpt-5-nano \
    --output final/result/IEMOCAP/mar-workshop/ensemble_gpt5-nano.txt \
    --dataset IEMOCAP \
    --workers 10 \
    --use-all-emotions \
    --use-data-quality

# GPT-4o-mini ensemble (all emotions + data quality)
python prob_experiment_baseline_main.py \
    --input final/data/IEMOCAP_annotation.json \
    --model gpt-4o-mini \
    --output final/result/IEMOCAP/mar-workshop/ensemble_gpt4-mini.txt \
    --dataset IEMOCAP \
    --workers 10 \
    --use-all-emotions \
    --use-data-quality
```

#### Analysis

```bash
python analyze_ensemble_results.py \
    final/result/IEMOCAP/mar-workshop/ensemble_gpt5-nano_20250830_025142.json \
    --confidence-threshold 70
```

## Key Findings

Based on the analysis of ensemble results:

1. **Visual modality (V)** shows the highest accuracy across datasets
2. **Audio modality (A)** tends to sabotage the ensemble performance the most
3. **Ensemble approach** generally improves performance over single-modality baselines
4. **Data quality weighting** helps reduce the impact of poor-quality modalities
5. **Multi-shot accuracy** provides more nuanced evaluation of model performance

## Output File Structure

### Text Output (.txt)

Contains summary results with:

- Individual sample predictions
- Overall accuracy metrics
- Per-label precision, recall, and F1 scores
- Multi-shot accuracy metrics

### JSON Output (.json)

Contains detailed results with:

- Individual modality predictions and confidences
- Data quality scores and issues
- Ensemble voting details
- Primary and secondary predictions
- All emotion scores for transparency

### Timestamped Filenames

All output files include timestamps in format `YYYYMMDD_HHMMSS` for version control and experiment tracking.

## Dependencies

Both scripts require:

- `openai`: For LLM API calls
- `sklearn`: For evaluation metrics
- `python-dotenv`: For environment variable loading
- Dataset-specific constants modules in the `neurips/` directory

## Error Handling

Both scripts implement robust error handling:

- API failures are caught and logged
- JSON parsing errors fall back to text parsing
- Missing modalities are handled gracefully
- Invalid predictions default to "neutral" or "error"

This experimental setup provides a comprehensive evaluation of different approaches to multimodal emotion recognition, from simple baselines to sophisticated ensemble methods with confidence scoring and data quality assessment.
