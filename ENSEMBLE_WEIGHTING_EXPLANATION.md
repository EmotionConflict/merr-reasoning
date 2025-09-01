# Ensemble Weighting with Accuracy and Sabotage Masks

This document explains how the new `--use-accuracy-mask` and `--use-sabotage-mask` flags work in the ensemble experiment script.

## Overview

The ensemble system now supports four different weighting schemes that can be combined:

1. **Confidence Weighting** (always active)
2. **Data Quality Weighting** (`--use-data-quality`)
3. **Accuracy Mask Weighting** (`--use-accuracy-mask`)
4. **Sabotage Mask Weighting** (`--use-sabotage-mask`)

## How the Weighting Works

### Final Weight Calculation

The final weight for each modality is calculated as:

```
combined_weight = quality_weight × accuracy_weight × sabotage_weight
```

Where:

- `quality_weight` = data quality score / 100 (if `--use-data-quality` is enabled)
- `accuracy_weight` = modality accuracy score (if `--use-accuracy-mask` is enabled)
- `sabotage_weight` = sabotage penalty (if `--use-sabotage-mask` is enabled)

If any flag is disabled, its corresponding weight defaults to 1.0.

### Dataset-Specific Weights

The script automatically uses different weights based on the dataset:

#### MER Dataset

```python
# Accuracy weights (from GPT-5-nano results)
accuracy_weights = {
    "T": 0.186,    # 18.6% accuracy
    "A": 0.250,    # 25.0% accuracy
    "V": 0.483,    # 48.3% accuracy
    "TAV": 0.333   # 33.3% accuracy
}

# Sabotage penalty weights (1 - sabotage percentage)
sabotage_weights = {
    "T": 0.883,    # 1 - 0.117 (7/60 sabotage cases)
    "A": 0.383,    # 1 - 0.617 (37/60 sabotage cases)
    "V": 0.783,    # 1 - 0.217 (13/60 sabotage cases)
    "TAV": 0.683   # 1 - 0.317 (19/60 sabotage cases)
}
```

#### MELD Dataset

```python
# Accuracy weights
accuracy_weights = {
    "T": 0.508,    # 50.8% accuracy
    "A": 0.298,    # 29.8% accuracy
    "V": 0.188,    # 18.8% accuracy
    "TAV": 0.236   # 23.6% accuracy
}

# Sabotage penalty weights
sabotage_weights = {
    "T": 0.885,    # 1 - 0.115 (22/191 sabotage cases)
    "A": 0.518,    # 1 - 0.482 (92/191 sabotage cases)
    "V": 0.529,    # 1 - 0.471 (90/191 sabotage cases)
    "TAV": 0.576   # 1 - 0.424 (81/191 sabotage cases)
}
```

#### IEMOCAP Dataset

```python
# Accuracy weights
accuracy_weights = {
    "T": 0.326,    # 32.6% accuracy
    "A": 0.248,    # 24.8% accuracy
    "V": 0.171,    # 17.1% accuracy
    "TAV": 0.279   # 27.9% accuracy
}

# Sabotage penalty weights
sabotage_weights = {
    "T": 0.837,    # 1 - 0.163 (21/129 sabotage cases)
    "A": 0.403,    # 1 - 0.597 (77/129 sabotage cases)
    "V": 0.488,    # 1 - 0.512 (66/129 sabotage cases)
    "TAV": 0.543   # 1 - 0.457 (59/129 sabotage cases)
}
```

## Detailed Example

Let's walk through a complete example using the MER dataset:

### Sample Input

```json
{
  "video_id": "sample_123",
  "true_label": "happy",
  "transcript": "I'm so excited about this!",
  "audio_description": "Clear, enthusiastic speech with rising intonation",
  "visual_expression_description": "Bright smile, wide eyes, animated gestures",
  "visual_objective_description": "Person appears joyful and energetic"
}
```

### Modality Predictions (with confidence scores)

```
T: happy (confidence: 85, data_quality: 90)
A: happy (confidence: 70, data_quality: 75)
V: happy (confidence: 95, data_quality: 85)
TAV: happy (confidence: 88, data_quality: 80)
```

### Weight Calculations

#### 1. Baseline (no additional weighting)

```
T: 85 × 1.0 × 1.0 × 1.0 = 85.0
A: 70 × 1.0 × 1.0 × 1.0 = 70.0
V: 95 × 1.0 × 1.0 × 1.0 = 95.0
TAV: 88 × 1.0 × 1.0 × 1.0 = 88.0
```

#### 2. With Data Quality Weighting (`--use-data-quality`)

```
T: 85 × (90/100) × 1.0 × 1.0 = 85 × 0.9 = 76.5
A: 70 × (75/100) × 1.0 × 1.0 = 70 × 0.75 = 52.5
V: 95 × (85/100) × 1.0 × 1.0 = 95 × 0.85 = 80.75
TAV: 88 × (80/100) × 1.0 × 1.0 = 88 × 0.8 = 70.4
```

#### 3. With Accuracy Mask (`--use-accuracy-mask`)

```
T: 85 × 1.0 × 0.186 × 1.0 = 85 × 0.186 = 15.81
A: 70 × 1.0 × 0.250 × 1.0 = 70 × 0.250 = 17.5
V: 95 × 1.0 × 0.483 × 1.0 = 95 × 0.483 = 45.89
TAV: 88 × 1.0 × 0.333 × 1.0 = 88 × 0.333 = 29.30
```

#### 4. With Sabotage Mask (`--use-sabotage-mask`)

```
T: 85 × 1.0 × 1.0 × 0.883 = 85 × 0.883 = 75.06
A: 70 × 1.0 × 1.0 × 0.383 = 70 × 0.383 = 26.81
V: 95 × 1.0 × 1.0 × 0.783 = 95 × 0.783 = 74.39
TAV: 88 × 1.0 × 1.0 × 0.683 = 88 × 0.683 = 60.10
```

#### 5. With All Weighting Schemes (`--use-data-quality --use-accuracy-mask --use-sabotage-mask`)

```
T: 85 × 0.9 × 0.186 × 0.883 = 85 × 0.9 × 0.186 × 0.883 = 12.56
A: 70 × 0.75 × 0.250 × 0.383 = 70 × 0.75 × 0.250 × 0.383 = 5.03
V: 95 × 0.85 × 0.483 × 0.783 = 95 × 0.85 × 0.483 × 0.783 = 30.47
TAV: 88 × 0.8 × 0.333 × 0.683 = 88 × 0.8 × 0.333 × 0.683 = 16.00
```

### Final Ensemble Prediction

After normalizing the scores, the ensemble would predict:

- **Baseline**: V (highest confidence)
- **With Data Quality**: V (highest weighted score)
- **With Accuracy Mask**: V (highest weighted score)
- **With Sabotage Mask**: T (highest weighted score)
- **With All Weighting**: V (highest weighted score)

## Command Line Usage Examples

### Basic Usage

```bash
# Run with accuracy mask only
python after_knowing_sabotage_experiment.py --dataset MER --use-accuracy-mask

# Run with sabotage mask only
python after_knowing_sabotage_experiment.py --dataset MELD --use-sabotage-mask

# Run with both masks
python after_knowing_sabotage_experiment.py --dataset IEMOCAP --use-accuracy-mask --use-sabotage-mask
```

### Advanced Usage

```bash
# Full ensemble with all weighting schemes
python after_knowing_sabotage_experiment.py \
    --dataset MER \
    --use-all-emotions \
    --use-data-quality \
    --use-accuracy-mask \
    --use-sabotage-mask \
    --model gpt-4o-mini \
    --workers 4

# Compare different approaches
python after_knowing_sabotage_experiment.py --dataset MELD --use-accuracy-mask
python after_knowing_sabotage_experiment.py --dataset MELD --use-sabotage-mask
python after_knowing_sabotage_experiment.py --dataset MELD --use-accuracy-mask --use-sabotage-mask
```

## Output Format

The script provides detailed output showing the weighting information:

```
Ensemble mode: Primary emotion only
Data quality weighting: Enabled
Accuracy weighting: Enabled
Sabotage weighting: Enabled
Accuracy weights: {'T': 0.186, 'A': 0.25, 'V': 0.483, 'TAV': 0.333}
Sabotage penalty weights: {'T': 0.883, 'A': 0.383, 'V': 0.783, 'TAV': 0.683}

Video sample_123: GT:happy, Ensemble:happy(0.456), Secondary:neutral(0.234),
Modalities:[T:happy(85)[Q:90][A:0.186][S:0.883], A:happy(70)[Q:75][A:0.250][S:0.383],
V:happy(95)[Q:85][A:0.483][S:0.783], TAV:happy(88)[Q:80][A:0.333][S:0.683]]
```

Where:

- `[Q:90]` = Data quality score
- `[A:0.186]` = Accuracy weight
- `[S:0.883]` = Sabotage penalty weight

## Expected Impact

### Accuracy Mask Impact

- **Higher accuracy modalities** get more weight in the ensemble
- **Lower accuracy modalities** get penalized
- Particularly effective when there's a significant accuracy gap between modalities

### Sabotage Mask Impact

- **Modalities with fewer sabotage cases** get more weight
- **Modalities with more sabotage cases** get penalized
- Helps reduce the influence of unreliable modalities

### Combined Impact

- Creates a more robust ensemble that considers both historical performance and data quality
- Can significantly improve overall accuracy by downweighting problematic modalities
- Provides a principled way to handle modality-specific issues

## Best Practices

1. **Start with individual masks** to understand their impact
2. **Combine masks gradually** to see cumulative effects
3. **Monitor the weights** to ensure they make sense for your data
4. **Compare results** across different weighting schemes
5. **Use cross-validation** to validate the effectiveness of different approaches
