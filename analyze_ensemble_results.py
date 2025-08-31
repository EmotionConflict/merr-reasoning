#!/usr/bin/env python3
"""
Analysis script for ensemble emotion recognition results.
Focuses on identifying modalities with confidence + data quality reporting
and detecting potential sabotage cases (high confidence but wrong predictions).
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import argparse
import os
from datetime import datetime
try:
    from scipy import stats
except ImportError:
    # Fallback if scipy is not available
    stats = None

class EnsembleAnalyzer:
    def __init__(self, file_path: str):
        """Initialize analyzer with the ensemble results file."""
        self.file_path = file_path
        self.data = self.load_data()
        self.modalities = ['T', 'A', 'V', 'TAV']
        
        # Generate filename prefix with input filename and timestamp
        input_filename = os.path.splitext(os.path.basename(file_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename_prefix = f"{input_filename}_{timestamp}"
        
    def load_data(self) -> List[Dict[str, Any]]:
        """Load the JSON data from file."""
        with open(self.file_path, 'r') as f:
            return json.load(f)
    
    def analyze_modality_reporting(self) -> Dict[str, Dict[str, Any]]:
        """Analyze which modalities report confidence and data quality."""
        reporting_stats = {}
        
        for modality in self.modalities:
            has_confidence = 0
            has_data_quality = 0
            has_reasoning = 0
            total_samples = 0
            
            for sample in self.data:
                for mod_result in sample['modality_results']:
                    if mod_result['modality'] == modality:
                        total_samples += 1
                        
                        # Check for confidence reporting
                        if 'primary_confidence' in mod_result:
                            has_confidence += 1
                        
                        # Check for data quality reporting
                        if 'data_quality' in mod_result:
                            has_data_quality += 1
                        
                        # Check for reasoning
                        if 'reasoning' in mod_result:
                            has_reasoning += 1
            
            reporting_stats[modality] = {
                'total_samples': total_samples,
                'has_confidence': has_confidence,
                'has_data_quality': has_data_quality,
                'has_reasoning': has_reasoning,
                'confidence_rate': has_confidence / total_samples if total_samples > 0 else 0,
                'data_quality_rate': has_data_quality / total_samples if total_samples > 0 else 0,
                'reasoning_rate': has_reasoning / total_samples if total_samples > 0 else 0
            }
        
        return reporting_stats
    
    def analyze_data_quality_vs_accuracy(self) -> Dict[str, Dict[str, Any]]:
        """Analyze the relationship between data quality scores and accuracy for each modality."""
        quality_accuracy_data = {mod: {'quality_scores': [], 'accuracies': [], 'cases': []} for mod in self.modalities}
        
        for sample in self.data:
            ground_truth = sample['ground_truth']
            
            for mod_result in sample['modality_results']:
                modality = mod_result['modality']
                primary_emotion = mod_result.get('primary_emotion')
                data_quality_score = mod_result.get('data_quality', {}).get('score', 0)
                
                # Handle None values for data quality score
                if data_quality_score is None:
                    data_quality_score = 0
                
                if primary_emotion and data_quality_score > 0:
                    is_correct = primary_emotion == ground_truth
                    
                    quality_accuracy_data[modality]['quality_scores'].append(data_quality_score)
                    quality_accuracy_data[modality]['accuracies'].append(1 if is_correct else 0)
                    quality_accuracy_data[modality]['cases'].append({
                        'video_id': sample['video_id'],
                        'quality_score': data_quality_score,
                        'is_correct': is_correct,
                        'predicted': primary_emotion,
                        'ground_truth': ground_truth,
                        'confidence': mod_result.get('primary_confidence', 0)
                    })
        
        # Calculate correlation and statistics for each modality
        quality_accuracy_stats = {}
        for modality, data in quality_accuracy_data.items():
            if len(data['quality_scores']) > 1:
                # Calculate correlation
                if stats is not None:
                    correlation, p_value = stats.pearsonr(data['quality_scores'], data['accuracies'])
                else:
                    # Simple correlation calculation if scipy is not available
                    correlation = np.corrcoef(data['quality_scores'], data['accuracies'])[0, 1]
                    p_value = 0.5  # Placeholder
                
                # Calculate accuracy by quality bins
                quality_bins = {
                    'low': {'range': (0, 50), 'correct': 0, 'total': 0},
                    'medium': {'range': (50, 80), 'correct': 0, 'total': 0},
                    'high': {'range': (80, 100), 'correct': 0, 'total': 0}
                }
                
                for quality, accuracy in zip(data['quality_scores'], data['accuracies']):
                    for bin_name, bin_data in quality_bins.items():
                        range_tuple = bin_data['range']
                        range_start, range_end = range_tuple
                        if range_start <= quality < range_end:
                            bin_data['total'] += 1
                            if accuracy == 1:
                                bin_data['correct'] += 1
                            break
                
                # Calculate accuracy for each bin
                bin_accuracies = {}
                for bin_name, bin_data in quality_bins.items():
                    total = bin_data['total']
                    if total > 0:
                        bin_accuracies[bin_name] = bin_data['correct'] / total
                    else:
                        bin_accuracies[bin_name] = 0
                
                quality_accuracy_stats[modality] = {
                    'correlation': correlation,
                    'p_value': p_value,
                    'total_cases': len(data['quality_scores']),
                    'mean_quality': np.mean(data['quality_scores']),
                    'std_quality': np.std(data['quality_scores']),
                    'overall_accuracy': np.mean(data['accuracies']),
                    'quality_bins': quality_bins,
                    'bin_accuracies': bin_accuracies,
                    'cases': data['cases']
                }
            else:
                quality_accuracy_stats[modality] = {
                    'correlation': 0,
                    'p_value': 1,
                    'total_cases': 0,
                    'mean_quality': 0,
                    'std_quality': 0,
                    'overall_accuracy': 0,
                    'quality_bins': {},
                    'bin_accuracies': {},
                    'cases': []
                }
        
        return quality_accuracy_stats
    
    def detect_sabotage_cases(self, confidence_threshold: float = 80) -> List[Dict[str, Any]]:
        """Detect potential sabotage cases: high confidence but wrong predictions."""
        sabotage_cases = []
        
        for sample in self.data:
            ground_truth = sample['ground_truth']
            
            for mod_result in sample['modality_results']:
                modality = mod_result['modality']
                primary_emotion = mod_result.get('primary_emotion')
                primary_confidence = mod_result.get('primary_confidence', 0)
                data_quality_score = mod_result.get('data_quality', {}).get('score', 0)
                
                # Handle None values for confidence
                if primary_confidence is None:
                    primary_confidence = 0
                if data_quality_score is None:
                    data_quality_score = 0
                
                # Check if this is a potential sabotage case
                if (primary_confidence >= confidence_threshold and 
                    primary_emotion != ground_truth):
                    
                    sabotage_case = {
                        'video_id': sample['video_id'],
                        'modality': modality,
                        'ground_truth': ground_truth,
                        'predicted_emotion': primary_emotion,
                        'confidence': primary_confidence,
                        'data_quality_score': data_quality_score,
                        'data_quality_issues': mod_result.get('data_quality', {}).get('issues', []),
                        'reasoning': mod_result.get('reasoning', ''),
                        'ensemble_prediction': sample.get('ensemble_prediction'),
                        'ensemble_confidence': sample.get('ensemble_confidence')
                    }
                    sabotage_cases.append(sabotage_case)
        
        return sabotage_cases
    
    def analyze_confidence_vs_accuracy(self) -> Dict[str, List[Tuple[float, bool]]]:
        """Analyze confidence vs accuracy for each modality."""
        confidence_accuracy = {mod: [] for mod in self.modalities}
        
        for sample in self.data:
            ground_truth = sample['ground_truth']
            
            for mod_result in sample['modality_results']:
                modality = mod_result['modality']
                primary_emotion = mod_result.get('primary_emotion')
                primary_confidence = mod_result.get('primary_confidence', 0)
                
                # Handle None values for confidence
                if primary_confidence is None:
                    primary_confidence = 0
                
                if primary_emotion and primary_confidence > 0:
                    is_correct = primary_emotion == ground_truth
                    confidence_accuracy[modality].append((primary_confidence, is_correct))
        
        return confidence_accuracy
    
    def analyze_data_quality_impact(self) -> Dict[str, List[Tuple[float, bool]]]:
        """Analyze data quality score vs accuracy for each modality."""
        quality_accuracy = {mod: [] for mod in self.modalities}
        
        for sample in self.data:
            ground_truth = sample['ground_truth']
            
            for mod_result in sample['modality_results']:
                modality = mod_result['modality']
                primary_emotion = mod_result.get('primary_emotion')
                data_quality_score = mod_result.get('data_quality', {}).get('score', 0)
                
                # Handle None values for data quality score
                if data_quality_score is None:
                    data_quality_score = 0
                
                if primary_emotion and data_quality_score > 0:
                    is_correct = primary_emotion == ground_truth
                    quality_accuracy[modality].append((data_quality_score, is_correct))
        
        return quality_accuracy
    
    def get_modality_accuracy(self) -> Dict[str, float]:
        """Calculate accuracy for each modality."""
        modality_correct = {mod: 0 for mod in self.modalities}
        modality_total = {mod: 0 for mod in self.modalities}
        
        for sample in self.data:
            ground_truth = sample['ground_truth']
            
            for mod_result in sample['modality_results']:
                modality = mod_result['modality']
                primary_emotion = mod_result.get('primary_emotion')
                
                if primary_emotion:
                    modality_total[modality] += 1
                    if primary_emotion == ground_truth:
                        modality_correct[modality] += 1
        
        return {mod: modality_correct[mod] / modality_total[mod] 
                for mod in self.modalities if modality_total[mod] > 0}
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report."""
        report = []
        report.append("=" * 80)
        report.append("ENSEMBLE EMOTION RECOGNITION ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Total samples analyzed: {len(self.data)}")
        report.append("")
        
        # Modality reporting analysis
        reporting_stats = self.analyze_modality_reporting()
        report.append("MODALITY REPORTING ANALYSIS:")
        report.append("-" * 40)
        for modality, stats in reporting_stats.items():
            report.append(f"{modality}:")
            report.append(f"  Total samples: {stats['total_samples']}")
            report.append(f"  Confidence reporting: {stats['confidence_rate']:.1%}")
            report.append(f"  Data quality reporting: {stats['data_quality_rate']:.1%}")
            report.append(f"  Reasoning reporting: {stats['reasoning_rate']:.1%}")
            report.append("")
        
        # Modality accuracy
        accuracy = self.get_modality_accuracy()
        report.append("MODALITY ACCURACY:")
        report.append("-" * 40)
        for modality, acc in accuracy.items():
            report.append(f"{modality}: {acc:.1%}")
        report.append("")
        
        # Data Quality vs Accuracy Analysis
        quality_accuracy_stats = self.analyze_data_quality_vs_accuracy()
        report.append("DATA QUALITY VS ACCURACY ANALYSIS:")
        report.append("-" * 40)
        for modality, stats in quality_accuracy_stats.items():
            if stats['total_cases'] > 0:
                report.append(f"{modality}:")
                report.append(f"  Total cases with quality scores: {stats['total_cases']}")
                report.append(f"  Correlation (quality vs accuracy): {stats['correlation']:.3f} (p={stats['p_value']:.3f})")
                report.append(f"  Mean quality score: {stats['mean_quality']:.1f} ± {stats['std_quality']:.1f}")
                report.append(f"  Overall accuracy: {stats['overall_accuracy']:.1%}")
                report.append("  Accuracy by quality bins:")
                for bin_name, bin_acc in stats['bin_accuracies'].items():
                    bin_data = stats['quality_bins'][bin_name]
                    if bin_data['total'] > 0:
                        report.append(f"    {bin_name} ({bin_data['range'][0]}-{bin_data['range'][1]}): "
                                   f"{bin_acc:.1%} ({bin_data['correct']}/{bin_data['total']})")
                report.append("")
        
        # Sabotage detection
        sabotage_cases = self.detect_sabotage_cases()
        report.append(f"SABOTAGE DETECTION (confidence >= 80%):")
        report.append("-" * 40)
        report.append(f"Total sabotage cases found: {len(sabotage_cases)}")
        
        if sabotage_cases:
            modality_sabotage = Counter(case['modality'] for case in sabotage_cases)
            report.append("Sabotage cases by modality:")
            for modality, count in modality_sabotage.items():
                report.append(f"  {modality}: {count}")
            
            # Show top 5 most confident wrong predictions
            report.append("")
            report.append("Top 5 most confident wrong predictions:")
            sorted_cases = sorted(sabotage_cases, key=lambda x: x['confidence'], reverse=True)[:5]
            for i, case in enumerate(sorted_cases, 1):
                report.append(f"  {i}. {case['video_id']} ({case['modality']}): "
                            f"{case['predicted_emotion']} (conf: {case['confidence']}%) "
                            f"vs {case['ground_truth']} (GT)")
        
        return "\n".join(report)
    
    def create_visualizations(self, output_dir: str = "analysis_plots"):
        """Create visualization plots for the analysis."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Create main analysis plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Ensemble Emotion Recognition Analysis', fontsize=16, fontweight='bold')
        
        # 1. Modality reporting rates
        reporting_stats = self.analyze_modality_reporting()
        modalities = list(reporting_stats.keys())
        confidence_rates = [reporting_stats[mod]['confidence_rate'] for mod in modalities]
        quality_rates = [reporting_stats[mod]['data_quality_rate'] for mod in modalities]
        
        x = np.arange(len(modalities))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, confidence_rates, width, label='Confidence', alpha=0.8)
        axes[0, 0].bar(x + width/2, quality_rates, width, label='Data Quality', alpha=0.8)
        axes[0, 0].set_xlabel('Modality')
        axes[0, 0].set_ylabel('Reporting Rate')
        axes[0, 0].set_title('Modality Reporting Rates')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(modalities)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Modality accuracy
        accuracy = self.get_modality_accuracy()
        acc_modalities = list(accuracy.keys())
        acc_values = list(accuracy.values())
        
        bars = axes[0, 1].bar(acc_modalities, acc_values, alpha=0.8, color='skyblue')
        axes[0, 1].set_xlabel('Modality')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Modality Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, acc_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.1%}', ha='center', va='bottom')
        
        # 3. Data Quality vs Accuracy correlation
        quality_accuracy_stats = self.analyze_data_quality_vs_accuracy()
        correlations = []
        p_values = []
        for modality in self.modalities:
            if modality in quality_accuracy_stats and quality_accuracy_stats[modality]['total_cases'] > 0:
                correlations.append(quality_accuracy_stats[modality]['correlation'])
                p_values.append(quality_accuracy_stats[modality]['p_value'])
            else:
                correlations.append(0)
                p_values.append(1)
        
        bars = axes[0, 2].bar(self.modalities, correlations, alpha=0.8, color='lightgreen')
        axes[0, 2].set_xlabel('Modality')
        axes[0, 2].set_ylabel('Correlation Coefficient')
        axes[0, 2].set_title('Data Quality vs Accuracy Correlation')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for bar, corr, p_val in zip(bars, correlations, p_values):
            significance = '*' if p_val < 0.05 else ''
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{corr:.3f}{significance}', ha='center', va='bottom')
        
        # 4. Confidence vs Accuracy scatter
        confidence_accuracy = self.analyze_confidence_vs_accuracy()
        for modality in self.modalities:
            if confidence_accuracy[modality]:
                confidences, accuracies = zip(*confidence_accuracy[modality])
                colors = ['green' if acc else 'red' for acc in accuracies]
                axes[1, 0].scatter(confidences, [1 if acc else 0 for acc in accuracies], 
                                 alpha=0.6, label=modality, s=30)
        
        axes[1, 0].set_xlabel('Confidence (%)')
        axes[1, 0].set_ylabel('Correct (1) / Incorrect (0)')
        axes[1, 0].set_title('Confidence vs Accuracy by Modality')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Data Quality vs Accuracy scatter
        quality_accuracy = self.analyze_data_quality_impact()
        for modality in self.modalities:
            if quality_accuracy[modality]:
                qualities, accuracies = zip(*quality_accuracy[modality])
                colors = ['green' if acc else 'red' for acc in accuracies]
                axes[1, 1].scatter(qualities, [1 if acc else 0 for acc in accuracies], 
                                 alpha=0.6, label=modality, s=30)
        
        axes[1, 1].set_xlabel('Data Quality Score')
        axes[1, 1].set_ylabel('Correct (1) / Incorrect (0)')
        axes[1, 1].set_title('Data Quality vs Accuracy by Modality')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Data Quality Distribution by Accuracy
        quality_accuracy_stats = self.analyze_data_quality_vs_accuracy()
        correct_qualities = []
        incorrect_qualities = []
        
        for modality, stats in quality_accuracy_stats.items():
            if stats['total_cases'] > 0:
                for case in stats['cases']:
                    if case['is_correct']:
                        correct_qualities.append(case['quality_score'])
                    else:
                        incorrect_qualities.append(case['quality_score'])
        
        if correct_qualities and incorrect_qualities:
            axes[1, 2].hist(correct_qualities, bins=20, alpha=0.7, label='Correct', color='green')
            axes[1, 2].hist(incorrect_qualities, bins=20, alpha=0.7, label='Incorrect', color='red')
            axes[1, 2].set_xlabel('Data Quality Score')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].set_title('Data Quality Distribution by Accuracy')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{self.filename_prefix}_ensemble_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create detailed data quality vs accuracy plots
        self._create_detailed_quality_accuracy_plots(output_dir)
        
        # Create sabotage analysis plot
        sabotage_cases = self.detect_sabotage_cases()
        if sabotage_cases:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Sabotage Analysis (High Confidence, Wrong Predictions)', fontsize=14)
            
            # Sabotage by modality
            modality_sabotage = Counter(case['modality'] for case in sabotage_cases)
            ax1.bar(modality_sabotage.keys(), modality_sabotage.values(), alpha=0.8, color='red')
            ax1.set_xlabel('Modality')
            ax1.set_ylabel('Number of Sabotage Cases')
            ax1.set_title('Sabotage Cases by Modality')
            ax1.grid(True, alpha=0.3)
            
            # Confidence distribution of sabotage cases
            confidences = [case['confidence'] for case in sabotage_cases]
            ax2.hist(confidences, bins=20, alpha=0.8, color='red', edgecolor='black')
            ax2.set_xlabel('Confidence (%)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Confidence Distribution of Sabotage Cases')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{self.filename_prefix}_sabotage_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def _create_detailed_quality_accuracy_plots(self, output_dir: str):
        """Create detailed plots showing data quality vs accuracy relationships."""
        quality_accuracy_stats = self.analyze_data_quality_vs_accuracy()
        
        # Create subplots for each modality
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Detailed Data Quality vs Accuracy Analysis by Modality', fontsize=16)
        
        for idx, modality in enumerate(self.modalities):
            ax = axes[idx // 2, idx % 2]
            
            if modality in quality_accuracy_stats and quality_accuracy_stats[modality]['total_cases'] > 0:
                stats = quality_accuracy_stats[modality]
                
                # Scatter plot with different colors for correct/incorrect
                correct_qualities = [case['quality_score'] for case in stats['cases'] if case['is_correct']]
                correct_accuracies = [1] * len(correct_qualities)
                incorrect_qualities = [case['quality_score'] for case in stats['cases'] if not case['is_correct']]
                incorrect_accuracies = [0] * len(incorrect_qualities)
                
                if correct_qualities:
                    ax.scatter(correct_qualities, correct_accuracies, alpha=0.7, color='green', 
                             label='Correct', s=50)
                if incorrect_qualities:
                    ax.scatter(incorrect_qualities, incorrect_accuracies, alpha=0.7, color='red', 
                             label='Incorrect', s=50)
                
                # Add trend line
                if len(stats['cases']) > 1:
                    qualities = [case['quality_score'] for case in stats['cases']]
                    accuracies = [case['is_correct'] for case in stats['cases']]
                    z = np.polyfit(qualities, accuracies, 1)
                    p = np.poly1d(z)
                    ax.plot(qualities, p(qualities), "r--", alpha=0.8, linewidth=2)
                
                # Add quality bin accuracy bars
                bin_accuracies = stats['bin_accuracies']
                if bin_accuracies:
                    bin_names = list(bin_accuracies.keys())
                    bin_values = list(bin_accuracies.values())
                    bin_positions = np.arange(len(bin_names))
                    
                    # Add small bars at the top
                    for i, (name, value) in enumerate(zip(bin_names, bin_values)):
                        if value > 0:
                            ax.bar(i * 20 + 10, value, width=15, alpha=0.3, color='blue')
                
                ax.set_xlabel('Data Quality Score')
                ax.set_ylabel('Accuracy (1=Correct, 0=Incorrect)')
                ax.set_title(f'{modality} (r={stats["correlation"]:.3f}, p={stats["p_value"]:.3f})')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-0.1, 1.1)
            else:
                ax.text(0.5, 0.5, f'No data for {modality}', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(modality)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{self.filename_prefix}_detailed_quality_accuracy.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Analyze ensemble emotion recognition results')
    parser.add_argument('file_path', help='Path to the ensemble results JSON file')
    parser.add_argument('--output-dir', default='analysis_plots', help='Output directory for plots')
    parser.add_argument('--confidence-threshold', type=float, default=80, 
                       help='Confidence threshold for sabotage detection')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = EnsembleAnalyzer(args.file_path)
    
    # Generate and print report
    report = analyzer.generate_summary_report()
    print(report)
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save report to file
    with open(f'{args.output_dir}/{analyzer.filename_prefix}_analysis_report.txt', 'w') as f:
        f.write(report)
    
    # Generate visualizations
    if not args.no_plots:
        analyzer.create_visualizations(args.output_dir)
    
    # Additional detailed analysis
    print("\n" + "=" * 80)
    print("DETAILED SABOTAGE ANALYSIS")
    print("=" * 80)
    
    sabotage_cases = analyzer.detect_sabotage_cases(args.confidence_threshold)
    
    if sabotage_cases:
        # Create detailed DataFrame for further analysis
        df_sabotage = pd.DataFrame(sabotage_cases)
        print(f"\nDetailed sabotage cases (confidence >= {args.confidence_threshold}%):")
        print(df_sabotage[['video_id', 'modality', 'ground_truth', 'predicted_emotion', 
                          'confidence', 'data_quality_score']].to_string(index=False))
        
        # Save detailed results
        df_sabotage.to_csv(f'{args.output_dir}/{analyzer.filename_prefix}_sabotage_cases.csv', index=False)
        
        # Analyze data quality issues in sabotage cases
        print(f"\nData quality issues in sabotage cases:")
        all_issues = []
        for case in sabotage_cases:
            all_issues.extend(case['data_quality_issues'])
        
        issue_counts = Counter(all_issues)
        for issue, count in issue_counts.most_common():
            print(f"  {issue}: {count} occurrences")
    else:
        print(f"No sabotage cases found with confidence threshold >= {args.confidence_threshold}%")
    
    # Detailed data quality vs accuracy analysis
    print("\n" + "=" * 80)
    print("DETAILED DATA QUALITY VS ACCURACY ANALYSIS")
    print("=" * 80)
    
    quality_accuracy_stats = analyzer.analyze_data_quality_vs_accuracy()
    for modality, stats in quality_accuracy_stats.items():
        if stats['total_cases'] > 0:
            print(f"\n{modality} Modality:")
            print(f"  Correlation: {stats['correlation']:.3f} (p={stats['p_value']:.3f})")
            print(f"  Mean quality score: {stats['mean_quality']:.1f} ± {stats['std_quality']:.1f}")
            print(f"  Overall accuracy: {stats['overall_accuracy']:.1%}")
            
            # Show cases with low quality but correct predictions
            low_quality_correct = [case for case in stats['cases'] 
                                 if case['quality_score'] < 50 and case['is_correct']]
            if low_quality_correct:
                print(f"  Low quality but correct predictions: {len(low_quality_correct)}")
                for case in low_quality_correct[:3]:  # Show first 3
                    print(f"    {case['video_id']}: quality={case['quality_score']}, "
                          f"predicted={case['predicted']}, GT={case['ground_truth']}")
            
            # Show cases with high quality but incorrect predictions
            high_quality_incorrect = [case for case in stats['cases'] 
                                    if case['quality_score'] >= 80 and not case['is_correct']]
            if high_quality_incorrect:
                print(f"  High quality but incorrect predictions: {len(high_quality_incorrect)}")
                for case in high_quality_incorrect[:3]:  # Show first 3
                    print(f"    {case['video_id']}: quality={case['quality_score']}, "
                          f"predicted={case['predicted']}, GT={case['ground_truth']}")

if __name__ == "__main__":
    main()
