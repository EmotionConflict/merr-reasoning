import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data from the results
datasets = ['MER', 'MELD', 'IEMOCAP']
modalities = ['T', 'A', 'V', 'TAV']

# Modality accuracy data (in percentages)
accuracy_data = {
    'MER': [18.6, 25.0, 48.3, 33.3],
    'MELD': [50.8, 29.8, 18.8, 23.6],
    'IEMOCAP': [32.6, 24.8, 17.1, 27.9]
}

# Sabotage cases data (sabotage_count / total_samples)
sabotage_data = {
    'MER': [7, 37, 13, 19],  # sabotage counts
    'MELD': [22, 92, 90, 81],  # sabotage counts
    'IEMOCAP': [21, 77, 66, 59]  # sabotage counts
}

# Total samples for each dataset-modality combination
total_samples = {
    'MER': [60, 60, 60, 60],
    'MELD': [191, 191, 191, 191],
    'IEMOCAP': [129, 129, 129, 129]
}

# Calculate sabotage percentages
sabotage_percentages = {}
for dataset in datasets:
    sabotage_percentages[dataset] = []
    for i, modality in enumerate(modalities):
        sabotage_pct = (sabotage_data[dataset][i] / total_samples[dataset][i]) * 100
        sabotage_percentages[dataset].append(sabotage_pct)

# Create separate visualizations for each chart

# Set up the bar positions
x = np.arange(len(modalities))
width = 0.25

# Color scheme
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

# Chart 1: Modality Accuracy Comparison
fig1, ax1 = plt.subplots(figsize=(10, 6))
fig1.suptitle('GPT-5-nano Ensemble Results: Modality Accuracy by Dataset', fontsize=16, fontweight='bold')

for i, dataset in enumerate(datasets):
    bars = ax1.bar(x + i*width, accuracy_data[dataset], width, label=dataset, 
                   color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, accuracy_data[dataset]):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=12)

ax1.set_xlabel('Modality', fontsize=14)
ax1.set_ylabel('Accuracy (%)', fontsize=14)
ax1.set_xticks(x + width)
ax1.set_xticklabels(modalities, fontsize=12)
ax1.tick_params(axis='y', labelsize=12)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 60)

plt.tight_layout()
plt.subplots_adjust(top=0.85)

# Save the first chart
plt.savefig('modality_accuracy_chart.png', dpi=300, bbox_inches='tight')
plt.show()

# Chart 2: Sabotage Cases Comparison
fig2, ax2 = plt.subplots(figsize=(10, 6))
fig2.suptitle('GPT-5-nano Ensemble Results: Sabotage Cases by Dataset', fontsize=16, fontweight='bold')

for i, dataset in enumerate(datasets):
    bars = ax2.bar(x + i*width, sabotage_percentages[dataset], width, label=dataset,
                   color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for j, (bar, sabotage_pct, sabotage_count, total) in enumerate(zip(bars, sabotage_percentages[dataset], 
                                                                      sabotage_data[dataset], total_samples[dataset])):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{sabotage_count}/{total}\n({sabotage_pct:.1f}%)', ha='center', va='bottom', 
                fontsize=11)

ax2.set_xlabel('Modality', fontsize=14)
ax2.set_ylabel('Sabotage Cases (%)', fontsize=14)
ax2.set_xticks(x + width)
ax2.set_xticklabels(modalities, fontsize=12)
ax2.tick_params(axis='y', labelsize=12)
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 100)

plt.tight_layout()
plt.subplots_adjust(top=0.85)

# Save the second chart
plt.savefig('sabotage_cases_chart.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("=== SABOTAGE ANALYSIS SUMMARY ===")
print(f"{'Dataset':<10} {'Modality':<6} {'Accuracy':<10} {'Sabotage':<15} {'Total':<8}")
print("-" * 60)
for dataset in datasets:
    for i, modality in enumerate(modalities):
        print(f"{dataset:<10} {modality:<6} {accuracy_data[dataset][i]:<10.1f}% "
              f"{sabotage_data[dataset][i]}/{total_samples[dataset][i]:<15} "
              f"{sabotage_percentages[dataset][i]:<8.1f}%")

print("\n=== KEY INSIGHTS ===")
print("1. MER Dataset: V modality has highest accuracy (48.3%) but moderate sabotage (21.7%)")
print("2. MELD Dataset: T modality has highest accuracy (50.8%) and lowest sabotage (11.5%)")
print("3. IEMOCAP Dataset: T modality has highest accuracy (32.6%) and lowest sabotage (16.3%)")
print("4. Overall: Text modality (T) tends to have better performance across datasets")
