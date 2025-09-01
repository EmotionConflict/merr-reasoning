import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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

# Create DataFrames for heatmaps
accuracy_df = pd.DataFrame(accuracy_data, index=modalities).T
sabotage_df = pd.DataFrame(sabotage_percentages, index=modalities).T

# Create the figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Set the main title
# fig.suptitle('GPT-5-nano Ensemble Results: Dataset-Modality Matrix', fontsize=22, y=0.95)

# Plot 1: Accuracy Heatmap
sns.heatmap(accuracy_df, annot=True, fmt='.1f', cmap='RdYlGn', 
            cbar_kws={'label': 'Accuracy (%)'}, ax=ax1, 
            vmin=0, vmax=60, linewidths=0.5, linecolor='white',
            annot_kws={'size': 16, 'color': 'black'})
ax1.set_title('Accuracy (%)', fontsize=18, pad=20)
ax1.set_xlabel('Modality', fontsize=16)
ax1.set_ylabel('Dataset', fontsize=16)
ax1.tick_params(axis='both', which='major', labelsize=14, labelcolor='black')
ax1.tick_params(axis='y', which='major', labelsize=18, labelcolor='black')

# Set colorbar label size after creation
cbar1 = ax1.collections[0].colorbar
cbar1.ax.tick_params(labelsize=14)
cbar1.ax.set_ylabel('Accuracy (%)', fontsize=14)

# Plot 2: Sabotage Cases Heatmap
sabotage_annotations = []
for i, dataset in enumerate(datasets):
    row = []
    for j, modality in enumerate(modalities):
        sabotage_count = sabotage_data[dataset][j]
        total = total_samples[dataset][j]
        sabotage_pct = sabotage_percentages[dataset][j]
        annotation = f'{sabotage_count}/{total}\n({sabotage_pct:.1f}%)'
        row.append(annotation)
    sabotage_annotations.append(row)

sns.heatmap(sabotage_df, annot=sabotage_annotations, fmt='', cmap='Reds', 
            cbar_kws={'label': 'Sabotage Cases (%)'}, ax=ax2,
            vmin=0, vmax=100, linewidths=0.5, linecolor='white',
            annot_kws={'size': 14, 'color': 'black'})
ax2.set_title('Sabotage Cases (%)', fontsize=18, pad=20)
ax2.set_xlabel('Modality', fontsize=16)
ax2.set_ylabel('Dataset', fontsize=16)
ax2.tick_params(axis='both', which='major', labelsize=14, labelcolor='black')
ax2.tick_params(axis='y', which='major', labelsize=18, labelcolor='black')

# Set colorbar label size after creation
cbar2 = ax2.collections[0].colorbar
cbar2.ax.tick_params(labelsize=14)
cbar2.ax.set_ylabel('Sabotage Cases (%)', fontsize=14)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.85)

# Save the combined heatmap
plt.savefig('dataset_modality_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Print detailed analysis
print("=== DATASET-MODALITY MATRIX ANALYSIS ===")
print("\nAccuracy Matrix:")
print(accuracy_df.round(1))
print("\nSabotage Cases Matrix:")
print(sabotage_df.round(1))

print("\n=== KEY OBSERVATIONS ===")
print("1. Audio (A) modality shows the highest sabotage rates across all datasets")
print("2. Text (T) modality generally has the lowest sabotage rates")
print("3. Vision (V) sabotage behavior varies by dataset")
print("4. MER: V has highest accuracy (48.3%) but moderate sabotage (21.7%)")
print("5. MELD: T has highest accuracy (50.8%) and lowest sabotage (11.5%)")
print("6. IEMOCAP: T has highest accuracy (32.6%) and lowest sabotage (16.3%)")

# Additional analysis
print("\n=== SABOTAGE RANKING BY MODALITY ===")
avg_sabotage = sabotage_df.mean()
print("Average sabotage rates by modality:")
for modality, rate in avg_sabotage.items():
    print(f"{modality}: {rate:.1f}%")

print("\n=== ACCURACY RANKING BY MODALITY ===")
avg_accuracy = accuracy_df.mean()
print("Average accuracy by modality:")
for modality, acc in avg_accuracy.items():
    print(f"{modality}: {acc:.1f}%")
