import glob
import re
import matplotlib.pyplot as plt
import numpy as np
import os

def extract_evaluation_metrics(file_path):
    # List to collect results
    results = []

    # Process each .txt file in the folder
    for file in glob.glob(f"{file_path}/*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Extract the evaluation metrics block
        match = re.search(r"EVALUATION METRICS\s*\n=+\s*\n(.*)", content, re.DOTALL)
        if match:
            metrics_block = match.group(1).strip()
            results.append(f"File: {file}\nEVALUATION METRICS\n=================\n{metrics_block}\n")
        else:
            results.append(f"File: {file}\nNo Evaluation Metrics block found.\n")

    # Write results to data_analysis.txt
    with open(f"{file_path}/data_analysis.txt", "w", encoding="utf-8") as out_file:
        out_file.write("\n".join(results))

    print("Evaluation metrics have been extracted to data_analysis.txt")

def plot_evaluation_metrics(file_path):
    """
    Reads evaluation metrics from a file and plots radar charts separately for mini_baseline and other models,
    ignoring any filename that contains 'RTAV'. In total, 8 plots are created:
        - For each metric (F1, Precision, Recall), one radar chart for mini_baseline models (if any) and one for others (6 plots).
        - One combined radar chart for F1 (overlaying both groups).
        - One bar chart comparing the average metrics between mini_baseline and others.
    All images are saved to a folder.
    """
    # Create output folder if it doesn't exist
    output_folder = "plots"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 1. Read the data from the file
    with open(file_path, 'r', encoding="utf-8") as f:
        data = f.read()

    # 2. Split the data into blocks based on each model's results
    blocks = data.split("File:")[1:]  # Skip any content before the first "File:"
    blocks = ["File:" + block.strip() for block in blocks]

    # Dictionary to store metrics for each model, and a set to collect all label names
    model_metrics = {}
    all_labels = set()

    for block in blocks:
        # Extract filename
        filename_match = re.search(r'File:\s*(.+)', block)
        if not filename_match:
            continue
        filename = filename_match.group(1).strip()

        # Skip any file name containing 'RTAV' (case-insensitive)
        if "RTAV" in filename.upper() or "R_GPT40" in filename.upper() or "R_o3" in filename.upper():
            continue

        # Initialize dictionary for this model
        model_metrics[filename] = {}

        # Expected format:
        # Label: <label> -> Precision: <value>, Recall: <value>, F1 Score: <value>, Support: <value>
        label_data = re.findall(
            r'Label:\s*(\w+)\s*->.*?Precision:\s*([\d\.]+),\s*Recall:\s*([\d\.]+),\s*F1 Score:\s*([\d\.]+),\s*Support:\s*(\d+)',
            block
        )

        # Build dictionary for this model
        for label, precision, recall, f1_score, support in label_data:
            model_metrics[filename][label] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1_score)
            }
            all_labels.add(label)

    # 3. Prepare the labels (axes) for the radar charts
    labels = sorted(list(all_labels))  # Sorted for consistent ordering
    num_labels = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_labels, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    # 4. Separate mini_baseline from the other models
    mini_baseline_models = {}
    other_models = {}
    for filename, scores in model_metrics.items():
        if "mini_baseline" in filename.lower():
            mini_baseline_models[filename] = scores
        else:
            other_models[filename] = scores

    # 5. Define which metrics to plot and their display titles
    metrics_to_plot = ['f1', 'precision', 'recall']
    titles = {"f1": "F1 Score", "precision": "Precision", "recall": "Recall"}

    # 6. Plot separate radar charts for each metric: one for mini_baseline and one for others.
    for metric in metrics_to_plot:
        # Radar chart for mini_baseline models (if any)
        if mini_baseline_models:
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
            for filename, label_scores in mini_baseline_models.items():
                scores = [label_scores.get(lbl, {}).get(metric, 0) for lbl in labels]
                scores += scores[:1]  # Close the polygon
                # Clean up filename for display
                display_name = os.path.basename(filename).replace(".txt", "").replace("results", "")
                display_name = display_name.replace("_", " ").replace("-", " ")
                ax.plot(angles, scores, linewidth=2, label=display_name)
                ax.fill(angles, scores, alpha=0.1)
            ax.set_thetagrids([angle * 180 / np.pi for angle in angles[:-1]], labels)
            ax.set_title(f"Mini Baseline {titles[metric]}", pad=30)
            ax.tick_params(axis='x', pad=15)
            ax.set_ylim(0, 1)
            ax.legend(loc='center left', bbox_to_anchor=(1.15, 0.5), fontsize='small')
            fig.subplots_adjust(bottom=0.3)
            # Save the plot instead of displaying it
            plt.savefig(os.path.join(output_folder, f"mini_baseline_{metric}.png"))
            plt.close()

        # Radar chart for other models (if any)
        if other_models:
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
            for filename, label_scores in other_models.items():
                scores = [label_scores.get(lbl, {}).get(metric, 0) for lbl in labels]
                scores += scores[:1]  # Close the polygon
                display_name = os.path.basename(filename).replace(".txt", "").replace("results", "")
                display_name = display_name.replace("_", " ").replace("-", " ")
                ax.plot(angles, scores, linewidth=2, label=display_name)
                ax.fill(angles, scores, alpha=0.1)
            ax.set_thetagrids([angle * 180 / np.pi for angle in angles[:-1]], labels)
            ax.set_title(f"Other Models {titles[metric]}", pad=30)
            ax.tick_params(axis='x', pad=15)
            ax.set_ylim(0, 1)
            ax.legend(loc='center left', bbox_to_anchor=(1.15, 0.5), fontsize='small')
            fig.subplots_adjust(bottom=0.3)
            # Save the plot
            plt.savefig(os.path.join(output_folder, f"others_{metric}.png"))
            plt.close()

    # 7. Extra Plot: Combined radar chart for F1 Score overlaying both groups
    if mini_baseline_models or other_models:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
        if mini_baseline_models:
            for filename, label_scores in mini_baseline_models.items():
                scores = [label_scores.get(lbl, {}).get('f1', 0) for lbl in labels]
                scores += scores[:1]
                display_name = "Mini: " + os.path.basename(filename)
                ax.plot(angles, scores, linewidth=2, label=display_name)
                ax.fill(angles, scores, alpha=0.1)
        if other_models:
            for filename, label_scores in other_models.items():
                scores = [label_scores.get(lbl, {}).get('f1', 0) for lbl in labels]
                scores += scores[:1]
                display_name = "Other: " + os.path.basename(filename)
                ax.plot(angles, scores, linewidth=2, label=display_name)
                ax.fill(angles, scores, alpha=0.1)
        ax.set_thetagrids([angle * 180 / np.pi for angle in angles[:-1]], labels)
        ax.set_title("Combined F1 Score Comparison", pad=30)
        ax.tick_params(axis='x', pad=15)
        ax.set_ylim(0, 1)
        ax.legend(loc='center left', bbox_to_anchor=(1.15, 0.5), fontsize='small')
        fig.subplots_adjust(bottom=0.3)
        # Save the combined F1 plot
        plt.savefig(os.path.join(output_folder, "combined_f1.png"))
        plt.close()

    # 8. Extra Plot: Bar chart comparing average metrics between groups
    # Compute average scores for each metric
    group_averages = {'mini_baseline': {}, 'others': {}}
    for metric in metrics_to_plot:
        # mini_baseline group
        mini_scores = []
        for label_scores in mini_baseline_models.values():
            for lbl in labels:
                mini_scores.append(label_scores.get(lbl, {}).get(metric, 0))
        group_averages['mini_baseline'][metric] = np.mean(mini_scores) if mini_scores else 0

        # others group
        other_scores = []
        for label_scores in other_models.values():
            for lbl in labels:
                other_scores.append(label_scores.get(lbl, {}).get(metric, 0))
        group_averages['others'][metric] = np.mean(other_scores) if other_scores else 0

    # Create the grouped bar chart
    metrics_names = ['F1 Score', 'Precision', 'Recall']
    x = np.arange(len(metrics_to_plot))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    mini_bar = [group_averages['mini_baseline'][m] for m in metrics_to_plot]
    others_bar = [group_averages['others'][m] for m in metrics_to_plot]

    ax.bar(x - width/2, mini_bar, width, label='Mini Baseline')
    ax.bar(x + width/2, others_bar, width, label='Others')

    ax.set_ylabel('Average Score')
    ax.set_title('Average Evaluation Metrics by Group')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    # Save the bar chart
    plt.savefig(os.path.join(output_folder, "average_metrics_comparison.png"))
    plt.close()

if __name__ == "__main__":
    # First, you can extract metrics if needed:
    # extract_evaluation_metrics("results/")

    # Then plot the charts and save the images to the "plots" folder:
    plot_evaluation_metrics("results/data_analysis.txt")
