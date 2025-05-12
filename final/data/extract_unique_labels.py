import json

def extract_unique_labels(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    all_labels = [entry["true_label"] for entry in data if "true_label" in entry]
    unique_labels = list(set(all_labels))
    return unique_labels

# Example usage:
# unique_emotions = extract_unique_labels("MELD_annotation.json")
unique_emotions = extract_unique_labels("MELD_annotation.json")
print("Unique emotion labels:", unique_emotions)
