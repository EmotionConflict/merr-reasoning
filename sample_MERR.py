import json
import random

# Define your emotion labels
labels = ["happy", "sad", "neutral", "angry", "worried", "surprise", "fear", "contempt", "doubt"]

# Load the original JSON file
with open("MERR_fine_grained.json", "r") as f:
    data = json.load(f)

# Group entries by emotion
emotion_groups = {label: [] for label in labels}
for key, value in data.items():
    emotion = value.get("pseu_emotion")
    if emotion in emotion_groups:
        emotion_groups[emotion].append((key, value))

# Sample 10 entries for each emotion
sampled_data = {}
for emotion in labels:
    entries = emotion_groups[emotion]
    if len(entries) >= 10:
        sampled = random.sample(entries, 10)
    else:
        sampled = entries  # include all if fewer than 10
    for key, value in sampled:
        sampled_data[key] = value

# Save the sampled data to a new file
with open("10_per_emotion_MERR_fine_grained.json", "w") as f:
    json.dump(sampled_data, f, indent=4)

print("Saved sampled data to 10_per_emotion_MERR_fine_grained.json")
