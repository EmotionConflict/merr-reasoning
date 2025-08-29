import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
import openai
import time

# Load environment variables from .env
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError('OPENAI_API_KEY not found in environment or .env file.')
openai.api_key = OPENAI_API_KEY

IEMOCAP_PATH = 'final/data/IEMOCAP_annotation.json'
OUTPUT_PATH = 'final/data/IEMOCAP_reasoning_paths.json'
EMOTIONS = [
    'frustrated', 'excited', 'happy', 'fearful', 'neutral', 'sad', 'angry', 'surprised'
]

# Helper: call GPT-4o for reasoning

def get_reasoning(combined_data, emotion, max_retries=5):
    prompt = (
        f"Given the following information, explain why you think this is a {emotion} emotion. "
        f"Be as specific as possible.\n\nInformation:\n{combined_data}"
    )
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=256
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"OpenAI API error (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(2 * (attempt + 1))
    return f"[ERROR: Could not get reasoning for {emotion}]"

# Main processing

def main():
    with open(IEMOCAP_PATH, 'r') as f:
        data = json.load(f)

    results = []
    for entry in tqdm(data, desc="Processing entries"):
        visual = entry.get('visual_objective_description', '')
        audio = entry.get('audio_description', '')
        transcript = entry.get('transcript', '')
        combined = f"Visual: {visual}\nAudio: {audio}\nTranscript: {transcript}"
        result = {
            'video_id': entry.get('video_id', ''),
            'visual_objective_description': visual,
            'audio_description': audio,
            'transcript': transcript,
            'combined_data': combined
        }
        for emotion in EMOTIONS:
            reason = get_reasoning(combined, emotion)
            result[emotion] = reason
        results.append(result)
        # Optional: Save progress every 10 entries
        if len(results) % 10 == 0:
            with open(OUTPUT_PATH, 'w') as f:
                json.dump(results, f, indent=2)

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved all results to {OUTPUT_PATH}")

if __name__ == '__main__':
    main() 