import json
from typing import List, Dict, Any

def convert_object_to_array_with_video_id(json_path: str, output_path: str) -> None:
    """
    Reads a JSON file where the top-level structure is an object with keys like 'samplenew_00002396',
    and values are objects. Converts it to a list of objects, each with a 'video_id' key containing the original key,
    and writes the result to a new file.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    result = []
    for video_id, obj in data.items():
        new_obj = dict(obj)
        new_obj['video_id'] = video_id
        result.append(new_obj)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    input_path = 'final/data/MERR_annotation.json'
    output_path = 'final/data/MERR_annotation_array.json'
    convert_object_to_array_with_video_id(input_path, output_path)
    print(f"Converted JSON written to {output_path}") 