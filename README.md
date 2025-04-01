# chenyu-llm-experiment

## set up conda environment

- `conda env create -f environment.yaml`
- `conda activate how2ai-llm-experiment`

## set up .env

OPENAI_API_KEY=<your_key>

## how to run

## to run main.py

`python main.py --model gpt-4o-mini`

or

`python main.py --model o1`

## to run cot_and_tot.py

`python cot_and_tot.py.py --mode tree_of_thought`\

## to run baseline_main.py

`python baseline_main.py --model gpt-4o-mini --output output_file_name`
