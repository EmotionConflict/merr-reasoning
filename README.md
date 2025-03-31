# chenyu-llm-experiment

## set up conda environment

- `conda env create -f environment.yaml`
- `conda activate how2ai-llm-experiment`

## set up .env

OPENAI_API_KEY=<your_key>

## how to run

running main.py

`python main.py --model gpt-3.5-turbo`

or

`python main.py --model o1`

running cot_and_tot.py

`python cot_and_tot.py.py --mode tree_of_thought`
