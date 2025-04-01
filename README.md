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

## to run baseline_main.py

`python baseline_main.py --model gpt-4o-mini --output baseline_T_gpt40.txt --comb T`
`python baseline_main.py --model gpt-4o-mini --output baseline_TA_gpt40.txt --comb TA`
`python baseline_main.py --model gpt-4o-mini --output baseline_AV_gpt40.txt --comb AV`
`python baseline_main.py --model gpt-4o-mini --output baseline_TV_gpt40.txt --comb TV`
`python baseline_main.py --model gpt-4o-mini --output baseline_TAV_gpt40.txt --comb TAV`

## to run baseline_cot_and_tot.py

`python baseline_cot_and_tot.py --model gpt-4o-mini --output baseline_TAV_COT_gpt40.txt --comb TAV --prompt COT`
`python baseline_cot_and_tot.py --model gpt-4o-mini --output baseline_TAV_TOT_gpt40.txt --comb TAV --prompt TOT`
