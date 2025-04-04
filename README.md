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

`python baseline_main.py --model gpt-4o-mini --output baseline_T_gpt4o.txt --comb T`
`python baseline_main.py --model gpt-4o-mini --output baseline_TA_gpt4o.txt --comb TA`
`python baseline_main.py --model gpt-4o-mini --output baseline_AV_gpt4o.txt --comb AV`
`python baseline_main.py --model gpt-4o-mini --output baseline_TV_gpt4o.txt --comb TV`
`python baseline_main.py --model gpt-4o-mini --output baseline_TAV_gpt4o.txt --comb TAV`
`python baseline_main.py --model gpt-4o-mini --output baseline_RTAV_gpt4o.txt --comb RTAV`

## to run baseline_cot_and_tot.py

`python baseline_cot_and_tot.py --model gpt-4o-mini --output mini_baseline_TAV_COT_gpt4o.txt --comb TAV --prompt COT`
`python baseline_cot_and_tot.py --model gpt-4o-mini --output mini_baseline_TAV_TOT_gpt4o.txt --comb TAV --prompt TOT`
`python baseline_cot_and_tot.py --model gpt-4o-mini --output mini_baseline_TAV_COT_gpt4o.txt --comb RTAV --prompt COT`
`python baseline_cot_and_tot.py --model gpt-4o-mini --output mini_baseline_TAV_TOT_gpt4o.txt --comb RTAV --prompt TOT`

TOT combinations:
`python baseline_cot_and_tot.py --model gpt-4o-mini --output mini_baseline_TAV_TOT_gpt4o_3-expert-UNI.txt --comb TAV --prompt TOT-3-EXPERT-UNI`
`python baseline_cot_and_tot.py --model gpt-4o-mini --output mini_baseline_RTAV_TOT_gpt4o_4-expert-UNI.txt --comb RTAV --prompt TOT-4-EXPERT-UNI`
`python baseline_cot_and_tot.py --model gpt-4o-mini --output mini_baseline_TAV_TOT_gpt4o_3-expert-debate-UNI.txt --comb TAV --prompt TOT-3-EXPERT-DEBATE-UNI`
`python baseline_cot_and_tot.py --model gpt-4o-mini --output mini_baseline_RTAV_TOT_gpt4o_4-expert-debate-UNI.txt --comb RTAV --prompt TOT-4-EXPERT-DEBATE-UNI`

`python baseline_cot_and_tot.py --model gpt-4o-mini --output mini_baseline_TAV_TOT_gpt4o_3-expert-BI.txt --comb TAV --prompt TOT-3-EXPERT-BI`
`python baseline_cot_and_tot.py --model gpt-4o-mini --output mini_baseline_RTAV_TOT_gpt4o_4-expert-BI.txt --comb RTAV --prompt TOT-4-EXPERT-BI`
`python baseline_cot_and_tot.py --model gpt-4o-mini --output mini_baseline_TAV_TOT_gpt4o_3-expert-debate-BI.txt --comb TAV --prompt TOT-3-EXPERT-DEBATE-BI`
`python baseline_cot_and_tot.py --model gpt-4o-mini --output mini_baseline_RTAV_TOT_gpt4o_4-expert-debate-BI.txt --comb RTAV --prompt TOT-4-EXPERT-DEBATE-BI`
