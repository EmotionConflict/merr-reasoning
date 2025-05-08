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

## FINAL

## to run final_baseline_main.py (✅ MER, ✅ MELD)

`python final_baseline_main.py --input final/data/MER_annotation.json --model gpt-4o-mini --output final/result/MER/baseline_T_gpt4o.txt --comb T --dataset MER`

`python final_baseline_main.py --input final/data/MER_annotation.json --model gpt-4o-mini --output final/result/MER/baseline_TA_gpt4o.txt --comb TA --dataset MER`

`python final_baseline_main.py --input final/data/MER_annotation.json --model gpt-4o-mini --output final/result/MER/baseline_AV_gpt4o.txt --comb AV --dataset MER`

`python final_baseline_main.py --input final/data/MER_annotation.json --model gpt-4o-mini --output final/result/MER/baseline_TV_gpt4o.txt --comb TV --dataset MER`

`python final_baseline_main.py --input final/data/MER_annotation.json --model gpt-4o-mini --output final/result/MER/baseline_TAV_gpt4o.txt --comb TAV --dataset MER`

## to run final_baseline_cot_and_tot.py (✅ MER, ✅ MELD)

`python final_baseline_cot_and_tot.py --input final/data/MELD_annotation.json --model gpt-4o-mini --output final/result/MELD/mini_baseline_TAV_COT_gpt4o.txt --comb TAV --prompt COT --dataset MELD`

`python final_baseline_cot_and_tot.py --input final/data/MELD_annotation.json --model gpt-4o-mini --output final/result/MELD/mini_baseline_TAV_TOT_gpt4o.txt --comb TAV --prompt TOT --dataset MELD`

TOT combinations: (❌ MELD)
`python final_baseline_cot_and_tot.py --input final/data/MELD_annotation.json --model gpt-4o-mini --output final/result/MELD/mini_baseline_TAV_TOT_gpt4o_3-expert-UNI.txt --comb TAV --prompt TOT-3-EXPERT-UNI --dataset MELD`

`python final_baseline_cot_and_tot.py --input final/data/MELD_annotation.json --model gpt-4o-mini --output final/result/MELD/mini_baseline_TAV_TOT_gpt4o_3-expert-debate-UNI.txt --comb TAV --prompt TOT-3-EXPERT-DEBATE-UNI --dataset MELD`

`python final_baseline_cot_and_tot.py --input final/data/MELD_annotation.json --model gpt-4o-mini --output final/result/MELD/mini_baseline_TAV_TOT_gpt4o_3-expert-BI.txt --comb TAV --prompt TOT-3-EXPERT-BI --dataset MELD`

`python final_baseline_cot_and_tot.py --input final/data/MELD_annotation.json --model gpt-4o-mini --output final/result/MELD/mini_baseline_TAV_TOT_gpt4o_3-expert-debate-BI.txt --comb TAV --prompt TOT-3-EXPERT-DEBATE-BI --dataset MELD`

TOT combinations: (❌ MER)
`python final_baseline_cot_and_tot.py --input final/data/MER_annotation.json --model gpt-4o-mini --output final/result/MER/mini_baseline_TAV_TOT_gpt4o_3-expert-UNI.txt --comb TAV --prompt TOT-3-EXPERT-UNI --dataset MER`

`python final_baseline_cot_and_tot.py --input final/data/MER_annotation.json --model gpt-4o-mini --output final/result/MER/mini_baseline_TAV_TOT_gpt4o_3-expert-debate-UNI.txt --comb TAV --prompt TOT-3-EXPERT-DEBATE-UNI --dataset MER`

`python final_baseline_cot_and_tot.py --input final/data/MER_annotation.json --model gpt-4o-mini --output final/result/MER/mini_baseline_TAV_TOT_gpt4o_3-expert-BI.txt --comb TAV --prompt TOT-3-EXPERT-BI --dataset MER`

`python final_baseline_cot_and_tot.py --input final/data/MER_annotation.json --model gpt-4o-mini --output final/result/MER/mini_baseline_TAV_TOT_gpt4o_3-expert-debate-BI.txt --comb TAV --prompt TOT-3-EXPERT-DEBATE-BI --dataset MER`
