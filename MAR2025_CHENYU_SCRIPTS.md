## neurips

## to run neurips_final_baseline_main.py (❌ MER)

❌ `python neurips_baseline_main.py --input final/data/MER_annotation.json --model gpt-4o --output final/result/MER/chenyu-run/baseline_T_gpt4o.txt --comb T --dataset MER`

❌ `python neurips_baseline_main.py --input final/data/MER_annotation.json --model gpt-4o --output final/result/MER/chenyu-run/baseline_TA_gpt4o.txt --comb TA --dataset MER`

❌ `python neurips_baseline_main.py --input final/data/MER_annotation.json --model gpt-4o --output final/result/MER/chenyu-run/baseline_AV_gpt4o.txt --comb AV --dataset MER`

❌ `python neurips_baseline_main.py --input final/data/MER_annotation.json --model gpt-4o --output final/result/MER/chenyu-run/baseline_TV_gpt4o.txt --comb TV --dataset MER`

❌ `python neurips_baseline_main.py --input final/data/MER_annotation.json --model gpt-4o --output final/result/MER/chenyu-run/baseline_TAV_gpt4o.txt --comb TAV --dataset MER`

⭐️ swap to gpt-5-nano
`python neurips_baseline_main.py --input final/data/MER_annotation.json --model gpt-5-nano --output final/result/MER/mar-workshop/baseline_TAV_gpt5-nano.txt --comb TAV --dataset MER --workers=10`

⭐️⭐️ new experiment
single-emotion: `python prob_experiment_baseline_main.py --input final/data/MER_annotation.json --model gpt-5-nano --output final/result/MER/mar-workshop/test_ensemble_gpt5-nano.txt --dataset MER --workers 10`

⭐️⭐️ new experiment
all-emotions: `python prob_experiment_baseline_main.py --input final/data/MER_annotation.json --model gpt-5-nano --output final/result/MER/mar-workshop/all-emos_test_ensemble_gpt5-nano.txt --dataset MER --workers 10 --use-all-emotions`

## to run neurips_baseline_main.py (❌ MELD)

❌ `python neurips_baseline_main.py --input final/data/MELD_annotation.json --model gpt-4o --output final/result/MELD/chenyu-run/baseline_T_gpt4o.txt --comb T --dataset MELD`

❌ `python neurips_baseline_main.py --input final/data/MELD_annotation.json --model gpt-4o --output final/result/MELD/chenyu-run/baseline_TA_gpt4o.txt --comb TA --dataset MELD`

❌ `python neurips_baseline_main.py --input final/data/MELD_annotation.json --model gpt-4o --output final/result/MELD/chenyu-run/baseline_AV_gpt4o.txt --comb AV --dataset MELD`

❌ `python neurips_baseline_main.py --input final/data/MELD_annotation.json --model gpt-4o --output final/result/MELD/chenyu-run/baseline_TV_gpt4o.txt --comb TV --dataset MELD`

❌ `python neurips_baseline_main.py --input final/data/MELD_annotation.json --model gpt-4o --output final/result/MELD/chenyu-run/baseline_TAV_gpt4o.txt --comb TAV --dataset MELD`

⭐️ `python neurips_baseline_main.py --input final/data/MELD_annotation.json --model gpt-5-nano --output final/result/MELD/mar-workshop/baseline_TAV_gpt5-nano.txt --comb TAV --dataset MELD --workers=10`

## to run neurips_baseline_main.py (❌ IEMOCAP)

❌ `python neurips_baseline_main.py --input final/data/IEMOCAP_annotation.json --model gpt-4o --output final/result/IEMOCAP/chenyu-run/baseline_T_gpt4o.txt --comb T --dataset IEMOCAP`

❌ `python neurips_baseline_main.py --input final/data/IEMOCAP_annotation.json --model gpt-4o --output final/result/IEMOCAP/chenyu-run/baseline_TA_gpt4o.txt --comb TA --dataset IEMOCAP`

❌ `python neurips_baseline_main.py --input final/data/IEMOCAP_annotation.json --model gpt-4o --output final/result/IEMOCAP/chenyu-run/baseline_AV_gpt4o.txt --comb AV --dataset IEMOCAP`

❌ `python neurips_baseline_main.py --input final/data/IEMOCAP_annotation.json --model gpt-4o --output final/result/IEMOCAP/chenyu-run/baseline_TV_gpt4o.txt --comb TV --dataset IEMOCAP`

❌ `python neurips_baseline_main.py --input final/data/IEMOCAP_annotation.json --model gpt-4o --output final/result/IEMOCAP/chenyu-run/baseline_TAV_gpt4o.txt --comb TAV --dataset IEMOCAP`

⭐️ `python neurips_baseline_main.py --input final/data/IEMOCAP_annotation.json --model gpt-5-nano --output final/result/IEMOCAP/mar-workshop/baseline_TAV_gpt5-nano.txt --comb TAV --dataset IEMOCAP --workers=10`

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

## to run neurips_baseline_cot_and_tot.py (❌ MER)

❌ `python neurips_baseline_cot_and_tot.py --input final/data/MER_annotation.json --model gpt-4o --output final/result/MER/chenyu-run/mini_baseline_TAV_COT_gpt4o.txt --comb TAV --prompt COT --dataset MER`

❌ `python neurips_baseline_cot_and_tot.py --input final/data/MER_annotation.json --model gpt-4o --output final/result/MER/chenyu-run/mini_baseline_TAV_TOT_gpt4o.txt --comb TAV --prompt TOT --dataset MER`

## to run neurips_baseline_cot_and_tot.py (❌ MELD)

❌ `python neurips_baseline_cot_and_tot.py --input final/data/MELD_annotation.json --model gpt-4o --output final/result/MELD/chenyu-run/mini_baseline_TAV_COT_gpt4o.txt --comb TAV --prompt COT --dataset MELD`

❌ `python neurips_baseline_cot_and_tot.py --input final/data/MELD_annotation.json --model gpt-4o --output final/result/MELD/chenyu-run/mini_baseline_TAV_TOT_gpt4o.txt --comb TAV --prompt TOT --dataset MELD`

## to run neurips_baseline_cot_and_tot.py (❌ IEMOCAP)

❌ `python neurips_baseline_cot_and_tot.py --input final/data/IEMOCAP_annotation.json --model gpt-4o --output final/result/IEMOCAP/chenyu-run/mini_baseline_TAV_COT_gpt4o.txt --comb TAV --prompt COT --dataset IEMOCAP`

❌ `python neurips_baseline_cot_and_tot.py --input final/data/IEMOCAP_annotation.json --model gpt-4o --output final/result/IEMOCAP/chenyu-run/mini_baseline_TAV_TOT_gpt4o.txt --comb TAV --prompt TOT --dataset IEMOCAP`

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
TOT combinations: (❌ MELD)
❌ `python neurips_baseline_cot_and_tot.py --input final/data/MELD_annotation.json --model gpt-4o --output final/result/MELD/chenyu-run/mini_baseline_TAV_TOT_gpt4o_3-expert-UNI.txt --comb TAV --prompt TOT-3-EXPERT-UNI --dataset MELD`

❌ `python neurips_baseline_cot_and_tot.py --input final/data/MELD_annotation.json --model gpt-4o --output final/result/MELD/chenyu-run/mini_baseline_TAV_TOT_gpt4o_3-expert-debate-UNI.txt --comb TAV --prompt TOT-3-EXPERT-DEBATE-UNI --dataset MELD`

❌ `python neurips_baseline_cot_and_tot.py --input final/data/MELD_annotation.json --model gpt-4o --output final/result/MELD/chenyu-run/mini_baseline_TAV_TOT_gpt4o_3-expert-BI.txt --comb TAV --prompt TOT-3-EXPERT-BI --dataset MELD`

❌ `python neurips_baseline_cot_and_tot.py --input final/data/MELD_annotation.json --model gpt-4o --output final/result/MELD/chenyu-run/mini_baseline_TAV_TOT_gpt4o_3-expert-debate-BI.txt --comb TAV --prompt TOT-3-EXPERT-DEBATE-BI --dataset MELD`

TOT combinations: (❌ MER)
❌ `python neurips_baseline_cot_and_tot.py --input final/data/MER_annotation.json --model gpt-4o --output final/result/MER/chenyu-run/mini_baseline_TAV_TOT_gpt4o_3-expert-UNI.txt --comb TAV --prompt TOT-3-EXPERT-UNI --dataset MER`

❌ `python neurips_baseline_cot_and_tot.py --input final/data/MER_annotation.json --model gpt-4o --output final/result/MER/chenyu-run/mini_baseline_TAV_TOT_gpt4o_3-expert-debate-UNI.txt --comb TAV --prompt TOT-3-EXPERT-DEBATE-UNI --dataset MER`

❌ `python neurips_baseline_cot_and_tot.py --input final/data/MER_annotation.json --model gpt-4o --output final/result/MER/chenyu-run/mini_baseline_TAV_TOT_gpt4o_3-expert-BI.txt --comb TAV --prompt TOT-3-EXPERT-BI --dataset MER`

❌ `python neurips_baseline_cot_and_tot.py --input final/data/MER_annotation.json --model gpt-4o --output final/result/MER/chenyu-run/mini_baseline_TAV_TOT_gpt4o_3-expert-debate-BI.txt --comb TAV --prompt TOT-3-EXPERT-DEBATE-BI --dataset MER`

TOT combinations: (❌ IEMOCAP)
❌ `python neurips_baseline_cot_and_tot.py --input final/data/IEMOCAP_annotation.json --model gpt-4o --output final/result/IEMOCAP/chenyu-run/mini_baseline_TAV_TOT_gpt4o_3-expert-UNI.txt --comb TAV --prompt TOT-3-EXPERT-UNI --dataset IEMOCAP`

❌ `python neurips_baseline_cot_and_tot.py --input final/data/IEMOCAP_annotation.json --model gpt-4o --output final/result/IEMOCAP/chenyu-run/mini_baseline_TAV_TOT_gpt4o_3-expert-debate-UNI.txt --comb TAV --prompt TOT-3-EXPERT-DEBATE-UNI --dataset IEMOCAP`

❌ `python neurips_baseline_cot_and_tot.py --input final/data/IEMOCAP_annotation.json --model gpt-4o --output final/result/IEMOCAP/chenyu-run/mini_baseline_TAV_TOT_gpt4o_3-expert-BI.txt --comb TAV --prompt TOT-3-EXPERT-BI --dataset IEMOCAP`

❌ `python neurips_baseline_cot_and_tot.py --input final/data/IEMOCAP_annotation.json --model gpt-4o --output final/result/IEMOCAP/chenyu-run/mini_baseline_TAV_TOT_gpt4o_3-expert-debate-BI.txt --comb TAV --prompt TOT-3-EXPERT-DEBATE-BI --dataset IEMOCAP`
