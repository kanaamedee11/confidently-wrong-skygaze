#!/bin/sh
cd "$(dirname "$0")"
export SUPERGLUE_PATH=$(readlink -f ../SuperGluePretrainedNetwork)
nohup sh -c "python3 src/run_exp.py --exp axes --methods SuperGlue --nref 2 --nseed 6 --axes blur,occlusion,appearance --out results/axes_sg.jsonl && python3 src/run_exp.py --exp rot --methods SuperGlue --nref 3 --nseed 10 --rot_levels 50,55 --out results/rot_sg_fine.jsonl && python3 src/run_exp.py --exp rot --methods SuperGlue --ref texture --nseed 12 --rot_levels 0,30,45,60,90,135,180 --out results/tex_sg.jsonl" > results/resumeA.log 2>&1 &
nohup sh -c "python3 src/run_exp.py --exp rot --methods SuperGlueTTA --nref 3 --nseed 6 --rot_levels 180 --out results/rot_tta.jsonl && python3 src/run_exp.py --exp rot --methods SuperGlueTTA --ref texture --nseed 8 --rot_levels 45,60,90,135 --out results/tex_tta.jsonl" > results/resumeB.log 2>&1 &
