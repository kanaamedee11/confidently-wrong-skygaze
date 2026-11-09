#!/bin/sh
cd "$(dirname "$0")"
export SUPERGLUE_PATH=$(readlink -f ../SuperGluePretrainedNetwork)
nohup sh -c "python3 src/run_exp.py --exp rot --methods SIFT,ORB --nref 3 --nseed 10 --out results/rot_classical.jsonl && python3 src/run_exp.py --exp axes --methods SIFT,ORB --nref 3 --nseed 10 --out results/axes_classical.jsonl" > results/classical.log 2>&1 &
nohup sh -c "python3 src/run_exp.py --exp rot --methods SuperGlue --nref 3 --nseed 10 --out results/rot_sg.jsonl && python3 src/run_exp.py --exp axes --methods SuperGlue --nref 2 --nseed 6 --out results/axes_sg.jsonl" > results/sg.log 2>&1 &
nohup python3 src/run_exp.py --exp rot --methods SuperGlueTTA --nref 3 --nseed 6 --rot_levels 15,45,60,75,90,135,180 --out results/rot_tta.jsonl > results/tta.log 2>&1 &
