# SkyGaze v2 changes

## Corrections to the earlier draft
- Author line said "Augustin"; now Kana Amedee.
- The old ablation counted any returned homography as "success". Re-running the original 60 degree protocol
  (SuperGlue, 5 seeds) gives errors of 2.8, 119.7, 568.7 and 724.5 px plus one no-homography trial, so the earlier text
  ("four of five trials sub-pixel, one failed") did not match its own CSV (median 189 px over 4 returned homographies).
- The old localization estimate was the mean of the 4 projected corners, which is not the projection of the query
  centre under a perspective component. v2 projects the centre through H. This also removes a ~0.4 px bias in the
  reported errors.

## New
- src/gen2.py: extra degradation axes (scale, perspective, noise, blur, occlusion, appearance), exact ground truth.
- src/run_exp.py: resumable JSONL experiment runner; SuperGlue + rotation hypotheses (4 rotated copies, most inliers).
- src/analyze.py: accurate / silent failure / detected failure taxonomy, Wilson CIs, AUROC, risk-coverage.
- src/make_figs.py, make_tables.py, make_qual.py, bench_runtime.py, fill_paper.py: figures, tables, timing, paper.
- results_v2/: every trial record (3,474) + runtime + gate summaries. results_v1_synthetic_original/: earlier files.
- paper/: IEEEtran LaTeX source (paper.tex template, paper_final.tex built by fill_paper.py), figures, paper.pdf.

## Reproduce
SuperGlue is not bundled (non-commercial licence). Clone it next to this folder as SuperGluePretrainedNetwork, then:
  python3 src/run_exp.py --exp rot --methods SIFT,ORB --nref 3 --nseed 10 --out results_v2/rot_classical.jsonl
  python3 src/run_exp.py --exp rot --methods SuperGlue --nref 3 --nseed 10 --out results_v2/rot_sg.jsonl
  (see run_exp.py header; analyze.py reads results/ relative to the repo, so copy results_v2/*.jsonl to results/)
  python3 src/make_figs.py && python3 src/make_tables.py && python3 src/fill_paper.py

## Real-data check (UAV-VisLoc flight 03)
- real/common.py, run_real.py, analyze_real.py, fig_real.py; trial-level results in results_real/.
- 32 held-out frames x 3 tile offsets = 96 trials per method; 8 calibration frames used only to pick the drone resize width (280 px, via SIFT).
- Results at 25 m: SIFT 19% accurate, ORB 0%, SuperGlue 12% (61% silent), SuperGlue + rotation hypotheses 47%.
- Note: the satellite map was downsampled to 4500 px by the user's prepare script; the data itself is not included here (see the UAV-VisLoc release).
