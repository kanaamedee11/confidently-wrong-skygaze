# Confidently Wrong: silent failures of classical and learned feature matching for GPS-denied UAV localization

A GPS-denied UAV can match its camera view against a stored georeferenced image. Matchers are usually ranked by accuracy alone, but a navigation
system also needs to know when a match is **wrong**. **SkyGaze** runs SIFT, ORB and SuperPoint+SuperGlue behind one interface, scores every trial
as *accurate*, *detected failure* or *silent failure* (a wrong homography returned as a success), and tests whether the RANSAC inlier count can
tell right from wrong.

*Status: preprint draft, not peer reviewed.* Paper: [`paper/paper.pdf`](paper/paper.pdf) (LaTeX source in `paper/`).

## Main results (see the paper for the full protocol and caveats)
* 3,474 synthetic trials with exact ground truth. SuperGlue is accurate in 150/150 trials up to 45 degrees of rotation, but from 60 degrees upward it is accurate in only 3/150 and returns a wrong homography in 79%. SIFT and ORB stay above 97% accurate from 0 to 180 degrees.
* Evaluating four rotated copies of the query removes the failure (126/126 accurate) at four times the runtime. The RANSAC inlier count separates correct from wrong results with AUROC 0.999 on the synthetic data.
* A small real-data check (32 held-out UAV-VisLoc frames, 96 trials per method, one flight) shows the same structure at much lower accuracy: SuperGlue is accurate in 12% of trials and silently wrong in 61%; SIFT 19%; SuperGlue with rotation hypotheses 47%. This is one flight and should be read as an illustration, not a benchmark.

## Layout
| Path | Content |
|---|---|
| `src/` | matchers, synthetic generator with exact ground truth, experiment runner, analysis, figure and table scripts |
| `real/` | real-data (UAV-VisLoc) trial design, runner, analysis |
| `results_v2/`, `results_real/` | every trial record (JSONL) and summaries |
| `paper/` | IEEEtran LaTeX source, figures, compiled PDF |
| `demo/app.py` | small Streamlit demo (`streamlit run demo/app.py`) |

## Reproduce
1. `pip install -r requirements.txt` (Python 3.10 or newer).
2. Clone SuperGlue next to this folder (see `DATA.md`). SIFT/ORB experiments do not need it.
3. Smoke test: `python src/run_baseline.py`.
4. Experiments: see the header of `src/run_exp.py` and `CHANGELOG.md` for the exact commands. Copy `results_v2/*.jsonl` to `results/` to re-run `src/analyze.py`, `src/make_figs.py`, `src/make_tables.py`, `src/fill_paper.py`.
5. Real-data check: obtain UAV-VisLoc flight 03 (see `DATA.md`), then `python real/run_real.py` and `python real/analyze_real.py`.

## Licence and citation
Code: MIT (`LICENSE`). Paper text and figures: CC BY 4.0 with the exception noted in `LICENSE-PAPER.md`. Cite via `CITATION.cff`.
Contact: Kana Amédée, kanaamedee00@gmail.com.
