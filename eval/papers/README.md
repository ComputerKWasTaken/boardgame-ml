# Exam papers — the frozen evidence behind the headline tables

Every headline generalization/transfer number in README.md and
DESIGN.md was produced by an exam: a fixed set of held-out tracks
sat by every agent under identical conditions (see
`src/boardgame_ml/eval/racing_exam.py`). The result trees were
originally written to per-milestone scratch folders outside the
repo; they are versioned here since M10 so the headline tables
are reproducible — and auditable — from a clone.

| File | Milestone | Paper | Contents |
|---|---|---|---|
| `m7_f2_exam.json` | M7 | Generator-v2 tracks 50001–50005, 20 eps each | Generalist SAC (100% laps, 0% crashes), specialist SAC/PPO zero-shot, generalist PPO, heuristic control |
| `m9_f3_exam_zeroshot.json` | M9 | Generator-v3 tracks 60001–60005, 20 eps each | The M7 agents, zero-shot on the foreign family (specialists: 0% laps; F2 generalist SAC: 62%) |
| `m9_f3gen_on_f2.json` | M9 | F2 paper | The F3-trained generalist SAC cross-tested on F2 (100% laps — the 2×2's key cell) |
| `m9_f3gen_on_f3.json` | M9 | F3 paper | The F3-trained generalist SAC on its home paper (63% laps) |
| `m10_replication_*.json` | M10 | F2 paper / tournament / eval protocols | The three-seed replication measurements (see DESIGN.md, M10) |

Per-episode records are stripped (the harness's `--json-out`
form); per-track and pooled aggregates are complete. Re-running
an exam with the same harness version and the checkpoint paths
named in each file reproduces these trees.
