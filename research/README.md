# Research index

| Approach | Outcome and reusable pieces | Evidence |
| --- | --- | --- |
| v13 contribution EQ | Active. Preserve the linear-magnitude allocator, level-matched auditions, and per-track frequency plots. Compare the predictor against the training mean. | [Frozen comparison](../experiments/eq/README.md), [audition script](../v13/scripts/audition_v13.py) |
| v13 pan | Parked. The L/C/R classifier favored center. Changing reconstruction width weights changed spread, while placement quality remained unresolved. Retain bypass, per-class diagnostics and EQ-only audition controls. | [Classifier recipe](../v13/experiments/v13-pan.toml), [reconstruction recipe](../v13/experiments/v13-pan-recon.toml), local dmc-data/archive/runs-pre-manifests/v13-pan*.log |
| GraFX distillation | v11b improved spectral reconstruction on three auditioned sessions. Two sounded over-compressed. Retain level-matched comparisons and synthetic controls from v12. | [v11b](../grafx/experiments/grafx-distill-v11b.toml), [v12](../grafx/experiments/minimal-v12-synth.toml), local notes/journey_2026-05.md |
| Full console | v6.2 round 14 ended in parameter collapse. Keep the DSP implementation and parameter diagnostics for comparison. | [Round 14](../diff_console/experiments/v6.2-round14.toml), [original project description](diff-console.md), local notes/RUNS.md |

Historical scripts and recipes retain their original locations under the approach directories. The original project description predates that reorganization.
