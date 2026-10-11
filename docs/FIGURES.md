# Figures and tables (P7)

Produced by `python -m analysis.figures` from the P6 tables in `results/tables/`. Presentation decisions are in
`docs/DECISIONS.md` K-065. Every caption below is a proposal. The numbers in it come from the CSV named in the
"Source" column and must be re-read from that CSV when the paper text is written.

Conventions used in every figure:
- Δ = Q0 − Qk, so a positive value is a loss.
- Q0 = float32, Q1 = float16, Q2 = INT8.
- Test split. Each quantized version uses its own validation threshold (FPR = 1 %, K-036 a), except in Figure 8.
- Coarse types are orange filled squares; stealthy (masquerade) types are blue open circles.
- 1D-CNN is a green circle, conv-AE a purple square, XGBoost a grey diamond.

RQ1 = per-attack-type loss (H1, H2); RQ2 = latency/size/recall trade-off (H3); RQ3 = supervised vs unsupervised (H4).

## Figures

| File | Width | Supports | Caption proposal | Source |
|---|---|---|---|---|
| `fig1_system` | double | Method (Sec. IV) | System overview: CAN captures are split by source capture (ROAD) or by time block (Car-Hacking), encoded per frame, cut into 100-frame windows (non-overlapping at test time), classified by a 1D-CNN, a convolutional autoencoder or XGBoost, post-training quantized to fp16 and INT8, thresholded at 1 % validation FPR per version, and evaluated per attack type. | `configs/*.yaml` |
| `fig2_attack_signatures` | double | Threat model (Sec. III), coarse/stealthy split | Traffic criteria that define the coarse and stealthy groups: unseen-ID frame share (C1), window ID-distribution divergence relative to an attack-free null (C2), and target-ID rate and inter-arrival-time ratios (C3, C4); masquerade attacks stay inside every threshold (dotted line, grey band). | `attack_signatures.csv`, `configs/eda.yaml` |
| **`fig3_delta_recall_coarse_vs_stealthy`** (main) | double | **RQ1, H2 (not supported)** | Recall loss under quantization is small and not larger for stealthy attacks: per-type ΔRecall with 95 % hierarchical bootstrap intervals for (a) the 1D-CNN and (b) the conv-AE on ROAD (Cliff's δ and Holm-adjusted p of the group test under each level), and (c) the paired masquerade-minus-fabrication difference D for the 1D-CNN, whose INT8 mean is 0.0006 [−0.0035, 0.0057]; grey points have Q0 recall below 0.10, where ΔRecall is uninformative. | `stats.csv` (per_type_delta, group_mwu, as1_pair, as1_pooled), `delta_f1.csv`, `all_results.csv` |
| `fig4_recall_heatmap` | double | RQ1, H1 | Test recall per attack type, model and quantization level (cell text, 3-seed mean); cell colour is ΔRecall relative to the model's own Q0 (orange = loss, purple = gain); italic grey cells have Q0 recall below 0.10; † marks the test-only coolant pair. | `all_results.csv`, `delta_f1.csv` |
| `fig5_pareto` | double | RQ2, H3 | Latency–recall trade-off on the host CPU: (a) recall on ROAD stealthy types (95 % interval), (b) Car-Hacking pooled recall (± 1 s.d. over seeds), against median per-window latency, where the horizontal bar is the range of two measurement sessions and the marker sits at the slower session; marker area is file size; the dotted line is the robust Pareto front; dashed and dash-dotted lines are the 1-message and window latency budgets. | `pareto.csv` |
| `fig6_size_latency_gain` | single | RQ2, H3 | Gains of quantization relative to float32: (a) model file size ratio (fp16 0.52, INT8 0.38), (b) within-session median latency ratio Q2/Q0 as the range of two sessions (1D-CNN 0.64–0.75; conv-AE 1.01–1.08, no gain); fp16 latency could not be measured on the CPU. | `all_results.csv` |
| `fig7_supervised_vs_unsupervised` | double | RQ3, H4 (partially supported) | Supervised vs unsupervised sensitivity to INT8: (a) per-type ΔROC-AUC, (b) the pre-registered absolute-difference statistic (biased upward for the noisier model) next to the signed difference, whose interval contains 0, (c) share of attack windows whose decision changes and (d) 1 − Spearman ρ between Q0 and Q2 window scores, both larger for the conv-AE on ROAD. | `stats.csv` (per_type_delta, as3, as3_signed_diff, as3_churn) |
| `fig8_calibration_shift` | double | RQ3, H4 mechanism | Calibration shift when the float32 threshold is kept for the quantized model: attack-free (a) validation and (b) test FPR and (c) pooled recall per seed; INT8 moves the ROAD conv-AE reconstruction error so far that FPR rises 2–48-fold and the apparent recall gain is false alarms, while fp16 and the 1D-CNN do not shift. | `stats.csv` (calib_shift) |

## Tables

| File | Supports | Caption proposal | Source |
|---|---|---|---|
| `table_1_dataset` | Setup (Sec. V) | Dataset summary per attack type: captures per split, frames, injected frames and windows (stride 50 for training, 100 for validation and test); ROAD is split by source capture, Car-Hacking attack files by disjoint time blocks. | `data_profile.csv`, `split_summary.csv` |
| `table_2_recall` (+ `table_2b_f1`) | RQ1, H1 | Test recall (and F1) per attack type, model and quantization level, mean ± s.d. over three seeds. | `all_results.csv` |
| `table_3_resources` | RQ2, H3 | Model size, median and p95 latency as the range over two host sessions (p95 not interpreted), latency ratio to the float32 1D-CNN, peak memory, and compliance with the 1-message and window budgets. | `all_results.csv`, `pareto.csv` |
| `table_4_stats` | RQ1, RQ3 | Statistical comparisons with 95 % hierarchical bootstrap intervals: coarse vs stealthy at group level (Cliff's δ, exact Mann–Whitney p, Holm p, minimum attainable p), the paired fabrication–masquerade difference (no test, minimum p 0.25), and the supervised vs unsupervised comparison (pre-registered rule and signed check). | `stats.csv` |

## Figure selection for the 6-page paper (proposal, not final)

Seven of the eight figures are double-column, so not all of them fit in a 6-page IEEE paper. User proposal (2026-10-11):

- **Main text:** Figure 1 (system), Figure 3 (main result, RQ1), Figure 5 (Pareto, RQ2), Figure 8 (calibration shift).
- **Appendix / extended version:** Figure 2 (attack signatures), Figure 4 (recall heat map), Figure 7 (supervised vs unsupervised).
- **Rationale:** calibration shift is the strongest practical finding: a quantized autoencoder must be re-thresholded on its own validation scores. It gets a main-text figure. The H4 numbers of Figure 7 stay available in Table IV block C.
- The final choice is made when the paper is written.

## Notes for the paper text

- Figure 3 is the main figure. The text must say that H2 is **not** supported. It must not say "no stealthy-specific loss exists"; the supported wording is "in this design the stealthy-specific INT8 loss does not appear larger than ≈ 0.6 recall points" (P6 open issue 2).
- Figure 7 and Table IV block C carry the three stages of the H4 verdict together: the pre-registered rule (K-062), the signed check (K-063) and the final verdict (K-064). None of them is shown alone.
- Latency values are orders of magnitude only (K-050). The direction of the Q2/Q0 ratio is the claim, not the absolute value.
