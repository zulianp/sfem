# Figure data for restyling

This is a CSV snapshot of the committed figures, not a new source of measurement data. Use it with your own plotting code; the existing paper generators remain authoritative. Re-export the data if a figure changes.

Pinned on 16 November 2026. The manuscript and its figures have not been modified.

Main source: [`cvfem-refactor` at `4386b46065d2`](https://github.com/zulianp/sfem/tree/4386b46065d2b664ebfa5e4173f206f7b6b65ca7/wip/paper).

| Figure | Contents | Data or source | Units / aggregation |
|---|---|---|---|
| 1 (`fig:decomposition`) | Mesh decomposition schematic | [../figures/packed_decomposition.tex](../figures/packed_decomposition.tex) | Schematic; no performance CSV. |
| 2 (`fig:idspace`) | Pack-local index spaces | [../figures/packed_id_space.tex](../figures/packed_id_space.tex) | Schematic; no performance CSV. |
| 3 (`fig:reduction`) | Local accumulation and reduction | [../figures/packed_reduction.tex](../figures/packed_reduction.tex) | Schematic; no performance CSV. |
| 4 (`fig:schemes`) | Passes used by each scheme | [../figures/scheme_passes.tex](../figures/scheme_passes.tex) | Schematic; no performance CSV. |
| 5 (`fig:throughput`) | Residual throughput versus problem size | [figure-05-throughput_size.csv](figure-05-throughput_size.csv) | cells per cube edge; MDOF/s. maximum observed throughput selected by existing generator. |
| 6 (`fig:scaling`) | Residual thread scaling | [figure-06-scaling_res.csv](figure-06-scaling_res.csv) | threads; MDOF/s. reported summary throughput. |
| 7 (`fig:packsize`) | Jacobian throughput versus packs per thread | [figure-07-packsize.csv](figure-07-packsize.csv) | packs per thread; MDOF/s. reported summary throughput. |
| 8 (`fig:convho`) | Residual schemes and precisions | [figure-08-convho.csv](figure-08-convho.csv) | scheme; MDOF/s. reported summary throughput. |
| 9 (`fig:jacho`) | Consistent Jacobian schemes and precisions | [figure-09-jacho.csv](figure-09-jacho.csv) | scheme; MDOF/s. DOFs divided by arithmetic mean application time. |
| 10 (`fig:roofline`) | Matched empirical roofline | [figure-10-empirical_roofline.csv](figure-10-empirical_roofline.csv) | FLOP/transported byte; GFLOP/s. per-capture rates and arithmetic mean of per-capture coordinates. |

## How to use these files

- Each main CSV contains the coordinates currently plotted, at the precision stored in the figure source. `series` groups a line or bar series; `point` preserves its order. `x` is numeric except for the scheme bars.
- Reference lines are in `reference-lines.csv`. A horizontal or vertical reference is specified once; extend it across the plotting area. Roofline paths and the closed range polygon have ordered vertices. The ideal-scaling line remains in Figure 6 CSV with `role=reference`.
- `current-series-styles.csv` records the existing style for each plotted series. These are a translation aid, not a requested palette. The same color does not have the same meaning in every old figure.
- The four schematics and all six plot sources remain in [figures](../figures/). Their captions are in [main.tex](../main.tex); no LaTeX sources are duplicated here.
- Figures 8 and 9 store absolute FP32 and FP64 throughput. The current light FP32 bar is overlaid behind the opaque FP64 bar. Do not stack the two absolute values. The light extension is the FP32 gain.
- Preserve marker/line distinctions as well as color. If legends move into captions, state which style denotes each layout, scheme, precision and reference. Keep the figure labels stable even if numbering changes.

## Comparisons that must remain distinct

Figures 5–8 contain historical measurements. Their existing source and scope qualifications should remain in the captions/text. Figure 5 selects the maximum observed throughput through the existing generator; this export does not replace it with a mean.

Figure 9 uses the newer complete Jacobian layout campaign. Its height is DOFs / mean application time, not the mean of individual throughputs. State gradients are reused; only packed Darwish uses the coefficient cache. The direct packed Darwish control is not plotted; its results remain in the existing [Jacobian layout documentation](../README.md). Timing ranges are sample ranges, not confidence intervals.

The SpMV references in Figures 8 and 9 are lagged operators, not assembled consistent Darwish. Figure 8 borrows its SpMV reference from the Figure 5 campaign; Figure 9 uses its own matched campaign. Do not combine these references into a universal line.

Figure 10 uses the earlier matched counter campaign. It does not contain coefficient-cached Darwish or the assembled consistent CSR/BSR comparison. The main panel has six implementations, 18 samples each, and six mean markers. The underlying campaign has eleven measured implementations; only the six in the CSV belong in this panel.

Roofline FLOPs are speculative hardware floating-point operations across precisions, with fused multiply-add counted as two per active vector element. Bytes are transported memory traffic. The mean point is the arithmetic mean of each coordinate across captures. Copy and triad are two separate bandwidth benchmarks; DGEMM provides the shared measured plateau. The vendor FP64 roof is contextual. Shading is an envelope of sample ranges, not a confidence interval.

## Verification and provenance

All six numerical figure sources were regenerated byte-for-byte from the pinned inputs before export. The Jacobian archive passed the existing completeness, launch, log-hash and numerical-result validator. The roofline archive was replayed with its existing acceptance checks before this export. CSV values round-trip exactly to their source coordinate strings. No new cluster measurements were run.

Original inputs and generators:

| Figures | Measurement input | Generator |
|---|---|---|
| 5 | `data/campaign_grace_5003779.csv` | `python/perf_figures.py` |
| 6 | `data/tscale_5003759.out` | `python/perf_figures.py` |
| 7 | `data/packsize_5005778.out` | `python/perf_figures.py` |
| 8 | `data/convho_5006526.out`, plus Figure 5 SpMV references | `python/perf_figures.py` |
| 9 | `pasc27-jacobian-layouts-5138025.zip` | `python/jacobian_layout_figures.py` |
| 10 | `pasc27-matched-counters-5047137.tar.gz` | `python/matched_counter_figures.py`, `python/matched_roofline.py` |

Paths are relative to [the paper directory](../). The [paper README](../README.md) describes archive retrieval and regeneration. Large archives and duplicate figure sources are deliberately excluded from this directory.
