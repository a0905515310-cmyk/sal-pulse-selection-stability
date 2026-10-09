# Figure contract

Backend: Python (matplotlib; all previews and export QA also use Python).

Provisional scientific claim: Joint time-and-width selection maintains high correct-selection probability under HPRF-only interference, but its recovery and persistence trade-offs depend on interference type and comparator.

Archetype: quantitative grid plus paired-effect forest plots. Condition order follows the frozen design, never effect-size ranking.

Evidence hierarchy and panel map:

1. HPRF-only: four distinct outcomes (correct selection, correct-state persistence, recovery from error, observed non-correct run length); all five rates and all five methods.
2. F-only: same four outcomes and methods at all four delays; defines the boundary of the HPRF-only result.
3. Composite: four outcome rows by three HPRF-rate columns; all three delays per column and all five methods.
4. TW minus T: four aligned forest panels, with all 18 interference conditions and frozen paired intervals; two labelled enlargements make the small F-only correctness and persistence contrasts readable.
5. TW minus W: same four outcomes and conditions; exposes the persistence/recovery trade-off.
6. H300 true-width separation: descriptive correctness by stratum, alongside stratum support; no inferred intervals.
7. Supplementary precision audit: all 216 preregistered precision checks; actual maximum one-sided half-width divided by the fixed target.
8. No-interference consistency: descriptive table; undefined metrics remain undefined.

Statistics: n = 2,000 independent simulation repeats per condition, 200 cycles per repeat. Bootstrap B = 2,000, whole-repeat paired percentile 95% intervals; point estimates are ratios of pooled counts. Intervals are pointwise, not simultaneous, and no multiplicity-adjusted claims or p-values are added. The same bootstrap repeat indices are shared across methods within each condition. All frozen point estimates and CI endpoints are used unchanged. Cycle observations are not independent replicates.

Observed non-correct run length includes terminal runs truncated at cycle 200; it is not an uncensored recovery-time estimate. H300 strata use the absolute difference between the true H pulse width and the nominal reference width for the current cycle, not measured noisy width difference.

Export: 183 mm width, 110–205 mm height as needed; 6–7 pt Arial body text, 8 pt bold lowercase panel letters; SVG with editable text, embedded TrueType PDF, 600 dpi TIFF and 300 dpi PNG. Use common method colours and marker shapes, white backgrounds, explicit units, no significance stars. Deliver clean source data, scripts, captions, verification report and combined PDF.

Image integrity: all figures are drawn from numerical tables. No fabricated observations, smoothing, selective data removal or raster reconstruction. FIRST/LAST overlaps remain visible through marker shape and small horizontal display offsets, which are disclosed. Probability values and differences may be expressed in percent and percentage points, respectively; no change to stored numerical values.

Review risks: F-only does not support blanket TW superiority; TW–W recovery/run-length trade-offs must remain visible; 6 of 216 precision checks miss the frozen targets; terminal runs are censored by the observation window; H300 strata are descriptive; no original manuscript figures were available for before/after verification.

Current Nature figure guidance consulted 2026-10-02:
https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/
