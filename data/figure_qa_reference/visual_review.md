# Visual review

Seven actual PDF renders were inspected in Python-generated previews. A separate reviewer inspected the composite, both paired-contrast figures and the precision heatmap.

Corrections made after review:

- Added two F-only enlargements to the TW−T forest figure so that the small correctness deficits and intervals crossing zero can be resolved without changing the full-range panels.
- Changed forest markers to hollow circles and drew interval segments above the markers. Intervals remain at their original widths.
- Made observed non-correct length explicit in axis and precision-column labels; captions define terminal truncation at cycle 200.
- Raised and inset the six precision-failure outlines so that white group separators do not cover them.
- Filtered undrawn, out-of-range automatic tick artists before the canvas-bound check; this does not suppress in-range ticks or data.

Final visual inspection found no overlapping titles, legends or axis labels in the reviewed renders. Seven PDF pages retain selectable text; vector text and 600 dpi TIFF exports are checked programmatically. Markers and line styles provide redundant method encoding, with Python-generated grayscale previews included for inspection.

Limits: the original manuscript and old figures were not available, so original-to-redraw panel correspondence and manuscript cross-references could not be verified. This work verifies the plotted frozen tables and their display, not the physical validity of the simulation model or a new statistical analysis. Some very narrow intervals remain smaller than a plotting symbol at journal scale; exact endpoints are included in the source tables.
