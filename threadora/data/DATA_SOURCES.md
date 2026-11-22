# DMC color dataset provenance

File: `dmc_floss.csv` (489 DMC six-strand embroidery floss colors, art. 117).

- Origin: npm package `craft-color-codes` v1.1.0 by MakeBead, file `data/csv/dmc-floss.csv`.
- Tarball: https://registry.npmjs.org/craft-color-codes/-/craft-color-codes-1.1.0.tgz
- Tarball SHA-256: `2d8a6428c0b711f2dc4090176b3a002b11e50431f31b30c6fc69527d503d0cb3`
- Downloaded: 2026-11-22.
- License: CC BY 4.0 for the compilation. Credit: "Color data: craft-color-codes by MakeBead (https://makebead.com)".
- Upstream (per the package's SOURCES.md, status `cross-checked`): floss.maxxmint.com R/G/B
  columns, adrianj/CrossStitchCreator (the set behind the sharlagelfand/dmc R package),
  with xstitchify.com used as arbiter where they disagreed.

Limitations (stated by the source and repeated here): values are screen sRGB approximations,
not spectrophotometer readings of physical skeins. Thread on fabric looks different from a
screen swatch. Matching uses CIEDE2000 on these values.

The engine checks the file's SHA-256 at load time (see `threadora/dmc.py`) so an edited
dataset is detected.
