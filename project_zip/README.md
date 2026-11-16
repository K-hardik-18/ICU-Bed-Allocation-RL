# project_zip: self-contained report bundle

Contents: `main.tex`, `ref.bib`, `rlreport.cls` (the supplied class, unmodified), and `figures/` (all PNG figures referenced by `main.tex`, copied physically at 400 dpi).

Compile (pdfLaTeX -> BibTeX -> pdfLaTeX x2), from inside this folder:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Requirements: a LaTeX distribution with the packages loaded by `rlreport.cls` (amsmath, graphicx, booktabs, cite, enumitem, titlesec, caption, geometry, hyperref, mathptmx) plus `listings` and `xcolor`. No file outside this folder is needed. The bundle is regenerated from the project root with `python make_project_zip.py`.

Note: `main.tex` intentionally does not call `\displayStyle{article}`; see the root `decision.md`. Author names and contributor responsibilities are placeholders to be completed by the authors. **Compilation status:** compiles from inside this folder (verified with Tectonic 0.17.0: `tectonic main.tex`), 13 pages, no undefined references.
