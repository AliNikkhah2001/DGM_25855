---
name: dgm-study-platform
description: >-
  Adds or updates a Deep Generative Models theory guide as split LaTeX
  chapters, a colored PDF, and GitHub Pages sections assembled from a catalog.
  Use when the user asks for a new homework guide, more guide sections, the
  same PDF-and-pages setup for a later assignment, or changes to the study
  platform in DGM_25855.
---

# DGM study platform

The course site is static. A guide is a catalog entry plus a manifest plus one HTML fragment per chapter. The page at `study/guide/` fetches those files and typesets them with the KaTeX already in `assets/katex/`. Follow `study/README.md` for the file layout. Do not rewrite `study/guide/index.html`, `guide.css`, or `platform.js` to add a chapter.

## Find the course root

Work in the directory that contains `study/catalog.json`. From the MSc archive that is `MSc_Fall_2026/courses/DGM_25855`.

## Extend the current guide

1. Add `study/tex/chapters/NN-short-name.tex` beginning with `\chapter{Title}`.
2. `\input` that file from `study/dgm-theory-guide.tex` in reading order.
3. From `study/`, run `pdflatex` twice.
4. From the course root, run `python3 study/tools/build_guide.py`.
5. Check one generated section for a result box and a display equation before committing.

## Add a later guide

Create `study/guides/<id>/manifest.json` and `sections/*.html`, put the PDF in `study/`, and append one object to the `guides` array in `study/catalog.json`. Section files are fragments: one `<section class="chapter">` each. Box classes are `def`, `result`, `intuition`, `practice`, `solution`, and `watch`. The web page picks the guide with `?guide=<id>`.

## Writing rules

Teach the identity in original exposition first, then map each homework theory part into the matching chapter. For every homework theory question, use a labeled pair `\begin{hwqbox}[Homework N, problem X.Y]...\end{hwqbox}` followed by `\begin{hwsolbox}[Homework N, problem X.Y]...\end{hwsolbox}` so the web class is `hwq` / `hwsol`. Expand teaching before those boxes when a step is missing. Keep the PDF page background white (`paper` = `#FFFFFF`). Colored boxes: burgundy results (`#8d3b2e` / `#f6ebe7`), navy definitions (`#1f4e79` / `#e7f0f7`), teal intuition (`#1d6a5a` / `#e5f4ef`), amber practice and homework questions (`#8a5a12` / `#fbf3e4`), plum solutions and homework answers (`#5c3d6e` / `#f3ecf6`), and a white watch-out box with a burgundy frame.

Open the guide over HTTP. A `file://` page cannot fetch the sections.
