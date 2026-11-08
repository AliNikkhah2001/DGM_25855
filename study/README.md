# Study platform

The Homework 2 study map is a static page, `study/index.html`. Theory guides are a catalog of section files. GitHub Pages loads `study/catalog.json`, then each guide’s `manifest.json`, then concatenates the section HTML and typesets it with KaTeX. Adding a later homework does not require editing the page shell.

The printed guide and the web guide share one LaTeX source. Colors match the PDF: burgundy results, navy definitions, teal intuition, amber practice, plum solutions, and a white watch-out box with a burgundy edge.

## What is already here

| Piece | Path |
|---|---|
| Catalog | `study/catalog.json` |
| Web shell | `study/guide/index.html`, `guide.css`, `platform.js` |
| Homework 1–2 sections | `study/guides/hw1-hw2/sections/*.html` |
| Manifest | `study/guides/hw1-hw2/manifest.json` |
| LaTeX chapters | `study/tex/chapters/*.tex` |
| PDF driver | `study/dgm-theory-guide.tex` |
| PDF | `study/dgm-theory-guide.pdf` |
| HTML builder | `study/tools/build_guide.py` |

`build_guide.py` reads the chapter `.tex` files and rewrites the HTML sections and the manifest. It does not rebuild the PDF. Compile the PDF from `study/`:

```bash
pdflatex -interaction=nonstopmode dgm-theory-guide.tex
pdflatex -interaction=nonstopmode dgm-theory-guide.tex
```

## Add a section to the current guide

1. Write the mathematics in a new file `study/tex/chapters/NN-short-name.tex`, starting with `\chapter{Title}`.
2. Add `\input{tex/chapters/NN-short-name.tex}` to `study/dgm-theory-guide.tex` in reading order.
3. Recompile the PDF twice.
4. Run `python3 study/tools/build_guide.py` from the course root. It refreshes every section HTML file and `manifest.json` from the chapter files.
5. Commit the chapter, the PDF, the sections, and the manifest.

## Add a new guide later

1. Create `study/guides/<id>/sections/` and one HTML fragment per chapter. Each file is a single `<section class="chapter" id="...">`, not a full document. Use the same box classes as the current sections: `def`, `result`, `intuition`, `practice`, `solution`, `watch`.
2. Write `study/guides/<id>/manifest.json`:

```json
{
  "id": "hw3",
  "title": "Homework 3 theory guide",
  "pdf": "../../hw3-theory-guide.pdf",
  "sections": [
    { "id": "setup", "title": "Setup", "file": "sections/01-setup.html" }
  ]
}
```

3. Put the PDF next to `study/dgm-theory-guide.pdf` and append one object to `study/catalog.json`:

```json
{
  "id": "hw3",
  "title": "Homework 3 theory guide",
  "summary": "One sentence on what the guide teaches.",
  "pdf": "hw3-theory-guide.pdf",
  "manifest": "guides/hw3/manifest.json"
}
```

4. Open `study/guide/?guide=hw3` on GitHub Pages. The picker lists every catalog entry. The shell, the stylesheet, and KaTeX stay as they are.

Section files are fetched from the site origin. Opening `index.html` with a `file://` address will not load them.

Keep the guide as original exposition and short practice problems. Do not paste a filled-in homework submission into a section.
