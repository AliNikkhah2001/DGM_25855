# Deep Generative Models

Course repository for Fall 2026. The homework sheet is numbered **25732-1** (Dr. Sajjad Amini, Department of Electrical Engineering, Sharif University of Technology). This repository is the course folder `DGM_25855`.

It sits inside the semester repository as a submodule:

`AI_MSc` → `MSc_Fall_2026` → `DGM_25855`

## Homework 1

The site root is a GitHub Pages index. Open the published page, or start from `index.html` locally.

| | |
|---|---|
| Study sheet | [HW1_study.html](HW1_study.html) |
| Teaching guide (PDF) | [HW1_tutorial.pdf](HW1_tutorial.pdf) |
| Teaching guide (LaTeX) | [HW1_tutorial.tex](HW1_tutorial.tex) |
| Written solutions | [HW1_solutions.md](HW1_solutions.md) |
| Checked against the lectures | [HW1_validated.md](HW1_validated.md) |
| Figures | [HW1_figures](HW1_figures) |
| Coding assignment | [CHW-1](CHW-1) |

`HW1_katex/` is the local KaTeX used by the study sheet, so the page renders without a network fetch.

## Coding assignment

`CHW-1/` is order-agnostic MADE and a class-conditional gated PixelCNN, trained on Apple MPS. See [CHW-1/README.md](CHW-1/README.md). MNIST is downloaded on first run and is not stored here. TensorBoard event files are local run logs and are not stored here.

## Clone

```bash
git clone --recurse-submodules https://github.com/AliNikkhah2001/AI_MSc.git
```

That brings the semester and this course with it. To clone only the course:

```bash
git clone https://github.com/AliNikkhah2001/DGM_25855.git
```
