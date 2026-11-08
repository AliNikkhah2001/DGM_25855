# Deep Generative Models

Course repository for Fall 2026. The homework sheet is numbered **25732-1** (Dr. Sajjad Amini, Department of Electrical Engineering, Sharif University of Technology). This repository is the course folder `DGM_25855`.

It sits inside the semester repository as a submodule:

`AI_MSc` → `MSc_Fall_2026` → `DGM_25855`

## Site

GitHub Pages serves three levels. Later assignments are new folders, not edits to Homework 1.

| Page | Path |
|---|---|
| Course home | [index.html](index.html) |
| Calendar | [calendar/](calendar/index.html) |
| Lectures | [lectures/](lectures/index.html) |
| Homework list | [homeworks/](homeworks/index.html) |
| Homework 1 | [hw1/](hw1/index.html) |
| Homework 2 | [hw2/](hw2/index.html) |
| Homework 2 study map | [study/](study/index.html) |

Homework 1 is in `hw1/`: the study sheet, the teaching PDF and its LaTeX, the written solutions, the lecture check, the figures, and the coding assignment in `hw1/chw1/`. Homework 2 contains the theoretical sheet in `hw2/theory/`, the VAE and normalizing-flow notebooks in `hw2/simulation/`, and the study report in `study/`. Lecture PDFs are in `lectures/`. The 1404 calendar from Lecture 0 is in `calendar/`. Shared KaTeX lives in `assets/katex/` so another homework page can use the same fonts.

To add a homework: create `hwN/index.html` with the same navigation, put that assignment’s files in `hwN/`, and add a card on `homeworks/index.html`.

## Clone

```bash
git clone --recurse-submodules https://github.com/AliNikkhah2001/AI_MSc.git
```

That brings the semester and this course with it. To clone only the course:

```bash
git clone https://github.com/AliNikkhah2001/DGM_25855.git
```
