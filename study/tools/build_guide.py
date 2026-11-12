#!/usr/bin/env python3
"""Split the theory guide into chapter files and HTML sections.

The LaTeX chapters are the PDF source. The HTML sections are what GitHub
Pages assembles from guides/<id>/manifest.json.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "dgm-theory-guide.tex"
CHAPTER_DIR = ROOT / "tex" / "chapters"
GUIDE_ID = "hw1-hw2"
SECTION_DIR = ROOT / "guides" / GUIDE_ID / "sections"
MANIFEST = ROOT / "guides" / GUIDE_ID / "manifest.json"

LABELS = {
    "ch:map": "How to use this guide",
    "ch:prob": "Probability",
    "ch:calc": "Matrix calculus and Jacobians",
    "ch:info": "Divergences",
    "ch:gauss": "The Gaussian family",
    "ch:mc": "Monte Carlo",
    "ch:ar": "Autoregressive models",
    "ch:vae": "Variational autoencoders",
    "ch:hier": "Hierarchical latents",
    "ch:flow": "Normalizing flows",
    "ch:set": "A short practice set",
}

BOXES = {
    "defbox": ("def", "Definition"),
    "resultbox": ("result", "Result"),
    "intbox": ("intuition", "Intuition"),
    "practbox": ("practice", "Practice"),
    "solbox": ("solution", "Worked solution"),
    "pitbox": ("watch", "Watch out"),
    "hwqbox": ("hwq", "Homework question"),
    "hwsolbox": ("hwsol", "Homework answer"),
}

MACROS = [
    (r"\\E(?![A-Za-z])", r"\\mathbb{E}"),
    (r"\\R(?![A-Za-z])", r"\\mathbb{R}"),
    (r"\\Z(?![A-Za-z])", r"\\mathbb{Z}"),
    (r"\\N(?![A-Za-z])", r"\\mathcal{N}"),
    (r"\\KL(?![A-Za-z])", r"\\mathrm{KL}"),
    (r"\\JS(?![A-Za-z])", r"\\mathrm{JS}"),
    (r"\\diag(?![A-Za-z])", r"\\operatorname{diag}"),
    (r"\\tr(?![A-Za-z])", r"\\operatorname{tr}"),
    (r"\\given(?![A-Za-z])", r"\\mid"),
    (r"\\logit(?![A-Za-z])", r"\\operatorname{logit}"),
    (r"\\bm\{", r"\\boldsymbol{"),
    (r"\\textcolor\{muted\}\{", r"\\text{"),
    (r"\\textcolor\{[a-z]+\}\{([^}]*)\}", r"\1"),
    (r"\\tfrac", r"\\frac"),
    (r"\\bigl", r"\\left"),
    (r"\\bigr", r"\\right"),
    (r"\\Bigl", r"\\left"),
    (r"\\Bigr", r"\\right"),
    (r"\\lvert", r"|"),
    (r"\\rvert", r"|"),
    (r"\\Vert", r"\\|"),
    (r"\\mid", r"\\mid"),
    (r"\\label\{[^}]*\}", ""),
    (r"\\eqref\{eq:([^}]*)\}", r"(\\mathrm{\1})"),
    (r"\\ref\{([^}]*)\}", r"\\mathrm{\1}"),
    (r"\\S(?=\d)", r"\\S "),
    (r"~", " "),
    (r"\\,", " "),
    (r"\\;", " "),
    (r"\\quad", r"\\quad"),
    (r"\\emph\{([^}]*)\}", r"\\textit{\1}"),
]


def split_chapters(text: str) -> tuple[str, list[tuple[str, str]]]:
    begin = text.index("\\begin{document}")
    preamble = text[:begin]
    body = text[begin:]
    toc = body.index("\\tableofcontents")
    front = body[: toc + len("\\tableofcontents")]
    rest = body[toc + len("\\tableofcontents") :]
    rest = rest.replace("\\end{document}", "").strip() + "\n"
    parts = re.split(r"(?=\\chapter\*?(?:\[.*?\])?\{)", rest)
    chapters = []
    for part in parts:
        part = part.strip()
        if not part.startswith("\\chapter"):
            continue
        title = re.search(r"\\chapter\*?\{([^}]*)\}", part).group(1)
        chapters.append((title, part + "\n"))
    return preamble, front, chapters


def write_tex(preamble: str, front: str, chapters: list[tuple[str, str]]) -> None:
    CHAPTER_DIR.mkdir(parents=True, exist_ok=True)
    inputs = []
    for i, (title, body) in enumerate(chapters, 1):
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        name = f"{i:02d}-{slug}.tex"
        (CHAPTER_DIR / name).write_text(body, encoding="utf-8")
        inputs.append(f"\\input{{tex/chapters/{name}}}")
    main = (
        preamble
        + "\\begin{document}\n"
        + "\\maketitle\n"
        + "\\begin{center}\n"
        + "\\textcolor{muted}{Lecture numbers refer to the decks in \\texttt{lectures/}.\\\\\n"
        + "Slide numbers are the numbers printed on the slides.}\n"
        + "\\end{center}\n"
        + "\\tableofcontents\n\n"
        + "\n".join(inputs)
        + "\n\\end{document}\n"
    )
    TEX.write_text(main, encoding="utf-8")


def expand_macros(tex: str) -> str:
    for pattern, repl in MACROS:
        tex = re.sub(pattern, repl, tex)
    return tex


def protect_math(tex: str) -> tuple[str, list[tuple[str, str]]]:
    slots: list[tuple[str, str]] = []

    def stash(kind: str, inner: str) -> str:
        inner = expand_macros(inner.strip())
        inner = inner.replace("\\begin{align*}", "\\begin{aligned}")
        inner = inner.replace("\\end{align*}", "\\end{aligned}")
        inner = inner.replace("\\begin{psmallmatrix}", "\\begin{pmatrix}")
        inner = inner.replace("\\end{psmallmatrix}", "\\end{pmatrix}")
        slots.append((kind, inner))
        return f"@@M{len(slots) - 1}@@"

    def stash_display(match: re.Match) -> str:
        return stash("display", match.group(1))

    tex = re.sub(
        r"\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}",
        lambda m: stash("display", m.group(1)),
        tex,
        flags=re.S,
    )
    tex = re.sub(
        r"\\begin\{align\*?\}(.*?)\\end\{align\*?\}",
        lambda m: stash("display", "\\begin{aligned}" + m.group(1) + "\\end{aligned}"),
        tex,
        flags=re.S,
    )
    tex = re.sub(r"\\\[(.*?)\\\]", stash_display, tex, flags=re.S)
    tex = re.sub(r"\$\$(.*?)\$\$", stash_display, tex, flags=re.S)
    tex = re.sub(r"\$(.+?)\$", lambda m: stash("inline", m.group(1)), tex)
    return tex, slots


def restore_math(text: str, slots: list[tuple[str, str]]) -> str:
    def repl(match: re.Match) -> str:
        kind, inner = slots[int(match.group(1))]
        inner = html.escape(inner, quote=False)
        if kind == "display":
            return f"\\[{inner}\\]"
        return f"\\({inner}\\)"

    return re.sub(r"@@M(\d+)@@", repl, text)


def inline_format(text: str) -> str:
    text = text.replace("{\\o}", "ø")
    text = html.escape(text, quote=False)
    text = re.sub(r"\\textbf\{([^}]*)\}", r"<strong>\1</strong>", text)
    text = re.sub(r"\\emph\{([^}]*)\}", r"<em>\1</em>", text)
    text = re.sub(r"\\textit\{([^}]*)\}", r"<em>\1</em>", text)
    text = re.sub(r"\\texttt\{([^}]*)\}", r"<code>\1</code>", text)
    text = text.replace("\\&", "&amp;")
    text = re.sub(r"\\ ", " ", text)
    text = text.replace("~", " ")
    text = text.replace("---", "—").replace("--", "–")
    text = re.sub(r"\\S\s*", "§", text)
    text = re.sub(r"\\ref\{([^}]*)\}", lambda m: LABELS.get(m.group(1), m.group(1)), text)
    text = re.sub(r"\\eqref\{[^}]*\}", "", text)
    text = re.sub(r"\\label\{[^}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^}]*)\}", r"\1", text)
    text = text.replace("\\\\", " ")
    return text


def convert_lists(text: str) -> str:
    def list_block(match: re.Match) -> str:
        kind = match.group(1)
        body = match.group(2)
        tag = "ol" if kind == "enumerate" else "ul"
        items = re.split(r"\\item\s*", body)
        lis = []
        for item in items:
            item = item.strip()
            if not item:
                continue
            lis.append(f"<li>{blocks_to_html(item)}</li>")
        return f"<{tag}>{''.join(lis)}</{tag}>"

    return re.sub(
        r"\\begin\{(enumerate|itemize)\}(?:\[[^\]]*\])?(.*?)\\end\{\1\}",
        list_block,
        text,
        flags=re.S,
    )


def _brace_end(text: str, start: int) -> int:
    depth = 0
    for j in range(start, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return j
    raise ValueError("unbalanced brace")


def convert_table(text: str) -> str:
    def table_block(body: str) -> str:
        rows = []
        for raw in body.split("\\\\"):
            raw = re.sub(r"\\(toprule|midrule|bottomrule)", "", raw).strip()
            if not raw or raw.startswith("%"):
                continue
            cells = [inline_format(c.strip()) for c in raw.split("&")]
            if not any(cells):
                continue
            rows.append(cells)
        if not rows:
            return ""
        head = "".join(f"<th>{c}</th>" for c in rows[0])
        body_rows = "".join(
            "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows[1:]
        )
        return f'<table class="data"><thead><tr>{head}</tr></thead><tbody>{body_rows}</tbody></table>'

    out = []
    cursor = 0
    token = r"\begin{tabularx}"
    while True:
        start = text.find(token, cursor)
        if start < 0:
            out.append(text[cursor:])
            break
        out.append(text[cursor:start])
        i = start + len(token)
        i = _brace_end(text, i) + 1
        i = _brace_end(text, i) + 1
        end = text.find(r"\end{tabularx}", i)
        out.append(table_block(text[i:end]))
        cursor = end + len(r"\end{tabularx}")
    return "".join(out)


def convert_boxes(text: str) -> str:
    def box_block(match: re.Match) -> str:
        kind = match.group(1)
        title = (match.group(2) or "").strip()
        body = match.group(3).strip()
        css, fallback = BOXES[kind]
        label = f"{fallback}: {title}" if title else fallback
        return (
            f'<aside class="box {css}"><p class="box-title">{html.escape(label)}</p>'
            f"{blocks_to_html(body)}</aside>"
        )

    return re.sub(
        r"\\begin\{(defbox|resultbox|intbox|practbox|solbox|pitbox|hwqbox|hwsolbox)\}(?:\[([^\]]*)\])?(.*?)\\end\{\1\}",
        box_block,
        text,
        flags=re.S,
    )


def convert_diagrams(text: str) -> str:
    def diagram(match: re.Match) -> str:
        return '<figure class="diagram" aria-hidden="true"></figure>'

    text = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", diagram, text, flags=re.S)
    text = re.sub(
        r"\\begin\{center\}\s*\\textcolor\{muted\}\{\\small\s*(.*?)\}\s*\\end\{center\}",
        lambda m: f'<p class="caption">{inline_format(m.group(1))}</p>',
        text,
        flags=re.S,
    )
    text = re.sub(r"\\begin\{center\}|\\end\{center\}", "", text)
    return text


def blocks_to_html(text: str) -> str:
    text = convert_diagrams(text)
    text = convert_lists(text)
    text = convert_table(text)
    text = convert_boxes(text)
    text = re.sub(r"\\addcontentsline\{[^}]*\}\{[^}]*\}\{[^}]*\}", "", text)
    text = re.sub(r"\\(?:renewcommand|label|ref)\{[^}]*\}(?:\{[^}]*\})?", "", text)
    chunks = re.split(r"\n\s*\n", text.strip())
    out = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.startswith("<"):
            out.append(chunk)
            continue
        sec = re.match(r"\\(chapter\*?|section)\{([^}]*)\}", chunk)
        if sec:
            title = inline_format(sec.group(2))
            tag = "h2" if sec.group(1).startswith("chapter") else "h3"
            rest = chunk[sec.end() :].strip()
            out.append(f"<{tag}>{title}</{tag}>")
            if rest:
                out.append(blocks_to_html(rest))
            continue
        if "@@M" in chunk or "\\[" in chunk or "\\(" in chunk:
            # A paragraph may be only a display, or prose with inline math.
            parts = re.split(r"(\\\[.*?\\\]|\\\(.*?\\\))", chunk, flags=re.S)
            buf = []
            for part in parts:
                if part.startswith("\\["):
                    if buf:
                        out.append(f"<p>{inline_format(''.join(buf))}</p>")
                        buf = []
                    out.append(part)
                else:
                    buf.append(part)
            if buf and "".join(buf).strip():
                out.append(f"<p>{inline_format(''.join(buf))}</p>")
            continue
        out.append(f"<p>{inline_format(chunk)}</p>")
    return "\n".join(out)


def chapter_html(title: str, body: str) -> str:
    protected, slots = protect_math(body)
    # Expand references that sit outside math before escaping.
    protected = re.sub(
        r"\\ref\{([^}]*)\}",
        lambda m: LABELS.get(m.group(1), m.group(1)),
        protected,
    )
    rendered = blocks_to_html(protected)
    rendered = restore_math(rendered, slots)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return f'<section class="chapter" id="{slug}">\n{rendered}\n</section>\n'


def write_html(chapters: list[tuple[str, str]]) -> None:
    SECTION_DIR.mkdir(parents=True, exist_ok=True)
    manifest_sections = []
    for i, (title, body) in enumerate(chapters, 1):
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        filename = f"{i:02d}-{slug}.html"
        (SECTION_DIR / filename).write_text(chapter_html(title, body), encoding="utf-8")
        manifest_sections.append(
            {"id": slug, "title": title, "file": f"sections/{filename}"}
        )
    MANIFEST.write_text(
        json.dumps(
            {
                "id": GUIDE_ID,
                "title": "Homework 1 and 2 theory guide",
                "pdf": "../../dgm-theory-guide.pdf",
                "sections": manifest_sections,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def refresh_html() -> None:
    chapters = []
    for path in sorted(CHAPTER_DIR.glob("*.tex")):
        body = path.read_text(encoding="utf-8")
        title = re.search(r"\\chapter\*?\{([^}]*)\}", body).group(1)
        chapters.append((title, body))
    write_html(chapters)
    print(f"sections {len(chapters)} -> {SECTION_DIR}")


def main() -> None:
    refresh_html()


if __name__ == "__main__":
    main()
