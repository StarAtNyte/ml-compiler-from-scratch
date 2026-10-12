"""Build the book website: book/src/*.md -> book/site/

  index.html              landing page (hero shows a real .uir program and the C uopc generates for it)
  <chapter>.html          one page per chapter: sidebar, outline, search, copy buttons, prev/next
  book.html               every chapter on one page, styled for print (the PDF is made from it)
  assets/                 style.css, app.js, search.json

Directives in the markdown (each on its own line):
  @include path                   the whole file as a code block
  @include path:10-40             lines 10..40
  @include path#name              a top-level function/class/variable, or Class.method (extracted with ast)
  @diagram name                   inline book/src/diagrams/name.svg
  @source a.py b.py ...           whole files, each in a collapsed block (Appendix D)

  python tools/build_book.py          # site
  python tools/build_book.py --pdf    # also book/site/book.pdf (needs google-chrome or chromium)

Set REPO_URL to change where "view source" links point.
"""
from __future__ import annotations
import ast, html, json, os, re, sys, textwrap
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parent.parent
SRC, OUT = ROOT / "book" / "src", ROOT / "book" / "site"
TITLE = "Building an ML Compiler from Scratch"
SHORT = "ML Compiler from Scratch"
REPO = os.environ.get("REPO_URL", "https://github.com/StarAtNyte/ml-compiler-from-scratch").rstrip("/")

# chapter number -> course week
WEEKS = {"00": "", "01": "Week 1", "02": "Week 1", "03": "Week 1", "04": "Week 2", "05": "Week 3", "06": "Week 4",
         "07": "Week 5", "08": "Week 5", "09": "Week 5", "10": "Week 6", "11": "Week 6", "12": "Week 7",
         "13": "Week 8", "14": "Weeks 9–10", "15": "Weeks 9–14", "16": "Appendix A", "17": "Appendix B", "18": "Appendix C",
         "19": "Appendix D"}

# files each chapter teaches: shown, collapsed, at the end of the chapter ("Full code for this chapter")
CHAPTER_CODE = {
  "01": ["uopc/uop.py", "uopc/dtype.py", "uopc/uir.py", "tests/test_w01_uops.py"],
  "02": ["uopc/rewrite.py", "uopc/symbolic.py", "tests/test_w01_uops.py"],
  "03": ["uopc/symbolic.py", "tests/test_w01_uops.py"],
  "04": ["uopc/render_c.py", "uopc/runtime.py", "tests/test_w02_render.py", "tests/helpers.py"],
  "05": ["uopc/linearize.py", "tests/test_w03_loops.py"],
  "06": ["uopc/rangeify.py", "uopc/composite.py", "tests/test_w04_rangeify.py"],
  "07": ["uopc/kernelize.py", "tests/test_w05_tensor.py"],
  "08": ["uopc/autodiff.py", "tests/test_w05_tensor.py"],
  "09": ["uopc/tensor.py", "uopc/nn.py", "examples/mnist_mlp.py", "examples/datasets.py"],
  "10": ["uopc/opt.py", "examples/bench_gemm.py", "tests/test_w06_opt.py"],
  "11": ["examples/flash_attention.py"],
  "12": ["uopc/gpu.py", "uopc/cuda.py", "tests/test_w07_gpu.py"],
  "13": ["uopc/gpu.py"],
  "14": ["examples/gpt.py", "tests/test_w09_models.py"],
}


# ---------------------------------------------------------------------------
# markdown preprocessing

def extract(path: Path, sel: str | None) -> tuple[str, str, int, int]:
  """(code, label, first line, last line) for an @include."""
  text = path.read_text()
  rel = str(path.relative_to(ROOT))
  lines = text.splitlines()
  if sel is None: return text, rel, 1, len(lines)
  if re.fullmatch(r"\d+-\d+", sel):
    a, b = map(int, sel.split("-"))
    return "\n".join(lines[a - 1:b]), f"{rel}:{a}-{b}", a, b
  node, scope = None, ast.parse(text).body
  for part in sel.split("."):
    node = next((n for n in scope if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef, ast.Assign))
                 and (getattr(n, "name", None) == part or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) == part for t in n.targets)))), None)
    if node is None: raise KeyError(f"{sel} not found in {path}")
    scope = getattr(node, "body", [])
  start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno])
  return "\n".join(lines[start - 1:node.end_lineno]), f"{rel} · {sel}", start, node.end_lineno


def make_md() -> markdown.Markdown:
  return markdown.Markdown(extensions=["fenced_code", "codehilite", "tables", "toc", "attr_list", "md_in_html"],
                           extension_configs={"codehilite": {"guess_lang": False}, "toc": {"toc_depth": "2-3"}})


DETAILS = re.compile(r'<details[^>]*>\s*(?:<summary>(.*?)</summary>)?(.*?)</details>', re.S)

def stash_details(md: str) -> tuple[str, dict[str, str]]:
  """Python-Markdown leaves raw HTML blocks alone, so markdown inside <details> came out as literal ** and `.
  Render each block's body ourselves and put a placeholder paragraph in its place."""
  out: dict[str, str] = {}
  def repl(m):
    summary, body = (m.group(1) or "Details").strip(), m.group(2)
    low = summary.lower()
    kind = "fullcode" if "lines" in low and "·" in summary else "answers" if "answer" in low else "solutions" if ("solution" in low or "hint" in low) else "more"
    sub = make_md()
    inner = sub.convert(textwrap.dedent(body).strip("\n"))
    summ = re.sub(r"^<p>|</p>$", "", sub.reset().convert(summary))
    key = f"DETAILSBLOCK{len(out)}X"
    out[key] = f'<details class="{kind}"><summary>{summ}</summary><div class="details-body">{inner}</div></details>'
    return f"\n\n{key}\n\n"
  return DETAILS.sub(repl, md), out


def unstash(html_: str, blocks: dict[str, str]) -> str:
  for k, v in blocks.items(): html_ = html_.replace(f"<p>{k}</p>", v).replace(k, v)
  return html_


def code_section(files: list[str], heading: str) -> str:
  parts = [f"\n\n## {heading}\n",
           "Every file this chapter builds, exactly as it is in the repository. Click a file to expand it.\n"]
  for f in files:
    n = len((ROOT / f).read_text().splitlines())
    parts.append(f"<details><summary>{f} · {n} lines</summary>\n\n@include {f}\n\n</details>\n")
  return "\n".join(parts)


def preprocess(md: str, errors: list[str]) -> str:
  out = []
  for line in md.splitlines():
    m = re.fullmatch(r"@include\s+([^\s:#]+)(?:[:#](\S+))?\s*", line)
    d = re.fullmatch(r"@diagram\s+(\S+)\s*", line)
    src = re.fullmatch(r"@source\s+(.+)", line)
    if src:   # several whole files, each collapsed
      for f in src.group(1).split():
        try: n = len((ROOT / f).read_text().splitlines())
        except OSError as e: errors.append(f"{line}: {e}"); continue
        out += ["", f"<details><summary>{f} · {n} lines</summary>", "", *preprocess(f"@include {f}", errors).splitlines(), "", "</details>", ""]
    elif m:
      path = ROOT / m.group(1)
      try: code, label, a, b = extract(path, m.group(2))
      except Exception as e:
        errors.append(f"{line}: {e}"); out.append(f"**[missing include: {html.escape(line)}]**"); continue
      lang = "c" if path.suffix in (".c", ".cu") else "python" if path.suffix == ".py" else ""
      href = f"{REPO}/blob/main/{m.group(1)}#L{a}-L{b}"
      out += [f'<div class="listing"><a href="{href}" target="_blank" rel="noopener">{html.escape(label)}</a></div>', "",
              f"```{lang}", code.rstrip(), "```"]
    elif d:
      p = SRC / "diagrams" / f"{d.group(1)}.svg"
      if not p.exists(): errors.append(f"missing diagram {p.name}"); out.append(f"**[missing diagram {d.group(1)}]**"); continue
      out += ['<figure class="diagram">', re.sub(r"<\?xml[^>]*\?>", "", p.read_text()), "</figure>", ""]
    else: out.append(line)
  return "\n".join(out)


# ---------------------------------------------------------------------------
# templates

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@500;600;700'
         '&family=IBM+Plex+Mono:wght@400;500&family=Newsreader:opsz,wght@6..72,400;6..72,500&family=Geist:wght@400;600'
         '&family=Geist+Mono&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet">')

FAVICON = ("data:image/svg+xml," + "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' "
           "fill='%231a2030'/%3E%3Ctext x='16' y='22' font-family='monospace' font-size='15' font-weight='700' fill='%23f08a59' "
           "text-anchor='middle'%3E%25u%3C/text%3E%3C/svg%3E")

# set the theme before first paint, so a dark-mode reader never sees a light flash
THEME_BOOT = ('<script>try{var t=localStorage.getItem("theme");if(t==="light"||t==="dark")'
              'document.documentElement.setAttribute("data-theme",t)}catch(e){}</script>')

ICON = {
  "menu": '<svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
  "search": '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><circle cx="11" cy="11" r="6.5" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="m16 16 4 4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
  "theme": '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><circle cx="12" cy="12" r="7.5" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M12 4.5a7.5 7.5 0 0 1 0 15z" fill="currentColor"/></svg>',
  "github": '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="currentColor" d="M12 2a10 10 0 0 0-3.2 19.5c.5.1.7-.2.7-.5v-1.7c-2.8.6-3.4-1.3-3.4-1.3-.5-1.2-1.1-1.5-1.1-1.5-.9-.6.1-.6.1-.6 1 .1 1.5 1 1.5 1 .9 1.5 2.4 1.1 2.9.8.1-.6.4-1.1.6-1.3-2.2-.3-4.6-1.1-4.6-5a3.9 3.9 0 0 1 1-2.7 3.6 3.6 0 0 1 .1-2.7s.8-.3 2.8 1a9.6 9.6 0 0 1 5 0c1.9-1.3 2.8-1 2.8-1 .5 1.4.2 2.4.1 2.7a3.9 3.9 0 0 1 1 2.7c0 3.9-2.4 4.7-4.6 5 .4.3.7.9.7 1.9v2.8c0 .3.2.6.7.5A10 10 0 0 0 12 2z"/></svg>',
  "pdf": '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M7 3h7l5 5v13H7z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="M14 3v5h5M10 13h6M10 16.5h6" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>',
}


def head(title: str, desc: str) -> str:
  return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
          f'<title>{html.escape(title)}</title><meta name="description" content="{html.escape(desc)}">'
          f'<meta property="og:title" content="{html.escape(title)}"><meta property="og:description" content="{html.escape(desc)}">'
          f'<link rel="icon" href="{FAVICON}">{FONTS}<link rel="stylesheet" href="assets/style.css">{THEME_BOOT}</head>')


def topbar(home: bool = False) -> str:
  menu = '' if home else f'<button class="icon-btn nav-toggle" aria-label="Open the chapter list" aria-expanded="false">{ICON["menu"]}</button>'
  return f"""<header class="topbar">
  {menu}<a class="brand" href="index.html"><span class="brand-mark">%u</span><span class="brand-name">{SHORT}</span></a>
  <div class="search">
    <label class="search-box">{ICON["search"]}<input id="search" type="search" placeholder="Search the book" autocomplete="off" aria-label="Search the book"><kbd>/</kbd></label>
    <div class="search-results" id="search-results" hidden></div>
  </div>
  <nav class="top-actions">
    <a class="icon-btn" href="book.pdf" title="Download the PDF" aria-label="Download the PDF">{ICON["pdf"]}</a>
    <a class="icon-btn" href="{REPO}" title="Source on GitHub" aria-label="Source on GitHub">{ICON["github"]}</a>
    <button class="icon-btn theme-toggle" title="Theme: system" aria-label="Switch theme">{ICON["theme"]}</button>
  </nav>
</header><div class="progress" aria-hidden="true"><span></span></div>"""


def sidebar(meta, part_at, cur: str | None) -> str:
  items = []
  for stem, t, _ in meta:
    if stem[:2] in part_at: items.append(f'<div class="nav-part">{html.escape(part_at[stem[:2]])}</div>')
    num = "" if stem[:2] == "00" else (stem[:2].lstrip("0") if int(stem[:2]) < 16 else "ABCD"[int(stem[:2]) - 16])
    label = re.sub(r"^(\d+\.|Appendix [A-D][:.])\s*", "", t).replace("`", "")
    items.append(f'<a href="{stem}.html"{" class=cur aria-current=page" if stem == cur else ""}>'
                 f'<span class="nav-num">{num}</span><span>{html.escape(label)}</span></a>')
  return f'<aside class="sidebar" id="sidebar"><nav aria-label="Chapters">{"".join(items)}</nav></aside>'


def outline(tokens) -> str:
  if not tokens: return ""
  def li(t, depth):
    kids = "".join(li(c, depth + 1) for c in t.get("children", []) if depth < 1)
    return f'<li><a href="#{t["id"]}" data-depth="{depth}">{html.escape(html.unescape(t["name"]))}</a>{f"<ul>{kids}</ul>" if kids else ""}</li>'
  h2 = [c for t in tokens for c in t.get("children", [])] if tokens and tokens[0]["level"] == 1 else tokens
  return f'<aside class="outline"><div class="outline-title">On this page</div><ul>{"".join(li(t, 0) for t in h2)}</ul></aside>'


def strip_tags(s: str) -> str:
  s = re.sub(r"<(script|style|svg)[\s\S]*?</\1>", " ", s)
  return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s))).strip()


def sections_for_search(stem: str, ctitle: str, body: str) -> list[dict]:
  """One search record per h2 section: title, chapter, url, plain text."""
  out = []
  parts = re.split(r'(<h2 id="[^"]+">.*?</h2>)', body)
  cur_t, cur_id, buf = ctitle, "", parts[0]
  for p in parts[1:]:
    m = re.match(r'<h2 id="([^"]+)">(.*?)</h2>', p)
    if m:
      out.append({"t": strip_tags(cur_t), "c": ctitle, "u": f"{stem}.html" + (f"#{cur_id}" if cur_id else ""), "x": strip_tags(buf)[:6000]})
      cur_id, cur_t, buf = m.group(1), m.group(2), ""
    else: buf += p
  out.append({"t": strip_tags(cur_t), "c": ctitle, "u": f"{stem}.html" + (f"#{cur_id}" if cur_id else ""), "x": strip_tags(buf)[:6000]})
  return out


# ---------------------------------------------------------------------------
# the landing page's hero: compile a real program at build time

HERO_UIR = """# C = A @ B for 4x4 matrices, from movement ops only
%0 = buffer 16 : dtype=f32 slot=0
%1 = buffer 16 : dtype=f32 slot=1
%2 = reshape %0, (4, 1, 4)
%3 = reshape %1, (4, 4)
%4 = permute %3, (1, 0)
%5 = reshape %4, (1, 4, 4)
%6 = mul %2, %5
%7 = reduce %6 : op=add pop=1"""

def hero_exhibit() -> tuple[str, str]:
  sys.path.insert(0, str(ROOT))
  from uopc.uir import parse
  from uopc.uop import UOp, Ops, ParamArg
  from uopc.dtype import dtypes
  from uopc.rangeify import lower
  from uopc.render_c import render_c
  g = parse(HERO_UIR)
  c = render_c(lower(g, UOp(Ops.BUFFER, (), ParamArg(2, dtypes.f32, (16,)))), "gemm")
  c = c[c.index("void gemm"):]   # drop the 4-line prelude
  return HERO_UIR, c


def highlight(code: str, lang: str) -> str:
  from pygments import highlight as hl
  from pygments.lexers import CLexer, TextLexer
  from pygments.formatters import HtmlFormatter
  return hl(code, CLexer() if lang == "c" else TextLexer(), HtmlFormatter(cssclass="codehilite", nowrap=False))


# ---------------------------------------------------------------------------
# build

def chapters() -> list[Path]: return sorted(p for p in SRC.glob("*.md") if p.name[0].isdigit())

def title_of(md: str, fallback: str) -> str:
  m = re.search(r"^# (.+)$", md, re.M)
  return m.group(1).strip() if m else fallback


def chapter_page(meta, part_at, i, body, tokens, words) -> str:
  stem, t, _ = meta[i]
  part = next((part_at[s[:2]] for s, _, _ in reversed(meta[:i + 1]) if s[:2] in part_at), "")
  eyebrow = " · ".join(x for x in (part.split(": ")[0], WEEKS.get(stem[:2], ""), f"{max(1, round(words / 230))} min read") if x)
  prev = (f'<a class="pager-card prev" href="{meta[i-1][0]}.html"><span>Previous</span><strong>{html.escape(meta[i-1][1])}</strong></a>'
          if i else "<span></span>")
  nxt = (f'<a class="pager-card next" href="{meta[i+1][0]}.html"><span>Next</span><strong>{html.escape(meta[i+1][1])}</strong></a>'
         if i + 1 < len(meta) else "<span></span>")
  desc = f"{t} — {TITLE}"
  return f"""{head(f"{t} · {SHORT}", desc)}<body class="reader">{topbar()}
<div class="shell">{sidebar(meta, part_at, stem)}
<main class="content" id="content"><div class="eyebrow">{html.escape(eyebrow)}</div>
<article class="prose">{body}</article>
<div class="page-foot"><a href="{REPO}/edit/main/book/src/{stem}.md" target="_blank" rel="noopener">Suggest an edit to this chapter</a></div>
<nav class="pager" aria-label="Chapter navigation">{prev}{nxt}</nav></main>
{outline(tokens)}</div><div class="scrim" hidden></div><script src="assets/app.js" defer></script></body></html>"""


def landing(meta, part_at, stats) -> str:
  uir, c = hero_exhibit()
  groups, cur = [], None
  for stem, t, words in meta:
    if stem[:2] in part_at:
      cur = {"name": part_at[stem[:2]], "items": []}; groups.append(cur)
    cur["items"].append((stem, t, words))
  cards = []
  for g in groups:
    name = g["name"]
    head_, _, sub = name.partition(": ")
    lis = "".join(f'<li><a href="{s}.html"{"" if WEEKS.get(s[:2]) else " class=nolw"}>{f"<span class=lw>{WEEKS[s[:2]]}</span>" if WEEKS.get(s[:2]) else ""}'
                  f'{html.escape(re.sub(r"^(\d+\.|Appendix [A-D][:.])\s*", "", t).replace("`", ""))}</a></li>' for s, t, _ in g["items"])
    cards.append(f'<section class="part-card"><div class="part-head"><span>{html.escape(head_)}</span>'
                 f'{f"<h3>{html.escape(sub)}</h3>" if sub else ""}</div><ol>{lis}</ol></section>')
  results = [("MNIST, week 5", "94%", "test accuracy, 15 ms per step"),
             ("GEMM 1024³, week 6", "321", "GFLOP/s, from 5 naive (numpy: 449)"),
             ("Flash attention, week 6", "3 ways", "unfused, fused, online softmax"),
             ("GPT, weeks 9–10", "2.11", "val loss on Tiny Shakespeare, from 3.83")]
  res = "".join(f'<div class="result"><div class="r-label">{a}</div><div class="r-num">{b}</div><div class="r-note">{c_}</div></div>'
                for a, b, c_ in results)
  desc = ("A book and a reference implementation for building a small machine learning compiler in Python, "
          "from a graph IR to generated C and CUDA, autodiff, and training a GPT.")
  return f"""{head(TITLE, desc)}<body class="home">{topbar(home=True)}
<main class="landing">
  <section class="hero">
    <div class="hero-text">
      <div class="kicker">Course notes for a 14-week compiler course</div>
      <h1>Building an ML Compiler <em>from Scratch</em></h1>
      <p class="lede">The book follows one small compiler, <code>uopc</code>, as it is built week by week: a graph IR and a rewrite
      engine first, then C code generation, loops, kernel fusion, automatic differentiation, CPU and GPU optimization,
      and finally training a GPT. Code listings are taken from the repository and the outputs come from running it.</p>
      <div class="cta"><a class="btn primary" href="{meta[0][0]}.html">Start reading</a>
        <a class="btn" href="{meta[1][0]}.html">Jump to chapter 1</a>
        <a class="btn ghost" href="book.pdf">{"Download PDF" if stats["pages"] == "PDF" else f"PDF · {stats['pages']}"}</a></div>
      <dl class="facts">
        <div><dt>Chapters</dt><dd>{stats["chapters"]}</dd></div><div><dt>Diagrams</dt><dd>{stats["diagrams"]}</dd></div>
        <div><dt>Words</dt><dd>{stats["words"]}</dd></div><div><dt>Compiler</dt><dd>{stats["loc"]} lines</dd></div>
      </dl>
    </div>
    <figure class="exhibit" aria-label="A .uir program and the C code uopc generates for it">
      <div class="pane"><div class="pane-head"><span>gemm.uir</span><span class="pane-tag">input</span></div>{highlight(uir, "text")}</div>
      <div class="arrow-row"><span>parse → rangeify → simplify → linearize → render</span></div>
      <div class="pane"><div class="pane-head"><span>gemm.c</span><span class="pane-tag hot">generated by uopc</span></div>{highlight(c, "c")}</div>
    </figure>
  </section>
  <section class="results" aria-label="What the reference compiler achieves">{res}</section>
  <section class="toc-grid" aria-label="Contents"><h2>Contents</h2><div class="parts">{"".join(cards)}</div></section>
  <section class="howto">
    <div><h2>Run it yourself</h2><p>Python 3.11+, numpy and gcc with OpenMP. An NVIDIA GPU is optional; GPU kernels are also checked
    on the CPU through an emulator.</p></div>
<pre class="codehilite"><code>git clone {REPO}.git
cd ml-compiler-from-scratch
python -m venv .venv &amp;&amp; .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q              # {stats["tests"]} tests
.venv/bin/python examples/mnist_mlp.py     # train MNIST with your compiler
.venv/bin/python examples/gpt.py           # train a GPT, then sample from it</code></pre>
  </section>
  <footer class="site-foot">Text and code in <a href="{REPO}">the repository</a>. Built with <code>tools/build_book.py</code>.</footer>
</main><script src="assets/app.js" defer></script></body></html>"""


def for_print(stem: str, body: str) -> str:
  """On paper nothing can be clicked open: expand answers and solutions. Per-chapter code stays collapsed
  (it would repeat), except in Appendix D, which is where the printed book carries the full source."""
  body = re.sub(r'<details class="(answers|solutions|more)">', r'<details class="\1" open>', body)
  if stem.startswith("19"): body = body.replace('<details class="fullcode">', '<details class="fullcode" open>')
  return body


def single_page(meta, bodies) -> str:
  """Every chapter on one page, for printing to PDF: a cover, then chapters, no app chrome."""
  cover = (f'<section class="cover"><div class="kicker">Course notes for a 14-week compiler course</div><h1>{TITLE}</h1>'
           f'<p>A graph IR, rewriting, code generation, kernel fusion, autodiff, GPUs, and training a GPT.</p><p class="cover-url">{REPO}</p></section>')
  toc = "".join(f'<li><a href="#{s}">{html.escape(t)}</a></li>' for s, t, _ in meta)
  chs = "".join(f'<section id="{s}" class="chapter-break"><article class="prose">{for_print(s, b)}</article></section>'
                for (s, _, _), b in zip(meta, bodies))
  return f"""{head(TITLE, TITLE)}<body class="print-book"><main class="print-main">{cover}
<section class="print-toc chapter-break"><h2>Contents</h2><ol>{toc}</ol></section>{chs}</main></body></html>"""


def build() -> int:
  (OUT / "assets").mkdir(parents=True, exist_ok=True)
  conv = make_md()
  errors: list[str] = []
  part_at = {l.split("|")[0].strip(): l.split("|")[1].strip()
             for l in (SRC / "PARTS.txt").read_text().splitlines() if "|" in l}
  meta, bodies, tokens = [], [], []
  for p in chapters():
    md = p.read_text()
    if p.stem[:2] in CHAPTER_CODE: md += code_section(CHAPTER_CODE[p.stem[:2]], "Full code for this chapter")
    conv.reset()
    text, blocks = stash_details(preprocess(md, errors))
    body = unstash(conv.convert(text), blocks)
    body = re.sub(r'href="(\d\d-[\w-]+)\.md(#[^"]*)?"', r'href="\1.html\2"', body)
    words = len(re.findall(r"\w+", strip_tags(body)))
    meta.append((p.stem, title_of(md, p.stem).replace("`", ""), words)); bodies.append(body); tokens.append(conv.toc_tokens)
  search = []
  for i, (stem, t, words) in enumerate(meta):
    (OUT / f"{stem}.html").write_text(chapter_page(meta, part_at, i, bodies[i], tokens[i], words))
    search += sections_for_search(stem, t, bodies[i])
  (OUT / "assets" / "search.json").write_text(json.dumps(search, ensure_ascii=False, separators=(",", ":")))
  pdf_path = OUT / "book.pdf"
  pages = "PDF"
  try:
    import subprocess
    info = subprocess.run(["pdfinfo", str(pdf_path)], capture_output=True, text=True).stdout
    if (m := re.search(r"Pages:\s+(\d+)", info)): pages = f"{m.group(1)} pages"
  except OSError: pass
  stats = {"chapters": len(meta), "diagrams": len(list((SRC / "diagrams").glob("*.svg"))),
           "words": f"{round(sum(w for _, _, w in meta) / 1000)}k",
           "loc": f"{sum(len(f.read_text().splitlines()) for f in (ROOT / 'uopc').glob('*.py')):,}",
           "tests": sum(f.read_text().count("\ndef test_") for f in (ROOT / "tests").glob("test_*.py")),
           "pages": pages}
  (OUT / "index.html").write_text(landing(meta, part_at, stats))
  (OUT / "book.html").write_text(single_page(meta, bodies))
  (OUT / "assets" / "style.css").write_text((ROOT / "tools" / "site" / "style.css").read_text() + pygments_css())
  (OUT / "assets" / "app.js").write_text((ROOT / "tools" / "site" / "app.js").read_text())
  (OUT / ".nojekyll").write_text("")
  for e in errors: print("ERROR:", e, file=sys.stderr)
  print(f"built {len(meta)} chapters -> {OUT}  ({len(errors)} errors)")
  return 1 if errors else 0


def pygments_css() -> str:
  from pygments.formatters import HtmlFormatter
  light = HtmlFormatter(style="default").get_style_defs(".codehilite")
  dark = HtmlFormatter(style="github-dark" if "github-dark" in __import__("pygments.styles", fromlist=["x"]).get_all_styles() else "monokai").get_style_defs(".codehilite")
  dark_scoped = lambda sel: "\n".join(f"{sel} {l}" for l in dark.splitlines() if l.startswith(".codehilite"))
  return ("\n/* syntax colors (pygments) */\n" + "\n".join(l for l in light.splitlines() if l.startswith(".codehilite")) +
          "\n@media (prefers-color-scheme: dark) {\n" + dark_scoped(':root:not([data-theme="light"])') + "\n}\n" +
          dark_scoped(':root[data-theme="dark"]') +
          "\n.codehilite, .codehilite pre, .codehilite span { background: transparent !important; }"
          "\n.codehilite .err { border: none !important; color: inherit !important; }\n")


def pdf() -> int:
  import shutil, subprocess
  chrome = next((c for c in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser") if shutil.which(c)), None)
  if chrome is None: print("no chrome/chromium found: open book/site/book.html and print to PDF"); return 1
  out = OUT / "book.pdf"
  subprocess.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                  "--virtual-time-budget=30000", f"--print-to-pdf={out}", (OUT / "book.html").as_uri()],
                 check=True, capture_output=True)
  print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
  return 0


if __name__ == "__main__":
  rc = build()
  if "--pdf" in sys.argv:
    rc |= pdf()
    rc |= build()   # rebuild so the landing page shows the PDF's page count
  sys.exit(rc)
