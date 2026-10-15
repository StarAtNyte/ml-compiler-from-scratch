"""Validate generated HTML page links and fragments after building the book.

Run: python tools/check_book.py. External links are outside this local check.
"""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent / "book" / "site"


class Links(HTMLParser):
  def __init__(self):
    super().__init__()
    self.ids, self.links = set(), []

  def handle_starttag(self, tag, attrs):
    values = dict(attrs)
    if "id" in values: self.ids.add(values["id"])
    if tag == "a" and "href" in values: self.links.append(values["href"])


def main():
  pages = {}
  for path in ROOT.glob("*.html"):
    parser = Links()
    parser.feed(path.read_text())
    pages[path.resolve()] = parser
  if not pages: raise SystemExit("build the book first: no HTML pages found")
  errors, checked = [], 0
  for path, parser in pages.items():
    for href in parser.links:
      url = urlsplit(href)
      if url.scheme or url.netloc: continue
      target = (path.parent / unquote(url.path)).resolve() if url.path else path
      if target.suffix != ".html": continue
      checked += 1
      if target not in pages: errors.append(f"{path.name}: missing page {href}")
      elif url.fragment and unquote(url.fragment) not in pages[target].ids:
        errors.append(f"{path.name}: missing fragment {href}")
  for error in errors: print(error)
  print(f"checked {checked} local page links across {len(pages)} HTML files; {len(errors)} errors")
  return bool(errors)


if __name__ == "__main__": raise SystemExit(main())
