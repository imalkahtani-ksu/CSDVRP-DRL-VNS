"""
highlight_changes.py — build the "Highlighted PDF" required at resubmission.

    python highlight_changes.py

Compares the revised manuscript with the submitted one and writes
main_highlighted.pdf, in which every line of text that is not present in the
submitted version carries a yellow highlight annotation. Because both
versions are typeset with the same class and column width, an untouched
paragraph produces exactly the same line breaks, so this marks changed and
new text (including captions, tables, equations and text inside the figures)
and leaves unchanged text clean.

A short report of how much of each page is highlighted is printed at the end.
"""
import re
import sys
from collections import Counter
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "paper_ieee_access" / "main.pdf"   # the submitted version
NEW = HERE / "main.pdf"                                 # the revision
OUT = HERE / "main_highlighted.pdf"

sys.stdout.reconfigure(encoding="utf-8")


def norm(s: str) -> str:
    """Normalize a line for comparison: collapse spaces, drop soft hyphens."""
    s = s.replace("­", "").replace("ﬁ", "fi").replace("ﬂ", "fl")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def lines_of(doc):
    """Yield (page_number, bbox, normalized_text) for every text line."""
    for pno in range(doc.page_count):
        for block in doc[pno].get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = "".join(sp["text"] for sp in line.get("spans", []))
                t = norm(text)
                if t:
                    yield pno, line["bbox"], t


def main():
    if not OLD.exists():
        sys.exit(f"submitted version not found: {OLD}")
    if not NEW.exists():
        sys.exit(f"revised version not found: {NEW}")

    old = fitz.open(OLD)
    old_lines = Counter(t for _, _, t in lines_of(old))
    old.close()

    new = fitz.open(NEW)
    pool = Counter(old_lines)
    marks = []
    total = 0
    for pno, bbox, t in lines_of(new):
        total += 1
        if pool[t] > 0:
            pool[t] -= 1            # an unchanged line, consumed once
        else:
            marks.append((pno, bbox))

    per_page = Counter(p for p, _ in marks)
    for pno, bbox in marks:
        page = new[pno]
        r = fitz.Rect(bbox)
        r.y0 -= 0.5
        r.y1 += 0.5
        annot = page.add_highlight_annot(r)
        annot.set_colors(stroke=(1, 1, 0.25))
        annot.set_info(title="Revision", content="changed in revision")
        annot.update()

    new.set_metadata({"title": "Highlighted revision",
                      "subject": "Changes against Access-2026-33829"})
    new.save(OUT, garbage=3, deflate=True)
    print(f"{len(marks)} of {total} text lines highlighted "
          f"({len(marks)/max(total,1)*100:.0f}%)")
    print("highlighted lines per page:")
    for p in range(new.page_count):
        print(f"  p{p+1:>2}: {per_page.get(p,0)}")
    print("wrote", OUT)
    new.close()


if __name__ == "__main__":
    main()
