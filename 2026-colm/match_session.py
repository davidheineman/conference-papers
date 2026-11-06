"""Filter a poster session to papers with a RESEARCHERS author and render session_N.md.

Usage: python 2026-colm/match_session.py 3
"""

import argparse
import difflib
import importlib
import json
import os
import re
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constants import RESEARCHERS

HERE = os.path.dirname(os.path.abspath(__file__))
PAPERS_PATH = os.path.join(HERE, "colm-2026.json")


def normalize(title):
    title = unicodedata.normalize("NFKC", title)
    title = title.replace("\u2018", "'").replace("\u2019", "'")
    title = title.replace("\u201c", '"').replace("\u201d", '"')
    title = title.replace("\u2014", "-").replace("\u2013", "-")
    title = re.sub(r"[^a-z0-9]+", " ", title.lower())
    return title.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=int, help="poster session number")
    args = parser.parse_args()

    module = importlib.import_module(f"session_{args.session}")
    titles = getattr(module, f"SESSION_{args.session}")
    markdown_path = os.path.join(HERE, f"session_{args.session}.md")

    papers = json.load(open(PAPERS_PATH, encoding="utf-8"))
    by_title = {normalize(p["title"]): p for p in papers}
    researchers = set(RESEARCHERS)

    # Program titles sometimes differ from the camera-ready ones on OpenReview
    # (revised wording, LaTeX in the title), so fall back to a fuzzy match.
    matched, fuzzy, unmatched = [], [], []
    for title in titles:
        key = normalize(title)
        paper = by_title.get(key)
        if paper is None:
            close = difflib.get_close_matches(key, by_title.keys(), n=1, cutoff=0.8)
            if close:
                paper = by_title[close[0]]
                fuzzy.append((title, paper))
        if paper is None:
            unmatched.append(title)
        else:
            matched.append(paper)

    print(
        f"SESSION_{args.session}: {len(titles)} titles, {len(matched)} matched "
        f"({len(fuzzy)} fuzzy), {len(unmatched)} unmatched"
    )
    if fuzzy:
        print("\nFuzzy matches (verify):")
        for title, paper in fuzzy:
            print(f"  - {title}\n    -> {paper['title']}")
    if unmatched:
        print("\nUnmatched:")
        for title in unmatched:
            print(f"  - {title}")

    hits = [p for p in matched if any(a in researchers for a in p["authors"])]
    print(f"\n{len(hits)} / {len(matched)} papers have a RESEARCHERS author")

    entries = []
    for paper in hits:
        abstract = " ".join((paper["abstract"] or "").split())
        entries.append(
            f"### [{paper['title']}]({paper['url']})\n"
            f"*{', '.join(paper['authors'])}*\n\n"
            f"{abstract}\n"
        )

    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(entries))
    print(f"Wrote {len(entries)} papers to {markdown_path}")


if __name__ == "__main__":
    main()
