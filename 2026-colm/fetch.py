"""Pull COLM 2026 main conference papers (title + authors) from OpenReview."""

import json
import os

from scraper import init_client

VENUE = "colmweb.org/COLM/2026/Conference"
OUT_PATH = os.path.join(os.path.dirname(__file__), "colm-2026.json")


def main():
    _, client = init_client()

    notes = client.get_all_notes(content={"venueid": VENUE})
    print(f"Pulled {len(notes)} notes from {VENUE}")

    papers = []
    for note in notes:
        content = {
            k: (v["value"] if isinstance(v, dict) and "value" in v else v)
            for k, v in note.content.items()
        }
        papers.append(
            {
                "title": content.get("title", ""),
                "authors": content.get("authors", []),
                "abstract": content.get("abstract"),
                "venue": content.get("venue", ""),
                "url": f"https://openreview.net/forum?id={note.id}",
            }
        )

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2)
    print(f"Wrote {len(papers)} papers to {OUT_PATH}")


if __name__ == "__main__":
    main()
