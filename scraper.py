import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from functools import partial
from multiprocessing.pool import ThreadPool
from openreview.api.client import Note
from tqdm import tqdm

from openreview import Client
from openreview.api import OpenReviewClient


@dataclass
class ConferenceConfig:
    venue_keywords: List[str]
    venue_id_prefix: str


@dataclass
class Review:
    reviewer: str
    rating: str
    confidence: str
    review: str
    summary: str
    strengths: str
    weaknesses: str
    review_id: str
    date: int


@dataclass
class Paper:
    title: str
    abstract: Optional[str]
    year: Optional[int]
    url: str
    pdf: str
    authors: List[str]
    venue: str
    venueid: str
    bibtex: str
    bibkey: Optional[str]
    invitation: Optional[str]
    venue_type: str
    reviews: List[Review]
    num_reviews: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Rename bibtex/bibkey fields for backward compatibility
        data["_bibtex"] = data.pop("bibtex")
        data["_bibkey"] = data.pop("bibkey")
        return data


CONFERENCE_CONFIGS = {
    "colm": ConferenceConfig(venue_keywords=["colmweb"], venue_id_prefix="COLM"),
    "iclr": ConferenceConfig(venue_keywords=["ICLR.cc"], venue_id_prefix="ICLR"),
    "neurips": ConferenceConfig(
        venue_keywords=["NeurIPS.cc"], venue_id_prefix="NeurIPS"
    ),
}


def init_client():
    username = os.environ.get("OPENREVIEW_USER")
    password = os.environ.get("OPENREVIEW_PASS")

    if not username or not password:
        raise FileNotFoundError(
            "OpenReview credentials not found in environment variables. Please set OPENREVIEW_USER and OPENREVIEW_PASS."
        )

    old_client = Client(
        baseurl="https://api.openreview.net",
        username=username,
        password=password,
    )

    new_client = OpenReviewClient(
        baseurl="https://api2.openreview.net",
        username=username,
        password=password,
    )

    return old_client, new_client


def get_venues(client: Client, conferences: List[str], years: List[int]) -> List[str]:
    def filter_year(venue):
        return next((str(year) for year in years if str(year) in venue), None)

    venues = client.get_group(id="venues").members
    filtered_venues = [venue for venue in venues if filter_year(venue)]

    read_venues = [
        venue
        for venue in filtered_venues
        if any(conf.lower() in venue.lower() for conf in conferences)
    ]

    return [venue for venue in read_venues if filter_year(venue)]


def get_papers(
    clients: List[Client],
    venues: List[str],
    conference: str = "colm",
    only_accepted: bool = True,
) -> Dict[str, List[Any]]:
    """Fetch papers from OpenReview."""
    papers = {}
    config = CONFERENCE_CONFIGS.get(conference.lower(), CONFERENCE_CONFIGS["colm"])
    venue_keywords = config.venue_keywords

    for venue in venues:
        # Check if venue matches any of the conference keywords
        if not any(keyword.lower() in venue.lower() for keyword in venue_keywords):
            continue

        print(f"Pulling {conference.upper()} papers from OpenReview: {venue}")
        papers[venue] = []

        for client in clients:
            if len(papers[venue]) > 0:
                continue

            if only_accepted:
                submissions = client.get_all_notes(
                    content={"venueid": venue}, details="directReplies"
                )
            else:
                single_blind_submissions = client.get_all_notes(
                    invitation=f"{venue}/-/Submission", details="directReplies"
                )
                double_blind_submissions = client.get_all_notes(
                    invitation=f"{venue}/-/Blind_Submission", details="directReplies"
                )
                submissions = single_blind_submissions + double_blind_submissions

            papers[venue] += submissions

    return papers


def get_reviews_for_paper(client: Client, paper_id: str, venue: str) -> List[Review]:
    """Fetch reviews for a specific paper."""
    paper_number = (
        paper_id.split("Submission")[-1] if "Submission" in paper_id else "1"
    )

    review_invitation = f"{venue}/Paper{paper_number}/-/Official_Review"
    reviews = client.get_all_notes(invitation=review_invitation)

    review_data = []
    for review in reviews:
        review_data.append(
            Review(
                reviewer=review.signatures[0] if review.signatures else "",
                rating=review.content.get("rating", ""),
                confidence=review.content.get("confidence", ""),
                review=review.content.get("review", ""),
                summary=review.content.get("summary", ""),
                strengths=review.content.get("strengths", ""),
                weaknesses=review.content.get("weaknesses", ""),
                review_id=review.id,
                date=review.date,
            )
        )

    return review_data


def _process_single_paper(
    note: Note,  # OpenReview Note object
    year: Optional[str],
    config: ConferenceConfig,
    client: Optional[Client],
    conf_name: str,
) -> tuple[Optional[Paper], Optional[str]]:
    # Get content as dict - handle both v1 and v2 API formats
    content = {}
    if hasattr(note, 'content'):
        content = note.content if isinstance(note.content, dict) else {}
    
    # Normalize content values (v2 API wraps values in {"value": ...})
    for k, v in content.items():
        if isinstance(v, dict) and "value" in v.keys():
            content[k] = v["value"]

    bibtex = content.get("_bibtex", "")
    if bibtex != "":
        bibkey = bibtex.split("{")[1].split(",")[0].replace("\n", "")
    else:
        bibkey = None

    venue = content.get("venue", "Submitted")
    venueid = content.get("venueid")

    # Normalize venue ID based on conference
    if venueid:
        if ".cc" in venueid:
            venueid = venueid.split(".cc")[0]
        elif "colmweb" in venueid:
            venueid = "COLM"
        else:
            # Try to extract conference name from venueid
            for conf_key, conf_config in CONFERENCE_CONFIGS.items():
                if any(
                    keyword in venueid for keyword in conf_config.venue_keywords
                ):
                    venueid = conf_config.venue_id_prefix
                    break

    # Determine venue type (acceptance status)
    if venue == config.venue_id_prefix:
        venue_type = "poster"  # Default for main conference
    elif "Submitted" in venue:
        venue_type = "rejected"
        return (None, "skipped")  # Skip rejected papers
    elif "spotlight" in venue.lower() or "notable top 25%" in venue:
        venue_type = "spotlight"
    elif "oral" in venue.lower() or "notable top 5%" in venue:
        venue_type = "oral"
    elif "accept" in venue.lower() or "poster" in venue.lower():
        venue_type = "poster"
    elif "invite" in venue.lower():
        venue_type = "invite"
    elif len(venue.split(" ")) == 2:
        venue_type = "poster"
    else:
        venue_type = "poster"

    # Get invitation
    invitation = None
    if hasattr(note, 'invitation'):
        invitation = note.invitation
    elif hasattr(note, 'invitations'):
        invitation = str(note.invitations)

    reviews = []

    # Check for direct replies (from details parameter in API call)
    if hasattr(note, 'details') and note.details and 'directReplies' in note.details:
        for reply in note.details['directReplies']:
            invitation_str = reply.get('invitation', '') if isinstance(reply, dict) else getattr(reply, 'invitation', '')
            if invitation_str.endswith("/-/Official_Review"):
                reply_content = reply.get('content', {}) if isinstance(reply, dict) else getattr(reply, 'content', {})
                reviews.append(
                    Review(
                        reviewer=(
                            reply.get('signatures', [''])[0] if isinstance(reply, dict) else getattr(reply, 'signatures', [''])[0]
                            if (reply.get('signatures') if isinstance(reply, dict) else getattr(reply, 'signatures', None))
                            else ""
                        ),
                        rating=reply_content.get("rating", ""),
                        confidence=reply_content.get("confidence", ""),
                        review=reply_content.get("review", ""),
                        summary=reply_content.get("summary", ""),
                        strengths=reply_content.get("strengths", ""),
                        weaknesses=reply_content.get("weaknesses", ""),
                        review_id=reply.get('id', '') if isinstance(reply, dict) else getattr(reply, 'id', ''),
                        date=reply.get('date', 0) if isinstance(reply, dict) else getattr(reply, 'date', 0),
                    )
                )

    # Check for replies attribute (alternative location)
    if hasattr(note, 'replies') and note.replies:
        for reply in note.replies:
            invitation_str = reply.get('invitation', '') if isinstance(reply, dict) else getattr(reply, 'invitation', '')
            if invitation_str.endswith("/-/Official_Review"):
                reply_content = reply.get('content', {}) if isinstance(reply, dict) else getattr(reply, 'content', {})
                reviews.append(
                    Review(
                        reviewer=(
                            reply.get('signatures', [''])[0] if isinstance(reply, dict) else getattr(reply, 'signatures', [''])[0]
                            if (reply.get('signatures') if isinstance(reply, dict) else getattr(reply, 'signatures', None))
                            else ""
                        ),
                        rating=reply_content.get("rating", ""),
                        confidence=reply_content.get("confidence", ""),
                        review=reply_content.get("review", ""),
                        summary=reply_content.get("summary", ""),
                        strengths=reply_content.get("strengths", ""),
                        weaknesses=reply_content.get("weaknesses", ""),
                        review_id=reply.get('id', '') if isinstance(reply, dict) else getattr(reply, 'id', ''),
                        date=reply.get('date', 0) if isinstance(reply, dict) else getattr(reply, 'date', 0),
                    )
                )

    # Fallback: try to fetch reviews separately
    if not reviews and client and hasattr(note, 'id'):
        try:
            reviews = get_reviews_for_paper(client, note.id, conf_name)
        except Exception as e:
            pass  # Silently fail for review fetching

    # Get paper fields with safe defaults
    note_id = getattr(note, 'id', '')
    title = content.get("title", "")
    abstract = content.get("abstract")
    pdf_path = content.get("pdf", "")
    authors = content.get("authors", [])

    paper = Paper(
        title=title,
        abstract=abstract,
        year=int(year) if year else None,
        url="https://openreview.net/forum?id=" + note_id,
        pdf="https://openreview.net" + pdf_path if pdf_path else "",
        authors=authors,
        venue=venue,
        venueid=venueid,
        bibtex=bibtex,
        bibkey=bibkey,
        invitation=invitation,
        venue_type=venue_type,
        reviews=reviews,
        num_reviews=len(reviews),
    )

    return (paper, None)


def preprocess_papers(
    papers: Dict[str, List[Any]],
    conference: str = "colm",
    client: Client = None,
    num_workers: int = 8,
) -> List[Paper]:
    """ Process papers in parallel. """
    dataset = []
    config = CONFERENCE_CONFIGS[conference.lower()]

    for conf_name, notes in papers.items():
        print(f"Processing {len(notes)} papers from {conf_name}")

        # Extract year from venue name
        venue_parts = conf_name.split("/")
        year = venue_parts[1] if len(venue_parts) > 1 else None

        # Create a partially applied function with fixed arguments
        process_paper = partial(
            _process_single_paper,
            year=year,
            config=config,
            client=client,
            conf_name=conf_name,
        )

        # Process papers in parallel with progress bar
        with ThreadPool(num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(process_paper, notes),
                    total=len(notes),
                    desc=f"Processing {conf_name}",
                    unit="paper",
                )
            )

        # Collect results and count skipped/errors
        skipped = 0
        errors = 0
        for paper, error_msg in results:
            if error_msg == "skipped":
                skipped += 1
            elif error_msg is not None:
                errors += 1
            elif paper is not None:
                dataset.append(paper)

        print(
            f"Processed {len(notes)-skipped-errors} / {len(notes)} entries for {conf_name} "
            f"(skipped: {skipped}, errors: {errors})"
        )

    return dataset


def download_papers(
    conference: str = "colm",
    output_path: Optional[str] = None,
    years: Optional[List[int]] = None,
    only_accepted: bool = True,
):
    """ Download papers from OpenReview. """
    conference = conference.lower()
    if conference not in CONFERENCE_CONFIGS:
        raise ValueError(
            f"Unsupported conference: {conference}. Supported: {list(CONFERENCE_CONFIGS.keys())}"
        )

    config = CONFERENCE_CONFIGS[conference]

    if output_path is None:
        output_path = f"{conference}_papers.json"

    if years is None:
        years = list(range(2013, 2026))

    print("Initializing OpenReview clients...")
    clientv1, clientv2 = init_client()

    print(f"Getting {conference.upper()} venues...")
    conferences = config.venue_keywords
    venues = get_venues(clientv1, conferences, years)

    print(f"Found {conference.upper()} venues: {venues}")

    if not venues:
        print(f"No {conference.upper()} venues found for the specified years.")
        return

    print("Downloading papers...")
    papers = get_papers([clientv1, clientv2], venues, conference, only_accepted)

    print("Processing papers...")
    processed_papers = preprocess_papers(papers, conference, clientv1)

    # Convert Paper dataclasses to dictionaries for JSON serialization
    papers_as_dicts = [paper.to_dict() for paper in processed_papers]

    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(papers_as_dicts, json_file, indent=4)

    print(
        f"Successfully saved {len(processed_papers)} {conference.upper()} papers to {output_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Pull papers from OpenReview",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--conference",
        default="colm",
        choices=list(CONFERENCE_CONFIGS.keys()),
        help="Conference (default: colm)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output path (default: {conference}_papers.json)",
    )
    parser.add_argument(
        "--years", nargs="+", type=int, help="Years to search (default: 2013-2025)"
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Include all submissions (accepted, under review, and rejected)",
    )

    args = parser.parse_args()

    download_papers(
        conference=args.conference,
        output_path=args.output,
        years=args.years,
        only_accepted=not args.include_all,
    )


if __name__ == "__main__":
    main()
