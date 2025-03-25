import json
import os
import re
import time
from urllib.parse import parse_qs, urlparse

import PyPDF2
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


def parse_youtube_timestamp(url):
    """Extract and convert YouTube timestamp to seconds from various formats"""
    # Check for 't=' parameter
    if "t=" in url:
        # Extract the timestamp part
        parse_result = urlparse(url)
        query_params = parse_qs(parse_result.query)

        if "t" in query_params:
            timestamp = query_params["t"][0]
        elif parse_result.fragment and "t=" in parse_result.fragment:
            # Handle URL format with #t=1m7s
            fragment_params = parse_qs(parse_result.fragment)
            timestamp = fragment_params.get("t", ["0"])[0]
        else:
            return 0

        # Convert to seconds
        # Case 1: Simple seconds (t=123)
        if timestamp.isdigit():
            return int(timestamp)

        # Case 2: Complex format (1h2m3s, 1m7s, 7s, etc.)
        seconds = 0
        # Match hours, minutes, seconds
        hours_match = re.search(r"(\d+)h", timestamp)
        minutes_match = re.search(r"(\d+)m", timestamp)
        seconds_match = re.search(r"(\d+)s", timestamp)

        if hours_match:
            seconds += int(hours_match.group(1)) * 3600
        if minutes_match:
            seconds += int(minutes_match.group(1)) * 60
        if seconds_match:
            seconds += int(seconds_match.group(1))

        return seconds

    return 0


def get_youtube_video_titles(video_ids):
    """Get YouTube video titles in bulk with clear logging of the method used"""
    titles = {}

    # Method 1: Using YouTube Data API (preferred, requires API key)
    api_key = YOUTUBE_API_KEY
    if api_key:
        print(f"🔑 YouTube API Key found! Using YouTube Data API to fetch video titles")
        # Process video IDs in batches of 50 (API limit)
        batch_size = 50
        for i in range(0, len(video_ids), batch_size):
            batch = video_ids[i : i + batch_size]
            ids_string = ",".join(batch)

            print(
                f"Fetching batch {i//batch_size + 1}/{(len(video_ids)-1)//batch_size + 1} ({len(batch)} videos)"
            )
            url = f"https://www.googleapis.com/youtube/v3/videos?id={ids_string}&part=snippet&key={api_key}"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    for item in data.get("items", []):
                        video_id = item["id"]
                        title = item["snippet"]["title"]
                        titles[video_id] = title
                else:
                    print(f"API returned status code {response.status_code}")
            except Exception as e:
                print(f"Error fetching YouTube titles via API: {e}")
    else:
        print("No YouTube API key found in environment variables (YOUTUBE_API_KEY)")

    # Method 2: Web scraping as fallback (less reliable)
    missing_ids = [vid for vid in video_ids if vid not in titles]

    if missing_ids:
        if titles:
            print(
                f"Falling back to web scraping for {len(missing_ids)}/{len(video_ids)} videos that weren't fetched via API"
            )
        else:
            print(
                f"🕸️ Using web scraping to fetch titles for all {len(missing_ids)} videos (slower but no API key required)"
            )

        successful = 0
        for i, video_id in enumerate(missing_ids):
            # Add a small delay to avoid rate limiting
            time.sleep(0.2)

            if i % 10 == 0:
                print(f"Scraping progress: {i}/{len(missing_ids)} videos processed")

            url = f"https://www.youtube.com/watch?v={video_id}"
            try:
                response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    # Look for title in meta tags
                    meta_title = soup.find("meta", property="og:title")
                    if meta_title and "content" in meta_title.attrs:
                        titles[video_id] = meta_title["content"]
                        successful += 1
                    else:
                        # Fallback to title tag
                        title_tag = soup.find("title")
                        if title_tag:
                            # Remove " - YouTube" suffix if present
                            title = title_tag.text.replace(" - YouTube", "")
                            titles[video_id] = title
                            successful += 1
                        else:
                            print(f"Could not find title for video {video_id}")
            except Exception as e:
                print(f"Error scraping title for video {video_id}: {e}")

        print(f"Successfully scraped {successful}/{len(missing_ids)} video titles")

    print(f"Total video titles retrieved: {len(titles)}/{len(video_ids)}")
    return titles


def extract_links_from_pdf(pdf_path):
    """Extract all hyperlinks from the PDF document"""
    links = []
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()

            # Get annotations (links)
            if "/Annots" in page:
                for annot in page["/Annots"]:
                    annotation_object = annot.get_object()
                    if "/A" in annotation_object and "/URI" in annotation_object["/A"]:
                        uri = annotation_object["/A"]["/URI"]

                        # Convert PyPDF2 objects to native Python types
                        if isinstance(uri, PyPDF2.generic.ByteStringObject):
                            uri = str(uri)

                        # Similarly convert rect coordinates to standard Python types
                        rect = annotation_object.get("/Rect", [])
                        if rect:
                            rect = [
                                float(coord) for coord in rect
                            ]  # Convert to standard floats

                        links.append(
                            {
                                "page": page_num + 1,
                                "url": uri,
                                "position": rect,
                                "page_text": page_text,  # Save the entire page text for better matching
                            }
                        )
    return links


def convert_pdf_objects_to_native(obj):
    """Recursively convert PyPDF2 objects to native Python types"""
    if isinstance(obj, (PyPDF2.generic.NumberObject, PyPDF2.generic.FloatObject)):
        return float(obj)
    elif isinstance(
        obj, (PyPDF2.generic.ByteStringObject, PyPDF2.generic.TextStringObject)
    ):
        return str(obj)
    elif isinstance(obj, list):
        return [convert_pdf_objects_to_native(item) for item in obj]
    elif isinstance(obj, dict):
        # Skip 'page_text' key - it's large and we don't need to convert it
        return {
            convert_pdf_objects_to_native(k): (
                v if k == "page_text" else convert_pdf_objects_to_native(v)
            )
            for k, v in obj.items()
        }
    else:
        return obj


def extract_motifs_with_pages(pdf_path, full_text):
    """Extract motifs and identify which pages they appear on"""
    motifs = {}
    page_boundaries = []

    # First, identify page boundaries in the full text
    page_breaks = [m.start() for m in re.finditer(r"==PAGE BREAK==", full_text)]
    for i in range(len(page_breaks)):
        start = 0 if i == 0 else page_breaks[i - 1]
        end = page_breaks[i]
        page_boundaries.append((start, end, i + 1))  # start, end, page number

    # Add the last page
    if page_breaks:
        page_boundaries.append((page_breaks[-1], len(full_text), len(page_breaks) + 1))

    # Split the document into sections that likely contain motifs
    motif_pattern = r"(\d+[a-z]?)\.\s+([A-Z][A-Z\s&()/\-]+)"

    # Find all potential motif headers
    motif_headers = list(re.finditer(motif_pattern, full_text))

    for i in range(len(motif_headers)):
        current_match = motif_headers[i]
        next_match = motif_headers[i + 1] if i + 1 < len(motif_headers) else None

        motif_num = current_match.group(1)
        motif_name = current_match.group(2).strip()

        # Get content between this motif header and the next one
        start_pos = current_match.end()
        end_pos = next_match.start() if next_match else len(full_text)

        motif_content = full_text[start_pos:end_pos].strip()

        # Determine which page(s) this motif appears on
        motif_pages = []
        motif_start = current_match.start()
        motif_end = end_pos

        for start, end, page_num in page_boundaries:
            # If motif overlaps with this page
            if (
                (motif_start >= start and motif_start < end)
                or (motif_end > start and motif_end <= end)
                or (motif_start <= start and motif_end >= end)
            ):
                motif_pages.append(page_num)

        # Extract key components from the motif content
        appearances_match = re.search(
            r"Used In:(.+?)(?:First Usage:|$)", motif_content, re.DOTALL
        )
        appearances_text = (
            appearances_match.group(1).strip() if appearances_match else ""
        )
        appearances = re.findall(r"[IVX]+|[A-Z]", appearances_text)
        # Clean up appearances list to remove duplicates and invalid entries
        appearances = [
            a
            for a in appearances
            if a
            in [
                "I",
                "II",
                "III",
                "IV",
                "V",
                "VI",
                "VII",
                "VIII",
                "IX",
                "R",
                "S",
                "K",
                "M",
                "B",
            ]
        ]

        # Description is everything before "Used In:"
        description_match = re.search(r"^(.*?)(?:Used In:|$)", motif_content, re.DOTALL)
        description = description_match.group(1).strip() if description_match else ""
        # Clean up description - remove newlines and excessive whitespace
        description = re.sub(r"\s+", " ", description)

        # Extract first usage
        first_usage_match = re.search(
            r"First Usage:(.+?)(?:Key Features:|$)", motif_content, re.DOTALL
        )
        first_usage_text = (
            first_usage_match.group(1).strip() if first_usage_match else ""
        )

        timestamp_match = re.search(r"\[(\d+:\d+:\d+)\]", first_usage_text)
        timestamp = timestamp_match.group(1) if timestamp_match else ""

        cue_match = re.search(r'"([^"]+)"', first_usage_text)
        cue_name = cue_match.group(1) if cue_match else ""

        cue_number_match = re.search(
            r"\((\d+[a-z]\d+(?:\s*[A-Za-z]*)?)\)", first_usage_text
        )
        cue_number = cue_number_match.group(1) if cue_number_match else ""

        # Extract key features
        key_features_match = re.search(r"Key Features:(.+?)$", motif_content, re.DOTALL)
        key_features = key_features_match.group(1).strip() if key_features_match else ""
        # Clean up key features
        key_features = re.sub(r"\s+", " ", key_features)

        # Create motif entry
        film = appearances[0] if appearances else ""
        motifs[f"{motif_num}. {motif_name}"] = {
            "number": motif_num,
            "name": motif_name,
            "description": description,
            "appearances": appearances,
            "first_appearance": {
                "film": film,
                "timestamp": timestamp,
                "cue_name": cue_name,
                "cue_number": cue_number,
            },
            "key_features": key_features,
            "pages": motif_pages,  # Add page numbers where this motif appears
            "youtube_links": [],
        }

    return motifs


def match_links_by_page_proximity(motifs, links):
    """Match YouTube links to motifs based on page proximity"""
    youtube_links = [
        link
        for link in links
        if "youtube.com" in link["url"] or "youtu.be" in link["url"]
    ]
    print(
        f"Found {len(youtube_links)} YouTube links out of {len(links)} total links"
    )

    # Create a mapping of pages to motifs
    page_to_motifs = {}
    for motif_key, motif_data in motifs.items():
        for page in motif_data.get("pages", []):
            if page not in page_to_motifs:
                page_to_motifs[page] = []
            page_to_motifs[page].append(motif_key)

    # Collect video IDs for bulk title fetching
    all_video_ids = []
    for link in youtube_links:
        video_id = ""
        if "v=" in link["url"]:
            video_id_match = re.search(r"v=([^&]+)", link["url"])
            if video_id_match:
                video_id = video_id_match.group(1)
        elif "youtu.be/" in link["url"]:
            video_id_match = re.search(r"youtu\.be/([^?&]+)", link["url"])
            if video_id_match:
                video_id = video_id_match.group(1)

        if video_id:
            all_video_ids.append(video_id)

    # Fetch all video titles at once
    print(f"🔍 Fetching titles for {len(all_video_ids)} unique YouTube videos...")
    video_titles = get_youtube_video_titles(all_video_ids)

    # Match YouTube links to motifs based on page
    matched_count = 0
    for link in youtube_links:
        page = link["page"]

        # Find motifs on this page
        motif_keys = page_to_motifs.get(page, [])

        if not motif_keys:
            # If no motifs on this exact page, look for motifs on adjacent pages
            adjacent_motifs = []
            for p in [page - 1, page + 1]:
                adjacent_motifs.extend(page_to_motifs.get(p, []))
            motif_keys = adjacent_motifs

        # If still no motifs found, skip this link
        if not motif_keys:
            continue

        # Parse YouTube info
        video_id = ""
        if "v=" in link["url"]:
            video_id_match = re.search(r"v=([^&]+)", link["url"])
            if video_id_match:
                video_id = video_id_match.group(1)
        elif "youtu.be/" in link["url"]:
            video_id_match = re.search(r"youtu\.be/([^?&]+)", link["url"])
            if video_id_match:
                video_id = video_id_match.group(1)

        # Get the parsed timestamp
        timestamp_sec = parse_youtube_timestamp(link["url"])

        # Get the video title
        track_name = video_titles.get(video_id, "Unknown Title")

        # If there's only one motif on this page, assign the link to it
        if len(motif_keys) == 1:
            motif_key = motif_keys[0]

            motifs[motif_key]["youtube_links"].append(
                {
                    "url": link["url"],
                    "video_id": video_id,
                    "timestamp_seconds": timestamp_sec,
                    "page": link["page"],
                    "track_name": track_name,
                }
            )
            matched_count += 1
        else:
            # Multiple motifs on this page - try to match based on page text
            page_text = link.get("page_text", "").lower()

            best_match = None
            best_score = 0

            for motif_key in motif_keys:
                motif_data = motifs[motif_key]

                # Check for motif name or number in page text
                score = 0

                # Check for motif name
                motif_name = motif_data["name"].lower()
                if motif_name in page_text:
                    score += 5

                # Check for motif number
                motif_num = motif_data["number"].lower()
                if motif_num in page_text:
                    score += 3

                # Check for first appearance details
                cue_name = motif_data["first_appearance"]["cue_name"].lower()
                if cue_name and cue_name in page_text:
                    score += 4

                timestamp = motif_data["first_appearance"]["timestamp"]
                if timestamp and timestamp in page_text:
                    score += 4

                if score > best_score:
                    best_score = score
                    best_match = motif_key

            # If we found a good match, assign the link
            if best_match and best_score > 0:
                motifs[best_match]["youtube_links"].append(
                    {
                        "url": link["url"],
                        "video_id": video_id,
                        "timestamp_seconds": timestamp_sec,
                        "page": link["page"],
                        "track_name": track_name,
                        "match_score": best_score,
                    }
                )
                matched_count += 1

    print(f"Successfully matched {matched_count} YouTube links to motifs")

    # Count motifs with links
    motifs_with_links = sum(1 for motif in motifs.values() if motif["youtube_links"])
    print(
        f"{motifs_with_links}/{len(motifs)} motifs now have associated YouTube links"
    )

    return motifs


def process_star_wars_pdf(pdf_path):
    """Process the Star Wars thematic catalogue PDF with improved extraction"""
    print(f"Processing PDF: {pdf_path}")

    # Extract hyperlinks
    links = extract_links_from_pdf(pdf_path)
    print(f"🔗 Extracted {len(links)} hyperlinks from PDF")

    # Convert PDF objects to native Python types for JSON serialization
    links = convert_pdf_objects_to_native(links)

    # Extract all text
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        full_text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            full_text += page.extract_text() + "\n\n==PAGE BREAK==\n\n"

    print(f"Extracted {len(full_text)} characters of text")

    # Parse motifs with improved method that tracks pages
    motifs = extract_motifs_with_pages(pdf_path, full_text)
    print(f"🎵 Identified {len(motifs)} motifs")

    # Match links to motifs with improved page-based method
    motifs_with_links = match_links_by_page_proximity(motifs, links)

    # Save the raw links for inspection (just URLs, not page text)
    try:
        simple_links = [{"page": link["page"], "url": link["url"]} for link in links]

        with open("extracted_links_simple.json", "w") as f:
            json.dump(simple_links, f, indent=2)
        print("Saved links to extracted_links_simple.json")
    except Exception as e:
        print(f"Warning: Could not save links due to error: {e}")

    return motifs_with_links


# Example usage
pdf_path = "data/Star-Wars-Thematic-Catalogue.pdf"
output_dir = "data/mappings"
result = process_star_wars_pdf(pdf_path)

# Save the result to a JSON file

with open(f"{output_dir}/star_wars_motifs_complete.json", "w") as f:
    json.dump(result, f, indent=2)

print("Complete Star Wars motif data saved to star_wars_motifs_complete.json")
