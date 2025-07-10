#!/usr/bin/env python3
"""
cvdr_crawler.py

Crawls the Dutch CVDR SRU service, extracts plain text from each record’s URL,
and uploads the resulting JSONL to the Hugging Face dataset hub.

Requirements:
    pip install requests lxml huggingface_hub

Usage:
    export HF_TOKEN="your_hf_token"
    python cvdr_crawler.py
"""

import os
import time
import json
import logging
from datetime import datetime
from html.parser import HTMLParser

import requests
from lxml import etree
from huggingface_hub import HfApi, Repository

# --- Configuration (from Handleiding SRU 2.0) :contentReference[oaicite:0]{index=0} ---
SRU_ENDPOINT = "https://zoekservice.overheid.nl/sru/Search"
SRU_PARAMS = {
    "xconnection": "cvdr",            # collection identifier
    "operation": "searchRetrieve",
    "version": "2.0",
    "query": 'keyword=""',            # ← match‐all via the empty keyword
    "httpAccept": "application/xml",  # ensure XML response
}
BATCH_SIZE = 1000
STATE_FILE = "crawler_state.json"
OUTPUT_FILE = "cvdr_data.jsonl"
HF_REPO_ID = "vGassen/Dutch_Centrale_Voorziening_Decentrale_Regelgeving"
MAX_RUNTIME_SECONDS = 5.5 * 3600     # guard against CI timeout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class TextExtractor(HTMLParser):
    """Extracts visible text from HTML, skipping <script> and <style>."""
    def __init__(self):
        super().__init__()
        self._parts = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self):
        # collapse whitespace
        return " ".join(" ".join(self._parts).split())


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            logging.info(f"Loaded state: {state}")
            return state
    logging.info("Starting fresh crawl")
    return {"startRecord": 1, "completed": False}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    logging.info(f"State saved: {state}")


def fetch_with_retries(url, params=None, retries=5):
    """Simple GET with retries."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            logging.warning(f"Fetch error ({attempt}/{retries}): {e}")
            time.sleep(1)
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")


def parse_sru(xml_bytes):
    """Return (list_of_recordData_elements, total_number_of_records)."""
    ns = {
        "sru": "http://docs.oasis-open.org/ns/search-ws/sruResponse",
        "gzd": "http://standaarden.overheid.nl/sru/record/gzd/1.0/",
    }
    root = etree.fromstring(xml_bytes)
    # check for diagnostics or explainResponse
    if root.tag.endswith("explainResponse"):
        logging.error("Got explainResponse—check parameters")
        return [], 0

    count_elt = root.find("sru:numberOfRecords", namespaces=ns)
    total = int(count_elt.text) if count_elt is not None else 0
    records = root.findall(".//sru:recordData", namespaces=ns)
    return records, total


def extract_plain_text(html_str):
    parser = TextExtractor()
    parser.feed(html_str)
    return parser.get_text()


def crawl_url(url):
    """Fetch HTML and extract text, or return None."""
    try:
        html = requests.get(url, timeout=30).text
        return extract_plain_text(html)
    except Exception as e:
        logging.warning(f"Failed to crawl {url}: {e}")
        return None


def push_to_hf(output_path, repo_id, token):
    """
    Uses huggingface_hub: creates the dataset repo if needed,
    then commits the JSONL file.
    """
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset",
                    exist_ok=True, token=token)
    # Clone, add, commit, push
    repo = Repository(local_dir="hf_tmp", clone_from=repo_id,
                      repo_type="dataset", use_auth_token=token)
    dest = os.path.join("hf_tmp", os.path.basename(output_path))
    # overwrite
    if os.path.exists(dest):
        os.remove(dest)
    os.rename(output_path, dest)
    repo.push_to_hub(commit_message="Update CVDR crawl")


def main():
    start_time = datetime.utcnow()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logging.error("Please set HF_TOKEN")
        return

    state = load_state()
    if state.get("completed"):
        logging.info("Already completed. Remove state file to re-run.")
        return

    # clean previous output
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    total_new = 0
    start_rec = state["startRecord"]

    try:
        while True:
            # timeout guard
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            if elapsed > MAX_RUNTIME_SECONDS:
                logging.warning("Approaching max runtime—saving state.")
                break

            logging.info(f"Fetching records from {start_rec}")
            params = {
                **SRU_PARAMS,
                "startRecord": start_rec,
                "maximumRecords": BATCH_SIZE,
            }
            xml = fetch_with_retries(SRU_ENDPOINT, params=params)
            records, total = parse_sru(xml)
            if not records:
                logging.info("No more records; marking complete")
                state["completed"] = True
                break

            for rec in records:
                # look for preferredUrl then url
                url = rec.findtext(".//gzd:preferredUrl", namespaces=rec.nsmap) \
                   or rec.findtext(".//gzd:url", namespaces=rec.nsmap)
                if not url:
                    continue

                text = crawl_url(url)
                if not text:
                    continue

                out = {"URL": url, "Content": text, "Source": "CVDR"}
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                total_new += 1

            # update state
            start_rec += len(records)
            state["startRecord"] = start_rec
            save_state(state)
            time.sleep(1)

    finally:
        save_state(state)

    if total_new:
        logging.info(f"Pushing {total_new} new records to HF")
        push_to_hf(OUTPUT_FILE, HF_REPO_ID, hf_token)
    else:
        logging.info("No new records found; nothing to push.")


if __name__ == "__main__":
    main()
