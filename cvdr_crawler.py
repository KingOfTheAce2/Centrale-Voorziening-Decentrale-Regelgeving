# cvdr_crawler.py
import os
import time
import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime

import urllib.request
import urllib.error
from html.parser import HTMLParser
import subprocess
import tempfile
import shutil

# --- Configuration ---
SRU_ENDPOINT = "https://zoekservice.overheid.nl/sru/Search"
SRU_PARAMS = {
    # NOTE: according to the SRU 2.0 manual the parameter name is
    # ``xconnection`` (without a dash). Using the wrong parameter leads to a
    # ``406 Not Acceptable`` response from the API.  Setting ``httpAccept``
    # ensures we always request an XML response.
    "xconnection": "cvdr",
    "operation": "searchRetrieve",
    "version": "2.0",
    "query": 'cql.textAndIndexes="*"',
    "httpAccept": "application/xml",
}
HF_REPO_ID = "vGassen/Dutch_Centrale_Voorziening_Decentrale_Regelgeving"
OUTPUT_FILE = "cvdr_data.jsonl"
STATE_FILE = "crawler_state.json"
BATCH_SIZE = 1000
# Set a timeout to exit gracefully before GitHub Actions kills the job (5h 30m)
MAX_RUNTIME_SECONDS = 5.5 * 3600

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class TextExtractor(HTMLParser):
    """Simple HTML text extractor that skips script and style tags."""

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
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

    def get_text(self) -> str:
        return " ".join(" ".join(self._parts).split())


def fetch_with_retries(url: str, params: dict | None = None, *, retries: int = 5) -> bytes:
    """Fetches a URL with basic retry logic using urllib."""
    if params:
        query = urllib.parse.urlencode(params)
        url = f"{url}?{query}"

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cvdr-crawler"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except urllib.error.URLError as e:
            logging.warning("Fetch failed (%s). Attempt %d/%d", e, attempt + 1, retries)
            time.sleep(1)
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")


def load_crawler_state() -> dict:
    """Loads the crawler state from a local JSON file."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            logging.info(f"Loaded crawler state: {state}")
            return state
    logging.info("No state file found. Starting a fresh crawl.")
    return {"startRecord": 1, "completed": False}

def save_crawler_state(state: dict):
    """Saves the current crawler state to a local JSON file."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)
    logging.info(f"Saved crawler state: {state}")

def get_existing_urls_from_hf(token: str) -> set:
    """Fetches existing URLs from the Hugging Face dataset via the raw file."""

    logging.info(f"Fetching existing URLs from Hugging Face repo: {HF_REPO_ID}")
    existing_urls: set[str] = set()
    raw_url = (
        f"https://huggingface.co/datasets/{HF_REPO_ID}/raw/main/{OUTPUT_FILE}"
    )
    req = urllib.request.Request(raw_url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            for line in resp.read().decode("utf-8").splitlines():
                try:
                    record = json.loads(line)
                    if "URL" in record:
                        existing_urls.add(record["URL"])
                except json.JSONDecodeError:
                    continue
        logging.info("Found %d existing URLs in the dataset", len(existing_urls))
    except urllib.error.URLError as e:
        logging.warning("Could not load existing dataset from Hugging Face: %s", e)

    return existing_urls

def parse_sru_response(xml_content: str) -> (list, int):
    """Parses the SRU XML response to extract record data and total record count."""
    ns = {
        "sru": "http://docs.oasis-open.org/ns/search-ws/sruResponse",
        "gzd": "http://standaarden.overheid.nl/sru/record/gzd/1.0/",
    }
    try:
        root = ET.fromstring(xml_content)

        if root.tag.endswith("explainResponse"):
            logging.error(
                "Received explain response\u2014check query parameters"
            )
            return [], 0

        count_elem = root.find("sru:numberOfRecords", ns)
        if count_elem is None or count_elem.text is None:
            diag = root.find(".//sru:diagnostics", ns)
            if diag is not None:
                logging.error(
                    "SRU diagnostics: %s",
                    ET.tostring(diag, encoding="unicode"),
                )
            else:
                logging.error(
                    "SRU response missing <numberOfRecords> element"
                )
            return [], 0

        total_records = int(count_elem.text)

        records = (
            root.findall(".//sru:recordData", ns) if total_records > 0 else []
        )
        return records, total_records
    except (ET.ParseError, AttributeError, ValueError) as e:
        logging.error(f"Failed to parse SRU XML response: {e}")
        return [], 0

def crawl_and_clean_document(url: str) -> str | None:
    """Crawls a given URL and returns its clean text content."""
    try:
        html_bytes = fetch_with_retries(url)
        html_str = html_bytes.decode("utf-8", errors="ignore")
        parser = TextExtractor()
        parser.feed(html_str)
        return parser.get_text()
    except Exception as e:
        logging.warning("Failed to crawl or parse document %s: %s", url, e)
        return None


def push_dataset_to_hf(file_path: str, repo_id: str, token: str):
    """Pushes the given file to a Hugging Face dataset repository using git."""
    if not token:
        logging.warning("No HF_TOKEN provided; skipping upload.")
        return

    tmpdir = tempfile.mkdtemp()
    repo_url = f"https://{token}@huggingface.co/datasets/{repo_id}"

    try:
        subprocess.run(["git", "clone", repo_url, tmpdir], check=True)
        dest = os.path.join(tmpdir, os.path.basename(file_path))
        if os.path.exists(dest):
            with open(dest, "a", encoding="utf-8") as dst, open(file_path, "r", encoding="utf-8") as src:
                dst.write(src.read())
        else:
            shutil.copy(file_path, dest)
        subprocess.run(["git", "add", os.path.basename(file_path)], cwd=tmpdir, check=True)
        subprocess.run(["git", "commit", "-m", "Update dataset"], cwd=tmpdir, check=True)
        subprocess.run(["git", "push"], cwd=tmpdir, check=True)
        logging.info("Successfully pushed data to Hugging Face.")
    except subprocess.CalledProcessError as e:
        logging.error("Failed to push dataset to Hugging Face: %s", e)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def main():
    """Main function to run the crawler and uploader."""
    start_time = datetime.utcnow()
    logging.info("--- Starting CVDR Crawler ---")

    hf_token = os.getenv("HF_TOKEN", "")

    state = load_crawler_state()

    if state.get("completed", False):
        logging.info("Previous crawl was marked as complete. To re-crawl, delete the state file.")
        # If we want to check daily, we might reset the state here.
        # For this implementation, we assume a full crawl is a one-off until reset.
        return

    existing_urls = get_existing_urls_from_hf(hf_token)
    new_records_found = 0
    start_record = state.get("startRecord", 1)

    # Clean up previous output file if it exists
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    try:
        while True:
            # --- GitHub Actions Timeout Check ---
            elapsed_seconds = (datetime.utcnow() - start_time).total_seconds()
            if elapsed_seconds > MAX_RUNTIME_SECONDS:
                logging.warning("Approaching 6-hour timeout. Saving state and exiting.")
                break
            
            # --- Fetch a batch from SRU ---
            params = {**SRU_PARAMS, "startRecord": start_record, "maximumRecords": BATCH_SIZE}
            logging.info(f"Requesting batch from startRecord: {start_record}")
            
            try:
                response_content = fetch_with_retries(SRU_ENDPOINT, params=params)
            except Exception as e:
                logging.error("Failed to fetch SRU batch: %s. Exiting run.", e)
                break

            records, total_records = parse_sru_response(response_content)

            if not records:
                logging.info("No more records found. Crawl is complete.")
                state["completed"] = True
                break

            # --- Process each record in the batch ---
            for record in records:
                # Prefer 'preferredUrl', fallback to 'url'
                url_element = record.find(".//gzd:preferredUrl", ns={"gzd": "http://standaarden.overheid.nl/sru/record/gzd/1.0/"})
                if url_element is None:
                    url_element = record.find(".//gzd:url", ns={"gzd": "http://standaarden.overheid.nl/sru/record/gzd/1.0/"})

                if url_element is None or not url_element.text:
                    logging.warning("Record found without a URL.")
                    continue

                doc_url = url_element.text
                if doc_url in existing_urls:
                    logging.debug(f"Skipping already processed URL: {doc_url}")
                    continue

                content = crawl_and_clean_document(doc_url)
                if content:
                    json_record = {
                        "URL": doc_url,
                        "Content": content,
                        "Source": "CVDR"
                    }
                    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps(json_record, ensure_ascii=False) + "\n")
                    
                    existing_urls.add(doc_url) # Add to set to avoid duplicates within the same run
                    new_records_found += 1
            
            logging.info(f"Processed batch. Found {new_records_found} new records so far in this run.")
            start_record += len(records)
            state["startRecord"] = start_record
            save_crawler_state(state) # Save state after each successful batch

            # Be a good citizen
            time.sleep(1)

    finally:
        # --- Final state saving and upload ---
        save_crawler_state(state)
        logging.info("Crawler finished its run.")

        if new_records_found > 0:
            logging.info(
                f"Found a total of {new_records_found} new records. Pushing to Hugging Face Hub."
            )
            push_dataset_to_hf(OUTPUT_FILE, HF_REPO_ID, hf_token)
        else:
            logging.info("No new records found in this run. Nothing to upload.")

        logging.info("--- CVDR Crawler Finished ---")


if __name__ == "__main__":
    main()
