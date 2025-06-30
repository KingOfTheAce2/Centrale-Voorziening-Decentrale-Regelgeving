# cvdr_crawler.py
import os
import time
import json
import logging
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

from bs4 import BeautifulSoup
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, HfFolder
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# --- Configuration ---
SRU_ENDPOINT = "https://zoekservice.overheid.nl/sru/Search"
SRU_PARAMS = {
    "x-connection": "cvdr",
    "operation": "searchRetrieve",
    "version": "2.0",
    "query": 'cql.textAndIndexes="*"'
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

def setup_requests_session() -> requests.Session:
    """Sets up a requests session with retry logic."""
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

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

def get_existing_urls_from_hf() -> set:
    """
    Downloads the 'URL' column from the Hugging Face dataset to prevent duplicates.
    Uses streaming to avoid high memory usage on large datasets.
    """
    logging.info(f"Fetching existing URLs from Hugging Face repo: {HF_REPO_ID}")
    existing_urls = set()
    try:
        # Use streaming to efficiently get URLs without downloading the whole dataset
        dataset = load_dataset(HF_REPO_ID, split="train", streaming=True)
        for record in dataset:
            if "URL" in record:
                existing_urls.add(record["URL"])
        logging.info(f"Found {len(existing_urls)} existing URLs in the dataset.")
    except Exception as e:
        logging.warning(
            f"Could not load existing dataset from Hugging Face (maybe it's empty or new?): {e}"
        )
    return existing_urls

def parse_sru_response(xml_content: str) -> (list, int):
    """Parses the SRU XML response to extract record data and total record count."""
    ns = {
        "sru": "http://docs.oasis-open.org/ns/search-ws/sruResponse",
        "gzd": "http://standaarden.overheid.nl/sru/record/gzd/1.0/",
    }
    try:
        root = ET.fromstring(xml_content)
        total_records = int(root.find("sru:numberOfRecords", ns).text)
        records = root.findall(".//sru:recordData", ns)
        return records, total_records
    except (ET.ParseError, AttributeError) as e:
        logging.error(f"Failed to parse SRU XML response: {e}")
        return [], 0

def crawl_and_clean_document(session: requests.Session, url: str) -> str | None:
    """Crawls a given URL and returns its clean text content."""
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        # Ensure content is decoded correctly
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, "lxml")
        
        # Strip all script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Get text, use a separator to preserve context, and clean up whitespace
        text = soup.get_text(separator=' ', strip=True)
        return ' '.join(text.split()) # Normalize whitespace
    except requests.RequestException as e:
        logging.warning(f"Failed to crawl document URL {url}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while cleaning {url}: {e}")
    return None

def main():
    """Main function to run the crawler and uploader."""
    start_time = datetime.utcnow()
    logging.info("--- Starting CVDR Crawler ---")

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logging.error("HF_TOKEN environment variable not set. Aborting.")
        return
    HfFolder.save_token(hf_token)

    session = setup_requests_session()
    state = load_crawler_state()

    if state.get("completed", False):
        logging.info("Previous crawl was marked as complete. To re-crawl, delete the state file.")
        # If we want to check daily, we might reset the state here.
        # For this implementation, we assume a full crawl is a one-off until reset.
        return

    existing_urls = get_existing_urls_from_hf()
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
                response = session.get(SRU_ENDPOINT, params=params)
                response.raise_for_status()
            except requests.RequestException as e:
                logging.error(f"Failed to fetch SRU batch: {e}. Exiting run.")
                break
                
            records, total_records = parse_sru_response(response.content)

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

                content = crawl_and_clean_document(session, doc_url)
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
            logging.info(f"Found a total of {new_records_found} new records. Pushing to Hugging Face Hub.")
            try:
                # Load the newly created JSONL file
                new_dataset = load_dataset("json", data_files=OUTPUT_FILE, split="train")
                
                # Push to hub, appending to the existing dataset
                new_dataset.push_to_hub(HF_REPO_ID, private=False) # `private=False` makes it a public repo
                logging.info("Successfully pushed new data to Hugging Face Hub.")
            except Exception as e:
                logging.error(f"Failed to push to Hugging Face Hub: {e}")
        else:
            logging.info("No new records found in this run. Nothing to upload.")

        logging.info("--- CVDR Crawler Finished ---")


if __name__ == "__main__":
    main()
