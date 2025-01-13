import os
import random
import time
import typing

import bittensor as bt
import boto3
import numpy as np
import requests
import smart_open
from dotenv import load_dotenv
from taoverse.utilities import logging
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

load_dotenv()


class SubsetLoader(IterableDataset):
    """Base class for data-specific subset loader classes."""

    name: str = None  # Dataset name
    rows_base_url: str = "https://datasets-server.huggingface.co/rows"
    size_base_url: str = "https://datasets-server.huggingface.co/size"
    max_pages: int = None

    def __init__(
        self,
        batch_size=None,
        sequence_length=None,
        num_samples=None,
        tokenizer: AutoTokenizer = None,
        random_seed: typing.Optional[int] = None,
        config: str = "default",
        split: str = "train",
        requires_auth: bool = False,
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_samples = num_samples
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        self.requires_auth = requires_auth

        # Initialize with seed if provided
        if random_seed is not None:
            random.seed(random_seed)

        self.num_rows_per_page = 50
        self.duplicate_page_threshold = 100
        self.retry_limit = 10
        self.retry_delay = 5

        # List of pages used to fill buffers.
        self.pages = []

        # Buffers
        self.buffer = []
        self.used_buffer = []
        self.padded_buffer = []

        # Get HF token if needed
        self.hf_token = None
        if self.requires_auth:
            self.hf_token = os.getenv("HF_TOKEN")
            if not self.hf_token:
                raise ValueError("HF_TOKEN environment variable not found")

        # Initialize request params
        self.params = self._get_default_params()

        # Fetch pages if specified (individually retrying each page)
        # If we fail to fill the whole buffer, then try to fill again (without wiping previous data)
        # If we fail to fill the whole buffer after retry limit fetch attempts we fail overall.
        fetch_attempt = 0

        if self.num_samples:
            while fetch_attempt <= self.retry_limit:
                fetch_attempt += 1
                try:
                    # Try to fill the buffer in one go.
                    self._fill_buffer()
                    # Technically it is fine to keep calling _fill_buffer() if it is already done but we break early.
                    break
                except Exception:
                    # We already log and swallow the specific exception as part of _fill_buffer() so pass here.
                    pass

            # If we hit retry limit and did not finish fetching the required number of samples throw.
            # TODO: Consider just logging and continuing if we hit ~90% of desired samples or similar.
            if self._get_loaded_sample_count() < self.num_samples:
                raise ValueError(
                    f"Maximum retry limit for fetching data reached. Only loaded {self._get_loaded_sample_count()}/{self.num_samples} samples."
                )

    def _get_default_params(self):
        """Get default request parameters. Override if needed."""
        return {
            "dataset": self.name,
            "config": self.config,
            "split": self.split,
        }

    def _get_request_headers(self):
        """Get request headers. Override if needed."""
        headers = {}
        if self.requires_auth:
            headers["Authorization"] = f"Bearer {self.hf_token}"
        return headers

    def _get_loaded_sample_count(self):
        """Checks how many samples worth of data have been loaded into the buffer."""
        return len(self.buffer) // self.sequence_length

    def _fill_buffer(self):
        """Fills buffer with data based on loader type."""
        # Note: This will can overfill the buffer by up to ~1 page worth of content. Consider truncating.
        if hasattr(self, "fetch_dataset_configs"):
            # For FineWebEdu2 style loaders
            self.configs_data = self.fetch_dataset_configs()
            self._fetch_data_to_buffer(self.num_samples)
        else:
            # For simple page-based loaders
            # Try to get data one page at a time with retries per page.
            while self._get_loaded_sample_count() < self.num_samples:
                self._fetch_data_for_page(self._random_page())

    def _fetch_data_for_page(self, page):
        """Fetch data for a single page"""
        # Handle different page types (tuple vs int)
        if isinstance(page, tuple):
            config_name, page_num, split = page
            self.params.update(
                {
                    "config": config_name,
                    "split": split,
                    "offset": page_num,
                }
            )
        else:
            self.params["offset"] = page

        self.params["length"] = self.num_rows_per_page

        attempt = 0
        while attempt < self.retry_limit:
            try:
                response = requests.get(
                    self.rows_base_url,
                    params=self.params,
                    headers=self._get_request_headers(),
                )
                response.raise_for_status()

                # Add this to the list of pages that the buffer was filled from.
                self.pages.append(page)

                for row in response.json()["rows"]:
                    content = self._get_content_from_row(row)
                    input_ids = self.tokenizer(content, truncation=True)["input_ids"]
                    self.buffer += input_ids
                    self.buffer += [self.tokenizer.eos_token_id]

                break

            except requests.exceptions.RequestException as e:
                attempt += 1
                logging.warning(
                    f"Failed to fetch data for page {page}, retrying. Attempt {attempt}/{self.retry_limit}"
                )
                if attempt < self.retry_limit:
                    time.sleep(self.retry_delay)
                else:
                    logging.error("Maximum retry limit reached. Unable to fetch data.")
                    raise

    def _get_content_from_row(self, row):
        """Extract content from row based on dataset format. Override if needed."""
        return row["row"].get("text", row["row"].get("content"))

    def _random_page(self):
        """Select a random page. Override for custom sampling logic."""
        # Note: this does not prevent us from selecting the same page twice.
        return random.randint(1, self.max_pages)

    def get_page_names(self):
        """Get page names in consistent format"""
        if not hasattr(self, "pages"):
            return []

        if isinstance(self.pages[0], tuple):
            return [
                f"{cfg_name}_{num_rows}_{split}"
                for cfg_name, num_rows, split in self.pages
            ]
        return self.pages

    def _refill_padded_buffer(self):
        """Refill the padded buffer from the main buffer."""
        while self.buffer and len(self.padded_buffer) < self.sequence_length:
            input_ids = []
            EOS_index = self.buffer.index(self.tokenizer.eos_token_id)
            input_ids = self.buffer[: EOS_index + 1]
            self.buffer = self.buffer[EOS_index + 1 :]
            self.used_buffer += input_ids
            self.padded_buffer += input_ids[:-1]
            self.padded_buffer += [self.tokenizer.eos_token_id]

    def __iter__(self):
        self.buffer = self.used_buffer + self.buffer
        self.padded_buffer = []
        self._refill_padded_buffer()
        return self

    def __next__(self):
        batch = []
        while len(self.padded_buffer) >= self.sequence_length:
            batch.append(self.padded_buffer[: self.sequence_length])
            self.padded_buffer = self.padded_buffer[self.sequence_length :]
            self._refill_padded_buffer()
            if len(batch) == self.batch_size:
                return np.stack(batch)
        raise StopIteration


class SubsetPes2oXLoader(SubsetLoader):
    max_pages: int = 8242000
    name: str = "laion/Pes2oX-fulltext"

    def __init__(self, **kwargs):
        super().__init__(config="pes2ov2", **kwargs)


class SubsetStackV1DedupLoader(SubsetLoader):
    max_pages: int = 236655813
    name: str = "bigcode/the-stack-dedup"

    def __init__(self, **kwargs):
        super().__init__(requires_auth=True, **kwargs)


class SubsetStackV2DedupLoader(SubsetLoader):
    max_pages: int = 5_451_114_734
    name: str = "bigcode/the-stack-v2-dedup"

    def __init__(self, **kwargs):

        # Create an AWS S3 session to enable reading data
        session = boto3.Session(
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )

        self.s3_sess = session.client("s3")

        super().__init__(requires_auth=True, **kwargs)

    def _download_row_content(self, blob_id, src_encoding):
        """Download the row content from S3."""

        s3_url = f"https://softwareheritage.s3.amazonaws.com/content/{blob_id}"

        with smart_open.open(
            s3_url, "rb", compression=".gz", transport_params={"client": self.s3_sess}
        ) as fin:
            content = fin.read().decode(src_encoding)

        return content

    def _get_content_from_row(self, row):
        """Extract row content by downloading from S3"""

        content = self._download_row_content(
            row["row"]["blob_id"], row["row"]["src_encoding"]
        )
        return content


class SubsetFalconLoader(SubsetLoader):
    max_pages: int = 968000015
    name: str = "tiiuae/falcon-refinedweb"


class SubsetFineWebEdu2Loader(SubsetLoader):
    name: str = "HuggingFaceFW/fineweb-edu-score-2"

    def fetch_dataset_configs(self) -> typing.Dict[str, typing.Dict]:
        """
        Fetch dataset configs and their metadata.
        Returns a dictionary with config names as keys and metadata as values.
        """
        params = dict(dataset=self.name)

        attempt = 0
        while attempt < self.retry_limit:
            try:
                response = requests.get(self.size_base_url, params=params)
                response.raise_for_status()

                configs_dict = response.json()["size"]["splits"]
                configs_data = {
                    entry["config"]: {
                        "num_rows": entry["num_rows"],
                        "split": entry["split"],
                    }
                    for entry in configs_dict
                    if entry["config"] != "default"
                }

                return configs_data

            except requests.exceptions.RequestException as e:
                attempt += 1
                logging.warning(
                    f"Failed to fetch dataset configs, retrying. Attempt {attempt}/{self.retry_limit}"
                )
                if attempt < self.retry_limit:
                    time.sleep(self.retry_delay)
                else:
                    logging.error("Maximum retry limit reached. Unable to fetch data.")
                    raise

    def _fetch_data_to_buffer(self, num_samples):
        """Fetch data to buffer with support for multiple configs."""
        attempts = 0
        duplicates = 0
        initial_offset = random.randint(0, self.num_rows_per_page - 1)

        while self._get_loaded_sample_count() < num_samples:
            page = self.get_random_pages(num_pages=1, initial_offset=initial_offset)[0]

            if page in self.pages:
                duplicates += 1
                if duplicates >= self.duplicate_page_threshold:
                    logging.debug(
                        f"Hit duplicate page threshold of {self.duplicate_page_threshold}. "
                        f"Stopping early at: {len(self.pages)} pages."
                    )
                    break
                continue

            config_name, page_row_start, split = page
            params = {
                "dataset": self.name,
                "config": config_name,
                "split": split,
                "offset": page_row_start,
                "length": self.num_rows_per_page,
            }

            try:
                response = requests.get(self.rows_base_url, params=params)
                response.raise_for_status()
                self.pages.append(page)

                for row in response.json()["rows"]:
                    content = row["row"]["text"]
                    input_ids = self.tokenizer(content, truncation=True)["input_ids"]
                    self.buffer += input_ids
                    self.buffer += [self.tokenizer.eos_token_id]

            except requests.exceptions.RequestException as e:
                attempts += 1
                logging.warning(
                    f"Failed to fetch data, retrying. Attempt {attempts}/{self.retry_limit}"
                )
                if attempts >= self.retry_limit:
                    logging.error("Maximum retry limit reached. Unable to fetch data.")
                    raise

    def get_random_pages(self, num_pages, initial_offset):
        """Get random pages across different configs."""
        pages = []
        for _ in range(num_pages):
            config_name = random.choice(list(self.configs_data.keys()))
            data_row_count = self.configs_data[config_name]["num_rows"] - initial_offset
            data_page_count = (data_row_count + 1) // self.num_rows_per_page
            selected_page_start = initial_offset + (
                random.randint(0, data_page_count - 1) * self.num_rows_per_page
            )
            split = self.configs_data[config_name]["split"]
            pages.append((config_name, selected_page_start, split))
        return pages
