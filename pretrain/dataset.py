# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import typing
import random
import time
import requests
import numpy as np
import bittensor as bt
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
from datasets import load_dataset
from pprint import pprint

import os
from dotenv import load_dotenv  
load_dotenv()


class SubsetLoader(IterableDataset):
    """
    Base class for data-specific subset loader classes.

    # TODO: Make this class abstract
    """

    def __init__(
        self,
        batch_size=None,
        sequence_length=None,
        num_pages=None,
        tokenizer: AutoTokenizer = None,
        pack_samples: bool = False,
        random_seed: typing.Optional[int] = None,
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_pages = num_pages
        self.tokenizer = tokenizer
        self.pack_samples = pack_samples

        # Initialize with seed if provided.
        if random_seed is not None:
            random.seed(random_seed)

        self.num_rows_per_page = 100
        self.duplicate_page_threshold = 100

        # Buffer to hold pages loaded from the api
        self.buffer = []

        # Buffer to hold pages already loaded into a batch
        self.used_buffer = []

        # Buffer to hold padded pages
        self.padded_buffer = []

    def fetch_data_for_pages(self, pages):
        """
        Set the pages to be used to fill the buffer. Then fetch the page data
        to the buffer.
        """

        self.pages = pages

        # Empty the buffer if it is not.
        self.buffer = []

        for page in self.pages:
            self._fetch_data_for_page(page)

    def _get_pad_size(self, input_ids):
        """
        Get the number of tokens to be padded to the sample to match
        the max allowed sequence length.
        If sample packing is activated, then return 1
        """

        if self.pack_samples:
            return 1

        sample_size = len(input_ids)

        remainder = sample_size % self.sequence_length
        pad_size = self.sequence_length - remainder

        # Apply modulo again to guarantee a pad size of 0 if remainder is 0
        pad_size = pad_size % self.sequence_length

        return pad_size

    def _refill_padded_buffer(self):
        """
        This methods pulls one page from `self.buffer`, pads it and pushs
        it to the `self.padded_buffer`.
        """

        while self.buffer and len(self.padded_buffer) < self.sequence_length:

            input_ids = []

            # search for EOS token index and cut the buffer at it.
            EOS_index = self.buffer.index(self.tokenizer.eos_token_id)
            input_ids = self.buffer[: EOS_index + 1]
            self.buffer = self.buffer[EOS_index + 1 :]

            self.used_buffer += input_ids

            # Add to padded buffer without the EOS token.
            self.padded_buffer += input_ids[:-1]

            # Pad
            self.padded_buffer += [self.tokenizer.eos_token_id] * self._get_pad_size(
                input_ids=input_ids[:-1]
            )

    def __iter__(self):
        self.buffer = self.used_buffer + self.buffer
        self.padded_buffer = []

        # Pad and prepare one page for batching
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
    base_url: str = "https://datasets-server.huggingface.co/rows"

    def __init__(
            self,
            batch_size,
            sequence_length,
            num_pages=None,
            tokenizer: AutoTokenizer=None,
            pack_samples: bool=True,
    ):
        super().__init__(batch_size,
                         sequence_length,
                         num_pages,
                         tokenizer,
                         pack_samples)

        self.params = {
            "dataset": self.name,
            "config": "pes2ov2",
            "split": "train",
        }

        self.retry_limit = 10  # Number of retries
        self.retry_delay = 5  # Seconds to wait between retries

        # Fetch pages only if the number of pages is specified
        if self.num_pages:
            pages = self._sample_pages()
            self.fetch_data_for_pages(pages)

    def _fetch_data_for_page(self, page):
        self.params["offset"] = page
        self.params["limit"] = self.num_rows_per_page
        attempt = 0
        while attempt < self.retry_limit:
            try:
                # Add authorization header
                response = requests.get(self.base_url, params=self.params)
                response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
                for row in response.json()["rows"]:
                    content = row["row"]["text"]
                    input_ids = self.tokenizer(content, truncation=True)["input_ids"]
                    self.buffer += input_ids
                    self.buffer += [self.tokenizer.eos_token_id]

                break  # If the request was successful, break out of the retry loop
            except requests.exceptions.RequestException as e:
                attempt += 1
                bt.logging.warning(
                    f"Failed to fetch data for page {page}, retrying. Attempt {attempt}/{self.retry_limit}"
                )
                if attempt < self.retry_limit:
                    time.sleep(self.retry_delay)  # Wait before the next retry
                else:
                    bt.logging.error(
                        "Maximum retry limit reached. Unable to fetch data."
                    )
                    raise

    def _sample_pages(self):
        """
        Randomly sample pages to be used in validation
        """
        pages = [
            random.randint(1, self.max_pages)
            for _ in range(self.num_pages)
        ]

        return pages


    def get_page_names(self):
        """
        This is a utility function that returns the page names that were used.
        Each page as a single string instead of a tuple
        """
        page_names = []

        if hasattr(self, 'pages'):
            page_names = self.pages

        return page_names


class SubsetStackV1DedupLoader(SubsetLoader):
    max_pages: int = 236655813
    name: str = "bigcode/the-stack-dedup"
    base_url: str = "https://datasets-server.huggingface.co/rows"

    def __init__(
            self,
            batch_size,
            sequence_length,
            num_pages=None,
            tokenizer: AutoTokenizer=None,
            pack_samples: bool=False,
    ):
        super().__init__(batch_size,
                         sequence_length,
                         num_pages,
                         tokenizer,
                         pack_samples)

        self.params = {
            "dataset": self.name,
            "config": "default",
            "split": "train",
        }

        self.retry_limit = 10  # Number of retries
        self.retry_delay = 5  # Seconds to wait between retries

        # Get HF token from environment
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("HF_TOKEN environment variable not found")

        # Fetch pages only if the number of pages is specified
        if self.num_pages:
            pages = self._sample_pages()
            self.fetch_data_for_pages(pages)

    def _fetch_data_for_page(self, page):
        self.params["offset"] = page
        self.params["limit"] = self.num_rows_per_page
        attempt = 0
        while attempt < self.retry_limit:
            try:
                # Add authorization header
                headers = {"Authorization": f"Bearer {self.hf_token}"}
                response = requests.get(self.base_url, params=self.params, headers=headers)
                response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
                for row in response.json()["rows"]:
                    content = row["row"]["content"]
                    input_ids = self.tokenizer(content, truncation=True)["input_ids"]
                    self.buffer += input_ids
                    self.buffer += [self.tokenizer.eos_token_id]

                break  # If the request was successful, break out of the retry loop
            except requests.exceptions.RequestException as e:
                attempt += 1
                bt.logging.warning(
                    f"Failed to fetch data for page {page}, retrying. Attempt {attempt}/{self.retry_limit}"
                )
                if attempt < self.retry_limit:
                    time.sleep(self.retry_delay)  # Wait before the next retry
                else:
                    bt.logging.error(
                        "Maximum retry limit reached. Unable to fetch data."
                    )
                    raise

    def _sample_pages(self):
        """
        Randomly sample pages to be used in validation
        """
        pages = [
            random.randint(1, self.max_pages)
            for _ in range(self.num_pages)
        ]

        return pages


    def get_page_names(self):
        """
        This is a utility function that returns the page names that were used.
        Each page as a single string instead of a tuple
        """
        page_names = []

        if hasattr(self, 'pages'):
            page_names = self.pages

        return page_names


class SubsetFineWebEdu2Loader(SubsetLoader):

    name: str = "HuggingFaceFW/fineweb-edu-score-2"
    rows_base_url: str = "https://datasets-server.huggingface.co/rows"
    size_base_url: str = "https://datasets-server.huggingface.co/size"

    retry_limit: int = 10  # Number of retries
    retry_delay: int = 5  # Seconds to wait between retries

    def __init__(
        self,
        batch_size=None,
        sequence_length=None,
        num_pages=None,
        tokenizer: AutoTokenizer = None,
        pack_samples: bool = False,
        random_seed: typing.Optional[int] = None,
    ):
        super().__init__(
            batch_size, sequence_length, num_pages, tokenizer, pack_samples, random_seed
        )

        # Get the dataset configs and their row sizes
        self.configs_data = self.fetch_dataset_configs()

        # We first need to fetch the data and fill the loader buffer.
        # Since some sample files are broken, we first try to find `num_pages`
        # responsive samples, then we add them to the found pages `self.pages`
        if self.num_pages:
            self._fetch_data_to_buffer(self.num_pages)

    def _fetch_data_to_buffer(self, num_pages):
        """
        Randomly sample unique pages and add their data to the buffer.
        If a page is inaccessible, another one is sampled.
        this method sets the `pages` property
        """

        self.pages = []
        attempts = 0
        duplicates = 0

        # Choose a consistent initial offset for the random pages so we do not overlap on each page get.
        initial_offset = random.randint(0, self.num_rows_per_page - 1)

        while len(self.pages) < num_pages:

            # randomly sample one page
            page = self.get_random_pages(num_pages=1, initial_offset=initial_offset)[0]

            # skip the page if we already have it
            if page in self.pages:
                duplicates += 1
                if duplicates >= self.duplicate_page_threshold:
                    bt.logging.debug(
                        f"Hit duplicate page threshold of {self.duplicate_page_threshold}. Stopping early at: {len(self.pages)} pages."
                    )
                    break
                else:
                    continue

            config_name, page_row_start, split = page

            # Create the request parameters
            params = dict(
                dataset=self.name,
                config=config_name,
                split=split,
                offset=page_row_start,
                limit=self.num_rows_per_page,
            )

            try:
                response = requests.get(self.rows_base_url, params=params)

                response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code

                # Add the page since the request was successful
                self.pages.append(page)

                for row in response.json()["rows"]:
                    content = row["row"]["text"]

                    # get the tokenized and encoded sample
                    input_ids = self.tokenizer(content, truncation=True)["input_ids"]
                    self.buffer += input_ids
                    self.buffer += [self.tokenizer.eos_token_id]

                response.close()

            except requests.exceptions.RequestException as e:

                response.close()
                attempts += 1
                bt.logging.warning(
                    f"Failed to fetch data, retrying with a newly sampled page. Attempt {attempts}/{self.retry_limit * num_pages}"
                )
                if attempts < num_pages * self.retry_limit:
                    pass

                else:
                    bt.logging.error(
                        "Maximum retry limit reached. Unable to fetch data."
                    )
                    raise

    def fetch_data_to_rows(self, num_pages):

        # This explicitly does not set the pages property, simply keeping track for duplicates.
        # We also use a set here as the order does not matter.
        downloaded_pages = set()
        rows = []
        attempts = 0
        duplicates = 0
        # Choose a consistent initial offset for the random pages so we do not overlap on each page get.
        initial_offset = random.randint(0, self.num_rows_per_page - 1)

        while len(downloaded_pages) < num_pages:

            # randomly sample one page
            page = self.get_random_pages(num_pages=1, initial_offset=initial_offset)[0]

            # skip the page if we already have it
            if page in downloaded_pages:
                duplicates += 1
                if duplicates >= self.duplicate_page_threshold:
                    bt.logging.debug(
                        f"Hit duplicate page threshold of {self.duplicate_page_threshold}. Stopping early at: {len(downloaded_pages)} pages."
                    )
                    break
                else:
                    continue

            config_name, page_row_start, split = page

            # Create the request parameters
            params = dict(
                dataset=self.name,
                config=config_name,
                split=split,
                offset=page_row_start,
                limit=self.num_rows_per_page,
            )

            try:
                response = requests.get(self.rows_base_url, params=params)

                response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code

                downloaded_pages.add(page)

                for row in response.json()["rows"]:
                    rows.append(row["row"]["text"])

            except requests.exceptions.RequestException as e:
                attempts += 1
                bt.logging.warning(
                    f"Failed to fetch data, retrying with a newly sampled page. Attempt {attempts}/{self.retry_limit * num_pages}"
                )
                if attempts < num_pages * self.retry_limit:
                    pass

                else:
                    bt.logging.error(
                        "Maximum retry limit reached. Unable to fetch data."
                    )
                    raise

        return rows

    def get_random_pages(self, num_pages, initial_offset):
        """
        Randomly sample num_pages with an intiial offset.
        A page is a row number of a given split of a given dataset dump offset by num_rows_per_page.
        """
        pages = []

        for _ in range(num_pages):

            # Choose a random config.
            config_name = random.choice(list(self.configs_data.keys()))

            # Choose a random page start.
            # We do so by chunking the rows of the config data into N pages of length num_rows_per_page.
            # We remove the initial offset from the total rows in doing this calculation to ensure we don't go over.
            data_row_count = self.configs_data[config_name]["num_rows"] - initial_offset
            # Add 1 to the row count as we are 0 indexed.
            data_page_count = (data_row_count + 1) // self.num_rows_per_page
            # Select a random page start by taking the randomly selected page and multiplying by num_rows_per_page.
            selected_page_start = initial_offset + (
                random.randint(0, data_page_count - 1) * self.num_rows_per_page
            )

            split = self.configs_data[config_name]["split"]

            pages.append((config_name, selected_page_start, split))

        return pages

    def get_page_names(self):
        """
        This is a utility function that returns the page names that were used.
        Each page as a single string instead of a tuple
        """

        page_names = []

        if hasattr(self, "pages"):
            page_names = [
                f"{cfg_name}_{num_rows}_{split}"
                for cfg_name, num_rows, split in self.pages
            ]

        return page_names

    def fetch_dataset_configs(self) -> typing.Dict[str, typing.Dict]:
        """
        Fetch the different dump names, aka configs, aka samples, of the
        dataset.
        The returned value is a dictionary with dump names as keys and
        a dict of the number of rows and the split as values.
        """
        # Request parameters
        params = dict(dataset=self.name)

        attempt = 0
        while attempt < self.retry_limit:
            try:
                response = requests.get(self.size_base_url, params=params)
                response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code

                # Extract the configs dict
                configs_dict = response.json()["size"]["splits"]

                # Now create a dict with config names (except 'default') as
                # keys, and the number of rows as values
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
                bt.logging.warning(
                    f"Failed to fetch dataset configs, retrying. Attempt {attempt}/{self.retry_limit}"
                )
                if attempt < self.retry_limit:
                    time.sleep(self.retry_delay)  # Wait before the next retry
                else:
                    bt.logging.error(
                        "Maximum retry limit reached. Unable to fetch data."
                    )
                    raise

    def _fetch_data_for_page(self, page):

        retry_limit = 10

        attempt = 0
        while attempt < retry_limit:
            config_name, page, split = page

            # Create the request parameters
            params = dict(
                dataset=self.name,
                config=config_name,
                split=split,
                offset=page,
                limit=self.num_rows_per_page,
            )

            try:

                response = requests.get(self.rows_base_url, params=params)

                response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code

                for row in response.json()["rows"]:
                    content = row["row"]["text"]
                    input_ids = self.tokenizer(content, truncation=True)["input_ids"]
                    self.buffer += input_ids
                    self.buffer += [self.tokenizer.eos_token_id]

                break  # If the request was successful, break out of the retry loop

            except requests.exceptions.RequestException as e:
                attempt += 1
                bt.logging.warning(
                    f"Failed to fetch data for page {page}, retrying. Attempt {attempt}/{self.retry_limit}"
                )
                if attempt < self.retry_limit:
                    time.sleep(self.retry_delay)  # Wait before the next retry
                else:
                    bt.logging.error(
                        "Maximum retry limit reached. Unable to fetch data."
                    )
                    raise


class SubsetFalconLoader(SubsetLoader):
    max_pages: int = 968000015
    name: str = "tiiuae/falcon-refinedweb"
    base_url: str = "https://datasets-server.huggingface.co/rows"

    def __init__(
        self,
        batch_size,
        sequence_length,
        num_pages=None,
        tokenizer: AutoTokenizer = None,
        pack_samples: bool = False,
        random_seed: typing.Optional[int] = None,
    ):
        super().__init__(
            batch_size, sequence_length, num_pages, tokenizer, pack_samples, random_seed
        )

        self.params = {
            "dataset": self.name,
            "config": "default",
            "split": "train",
        }

        self.retry_limit = 10  # Number of retries
        self.retry_delay = 5  # Seconds to wait between retries

        # Fetch pages only if the number of pages is specified
        if self.num_pages:
            pages = self._sample_pages()
            self.fetch_data_for_pages(pages)

    def _fetch_data_for_page(self, page):
        self.params["offset"] = page
        self.params["limit"] = self.num_rows_per_page
        attempt = 0
        while attempt < self.retry_limit:
            try:
                response = requests.get(self.base_url, params=self.params)
                response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
                for row in response.json()["rows"]:
                    content = row["row"]["content"]
                    input_ids = self.tokenizer(content, truncation=True)["input_ids"]
                    self.buffer += input_ids
                    self.buffer += [self.tokenizer.eos_token_id]

                break  # If the request was successful, break out of the retry loop
            except requests.exceptions.RequestException as e:
                attempt += 1
                bt.logging.warning(
                    f"Failed to fetch data for page {page}, retrying. Attempt {attempt}/{self.retry_limit}"
                )
                if attempt < self.retry_limit:
                    time.sleep(self.retry_delay)  # Wait before the next retry
                else:
                    bt.logging.error(
                        "Maximum retry limit reached. Unable to fetch data."
                    )
                    raise

    def _sample_pages(self):
        """
        Randomly sample pages to be used in validation
        """
        pages = [random.randint(1, self.max_pages) for _ in range(self.num_pages)]

        return pages

    def get_page_names(self):
        """
        This is a utility function that returns the page names that were used.
        Each page as a single string instead of a tuple
        """
        page_names = []

        if hasattr(self, "pages"):
            page_names = self.pages

        return page_names
