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

import torch
import typing
import random
import time
import requests
import bittensor as bt
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
from datasets import load_dataset
from pprint import pprint

class SubsetFineWebEdu2Loader(IterableDataset):

    name: str = "HuggingFaceFW/fineweb-edu-score-2"
    rows_base_url: str = "https://datasets-server.huggingface.co/rows"
    size_base_url: str = "https://datasets-server.huggingface.co/size"

    retry_limit: int = 10  # Number of retries
    retry_delay: int = 5  # Seconds to wait between retries

    def __init__(
        self,
        sequence_length=None,
        num_pages=None,
        tokenizer: AutoTokenizer = None,
        batch_size=1,
    ):
        self.sequence_length = sequence_length
        self.num_pages = num_pages
        self.num_rows_per_page = 100
        self.tokenizer = tokenizer
        self.batch_size = batch_size

        # Use a default PAD token ID if the tokenizer does not provide one
        # (e.g. GPT-2 does not have a padding token)
        # If the tokenizer does not have a padding token, use the EOS token.

        self.pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        self.buffer = []

        # Get the dataset configs and their row sizes
        self.configs_data = self.fetch_dataset_configs()

        # We first need to fetch the data and fill the loader buffer.
        # Since some sample files are broken, we first try to find `num_pages`
        # responsive samples, then we add them to the found pages `self.pages`
        if self.num_pages:
            self._fetch_data_to_buffer(self.num_pages)

    def _fetch_data_to_buffer(self, num_pages):
        """
        Randomly sample pages and add their data to the buffer.
        If a page is inaccessible, another one is sampled.
        this method sets the `pages` property
        """

        self.pages = []
        attempts = 0

        while len(self.pages) < num_pages:

            # randomly sample one page
            config_name, page, split = self.get_random_pages(num_pages=1)[0]

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

                # Add the page since the request was successful
                self.pages.append((config_name, page, split))

                page_buffer = []
                for row in response.json()["rows"]:
                    content = row["row"]["text"]
                    page_buffer += self.tokenizer(content, truncation=True)["input_ids"]
                    page_buffer += [self.tokenizer.eos_token_id]

                self.buffer.append(page_buffer)

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

                page_buffer = []
                for row in response.json()["rows"]:
                    content = row["row"]["text"]
                    page_buffer += self.tokenizer(content, truncation=True)["input_ids"]
                    page_buffer += [self.tokenizer.eos_token_id]

                self.buffer.append(page_buffer)
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

    def fetch_data_to_rows(self, num_pages):

        rows = []
        attempts = 0
        num_downloaded_pages = 0

        while num_downloaded_pages < num_pages:

            # randomly sample one page
            config_name, page, split = self.get_random_pages(num_pages=1)[0]

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

                num_downloaded_pages += 1

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

    def get_random_pages(self, num_pages):
        """
        Randomly sample one page.
        A page is a row number of a given split of a given dataset dump.
        """
        pages = []

        for _ in range(num_pages):

            # Choose a random config
            config_name = random.choice(list(self.configs_data.keys()))

            # Choose a random page (row)
            page = random.randint(
                0,
                self.configs_data[config_name]["num_rows"] - 1 - self.num_rows_per_page,
            )

            split = self.configs_data[config_name]["split"]

            pages.append((config_name, page, split))

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

    def __iter__(self):
        while self.buffer:
            batch = []
            while len(batch) < self.batch_size:
                if not self.buffer:
                    break
                page_buffer = self.buffer[0]
                if len(page_buffer) <= self.sequence_length:
                    batch.append(
                        page_buffer
                        + [self.pad_token_id]
                        * (self.sequence_length - len(page_buffer))
                    )
                    self.buffer.pop(0)
                else:
                    batch.append(page_buffer[: self.sequence_length])
                    self.buffer[0] = page_buffer[self.sequence_length :]
            if batch:
                yield torch.tensor(batch)

    def __next__(self):
        if not self.buffer:
            raise StopIteration
        batch = []
        while len(batch) < self.batch_size:
            if not self.buffer:
                break
            page_buffer = self.buffer[0]
            if len(page_buffer) <= self.sequence_length:
                batch.append(
                    page_buffer
                    + [self.pad_token_id] * (self.sequence_length - len(page_buffer))
                )
                self.buffer.pop(0)
            else:
                batch.append(page_buffer[: self.sequence_length])
                self.buffer[0] = page_buffer[self.sequence_length :]
        if batch:
            return torch.tensor(batch)
        else:
            raise StopIteration

class SubsetFalconLoader(IterableDataset):
    max_pages: int = 968000015

    def __init__(
        self,
        batch_size,
        sequence_length,
        num_pages=None,
        tokenizer: AutoTokenizer=None,
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_rows_per_page = 100
        self.tokenizer = tokenizer
        self.base_url = "https://datasets-server.huggingface.co/rows"
        self.params = {
            "dataset": "tiiuae/falcon-refinedweb",
            "config": "default",
            "split": "train",
        }
        self.num_pages = num_pages
        self.buffer = []
        self.retry_limit = 10  # Number of retries
        self.retry_delay = 5  # Seconds to wait between retries


        # Fetch pages only if the number of pages is specified
        if self.num_pages:
            pages = self._sample_pages()            
            self.fetch_data_for_pages(pages)

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
                    self.buffer += self.tokenizer(content, truncation=True)["input_ids"]
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
    
    def __iter__(self):
        while len(self.buffer) >= self.sequence_length * self.batch_size:
            batch = []
            for _ in range(self.batch_size):
                batch.append(torch.tensor(self.buffer[: self.sequence_length]))
                self.buffer = self.buffer[self.sequence_length :]
            yield torch.stack(batch)

    def __next__(self):
        batch = []
        for _ in range(self.batch_size):
            batch.append(torch.tensor(self.buffer[: self.sequence_length]))
            self.buffer = self.buffer[self.sequence_length :]
        return torch.stack(batch)