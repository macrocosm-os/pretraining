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
import requests
import bittensor as bt
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
from datasets import load_dataset
import time


class SubsetFalconLoader(IterableDataset):
    max_pages: int = 968000015

    def __init__(
        self,
        batch_size,
        sequence_length,
        pages: typing.List[int],
        tokenizer: AutoTokenizer,
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
        self.pages = pages
        self.buffer = []
        self.retry_limit = 10  # Number of retries
        self.retry_delay = 5  # Seconds to wait between retries

        for page in self.pages:
            self.fetch_data_for_page(page)

    def fetch_data_for_page(self, page):
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
                    f"Failed to fetch data, retrying. Attempt {attempt}/{self.retry_limit}"
                )
                if attempt < self.retry_limit:
                    time.sleep(self.retry_delay)  # Wait before the next retry
                else:
                    bt.logging.error(
                        "Maximum retry limit reached. Unable to fetch data."
                    )
                    raise

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


class SubsetFineWebEdu2Loader(IterableDataset):
    """
    A custom dataset loader for a subset of the FineWeb Edu Score 2 dataset.

    Args:
        sequence_length (int): The maximum sequence length for tokenization.
        batch_size (int): The batch size for data loading.
        pages (List[int]): A list of page indices to fetch from the dataset.
        tokenizer (AutoTokenizer): The tokenizer to use for tokenization.
        return_attention_mask (bool, optional): Whether to return attention masks along with input IDs. Defaults to False.
    """

    def __init__(
        self,
        sequence_length,
        batch_size,
        pages: typing.List[int],
        tokenizer: AutoTokenizer,
        return_attention_mask: bool = False,
    ):
        # Number of rows, read from https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-score-2
        self.max_pages: int = 11816970552
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.pages = pages
        self.tokenizer = tokenizer
        self.return_attention_mask = return_attention_mask

        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu-score-2", split="train", streaming=True
        )

    def fetch_pages(self):
        """
        Fetches the pages from the dataset based on the specified page indices.

        Returns:
            List[str]: A list of fetched pages.
        """
        fetched_pages = []
        for i, sample in enumerate(self.dataset):
            if i in self.pages:
                fetched_pages.append(sample["text"])
            if len(fetched_pages) == len(self.pages):
                break
        return fetched_pages

    def tokenize_page(self, page: str) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenizes a page using the specified tokenizer.

        Args:
            page (str): The page content to tokenize.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The tokenized input IDs and attention mask.
        """
        # Tokenize the page content with padding and truncation to the specified sequence length
        tokenized = self.tokenizer(
            page,
            truncation=True,
            padding="max_length",
            max_length=self.sequence_length,
            return_tensors="pt",
        )

        # Extract input IDs and attention mask tensors, removing the batch dimension
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        # Ensure the tokenized sequences match the specified sequence length
        if input_ids.size(0) != self.sequence_length:
            if input_ids.size(0) > self.sequence_length:
                # Truncate sequences if they exceed the maximum length
                input_ids = input_ids[: self.sequence_length]
                attention_mask = attention_mask[: self.sequence_length]
            else:
                # Pad sequences if they are shorter than the maximum length
                padding_length = self.sequence_length - input_ids.size(0)
                input_ids = torch.cat(
                    [
                        input_ids,
                        torch.full(
                            (padding_length,),
                            self.tokenizer.pad_token_id,
                            dtype=torch.long,
                        ),
                    ]
                )
                attention_mask = torch.cat(
                    [attention_mask, torch.zeros(padding_length, dtype=torch.long)]
                )

        return input_ids, attention_mask

    def __iter__(self):
        fetched_pages = self.fetch_pages()
        for page in fetched_pages:
            input_ids, attention_mask = self.tokenize_page(page)
            if self.return_attention_mask:
                yield input_ids, attention_mask
            else:
                yield input_ids
