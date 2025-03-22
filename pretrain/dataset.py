import os
import random
import time
import typing


import bittensor as bt
import boto3
import numpy as np
import requests
import smart_open

from io import BytesIO
from dotenv import load_dotenv

import torch
import torchaudio

from taoverse.utilities import logging
from taoverse.model.tts.utils.e2tts import convert_char_to_pinyin
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

from pydub import AudioSegment

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
        num_pages=None,
        tokenizer: AutoTokenizer = None,
        pack_samples: bool = True,
        random_seed: typing.Optional[int] = None,
        config: str = "default",
        split: str = "train",
        requires_auth: bool = False,
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_pages = num_pages
        self.tokenizer = tokenizer
        self.pack_samples = pack_samples
        self.config = config
        self.split = split
        self.requires_auth = requires_auth

        # Initialize with seed if provided
        if random_seed is not None:
            random.seed(random_seed)

        self.num_rows_per_page = 50
        self.duplicate_page_threshold = 100
        self.retry_limit = 15
        self.retry_delay = 5

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

        # Fetch pages if specified
        # If the fetched pages are empty, try again until
        # we hit the retry limit.
        fetch_attempt = 1

        if self.num_pages:
            while fetch_attempt < self.retry_limit:
                self._initialize_pages()
                fetch_attempt += 1

                # Exit if the buffer has at least one batch
                if len(self.buffer) >= self.sequence_length:
                    break

                logging.warning(
                    f"All fetched pages seem to be empty or have an extremely low token count. "
                    f"Trying to fetch a new set of pages... (attempt {fetch_attempt}/{self.retry_limit})"
                )

            # If we exhaust all attempts and still don't have enough data, raise an error
            if len(self.buffer) < self.sequence_length:
                raise ValueError(
                    "Maximum retry limit for fetching pages reached. "
                    "All fetched pages seem to be empty or have an extremely low token count."
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

    def _initialize_pages(self):
        """Initialize pages based on loader type"""
        if hasattr(self, "fetch_dataset_configs"):
            # For FineWebEdu2 style loaders
            self.configs_data = self.fetch_dataset_configs()
            self._fetch_data_to_buffer(self.num_pages)
        else:
            # For simple page-based loaders
            pages = self._sample_pages()
            print(pages)
            self.fetch_data_for_pages(pages)

    def fetch_data_for_pages(self, pages):
        """Set the pages and fetch their data to the buffer."""
        self.pages = pages
        self.buffer = []
        for page in self.pages:
            self._fetch_data_for_page(page)

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
                    timeout=15,
                )
                response.raise_for_status()

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

    def _sample_pages(self):
        """Sample random pages. Override for custom sampling logic."""
        return [random.randint(1, self.max_pages) for _ in range(self.num_pages)]

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

    def _get_pad_size(self, input_ids):
        """Get padding size for input tokens."""
        if self.pack_samples:
            return 1

        sample_size = len(input_ids)
        remainder = sample_size % self.sequence_length
        pad_size = self.sequence_length - remainder
        pad_size = pad_size % self.sequence_length
        return pad_size

    def _refill_padded_buffer(self):
        """Refill the padded buffer from the main buffer."""
        while self.buffer and len(self.padded_buffer) < self.sequence_length:
            input_ids = []
            EOS_index = self.buffer.index(self.tokenizer.eos_token_id)
            input_ids = self.buffer[: EOS_index + 1]
            self.buffer = self.buffer[EOS_index + 1 :]
            self.used_buffer += input_ids
            self.padded_buffer += input_ids[:-1]
            self.padded_buffer += [self.tokenizer.eos_token_id] * self._get_pad_size(
                input_ids=input_ids[:-1]
            )

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


class SubsetFineMathLoader(SubsetLoader):
    max_pages: int = 21_400_000
    name: str = "HuggingFaceTB/finemath"

    def __init__(self, **kwargs):
        super().__init__(config="finemath-3plus", **kwargs)


class SubsetInfiWebMathLoader(SubsetLoader):
    max_pages: int = 13_900_000
    name: str = "HuggingFaceTB/finemath"

    def __init__(self, **kwargs):
        super().__init__(config="infiwebmath-3plus", **kwargs)


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
                response = requests.get(self.size_base_url, params=params, timeout=15)
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

    def _fetch_data_to_buffer(self, num_pages):
        """Fetch data to buffer with support for multiple configs."""
        self.pages = []
        attempts = 0
        duplicates = 0
        initial_offset = random.randint(0, self.num_rows_per_page - 1)

        while len(self.pages) < num_pages:
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
                response = requests.get(self.rows_base_url, params=params, timeout=15)
                response.raise_for_status()
                self.pages.append(page)

                for row in response.json()["rows"]:
                    input_ids = self._get_content_from_row(row)
                    self.buffer += input_ids

            except requests.exceptions.RequestException as e:
                attempts += 1
                logging.warning(
                    f"Failed to fetch data, retrying. Attempt {attempts}/{self.retry_limit}"
                )
                if attempts < self.retry_limit:
                    time.sleep(self.retry_delay)
                else:
                    logging.error("Maximum retry limit reached. Unable to fetch data.")
                    raise

    def _get_content_from_row(self, row):

        content = row["row"]["text"]
        input_ids = self.tokenizer(content, truncation=True)["input_ids"]
        input_ids += [self.tokenizer.eos_token_id]
        return input_ids

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

    def fetch_data_to_rows(self, num_pages):
        """Fetch data and return raw text rows instead of adding to buffer."""
        downloaded_pages = set()
        rows = []
        attempts = 0
        duplicates = 0
        initial_offset = random.randint(0, self.num_rows_per_page - 1)

        while len(downloaded_pages) < num_pages:
            page = self.get_random_pages(num_pages=1, initial_offset=initial_offset)[0]

            if page in downloaded_pages:
                duplicates += 1
                if duplicates >= self.duplicate_page_threshold:
                    logging.debug(
                        f"Hit duplicate page threshold of {self.duplicate_page_threshold}. "
                        f"Stopping early at: {len(downloaded_pages)} pages."
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
                response = requests.get(self.rows_base_url, params=params, timeout=15)
                response.raise_for_status()
                downloaded_pages.add(page)

                for row in response.json()["rows"]:
                    rows.append(row["row"]["text"])

            except requests.exceptions.RequestException as e:
                attempts += 1
                logging.warning(
                    f"Failed to fetch data, retrying with a newly sampled page. "
                    f"Attempt {attempts}/{self.retry_limit}"
                )
                if attempts < self.retry_limit:
                    time.sleep(self.retry_delay)
                else:
                    logging.error("Maximum retry limit reached. Unable to fetch data.")
                    raise

        return rows


class SubsetFineWebLoader(SubsetFineWebEdu2Loader):
    name: str = "HuggingFaceFW/fineweb"

    def __init__(self, **kwargs):
        super().__init__(requires_auth=False, **kwargs)



class SubsetPeopleSpeechLoader(SubsetFineWebEdu2Loader):
    name: str = "MLCommons/peoples_speech"

    def __init__(
        self,
        batch_size=None,
        sequence_length=None,
        num_pages=None,
        tokenizer= None, # Not used yet
        pack_samples: bool = True,
        random_seed: typing.Optional[int] = None,
        config: str = "default",
        split: str = "train",
        requires_auth: bool = False,
        target_sr: int = 24000,
        target_rms: float = 0.1,
        ref_audio_max_duration: int = 15,
        hop_length: int = 256

    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_pages = num_pages
        self.tokenizer = tokenizer
        self.pack_samples = pack_samples
        self.config = config
        self.split = split
        self.requires_auth = requires_auth

        # Initialize with seed if provided
        if random_seed is not None:
            random.seed(random_seed)

        # TODO: These preprocessing-specific parameters should eventually by performed by the model's tokenizer object
        self.ref_audio_max_duration = ref_audio_max_duration # seconds
        self.target_rms = target_rms
        self.target_sr = target_sr
        self.hop_length = hop_length

        self.num_rows_per_page = 40
        self.duplicate_page_threshold = 100
        self.retry_limit = 15
        self.retry_delay = 5

        # Buffers
        self.buffer = []

        # Get HF token if needed
        self.hf_token = None
        if self.requires_auth:
            self.hf_token = os.getenv("HF_TOKEN")
            if not self.hf_token:
                raise ValueError("HF_TOKEN environment variable not found")

        # Initialize request params
        self.params = self._get_default_params()

        # Fetch pages if specified
        # If the fetched pages are empty, try again until
        # we hit the retry limit.
        fetch_attempt = 1

        if self.num_pages:
            while fetch_attempt < self.retry_limit:
                self._initialize_pages()
                fetch_attempt += 1

                # Exit if the buffer has at least one sample
                if len(self.buffer) >= 1:
                    break

                logging.warning(
                    f"All fetched pages seem to be empty or have an extremely low token count. "
                    f"Trying to fetch a new set of pages... (attempt {fetch_attempt}/{self.retry_limit})"
                )

            # If we exhaust all attempts and still don't have enough data, raise an error
            if len(self.buffer) == 0:
                raise ValueError(
                    "Maximum retry limit for fetching pages reached. "
                    "All fetched pages seem to be empty or have an extremely low token count."
                )

            # Add the query texts to each sample
            self.add_queries_to_buffer()

            # Compose the final tokenized text
            self.preprocess_text()

            # Estimate the generated audio duration for each entry
            self.add_generation_duration()



    def _get_content_from_row(self, row):
        "Parse the data row into a format the TTS model can use."

        row_data = row['row']

        # Get the condition text and audio
        ref_text = row_data['text']
        audio_url = row_data['audio'][0]['src']
        audio_type = row_data['audio'][0]['type'].split('/')[1]
        ref_audio, ref_audio_sr, ref_audio_duration = self.download_audio_torch(audio_url, audio_type)

        ref_audio, ref_audio_sr, ref_audio_duration = self.preprocess_audio(ref_audio, ref_audio_sr)

        # Only accepts the audio if it is less than a max duration
        # For E2TTS, this threshold is about 15seconds
        # TODO: All preprocessing should eventually taken over by the tokenizer.
        if (ref_audio_duration >= self.ref_audio_max_duration or
            len(ref_text) < 5):
            return []

        # Compose the sample dict
        sample = dict(
            ref_text = ref_text,
            ref_audio = ref_audio,
            ref_audio_sr = ref_audio_sr,
            ref_audio_duration = ref_audio_duration
        )


        return [sample]


    def download_audio_segment(self,
                               url: str,
                               format: str = "wav") -> AudioSegment:
        """
        Download an audio file from the given URL and load it as a pydub AudioSegment.

        Args:
            url (str): The URL to download the audio file from.
            format (str): The audio format (e.g. "wav", "mp3"). Default is "wav".

        Returns:
            AudioSegment: The loaded audio segment.

        Raises:
            Exception: If the download fails or the file cannot be parsed.
        """
        try:
            response = requests.get(url)
        except Exception as e:
            raise Exception(f"Error during HTTP request to get audio file: {e}")

        if response.status_code != 200:
            raise Exception(f"Failed to download audio file: HTTP {response.status_code}")

        audio_bytes = BytesIO(response.content)
        try:
            if format.lower() == "wav":
                audio = AudioSegment.from_wav(audio_bytes)
            elif format.lower() == "mp3":
                audio = AudioSegment.from_mp3(audio_bytes)
            else:
                raise Exception(f"Unsupported audio format: {format}")
        except Exception as e:
            raise Exception(f"Failed to parse audio file: {e}")

        return audio

    def download_audio_torch(self, url: str, format: str = "wav"):
        """
        Download an audio file from the given URL and load it as a torch tensor using torchaudio.

        Args:
            url (str): The URL to download the audio file from.
            format (str): The expected audio format (e.g. "wav", "mp3").

        Returns:
            tuple: A tuple containing:
                - audio (torch.Tensor): The loaded audio tensor of shape (channels, samples).
                - sample_rate (int): The sample rate of the audio.
                - duration (float): The duration of the audio in seconds.

        Raises:
            Exception: If the download fails or the file cannot be loaded.
        """
        try:
            response = requests.get(url)
        except Exception as e:
            raise Exception(f"Error during HTTP request to get audio file: {e}")

        if response.status_code != 200:
            raise Exception(f"Failed to download audio file: HTTP {response.status_code}")

        # Use BytesIO to wrap the downloaded content as a file-like object.
        audio_bytes = BytesIO(response.content)

        try:
            # torchaudio.load supports file-like objects
            audio, sample_rate = torchaudio.load(audio_bytes)
        except Exception as e:
            raise Exception(f"Failed to load audio with torchaudio: {e}")

        # Calculate duration in seconds: number of samples divided by sample rate.
        duration = audio.shape[1] / sample_rate
        return audio, sample_rate, duration

    def fetch_data_to_row(self):
        raise NotImplementedError("Method is not supported by {self.name} loader.")


    def add_queries_to_buffer(self):
        """
        For each dictionary in self.buffer, randomly choose another dictionary
        (different from the current one) and add its 'ref_text' value to the current
        dictionary under the key 'query_text'.

        Raises:
            ValueError: if the buffer has fewer than two entries.
        """
        if len(self.buffer) < 2:
            raise ValueError("Buffer must have at least two entries to add a query from a different dict.")

        for entry in self.buffer:
            # Exclude the current dictionary from the candidates.
            candidates = [d for d in self.buffer if d is not entry]
            chosen = random.choice(candidates)
            entry['query_text'] = chosen['ref_text']

    def preprocess_audio(self, audio: torch.Tensor, sr: int) -> np.ndarray:
        """
        Process the input audio tensor and return a NumPy array with the processed audio.

        The processing steps include:
          - Converting stereo to mono by averaging channels (if necessary).
          - Adjusting the volume so that the RMS is at least TARGET_RMS.
          - Resampling the audio to TARGET_SR if needed.

        Args:
            audio (torch.Tensor): Input audio tensor of shape (channels, samples).
            sr (int): The sample rate of the input audio.

        Returns:
            np.ndarray: The processed audio as a NumPy array.
            sample_rate (int): The sample rate of the audio.
            duration (float): The duration of the audio in seconds.


        TODO: in future releases, processing audio should be owned
              by the model. This can be handled by a specialized
              tokenizer object for instance.
        """
        # Convert to mono if the audio has more than one channel
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # Compute the RMS (root mean square) of the audio
        rms = torch.sqrt(torch.mean(torch.square(audio)))
        # Adjust volume if the RMS is below the target level
        if rms < self.target_rms:
            audio = audio * (self.target_rms / rms)


        # Resample the audio if its sample rate does not match the target sample rate
        # TODO: Check if resampling to a higher rate does not harm the audio quality
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            audio = resampler(audio)

        # Calculate duration in seconds: number of samples divided by sample rate.
        # This should not change wrt. to the old duration
        duration = audio.shape[1] / self.target_sr

        # Return the processed audio as a NumPy array
        return audio.cpu().numpy(), self.target_sr, duration

    def preprocess_text(self):
        """
        For each dictionary in self.buffer, take its 'ref_text' and 'query_text',
        merge them, then tokenize the combined text and store it in self.buffer.

        TODO: This should also be owned by the model's tokenizer object.
        """
        for entry in self.buffer:
            # Ensure there's a space after 'ref_text'
            ref_text = entry["ref_text"] + " "

            # Combine ref_text and query_text
            text_list = [ref_text + entry["query_text"]]

            # Process the text (e.g., pinyin conversion, spacing, etc.)
            final_text_list = convert_char_to_pinyin(text_list)

            # Store the processed text under 'tokenized_text'
            entry["tokenized_text"] = final_text_list

    def add_generation_duration(self):
        """
        Compute the generated audio duration using both ref and query texts
        """
        for entry in self.buffer:
            audio = entry["ref_audio"]
            ref_text = entry["ref_text"]
            query_text = entry["query_text"]

            # The number of reference audio 'sample' groups/tokens
            ref_audio_len = audio.shape[-1] // self.hop_length

            # Local speed is the speed of the generated audio.
            # For short texts, speech should be slower (0.3 here)
            # TODO: This should probably be given to miners to decide.
            local_speed = 1 if len(query_text.encode("utf-8")) >= 10 else 0.3
            ref_text_len = len(ref_text.encode("utf-8"))
            query_text_len = len(query_text.encode("utf-8"))

            # Compute estimated duration duration
            gen_audio_len = ref_audio_len + int((ref_audio_len / ref_text_len) * query_text_len / local_speed)

            entry["gen_audio_len"] = gen_audio_len # hops not seconds
            entry["ref_audio_len"] = ref_audio_len # hops not seconds

    def __iter__(self):
        return self

    def __next__(self):
        if not self.buffer:
            raise StopIteration
        return self.buffer.pop(0)
