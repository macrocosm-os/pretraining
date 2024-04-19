"""Tool to periodically run benchmarks on the best model of the subnet and some well-known models."""

import abc
from argparse import ArgumentParser
import asyncio
import time
from typing import Dict, List, Tuple
import requests
import wandb
import torch
import random
from tqdm import tqdm
from model.data import ModelMetadata, TokenizerIdentifier
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
import pretrain as pt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss
from collections import defaultdict
import os
import wandb
import pandas as pd
from dotenv import load_dotenv
import bittensor as bt
import constants
import model.utils as model_utils

load_dotenv()  # take environment variables from .env.

PROJECT = "pretraining-leaderboard-data"
ENTITY = "raofoundation"
WANDB_TOKEN = os.getenv("WANDB_API_KEY")


def compute_ppl(
    text,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    stride: int = 512,
    max_length: int = 2048,
) -> float:
    """Returns the perplexity of the model on the given text."""

    encodings = tokenizer(
        text,
        truncation=False,
        return_tensors="pt",
    ).to("cuda")

    loss_fct = CrossEntropyLoss(reduction="none", ignore_index=-100)
    seq_len = encodings.input_ids.size(1)

    losses = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        # Computes how much overlap there is with the previous batch.
        new_tokens_len = end_loc - prev_end_loc
        if end_loc - begin_loc < max_length:
            bt.logging.info(
                f"Skipping batch as it has less than max_length tokens: {begin_loc}:{end_loc}."
            )
            break
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to("cuda")
        # attn_mask = encodings.attention_mask[:, begin_loc:end_loc].to(device)
        labels = input_ids.clone()
        # Ignore the tokens we've processed on a previous batch. -100 is a magic
        # value that is ignored by the CrossEntropyLoss function
        labels[:, :-new_tokens_len] = -100

        with torch.no_grad():
            out_logits = model(input_ids).logits

        # Shift by 1 token.
        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tensors
        shift_logits = shift_logits.view(-1, model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        losses.append(loss_fct(shift_logits, shift_labels))
    return torch.exp(torch.stack(losses).mean()).item()


class ModelProvider(abc.ABC):
    """Base class for a provider of a model and its tokenizer."""

    @abc.abstractmethod
    def get_model(self) -> AutoModelForCausalLM:
        pass

    @abc.abstractmethod
    def get_tokenizer(self) -> AutoTokenizer:
        pass

    @abc.abstractmethod
    def get_sequence_length(self) -> int:
        pass


class HuggingFaceModelProvider(ModelProvider):
    """Provides a well-known model from hugging face."""

    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        sequence_length: int = 2048,
        use_flash=True,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.sequence_length = sequence_length
        self.use_flash = use_flash

    def get_model(self) -> AutoModelForCausalLM:
        if self.use_flash:
            return AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        return AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            torch_dtype=torch.bfloat16,
        )

    def get_tokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)

    def get_sequence_length(self) -> int:
        return self.sequence_length


class SubnetModelProvider(ModelProvider):
    """Provides models from the subnet."""

    def __init__(self, model_metadata: ModelMetadata, cache_dir: str):
        self.model_metadata = model_metadata
        self.cache_dir = cache_dir

    def get_model(self) -> AutoModelForCausalLM:
        store = HuggingFaceModelStore()
        model = asyncio.run(
            store.download_model(self.model_metadata.id, self.cache_dir)
        )
        return model.pt_model

    def get_tokenizer(self) -> AutoTokenizer:
        # Note that AutoTokenizer maps to either PretrainedTokenizer | PretrainedTokenizerFast.
        # Both methods return a type that corresponds to one of those.
        if (
            model_utils.get_model_criteria(
                self.model_metadata.block
            ).tokenizer_identifier
            == TokenizerIdentifier.DISTILGPT_2
        ):
            return pt.model.get_old_tokenizer(cache_dir=self.cache_dir)
        return pt.model.get_tokenizer(cache_dir=self.cache_dir)

    def get_sequence_length(self) -> int:
        return model_utils.get_model_criteria(self.model_metadata.block).sequence_length


def get_best_model_provider(
    cache_dir: str, config: bt.config
) -> Tuple[str, SubnetModelProvider]:
    """Returns a provider to fetch the subnets best model.

    Returns:
        Tuple[str, SubnetModelProvider]: A tuple containing the models' HF repo and the model provider.
    """
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=constants.SUBNET_UID)
    best_uid = pt.graph.best_uid(metagraph=metagraph)
    hotkey = metagraph.hotkeys[best_uid]

    metagraph_store = ChainModelMetadataStore(subtensor)
    metadata = asyncio.run(metagraph_store.retrieve_model_metadata(hotkey))
    if metadata is None:
        raise ValueError(f"No model metadata found for miner {best_uid}")

    return (
        f"{metadata.id.namespace}/{metadata.id.name}",
        SubnetModelProvider(metadata, cache_dir),
    )


def get_wikitext103(cache_dir: str) -> str:
    """Returns the wikitext103 dataset.

    Args:
        cache_dir (str): The directory to cache the dataset.
    """
    wikitext_dataset = load_dataset(
        "wikitext", "wikitext-103-raw-v1", split="test", cache_dir=cache_dir
    )
    return "\n\n".join(wikitext_dataset["text"])


def get_lambada(cache_dir: str) -> str:
    """Returns the lambada dataset.

    Args:
        cache_dir (str): The directory to cache the dataset.
    """
    lambada_dataset = load_dataset("lambada", split="test", cache_dir=cache_dir)
    return "\n\n".join(lambada_dataset["text"])


def get_ptb(cache_dir: str) -> str:
    """Returns the Penn Treebank dataset.

    Args:
        cache_dir (str): The directory to cache the dataset.
    """
    ptb_dataset = load_dataset("ptb", split="test", cache_dir=cache_dir)
    return "\n\n".join(ptb_dataset["text"])


def get_falcon() -> str:
    """Returns a random subset of text from the Falcon Refined Web dataset."""

    def _fetch_data_for_page(page: int, max_retries: int = 5) -> List[str]:
        params = {
            "dataset": "tiiuae/falcon-refinedweb",
            "config": "default",
            "split": "train",
            "offset": page,
            "limit": 100,
        }

        attempt = 0
        while attempt < max_retries:
            try:
                response = requests.get(
                    "https://datasets-server.huggingface.co/rows",
                    params=params,
                    timeout=60,
                )
                response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
                return [r["row"]["content"] for r in response.json()["rows"]]
            except requests.exceptions.RequestException:
                attempt += 1
                bt.logging.warning(
                    f"Failed to fetch data, retrying. Attempt {attempt}/{max_retries}"
                )
                if attempt < max_retries:
                    time.sleep(3)
                else:
                    bt.logging.error(
                        "Maximum retry limit reached. Unable to fetch data."
                    )
                    raise

    pages = [
        random.randint(1, pt.dataset.SubsetFalconLoader.max_pages) for _ in range(20)
    ]
    rows = []
    for page in pages:
        rows.extend(_fetch_data_for_page(page))
    return "\n\n".join(rows)


def format_model_size(size: int) -> str:
    """Formats a model size into a human readable format."""
    if size >= 1e12:
        return f"{size / 1e12:.1f}T"
    if size >= 1e9:
        return f"{size / 1e9:.1f}B"
    if size >= 1e6:
        return f"{int(size // 1e6)}M"
    return str(size)


def run_benchmarks(args: ArgumentParser, datasets: Dict[str, str], config: bt.config):
    """Performs a single run of the benchmarks on the given datasets."""
    # best_model_hf, best_model_provider = get_best_model_provider(args.cache_dir, config)
    models = {
        # best_model_hf: best_model_provider,
        "hardcode-jonathan": HuggingFaceModelProvider(
            "J0nathan18/7b_v1", args.cache_dir, sequence_length=4096
        ),
        "hardcode-lucia": HuggingFaceModelProvider(
            "Lucia-no/subnet9_6_9B", args.cache_dir, sequence_length=4096
        ),
        "gpt2": HuggingFaceModelProvider(
            "gpt2", args.cache_dir, sequence_length=1024, use_flash=False
        ),
        "gpt2-large": HuggingFaceModelProvider(
            "gpt2-large", args.cache_dir, sequence_length=1024, use_flash=False
        ),
        # # Also run a 3b for comparison.
        "phi-2": HuggingFaceModelProvider("microsoft/phi-2", args.cache_dir),
        "falcon-7b": HuggingFaceModelProvider("tiiuae/falcon-7b", args.cache_dir),
        # Intentionally use a sequence length of 2048, even though the model can support 32k.
        "Mistral-7B-v0.1 ": HuggingFaceModelProvider(
            "mistralai/Mistral-7B-v0.1", args.cache_dir, sequence_length=2048
        ),
    }

    ppls = defaultdict(list)
    model_sizes = []
    # For each model, compute PPL on each dataset.
    for model_name, provider in models.items():
        bt.logging.info(f"Computing benchmarks for model: {model_name}")
        get_model_start = time.time()
        model = provider.get_model().to("cuda")
        model.eval()
        model_size = sum(p.numel() for p in model.parameters())
        model_sizes.append(format_model_size(model_size))

        # Should be cached and reasonably fast.
        bt.logging.info(
            f"Finished getting model: {model_name} of size {model_size} in {round(time.time()- get_model_start, 2)}"
        )

        tokenizer = provider.get_tokenizer()
        for dataset_name, dataset in datasets.items():
            compute_start = time.time()
            bt.logging.info(
                f"Starting computing PPL for model: {model_name} on dataset: {dataset_name}"
            )
            ppls[dataset_name].append(
                compute_ppl(
                    dataset,
                    model,
                    tokenizer,
                    max_length=provider.get_sequence_length(),
                )
            )
            bt.logging.info(
                f"Finished computing PPL: {round(ppls[dataset_name][-1], 2)} for model: {model_name} on dataset: {dataset_name} in {round(time.time()- compute_start, 2)}"
            )
        del model
        del tokenizer
        torch.cuda.empty_cache()

    # Log to wandb.
    wandb.login(key=WANDB_TOKEN)
    with wandb.init(project=PROJECT, entity=ENTITY):
        table = wandb.Table(
            dataframe=pd.DataFrame(
                {"Model": models.keys(), "Size": model_sizes, **ppls}
            )
        )
        wandb.log({"benchmarks": table})


def main(args: ArgumentParser, config: bt.config):

    bt.logging.info("Loading datasets...")
    datasets = {
        "Wikitext103 (PPL)": get_wikitext103(args.cache_dir),
        "Falcon Refined Web (PPL)": get_falcon(),
    }

    while True:
        try:
            run_benchmarks(args, datasets, config)

            # Run every 12 hours.
            time.sleep(12 * 60 * 60)
        except Exception as e:
            bt.logging.error(f"Exception occurred: {e}")

            # Try again after 10 minutes.
            time.sleep(10 * 60)


if __name__ == "__main__":
    bt.logging()

    parser = ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default=None)

    bt.subtensor.add_args(parser)

    config = bt.config(parser=parser)
    args = parser.parse_args()

    main(args, config)
