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

from transformers import (
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    AutoTokenizer,
    GPT2TokenizerFast,
)
import bittensor as bt
import torch

config = GPTNeoXConfig()


# 769_782_400 param model as a sample.
def get_model():
    config = GPTNeoXConfig(
        vocab_size=10000,
        num_attention_heads=40,
        hidden_size=1600,
        intermediate_size=6400,
        num_hidden_layers=24,
        max_position_embeddings=2048,
        torch_dtype=torch.bfloat16,
        attention_type="flash_attention_2",
    )
    return GPTNeoXForCausalLM(config)


def get_old_tokenizer(cache_dir: str = None):
    """Returns the tokenizer used prior to 7B parameter models"""
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2", cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_tokenizer(cache_dir: str = None):
    """Returns the tokenizer used by the latest models."""
    bt.logging.info(
        "Getting gpt-4 tokenizer. Following logs about not matching GPT2TokenizerFast are expected."
    )
    tokenizer = GPT2TokenizerFast.from_pretrained("Xenova/gpt-4", cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
