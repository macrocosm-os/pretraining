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

from taoverse.model.competition.data import ModelConstraints
from transformers import (
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    AutoTokenizer,
    GPT2TokenizerFast,
)
from transformers import AutoTokenizer, PreTrainedTokenizer

import bittensor as bt

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
    )
    return GPTNeoXForCausalLM(config)


def load_tokenizer(
    model_constraints: ModelConstraints, cache_dir: str = None
) -> PreTrainedTokenizer:
    """Returns the fixed tokenizer for the given model constraints."""

    match model_constraints.tokenizer:
        # This is a temporary hack for tts models.
        # TODO: We should create an AutoTokenizer custom class to support tokenizer
        #       for custom model types
        case 'e2tts':
            return 'tts_tokenizer'

        # By default we use the HF AutoTokenizer class
        case _:
            return AutoTokenizer.from_pretrained(
                model_constraints.tokenizer, cache_dir=cache_dir
            )
