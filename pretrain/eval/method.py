import difflib
import math
import re
import traceback
import typing
from enum import IntEnum

import taoverse.utilities.logging as logging
import torch
import transformers
from transformers import DynamicCache, PreTrainedModel


class EvalMethodId(IntEnum):
    """Enumeration of evaluation methods."""

    NONE = 0

    # Evalutes the model's performance on a text generation task by computing average cross entropy loss
    # on the entirety of the provided text.
    TEXT_LOSS = 1
