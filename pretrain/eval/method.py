import math
import traceback
import typing
from enum import IntEnum

import numpy as np
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


def check_for_reasonable_output(
    model, input1: torch.Tensor, input2: torch.Tensor, pad_token_id: int
) -> bool:
    """Checks that a model generates reasonable outputs for two given inputs.

    Args:
        model (torch.nn.Module): The model for which outputs are to be checked. Already loaded to device.
        input1 (torch.Tensor]): Tokenized input1 to check. Already loaded to device.
        input2 (torch.Tensor]): Tokenized input2 to check. Already loaded to device.
        pad_token_id (int): Pad token id for the tokenizer used to generate inputs 1 and 2.

    Returns:
        bool: If the model generates reasonable outputs.
    """
    # Generate 20 tokens of output from the model for each prompt.
    output_length = 20
    # Only take the last 20 tokens since otherwise we also get the prompt ids.
    generate_id1s = model.generate(
        input1,
        min_new_tokens=output_length,
        max_new_tokens=output_length,
        pad_token_id=pad_token_id,
    )[:, -output_length:]
    generate_id2s = model.generate(
        input2,
        min_new_tokens=output_length,
        max_new_tokens=output_length,
        pad_token_id=pad_token_id,
    )[:, -output_length:]

    # Check if too many of the generated ids are the same between the two outputs.
    if torch.sum(torch.eq(generate_id1s, generate_id2s)).item() >= output_length / 2:
        logging.info(
            f"Model with config {model.config} had too much overlap between generated outputs."
        )
        return False

    # Check if internally both responses are too repetitive.
    most_common_counts = []
    for tensor in [generate_id1s, generate_id2s]:
        # Find unique elements and their counts
        _, counts = torch.unique(tensor, return_counts=True)
        # Find the index of the maximum count
        max_count_index = torch.argmax(counts)
        # Extract the count of the most common element
        most_common_counts.append(counts[max_count_index].item())

    if all(count > output_length / 2 for count in most_common_counts):
        logging.info(
            f"Model with config {model.config} had too much repetition in generated outputs."
        )
        return False

    # Passed all the checks, return True.
    return True


def compute_text_loss(
    model: PreTrainedModel,
    batches: typing.List[np.ndarray],
    device: str,
    pad_token_id: int,
) -> float:
    """
    Computes the average loss for a given model on provided batches.

    Parameters:
        model (torch.nn.Module): The model for which losses are to be computed.
        batches (List): A list of batches.
        device (str): The device to use for computation (e.g., 'cpu', 'gpu').
        pad_token_id int: Pad token id for the tokenizer used to tokenize the batches.

    Returns:
        list: A list of losses for each batch.
    """
    model.to(device)
    model.eval()

    # First check that model generates reasonable looking outputs.
    # Grab 100 tokens from the first two batches as 'prompts'. (1 x Seq Length tensors.)
    prompt_length = 100
    token_inputs_1 = torch.tensor(batches[0][:, :prompt_length]).to(device)
    token_inputs_2 = torch.tensor(batches[1][:, :prompt_length]).to(device)

    if not check_for_reasonable_output(
        model, token_inputs_1, token_inputs_2, pad_token_id
    ):
        return [math.inf for _ in range(len(batches))]

    # Everything looks good! Continue to computing actual losses.

    # Iterate over each page and corresponding batches
    losses = []
    with torch.no_grad():
        for batch in batches:
            try:
                # Context and ref are 1 dimensional tensors.
                inputs = torch.tensor(batch).to(device)
                # Prepare a cache class and pass it to the model's forward.
                past_key_values = DynamicCache()
                logits = model(inputs, past_key_values=past_key_values).logits

                # Shift the logits and labels to compute the loss.
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs[..., 1:].contiguous()

                # Flatten the tokens
                loss_fct = torch.nn.CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                loss = loss_fct(shift_logits, shift_labels).item()

                losses.append(loss)
            except Exception as e:
                logging.error(
                    f"Exception occurred in reference loss computation: {traceback.format_exc()}"
                )
                return math.inf

    return sum(losses) / len(losses) if losses else math.inf


def generate_output(
    model,
    input_ids: torch.Tensor,
    generation_config: transformers.GenerationConfig,
    device: str,
    tokenizer: transformers.PreTrainedTokenizer,
) -> str:
    """
    Generates the tokenized output for a model given a tokenized input and generation config.

    Args:
        model (torch.nn.Module): The model for which losses are to be computed.
        input_ids (torch.Tensor): Input tokens to generate a response to.
        generation_config (transformers.GenerationConfig): Configuration parameters for generating output.
        device (str): The device to use for computation (e.g., 'cpu', 'gpu').
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to tokenize the output with before returning.

    Returns:
        str: Generated tokenized output from the model.
    """
    input_ids = input_ids.to(device)
    output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
    )
    response = tokenizer.decode(
        output[0][len(input_ids[0]) :], skip_special_tokens=True
    )
    return response
