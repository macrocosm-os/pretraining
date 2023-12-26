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

# Tools for performing validation over models.

import math
import torch
import typing
import pretrain
import traceback
import bittensor as bt


def iswin(loss_i, loss_j, time_i, time_j):
    """
    Determines the winner between two models based on the epsilon adjusted loss.

    Parameters:
        loss_i (float): Loss of uid i on batch
        loss_j (float): Loss of uid j on batch.
        time_i (float): Timestamp of uid i.
        time_j (float): Timestamp of uid j.
    Returns:
        bool: True if loss i is better, False otherwise.
    """
    # Adjust loss based on timestamp and pretrain epsilon
    loss_i = (1 - pretrain.timestamp_epsilon) * loss_i if time_i < time_j else loss_i
    loss_j = (1 - pretrain.timestamp_epsilon) * loss_j if time_j < time_i else loss_j
    return loss_i < loss_j


def compute_wins(
    uids: typing.List[int],
    losses: typing.Dict[int, typing.List[float]],
    batches: typing.List[torch.FloatTensor],
    timestamps: typing.Dict[int, float],
):
    """
    Computes the wins and win rate for each model based on loss comparison.

    Parameters:
        losses (dict): A dictionary of losses for each uid.
        batches (dict): A dictionary of data batches.
        timestamps (dict): A dictionary of timestamps for each uid.

    Returns:
        tuple: A tuple containing two dictionaries, one for wins and one for win rates.
    """
    uids = losses.keys()
    wins = {uid: 0 for uid in uids}
    win_rate = {uid: 0 for uid in uids}

    # Iterate over each pair of models
    for i in uids:
        time_i = timestamps[i]
        loss_i = losses[i]
        total_matches = 0
        for j in uids:
            if i == j:
                continue
            time_j = timestamps[i]
            loss_j = losses[i]
            for b, _ in enumerate(batches):
                # Increment wins for model i if it's better than model j
                wins[i] += 1 if iswin(loss_i, loss_j, time_i, time_j) else 0
                total_matches += 1

        # Calculate win rate for model i
        win_rate[i] = wins[i] / total_matches if total_matches > 0 else 0

    return wins, win_rate


def compute_losses(
    model, batches: typing.List[torch.Tensor], device: str
) -> typing.List[float]:
    """
    Computes the losses for a given model on provided batches.

    Parameters:
        model (torch.nn.Module): The model for which losses are to be computed.
        batches (dict): A list of batches.
        device (str): The device to use for computation (e.g., 'cpu', 'gpu').

    Returns:
        dict: A dictionary with page indices as keys and lists of loss values as values.
    """
    model.to(device)
    model.eval()

    # Iterate over each page and corresponding batches
    losses = []
    for batch in batches:
        try:
            inputs = batch.to(device)
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss.item()  # Extract scalar loss value
            losses.append(loss)
        except Exception as e:
            bt.logging.error(f"Exception occurred: {e}")
            traceback.print_exc()  # Print the stack trace
            losses.append(math.inf)  # Use infinity to indicate failure

    return losses
