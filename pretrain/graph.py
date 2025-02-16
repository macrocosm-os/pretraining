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

import asyncio
from typing import Optional

import bittensor as bt
from taoverse.model.storage.chain.chain_model_metadata_store import (
    ChainModelMetadataStore,
)
from taoverse.model.storage.model_metadata_store import ModelMetadataStore

import constants
from competitions.data import CompetitionId


def best_uid(
    competition_id: CompetitionId,
    subtensor: Optional[bt.subtensor] = None,
    metagraph: Optional[bt.metagraph] = None,
    metadata_store: Optional[ModelMetadataStore] = None,
) -> Optional[int]:
    """Returns the best performing UID in the metagraph for the given competition.

    Returns:
        int: The UID of the best performing miner in the metagraph for the given competition or None if no miner for the given competition is found.
    """
    if not subtensor:
        subtensor = bt.subtensor()

    if not metagraph:
        metagraph = subtensor.metagraph(constants.SUBNET_UID)

    if not metadata_store:
        metadata_store = ChainModelMetadataStore(
            subtensor=subtensor, subnet_uid=constants.SUBNET_UID
        )

    incentives = [(metagraph.I[uid].item(), uid) for uid in range(metagraph.n)]
    # With a winner takes all model, we expect ~ 1 model per competition.
    # So the top 5 miners should provide sufficient coverage.
    top_miners = sorted(incentives, reverse=True)[:5]

    for _, miner_uid in top_miners:
        metadata = asyncio.run(
            metadata_store.retrieve_model_metadata(
                miner_uid, metagraph.hotkeys[miner_uid]
            )
        )
        if metadata and metadata.id.competition_id == competition_id:
            return miner_uid

    return None
