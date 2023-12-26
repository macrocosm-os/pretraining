import asyncio
import bittensor as bt
import os
from model.data import ModelId, ModelMetadata
from model.storage.chain import constants
from model.storage.model_metadata_store import ModelMetadataStore
from typing import Optional


class ChainModelMetadataStore(ModelMetadataStore):
    """Chain based implementation for storing and retrieving metadata about a model."""

    def __init__(
        self,
        subtensor: bt.subtensor,
        wallet: Optional[bt.wallet] = None,
        subnet_uid: int = constants.SUBNET_UID,
    ):
        self.subtensor = subtensor
        self.wallet = (
            wallet  # Wallet is only needed to write to the chain, not to read.
        )
        self.subnet_uid = subnet_uid
        self.metagraph = self.subtensor.metagraph(self.subnet_uid)

    # TODO actually make this asynchronous with threadpools etc.
    async def store_model_metadata(self, uid: int, model_id: ModelId):
        """Stores model metadata on this subnet for a specific wallet."""
        if self.wallet is None:
            raise ValueError("No wallet available to write to the chain.")

        # TODO: Confirm that the hotkey matches the wallet using self.metagraph.hotkeys[uid]
        self.subtensor.commit(
            wallet=self.wallet,
            netuid=self.subnet_uid,
            data=model_id.to_compressed_str(),
        )

    async def retrieve_model_metadata(self, uid: int) -> ModelMetadata:
        """Retrieves model metadata on this subnet for specific hotkey"""
        hotkey = self.metagraph.hotkeys[uid]
        metadata = bt.extrinsics.serving.get_metadata(
            self.subtensor, self.subnet_uid, hotkey
        )

        commitment = metadata["info"]["fields"][0]
        hex_data = commitment[list(commitment.keys())[0]][2:]

        chain_str = bytes.fromhex(hex_data).decode()
        model_id = ModelId.from_compressed_str(chain_str)
        model_metadata = ModelMetadata(id=model_id, block=metadata["block"])

        return model_metadata


# Can only commit data every ~20 minutes.
async def test_store_model_metadata():
    """Verifies that the ChainModelMetadataStore can store data on the chain."""
    model_id = ModelId(
        path="TestPath", name="TestModel", hash="TestHash1", commit="1.0"
    )

    # Use a different subnet that does not leverage chain storage to avoid conflicts.
    # TODO switch to a mocked version when it supports commits.
    subtensor = bt.subtensor()

    # Uses .env configured wallet/hotkey/uid for the test.
    coldkey = os.getenv("TEST_COLDKEY")
    hotkey = os.getenv("TEST_HOTKEY")
    net_uid = int(os.getenv("TEST_SUBNET_UID"))
    uid = int(os.getenv("TEST_UID"))

    wallet = bt.wallet(name=coldkey, hotkey=hotkey)

    metadata_store = ChainModelMetadataStore(
        subtensor=subtensor, wallet=wallet, subnet_uid=net_uid
    )

    # Store the metadata on chain.
    await metadata_store.store_model_metadata(uid=uid, model_id=model_id)

    print(f"Finished storing {model_id} on the chain.")


async def test_retrieve_model_metadata():
    """Verifies that the ChainModelMetadataStore can retrieve data from the chain."""
    expected_model_id = ModelId(
        path="TestPath", name="TestModel", hash="TestHash1", commit="1.0"
    )

    # Use a different subnet that does not leverage chain storage to avoid conflicts.
    # TODO switch to a mocked version when it supports commits.
    subtensor = bt.subtensor()

    # Uses .env configured hotkey/uid for the test.
    net_uid = int(os.getenv("TEST_SUBNET_UID"))
    uid = int(os.getenv("TEST_UID"))

    # Do not require a wallet for retrieving data.
    metadata_store = ChainModelMetadataStore(
        subtensor=subtensor, wallet=None, subnet_uid=net_uid
    )

    # Retrieve the metadata from the chain.
    model_metadata = await metadata_store.retrieve_model_metadata(uid)

    print(f"Expecting matching model id: {expected_model_id == model_metadata.id}")


# Can only commit data every ~20 minutes.
async def test_roundtrip_model_metadata():
    """Verifies that the ChainModelMetadataStore can roundtrip data on the chain."""
    model_id = ModelId(
        path="TestPath", name="TestModel", hash="TestHash1", commit="1.0"
    )

    # Use a different subnet that does not leverage chain storage to avoid conflicts.
    # TODO switch to a mocked version when it supports commits.
    subtensor = bt.subtensor()

    # Uses .env configured wallet/hotkey/uid for the test.
    coldkey = os.getenv("TEST_COLDKEY")
    hotkey = os.getenv("TEST_HOTKEY")
    net_uid = int(os.getenv("TEST_SUBNET_UID"))
    uid = int(os.getenv("TEST_UID"))

    wallet = bt.wallet(name=coldkey, hotkey=hotkey)

    metadata_store = ChainModelMetadataStore(
        subtensor=subtensor, wallet=wallet, subnet_uid=net_uid
    )

    # Store the metadata on chain.
    await metadata_store.store_model_metadata(uid=uid, model_id=model_id)

    # May need to use the underlying publish_metadata function with wait_for_inclusion: True to pass here.
    # Otherwise it defaults to False and we only wait for finalization not necessarily inclusion.

    # Retrieve the metadata from the chain.
    model_metadata = await metadata_store.retrieve_model_metadata(uid)

    print(f"Expecting matching metadata: {model_id == model_metadata.id}")


if __name__ == "__main__":
    # Can only commit data every ~20 minutes.
    # asyncio.run(test_roundtrip_model_metadata())
    # asyncio.run(test_store_model_metadata())
    asyncio.run(test_retrieve_model_metadata())
