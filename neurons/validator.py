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

from collections import defaultdict
import datetime as dt
import functools
import os
import json
import math
import pickle
import time
import torch
import random
import asyncio
import argparse

import wandb
import constants
from model.data import TokenizerIdentifier
from model.model_tracker import ModelTracker
from model.model_updater import ModelUpdater
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from model.storage.disk.disk_model_store import DiskModelStore
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
import model.utils as model_utils
import traceback
import threading
import multiprocessing
from rich.table import Table
from rich.console import Console

import bittensor as bt
import pretrain as pt
from utilities.miner_iterator import MinerIterator
from utilities import utils
from utilities.perf_monitor import PerfMonitor

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class Validator:
    TRACKER_FILENAME = "model_tracker_2.pickle"
    UIDS_FILENAME = "uids_2.pickle"
    VERSION_FILENAME = "version.txt"

    @staticmethod
    def config():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--device",
            type=str,
            default="cuda" if torch.cuda.is_available() else "cpu",
            help="Device name.",
        )
        parser.add_argument(
            "--wandb.off",
            dest="wandb.on",
            action="store_false",
            help="Turn off wandb logging.",
        )
        parser.add_argument(
            "--blocks_per_epoch",
            type=int,
            default=50,
            help="Number of blocks to wait before setting weights.",
        )
        parser.add_argument(
            "--pages_per_eval",
            type=int,
            default=constants.n_eval_pages,
            help="Number of pages used to eval each step.",
        )
        parser.add_argument(
            "--sample_min",
            type=int,
            default=constants.sample_min,
            help="Number of uids to bring to next eval.",
        )
        parser.add_argument(
            "--sample_max",
            type=int,
            default=constants.sample_max,
            help="Maximum number of new uids to eval each step.",
        )
        parser.add_argument(
            "--dont_set_weights",
            action="store_true",
            help="Validator does not set weights on the chain.",
        )
        parser.add_argument(
            "--offline",
            action="store_true",
            help="Does not launch a wandb run, does not set weights, does not check that your key is registered.",
        )
        parser.add_argument(
            "--model_dir",
            default=os.path.join(constants.ROOT_DIR, "model-store/"),
            help="Where to store downloaded models",
        )
        parser.add_argument(
            "--netuid",
            type=str,
            default=constants.SUBNET_UID,
            help="The subnet UID.",
        )

        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        bt.axon.add_args(parser)
        config = bt.config(parser)
        return config

    def state_path_old(self) -> str:
        """
        Constructs the old file path for storing validator state.

        This will soon be deprecated.

        Returns:
        str: A string representing the file path.
        """
        return os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                bt.logging.config().logging.logging_dir,
                self.wallet.name,
                self.wallet.hotkey_str,
                self.config.netuid,
                "vali-state",
            )
        )

    def state_path(self) -> str:
        """
        Returns the file path for storing validator state.

        Returns:
        str: A string representing the file path.
        """
        return os.path.join(self.config.model_dir, "vali-state")

    def __init__(self):
        self.config = Validator.config()
        bt.logging(config=self.config)

        bt.logging.info(f"Starting validator with config: {self.config}")

        # === Bittensor objects ====
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        torch.backends.cudnn.benchmark = True

        # Dont check registration status if offline.
        if not self.config.offline:
            self.uid = utils.assert_registered(self.wallet, self.metagraph)

        # Track how may run_steps this validator has completed.
        self.run_step_count = 0

        # Dont log to wandb if offline.
        if not self.config.offline and self.config.wandb.on:
            self.new_wandb_run()

        # === Running args ===
        self.weights = torch.zeros_like(self.metagraph.S)
        self.epoch_step = 0
        self.global_step = 0
        self.last_epoch = self.metagraph.block.item()

        self.uids_to_eval = []

        # Create a set of newly added uids that should be evaluated on the next loop.
        self.pending_uids_to_eval_lock = threading.RLock()
        self.pending_uids_to_eval = set()

        # Setup a model tracker to track which miner is using which model id.
        self.model_tracker = ModelTracker()

        # Construct the filepaths to save/load state.
        state_dir = self.state_path()
        os.makedirs(state_dir, exist_ok=True)

        self.uids_filepath = os.path.join(state_dir, Validator.UIDS_FILENAME)
        self.tracker_filepath = os.path.join(state_dir, Validator.TRACKER_FILENAME)
        self.version_filepath = os.path.join(state_dir, Validator.VERSION_FILENAME)

        # Perform a one-time migration of the state files from the old path to the new path
        self.maybe_migrate_state_files()

        # Check if the version has changed since we last restarted.
        previous_version = utils.get_version(self.version_filepath)
        utils.save_version(self.version_filepath, constants.__spec_version__)

        # If this is an upgrade, blow away state so that everything is re-evaluated.
        if previous_version != constants.__spec_version__:
            bt.logging.info(
                f"Validator updated. Previous version={previous_version}. Current version={constants.__spec_version__}"
            )
            if os.path.exists(self.uids_filepath):
                bt.logging.info(
                    f"Because the validator updated, deleting {self.uids_filepath} so everything is re-evaluated."
                )
                os.remove(self.uids_filepath)
            if os.path.exists(self.tracker_filepath):
                bt.logging.info(
                    f"Because the validator updated, deleting {self.tracker_filepath} so everything is re-evaluated."
                )
                os.remove(self.tracker_filepath)

        # Initialize the model tracker.
        if not os.path.exists(self.tracker_filepath):
            bt.logging.warning("No tracker state file found. Starting from scratch.")
        else:
            self.model_tracker.load_state(self.tracker_filepath)

        # Initialize the UIDs to eval.
        if not os.path.exists(self.uids_filepath):
            bt.logging.warning("No uids state file found. Starting from scratch.")
            hotkeys = (
                self.model_tracker.get_miner_hotkey_to_model_metadata_dict().keys()
            )
            uids = []
            for hotkey in hotkeys:
                if hotkey in self.metagraph.hotkeys:
                    uids.append(self.metagraph.hotkeys.index(hotkey))
            self.uids_to_eval = set(uids)
        else:
            with open(self.uids_filepath, "rb") as f:
                self.uids_to_eval = pickle.load(f)
                self.pending_uids_to_eval = pickle.load(f)

        # Setup a miner iterator to ensure we update all miners.
        # This subnet does not differentiate between miner and validators so this is passed all uids.
        self.miner_iterator = MinerIterator(self.metagraph.uids.tolist())

        # Setup a ModelMetadataStore
        self.metadata_store = ChainModelMetadataStore(
            self.subtensor, self.wallet, self.config.netuid
        )

        # Setup a RemoteModelStore
        self.remote_store = HuggingFaceModelStore()

        # Setup a LocalModelStore
        self.local_store = DiskModelStore(base_dir=self.config.model_dir)

        # Setup a model updater to download models as needed to match the latest provided miner metadata.
        self.model_updater = ModelUpdater(
            metadata_store=self.metadata_store,
            remote_store=self.remote_store,
            local_store=self.local_store,
            model_tracker=self.model_tracker,
        )

        # Create a metagraph lock to avoid cross thread access issues in the update and clean loop.
        self.metagraph_lock = threading.RLock()

        # == Initialize the update thread ==
        self.stop_event = threading.Event()
        self.update_thread = threading.Thread(target=self.update_models, daemon=True)
        self.update_thread.start()

        # == Initialize the cleaner thread to remove outdated models ==
        self.clean_thread = threading.Thread(target=self.clean_models, daemon=True)
        self.clean_thread.start()

    def __del__(self):
        if hasattr(self, "stop_event"):
            self.stop_event.set()
            self.update_thread.join()
            self.clean_thread.join()

    def new_wandb_run(self):
        """Creates a new wandb run to save information to."""
        # Create a unique run id for this run.
        run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = "validator-" + str(self.uid) + "-" + run_id
        self.wandb_run = wandb.init(
            name=name,
            project=constants.WANDB_PROJECT,
            entity="opentensor-dev",
            config={
                "uid": self.uid,
                "hotkey": self.wallet.hotkey.ss58_address,
                "run_name": run_id,
                "version": constants.__version__,
                "type": "validator",
            },
            allow_val_change=True,
        )

        bt.logging.debug(f"Started a new wandb run: {name}")

    def maybe_migrate_state_files(self):
        """Performs a one-time migration of the state files from the old path to the new path."""
        if utils.move_file_if_exists(
            os.path.join(self.state_path_old(), Validator.UIDS_FILENAME),
            self.uids_filepath,
        ):
            bt.logging.success(
                f"Moved {Validator.UIDS_FILENAME} from old state path to new state path."
            )
        if utils.move_file_if_exists(
            os.path.join(self.state_path_old(), Validator.TRACKER_FILENAME),
            self.tracker_filepath,
        ):
            bt.logging.success(
                f"Moved {Validator.TRACKER_FILENAME} from old state path to new state path."
            )
        if utils.move_file_if_exists(
            os.path.join(self.state_path_old(), Validator.VERSION_FILENAME),
            self.version_filepath,
        ):
            bt.logging.success(
                f"Moved {Validator.VERSION_FILENAME} from old state path to new state path."
            )

    def save_state(self):
        """Saves the state of the validator to a file."""

        bt.logging.trace("Saving validator state.")
        if not os.path.exists(self.state_path()):
            os.makedirs(self.state_path())

        with self.pending_uids_to_eval_lock:
            # Save the state of the validator uids to file.
            with open(self.uids_filepath, "wb") as f:
                pickle.dump(self.uids_to_eval, f)
                pickle.dump(self.pending_uids_to_eval, f)

        # Save the state of the tracker to file.
        self.model_tracker.save_state(self.tracker_filepath)

    def update_models(self):
        # Track how recently we updated each uid from sequential iteration.
        uid_last_checked_sequential = dict()
        # Track how recently we updated each uid from incentives.
        uid_last_checked_incentive = dict()
        # Track how recently we updated each uid from sequential iteration.
        uid_last_checked_sequential = dict()
        # Track how recently we updated each uid from incentives.
        uid_last_checked_incentive = dict()

        # The below loop iterates across all miner uids and checks to see
        # if they should be updated.
        while not self.stop_event.is_set():
            try:
                # Limit the number of pending uids, waiting for the eval loop to process them.
                pending_uid_count = 0
                current_uid_count = 0
                with self.pending_uids_to_eval_lock:
                    pending_uid_count = len(self.pending_uids_to_eval)
                    current_uid_count = len(self.uids_to_eval)

                # Only allow at most sample max models. Typically this will be carryover from sample_min + new models.
                while pending_uid_count + current_uid_count >= self.config.sample_max:
                    # Wait 5 minutes for the eval loop to process them.
                    bt.logging.info(
                        f"Update loop: Already {self.config.sample_max} synced models pending eval. Checking again in 5 minutes."
                    )
                    time.sleep(300)
                    # Check to see if the pending uids have been cleared yet.
                    with self.pending_uids_to_eval_lock:
                        pending_uid_count = len(self.pending_uids_to_eval)
                        current_uid_count = len(self.uids_to_eval)

                # Get the next uid to check
                next_uid = None

                # First check for any uids with incentives above threshold on the chain.
                # This will catch updates to current best models and models other valis have incentivized faster.
                with self.metagraph_lock:
                    incentives = self.metagraph.I

                for uid, incentive in enumerate(incentives):
                    # Use .item() to get the number value since this is a tensor.
                    if (
                        incentive.item()
                        >= constants.update_priority_incentive_threshold
                    ):
                        # Confirm that we haven't checked it within the chain update cadence in this path.
                        time_diff = (
                            dt.datetime.now() - uid_last_checked_incentive[uid]
                            if uid in uid_last_checked_incentive
                            else constants.chain_update_cadence  # Default to being stale enough to check again.
                        )
                        if time_diff >= constants.chain_update_cadence:
                            # Check this uid next and update that we have checked it in this path..
                            next_uid = uid
                            uid_last_checked_incentive[uid] = dt.datetime.now()
                            break

                # Then iterate sequentially for new models.
                # In this case if we have seen the uid in chain update cadence we have seen all of them and should wait.
                if next_uid is None:
                    next_uid = next(self.miner_iterator)

                    # Confirm that we haven't checked it in the chain update cadence
                    time_diff = (
                        dt.datetime.now() - uid_last_checked_sequential[next_uid]
                        if next_uid in uid_last_checked_sequential
                        else None
                    )

                    if time_diff and time_diff < constants.chain_update_cadence:
                        # If we have seen it within chain update cadence then sleep until it has been at least that long.
                        time_to_sleep = (
                            constants.chain_update_cadence - time_diff
                        ).total_seconds()
                        bt.logging.trace(
                            f"Update loop has already processed all UIDs in the last {constants.chain_update_cadence}. Sleeping {time_to_sleep} seconds."
                        )
                        time.sleep(time_to_sleep)

                    uid_last_checked_sequential[next_uid] = dt.datetime.now()
                    uid_last_checked_sequential[next_uid] = dt.datetime.now()

                # Get their hotkey from the metagraph.
                with self.metagraph_lock:
                    hotkey = self.metagraph.hotkeys[next_uid]

                # Compare metadata and tracker, syncing new model from remote store to local if necessary.
                updated = asyncio.run(self.model_updater.sync_model(hotkey))

                if updated:
                    bt.logging.trace(
                        f"Updated model for UID={next_uid}. Was new = {updated}"
                    )

                # Ensure we eval the new model on the next loop.
                if updated:
                    with self.pending_uids_to_eval_lock:
                        self.pending_uids_to_eval.add(next_uid)
                        bt.logging.debug(
                            f"Found a new model for UID={next_uid}. It will be evaluated on the next loop."
                        )

            except Exception as e:
                bt.logging.error(
                    f"Error in update loop: {e} \n {traceback.format_exc()}"
                )

        bt.logging.info("Exiting update models loop.")

    def clean_models(self):
        # Delay the clean-up thread until the update loop has had time to run one full pass after an upgrade.
        # This helps prevent unnecessarily deleting a model which is on disk, but hasn't yet been re-added to the
        # model tracker by the update loop.
        time.sleep(dt.timedelta(hours=1).total_seconds())

        # The below loop checks to clear out all models in local storage that are no longer referenced.
        while not self.stop_event.is_set():
            try:
                bt.logging.trace("Starting cleanup of stale models.")

                # Get a mapping of all hotkeys to model ids.
                hotkey_to_model_metadata = (
                    self.model_tracker.get_miner_hotkey_to_model_metadata_dict()
                )
                hotkey_to_model_id = {
                    hotkey: metadata.id
                    for hotkey, metadata in hotkey_to_model_metadata.items()
                }

                # Find all hotkeys that are currently being evaluated or pending eval.
                uids_to_keep = set()
                with self.pending_uids_to_eval_lock:
                    uids_to_keep = set(self.uids_to_eval).union(
                        self.pending_uids_to_eval
                    )

                hotkeys_to_keep = set()
                with self.metagraph_lock:
                    for uid in uids_to_keep:
                        hotkeys_to_keep.add(self.metagraph.hotkeys[uid])

                # Only keep those hotkeys.
                evaluated_hotkeys_to_model_id = {
                    hotkey: model_id
                    for hotkey, model_id in hotkey_to_model_id.items()
                    if hotkey in hotkeys_to_keep
                }

                self.local_store.delete_unreferenced_models(
                    valid_models_by_hotkey=evaluated_hotkeys_to_model_id,
                    grace_period_seconds=300,
                )
            except Exception as e:
                bt.logging.error(f"Error in clean loop: {e}")

            # Only check every 5 minutes.
            time.sleep(dt.timedelta(minutes=5).total_seconds())

        bt.logging.info("Exiting clean models loop.")

    async def try_set_weights(self, ttl: int):
        async def _try_set_weights():
            try:
                self.weights.nan_to_num(0.0)
                self.subtensor.set_weights(
                    netuid=self.config.netuid,
                    wallet=self.wallet,
                    uids=self.metagraph.uids,
                    weights=self.weights,
                    wait_for_inclusion=False,
                    version_key=constants.weights_version_key,
                )
            except:
                bt.logging.warning("Failed to set weights. Trying again later.")

            ws, ui = self.weights.topk(len(self.weights))
            table = Table(title="All Weights")
            table.add_column("uid", justify="right", style="cyan", no_wrap=True)
            table.add_column("weight", style="magenta")
            for index, weight in list(zip(ui.tolist(), ws.tolist())):
                table.add_row(str(index), str(round(weight, 4)))
            console = Console()
            console.print(table)

        try:
            bt.logging.debug(f"Setting weights.")
            await asyncio.wait_for(_try_set_weights(), ttl)
            bt.logging.debug(f"Finished setting weights.")
        except asyncio.TimeoutError:
            bt.logging.error(f"Failed to set weights after {ttl} seconds")

    async def try_sync_metagraph(self, ttl: int):
        def sync_metagraph(endpoint):
            metagraph = bt.subtensor(endpoint).metagraph(self.config.netuid)
            metagraph.save()

        process = multiprocessing.Process(
            target=sync_metagraph, args=(self.subtensor.chain_endpoint,)
        )
        process.start()
        process.join(timeout=ttl)
        if process.is_alive():
            process.terminate()
            process.join()
            bt.logging.error(f"Failed to sync metagraph after {ttl} seconds")
            return

        bt.logging.info("Synced metagraph")
        with self.metagraph_lock:
            self.metagraph.load()
            self.miner_iterator.set_miner_uids(self.metagraph.uids.tolist())
            self.model_tracker.on_hotkeys_updated(set(self.metagraph.hotkeys))

    async def try_run_step(self, ttl: int):
        async def _try_run_step():
            await self.run_step()

        try:
            bt.logging.trace("Running step.")
            await asyncio.wait_for(_try_run_step(), ttl)
            bt.logging.trace("Finished running step.")
        except asyncio.TimeoutError:
            bt.logging.error(f"Failed to run step after {ttl} seconds")

    async def run_step(self):
        """
        Executes a step in the evaluation process of models. This function performs several key tasks:
        1. Identifies valid models for evaluation (top 30 from last run + newly updated models).
        2. Generates random pages for evaluation and prepares batches for each page from the dataset.
        3. Computes the scoring for each model based on the losses incurred on the evaluation batches.
        4. Calculates wins and win rates for each model to determine their performance relative to others.
        5. Updates the weights of each model based on their performance and applies a softmax normalization.
        6. Implements a blacklist mechanism to remove underperforming models from the evaluation set.
        7. Logs all relevant data for the step, including model IDs, pages, batches, wins, win rates, and losses.
        """

        # Add uids with newly updated models to the upcoming batch of evaluations.
        with self.pending_uids_to_eval_lock:
            self.uids_to_eval.update(self.pending_uids_to_eval)
            self.pending_uids_to_eval.clear()

        # Pull relevant uids for step. If they aren't found in the model tracker on eval they will be skipped.
        uids = list(self.uids_to_eval)

        if not uids:
            bt.logging.debug(
                "No uids to eval. Waiting 5 minutes to download some models."
            )
            time.sleep(300)
            return

        # Keep track of which block this uid last updated their model.
        # Default to an infinite block if we can't retrieve the metadata for the miner.
        uid_to_block = defaultdict(lambda: math.inf)

        # Generate random pages for evaluation and prepare batches for each page
        # the dataset contains >900 million pages to eval over.
        pages = [
            random.randint(1, pt.dataset.SubsetFalconLoader.max_pages)
            for _ in range(self.config.pages_per_eval)
        ]

        # Temporary ugliness to load the batches with both the previous tokenizer
        # and the new tokenizer. batches_old can be removed once the block is newer
        # than the point we allow 7B parameter models.
        batches_old = list(
            pt.dataset.SubsetFalconLoader(
                batch_size=constants.batch_size,
                sequence_length=constants.SEQUENCE_LENGTH_1,
                pages=pages,
                tokenizer=pt.model.get_old_tokenizer(cache_dir=self.config.model_dir),
            )
        )

        batches = list(
            pt.dataset.SubsetFalconLoader(
                batch_size=constants.batch_size,
                sequence_length=constants.SEQUENCE_LENGTH_2,
                pages=pages,
                tokenizer=pt.model.get_tokenizer(cache_dir=self.config.model_dir),
            )
        )

        bt.logging.debug(f"Computing losses on {uids} with pages {pages}")

        # Compute model losses on batches.
        losses_per_uid = {muid: None for muid in uids}

        load_model_perf = PerfMonitor("Eval: Load model")
        compute_loss_perf = PerfMonitor("Eval: Compute loss")

        for uid_i in uids:
            bt.logging.trace(f"Computing model losses for uid:{uid_i}.")

            # Check that the model is in the tracker.
            hotkey = self.metagraph.hotkeys[uid_i]
            model_i_metadata = self.model_tracker.get_model_metadata_for_miner_hotkey(
                hotkey
            )

            losses = [math.inf for _ in range(len(batches))]

            if model_i_metadata != None:
                try:
                    # Update the block this uid last updated their model.
                    uid_to_block[uid_i] = model_i_metadata.block
                    # Get criteria to evaluate model with based on block.
                    criteria = model_utils.get_model_criteria(model_i_metadata.block)
                    # Use bfloat16 and flash attention optimization based on block.
                    optimized = criteria.optimized
                    # Use tokenizer based on block.
                    tokenizer_identifier = criteria.tokenizer_identifier

                    # Get the model locally and evaluate its loss.
                    model_i = None
                    with load_model_perf.sample():
                        model_i = self.local_store.retrieve_model(
                            hotkey,
                            model_i_metadata.id,
                            optimized,
                        )

                    with compute_loss_perf.sample():
                        # Run each computation in a subprocess so that the GPU is reset between each model.
                        batches_to_use = (
                            batches_old
                            if tokenizer_identifier == TokenizerIdentifier.DISTILGPT_2
                            else batches
                        )
                        losses = utils.run_in_subprocess(
                            functools.partial(
                                pt.validation.compute_losses,
                                model_i.pt_model,
                                batches_to_use,
                                self.config.device,
                            ),
                            ttl=360,
                            mode="spawn",
                        )
                    del model_i
                except Exception as e:
                    bt.logging.error(
                        f"Error in eval loop: {e}. Setting losses for uid: {uid_i} to infinity."
                    )
            else:
                bt.logging.debug(
                    f"Unable to load the model for {uid_i}. Setting loss to inifinity."
                )

            losses_per_uid[uid_i] = losses
            average_model_loss = sum(losses) / len(losses)
            bt.logging.trace(
                f"Computed model losses for uid:{uid_i} with average loss: {average_model_loss}"
            )

        # Compute wins and win rates per uid.
        wins, win_rate = pt.validation.compute_wins(
            uids, losses_per_uid, batches, uid_to_block
        )

        # Compute softmaxed weights based on win rate.
        model_weights = torch.tensor(
            [win_rate[uid] for uid in uids], dtype=torch.float32
        )
        step_weights = torch.softmax(model_weights / constants.temperature, dim=0)

        # Update weights based on moving average.
        new_weights = torch.zeros_like(self.weights)
        for i, uid_i in enumerate(uids):
            new_weights[uid_i] = step_weights[i]
        new_weights /= new_weights.sum()
        self.weights = (
            constants.alpha * self.weights + (1 - constants.alpha) * new_weights
        )
        self.weights = self.weights.nan_to_num(0.0)

        # Filter based on win rate removing all by the sample_min best models for evaluation.
        # First remove any models that have an infinite loss and 0 weight.
        filtered_win_rate = {
            uid: wr
            for uid, wr in win_rate.items()
            if not all(math.isinf(x) for x in losses_per_uid.get(uid, [math.inf]))
            or self.weights[uid] > 0
        }
        self.uids_to_eval = set(
            sorted(filtered_win_rate, key=filtered_win_rate.get, reverse=True)[
                : self.config.sample_min
            ]
        )

        # Save state
        self.save_state()

        # Log the performance of the eval loop.
        bt.logging.debug(load_model_perf.summary_str())
        bt.logging.debug(compute_loss_perf.summary_str())

        # Log to screen and wandb.
        self.log_step(
            uids,
            uid_to_block,
            pages,
            batches,
            wins,
            win_rate,
            losses_per_uid,
            load_model_perf.summary_str(),
            compute_loss_perf.summary_str(),
        )

        # Increment the number of completed run steps by 1
        self.run_step_count += 1

    def log_step(
        self,
        uids,
        uid_to_block,
        pages,
        batches,
        wins,
        win_rate,
        losses_per_uid,
        load_model_perf_str,
        compute_loss_perf_str,
    ):
        # Build step log
        step_log = {
            "timestamp": time.time(),
            "pages": pages,
            "uids": uids,
            "uid_data": {},
        }
        for i, uid in enumerate(uids):
            step_log["uid_data"][str(uid)] = {
                "uid": uid,
                "block": uid_to_block[uid],
                "average_loss": sum(losses_per_uid[uid]) / len(batches),
                "win_rate": win_rate[uid],
                "win_total": wins[uid],
                "weight": self.weights[uid].item(),
            }
        table = Table(title="Step")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("average_loss", style="magenta")
        table.add_column("win_rate", style="magenta")
        table.add_column("win_total", style="magenta")
        table.add_column("weights", style="magenta")
        table.add_column("block", style="magenta")
        for uid in uids:
            try:
                table.add_row(
                    str(uid),
                    str(round(step_log["uid_data"][str(uid)]["average_loss"], 4)),
                    str(round(step_log["uid_data"][str(uid)]["win_rate"], 4)),
                    str(step_log["uid_data"][str(uid)]["win_total"]),
                    str(round(self.weights[uid].item(), 4)),
                    str(step_log["uid_data"][str(uid)]["block"]),
                )
            except:
                pass
        console = Console()
        console.print(table)

        ws, ui = self.weights.topk(len(self.weights))
        table = Table(title="Weights > 0.001")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("weight", style="magenta")
        for index, weight in list(zip(ui.tolist(), ws.tolist())):
            if weight > 0.001:
                table.add_row(str(index), str(round(weight, 4)))
        console = Console()
        console.print(table)

        # Sink step log.
        bt.logging.trace(f"Step results: {step_log}")

        if self.config.wandb.on and not self.config.offline:
            # If we have already completed X steps then we will complete the current wandb run and make a new one.
            if (
                self.run_step_count
                and self.run_step_count % constants.MAX_RUN_STEPS_PER_WANDB_RUN == 0
            ):
                bt.logging.trace(
                    f"Validator has completed {self.run_step_count} run steps. Creating a new wandb run."
                )
                self.wandb_run.finish()
                self.new_wandb_run()

            original_format_json = json.dumps(step_log)
            uids = step_log["uids"]
            uid_data = step_log["uid_data"]

            # Create a new dictionary with the required format
            graphed_data = {
                "time": time.time(),
                "block": self.metagraph.block.item(),
                "uid_data": {
                    str(uid): uid_data[str(uid)]["average_loss"] for uid in uids
                },
                "weight_data": {str(uid): self.weights[uid].item() for uid in uids},
                "load_model_perf_log": load_model_perf_str,
                "compute_model_perf_log": compute_loss_perf_str,
            }
            bt.logging.trace("Logging to Wandb")
            self.wandb_run.log(
                {**graphed_data, "original_format_json": original_format_json},
                step=self.global_step,
            )

    async def run(self):
        while True:
            try:
                while (
                    self.metagraph.block.item() - self.last_epoch
                    < self.config.blocks_per_epoch
                ):
                    await self.try_run_step(ttl=60 * 20)
                    await self.try_sync_metagraph(ttl=60)
                    self.save_state()
                    bt.logging.debug(
                        f"{self.metagraph.block.item() - self.last_epoch } / {self.config.blocks_per_epoch} blocks until next epoch."
                    )
                    self.global_step += 1

                if not self.config.dont_set_weights and not self.config.offline:
                    await self.try_set_weights(ttl=60)
                self.last_epoch = self.metagraph.block.item()
                self.epoch_step += 1

            except KeyboardInterrupt:
                bt.logging.info(
                    "KeyboardInterrupt caught, gracefully closing the wandb run..."
                )
                if self.wandb_run:
                    self.wandb_run.finish()
                exit()

            except Exception as e:
                bt.logging.error(
                    f"Error in validator loop \n {e} \n {traceback.format_exc()}"
                )


if __name__ == "__main__":
    asyncio.run(Validator().run())
