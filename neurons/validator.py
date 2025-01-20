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

# Due to the implementation of disable_progress_bars(), this has to be the first import+call in the application relating to huggingface
from huggingface_hub.utils import disable_progress_bars

disable_progress_bars()

import asyncio
import copy
import dataclasses
import datetime as dt
import functools
import json
import logging
import math
import os
import pickle
import threading
import time
import random
import traceback
import typing
from collections import defaultdict
from websockets.exceptions import InvalidStatus

import bittensor as bt
import torch
import wandb
from bittensor.utils.btlogging.defines import BITTENSOR_LOGGER_NAME
from bittensor.utils.btlogging.helpers import all_loggers
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
from retry import retry
from rich.console import Console
from rich.table import Table
from taoverse.metagraph import utils as metagraph_utils
from taoverse.metagraph.metagraph_syncer import MetagraphSyncer
from taoverse.metagraph.miner_iterator import MinerIterator
from taoverse.model import utils as model_utils
from taoverse.model.competition import utils as competition_utils
from taoverse.model.competition.competition_tracker import CompetitionTracker
from taoverse.model.competition.data import Competition
from taoverse.model.competition.epsilon import EpsilonFunc, FixedEpsilon
from taoverse.model.data import EvalResult
from taoverse.model.model_tracker import ModelTracker
from taoverse.model.model_updater import MinerMisconfiguredError, ModelUpdater
from taoverse.model.storage.chain.chain_model_metadata_store import (
    ChainModelMetadataStore,
)
from taoverse.model.storage.disk.disk_model_store import DiskModelStore
from taoverse.model.storage.hugging_face.hugging_face_model_store import (
    HuggingFaceModelStore,
)
from taoverse.utilities import logging, utils
from taoverse.utilities.perf_monitor import PerfMonitor

import constants
import pretrain as pt
from competitions.data import CompetitionId
from model.retry import should_retry_model
from neurons import config

from taoverse.model.eval.task import EvalTask
from pretrain.datasets.factory import DatasetLoaderFactory
from pretrain.datasets.ids import DatasetId
from pretrain.dataset import SubsetLoader
from pretrain.eval.sample import EvalSample
from pretrain.validation import ScoreDetails

os.environ["TOKENIZERS_PARALLELISM"] = "true"


@dataclasses.dataclass
class PerUIDEvalState:
    """State tracked per UID in the eval loop"""

    # The block the model was submitted.
    block: int = math.inf

    # The hotkey for the UID at the time of eval.
    hotkey: str = "Unknown"

    # The hugging face repo name.
    repo_name: str = "Unknown"

    # The model's score
    score: float = math.inf

    # Details about the model's score.
    score_details: typing.Dict[str, ScoreDetails] = dataclasses.field(
        default_factory=dict
    )


class Validator:
    MODEL_TRACKER_FILENAME = "model_tracker.pickle"
    COMPETITION_TRACKER_FILENAME = "competition_tracker.pickle"
    UIDS_FILENAME = "uids.pickle"
    VERSION_FILENAME = "version.txt"

    def state_path(self) -> str:
        """
        Returns the file path for storing validator state.

        Returns:
        str: A string representing the file path.
        """
        return os.path.join(self.config.model_dir, "vali-state")

    def _configure_logging(self, config: bt.config) -> None:
        # BT logging is noisy, so set it to only log errors.
        bt.logging.set_warning()

        # Setting logging level on bittensor messes with all loggers, which we don't want, so set explicitly to warning here.
        for logger in all_loggers():
            if not logger.name.startswith(BITTENSOR_LOGGER_NAME):
                logger.setLevel(logging.WARNING)

        # Configure the Taoverse logger, which is our primary logger.
        utils.logging.reinitialize()
        utils.configure_logging(config)

    def __init__(self):
        self.config = config.validator_config()
        self._configure_logging(self.config)

        logging.info(f"Starting validator with config: {self.config}")

        # === Bittensor objects ====
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.weights_subtensor = bt.subtensor(config=self.config)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        # self.metagraph = self.subtensor.metagraph(self.config.netuid, lite=False)

        # Setup metagraph syncer for the subnet based on config. This is non-lite for getting weights by vali.
        syncer_subtensor = bt.subtensor(config=self.config)
        self.subnet_metagraph_syncer = MetagraphSyncer(
            syncer_subtensor,
            config={
                self.config.netuid: dt.timedelta(minutes=20).total_seconds(),
            },
            lite=False,
        )
        # Perform an initial sync of all tracked metagraphs.
        self.subnet_metagraph_syncer.do_initial_sync()
        self.subnet_metagraph_syncer.start()
        # Get initial metagraphs.
        self.metagraph: bt.metagraph = self.subnet_metagraph_syncer.get_metagraph(
            self.config.netuid
        )

        # Register a listener for metagraph updates.
        self.subnet_metagraph_syncer.register_listener(
            self._on_subnet_metagraph_updated,
            netuids=[self.config.netuid],
        )

        torch.backends.cudnn.benchmark = True

        # Dont check registration status if offline.
        if not self.config.offline:
            self.uid = metagraph_utils.assert_registered(self.wallet, self.metagraph)

        # Track how may run_steps this validator has completed.
        self.run_step_count = 0

        # Dont log to wandb if offline.
        if not self.config.offline and self.config.wandb.on:
            self._new_wandb_run()

        # === Running args ===
        self.weight_lock = threading.RLock()
        self.weights = torch.zeros_like(torch.from_numpy(self.metagraph.S))
        self.global_step = 0

        self.uids_to_eval: typing.Dict[CompetitionId, typing.Set] = defaultdict(set)

        # Create a set of newly added uids that should be evaluated on the next loop.
        self.pending_uids_to_eval_lock = threading.RLock()
        self.pending_uids_to_eval: typing.Dict[CompetitionId, typing.Set] = defaultdict(
            set
        )

        # Setup a model tracker to track which miner is using which model id.
        self.model_tracker = ModelTracker()

        # Setup a competition tracker to track weights across different competitions.
        self.competition_tracker = CompetitionTracker(
            num_neurons=len(self.metagraph.uids), alpha=constants.alpha
        )

        # Construct the filepaths to save/load state.
        state_dir = self.state_path()
        os.makedirs(state_dir, exist_ok=True)

        self.uids_filepath = os.path.join(state_dir, Validator.UIDS_FILENAME)
        self.model_tracker_filepath = os.path.join(
            state_dir, Validator.MODEL_TRACKER_FILENAME
        )
        self.competition_tracker_filepath = os.path.join(
            state_dir, Validator.COMPETITION_TRACKER_FILENAME
        )
        self.version_filepath = os.path.join(state_dir, Validator.VERSION_FILENAME)

        # Check if the version has changed since we last restarted.
        previous_version = utils.get_version(self.version_filepath)
        utils.save_version(self.version_filepath, constants.__spec_version__)

        # If this is an upgrade, blow away state so that everything is re-evaluated.
        if previous_version != constants.__spec_version__:
            logging.info(
                f"Validator updated. Previous version={previous_version}. Current version={constants.__spec_version__}"
            )
            if os.path.exists(self.uids_filepath):
                logging.info(
                    f"Because the validator updated, deleting {self.uids_filepath} so everything is re-evaluated."
                )
                os.remove(self.uids_filepath)
            if os.path.exists(self.model_tracker_filepath):
                logging.info(
                    f"Because the validator updated, deleting {self.model_tracker_filepath} so everything is re-evaluated."
                )
                os.remove(self.model_tracker_filepath)

        # Initialize the model tracker.
        if not os.path.exists(self.model_tracker_filepath):
            logging.warning("No model tracker state file found. Starting from scratch.")
        else:
            try:
                self.model_tracker.load_state(self.model_tracker_filepath)
            except Exception as e:
                logging.warning(
                    f"Failed to load model tracker state. Reason: {e}. Starting from scratch."
                )

        # Initialize the competition tracker.
        if not os.path.exists(self.competition_tracker_filepath):
            logging.warning(
                "No competition tracker state file found. Starting from scratch."
            )
        else:
            try:
                self.competition_tracker.load_state(self.competition_tracker_filepath)
            except Exception as e:
                logging.warning(
                    f"Failed to load competition tracker state. Reason: {e}. Starting from scratch."
                )

        # Also update our internal weights based on the tracker.
        cur_block = self._get_current_block()

        # Get the competition schedule for the current block.
        # This is a list of competitions
        competition_schedule: typing.List[Competition] = (
            competition_utils.get_competition_schedule_for_block(
                block=cur_block,
                schedule_by_block=constants.COMPETITION_SCHEDULE_BY_BLOCK,
            )
        )
        with self.weight_lock:
            self.weights = self.competition_tracker.get_subnet_weights(
                competition_schedule
            )

        # Initialize the UIDs to eval.
        if not os.path.exists(self.uids_filepath):
            logging.warning("No uids state file found. Starting from scratch.")
        else:
            try:
                with open(self.uids_filepath, "rb") as f:
                    self.uids_to_eval = pickle.load(f)
                    self.pending_uids_to_eval = pickle.load(f)
            except Exception as e:
                logging.warning(
                    f"Failed to load uids to eval state. Reason: {e}. Starting from scratch."
                )
                # We also need to wipe the model tracker state in this case to ensure we re-evaluate all the models.
                self.model_tracker = ModelTracker()
                if os.path.exists(self.model_tracker_filepath):
                    logging.warning(
                        f"Because the uids to eval state failed to load, deleting model tracker state at {self.model_tracker_filepath} so everything is re-evaluated."
                    )
                    os.remove(self.model_tracker_filepath)

        # Setup a miner iterator to ensure we update all miners.
        # This subnet does not differentiate between miner and validators so this is passed all uids.
        self.miner_iterator = MinerIterator(self.metagraph.uids.tolist())

        # Setup a ModelMetadataStore
        chain_store_subtensor = bt.subtensor(config=self.config)
        self.metadata_store = ChainModelMetadataStore(
            subtensor=chain_store_subtensor,
            subnet_uid=self.config.netuid,
            wallet=self.wallet,
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

        # == Initialize the weight setting thread ==
        if not self.config.offline and not self.config.dont_set_weights:
            self.weight_thread = threading.Thread(target=self.set_weights, daemon=True)
            self.weight_thread.start()

    def __del__(self):
        if hasattr(self, "stop_event"):
            self.stop_event.set()
            self.update_thread.join()
            self.clean_thread.join()

    def _new_wandb_run(self):
        """Creates a new wandb run to save information to."""

        # Create a unique run id for this run.
        run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = "validator-" + str(self.uid) + "-" + run_id
        self.wandb_run = wandb.init(
            name=name,
            project=self.config.wandb_project,
            entity="macrocosmos",
            config={
                "uid": self.uid,
                "hotkey": self.wallet.hotkey.ss58_address,
                "run_name": run_id,
                "version": constants.__version__,
                "validator version": constants.__validator_version__,
                "type": "validator",
            },
            allow_val_change=True,
        )

        logging.debug(f"Started a new wandb run: {name}")

    def save_state(self):
        """Saves the state of the validator to a file."""

        logging.trace("Saving validator state.")
        if not os.path.exists(self.state_path()):
            os.makedirs(self.state_path())

        with self.pending_uids_to_eval_lock:
            # Save the state of the validator uids to file.
            with open(self.uids_filepath, "wb") as f:
                pickle.dump(self.uids_to_eval, f)
                pickle.dump(self.pending_uids_to_eval, f)

        # Save the state of the trackers to file.
        self.model_tracker.save_state(self.model_tracker_filepath)
        self.competition_tracker.save_state(self.competition_tracker_filepath)

    def get_pending_and_current_uid_counts(self) -> typing.Tuple[int, int]:
        """Gets the total number of uids pending eval and currently being evaluated across all competitions.

        Returns:
            typing.Tuple[int, int]: Pending uid count, Current uid count.
        """
        pending_uid_count = 0
        current_uid_count = 0

        with self.pending_uids_to_eval_lock:
            # Loop through the uids across all competitions.
            for uids in self.pending_uids_to_eval.values():
                pending_uid_count += len(uids)
            for uids in self.uids_to_eval.values():
                current_uid_count += len(uids)

        return pending_uid_count, current_uid_count

    def update_models(self):
        """Updates the models in the local store based on the latest metadata from the chain."""

        # Track how recently we updated each uid from sequential iteration.
        uid_last_checked_sequential = dict()
        # Track how recently we checked the list of top models.
        last_checked_top_models_time = None

        # Delay the first update loop until the metagraph has been synced.
        time.sleep(60)

        # The below loop iterates across all miner uids and checks to see
        # if they should be updated.
        while not self.stop_event.is_set():
            try:
                # At most once per `scan_top_model_cadence`, check which models are being assigned weight by
                # the top validators and ensure they'll be evaluated soon.
                if (
                    not last_checked_top_models_time
                    or dt.datetime.now() - last_checked_top_models_time
                    > constants.scan_top_model_cadence
                ):
                    last_checked_top_models_time = dt.datetime.now()
                    self._queue_top_models_for_eval()

                # Top model check complete. Now continue with the sequential iterator to check for the next miner
                # to update.

                self._wait_for_open_eval_slot()

                # We have space to add more models for eval. Process the next UID.
                next_uid = next(self.miner_iterator)

                # Confirm that we haven't already checked it within the chain update cadence.
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
                    logging.trace(
                        f"Update loop has already processed all UIDs in the last {constants.chain_update_cadence}. Sleeping {time_to_sleep} seconds."
                    )
                    time.sleep(time_to_sleep)

                uid_last_checked_sequential[next_uid] = dt.datetime.now()
                curr_block = self._get_current_block()

                # Get their hotkey from the metagraph.
                with self.metagraph_lock:
                    hotkey = self.metagraph.hotkeys[next_uid]

                # Check if we should retry this model and force a sync if necessary.
                force_sync = False
                model_metadata = self.model_tracker.get_model_metadata_for_miner_hotkey(
                    hotkey
                )

                if model_metadata:
                    # Check if the model is already queued for eval.
                    is_queued_for_eval = False
                    with self.pending_uids_to_eval_lock:
                        is_queued_for_eval = (
                            next_uid
                            in self.pending_uids_to_eval[
                                model_metadata.id.competition_id
                            ]
                            or next_uid
                            in self.uids_to_eval[model_metadata.id.competition_id]
                        )

                    competition = competition_utils.get_competition_for_block(
                        model_metadata.id.competition_id,
                        curr_block,
                        constants.COMPETITION_SCHEDULE_BY_BLOCK,
                    )
                    if competition is not None and not is_queued_for_eval:
                        eval_history = (
                            self.model_tracker.get_eval_results_for_miner_hotkey(
                                hotkey, competition.id
                            )
                        )
                        force_sync = should_retry_model(
                            competition.constraints.epsilon_func,
                            curr_block,
                            eval_history,
                        )
                        if force_sync:
                            logging.debug(
                                f"Force downloading model for UID {next_uid} because it should be retried. Eval_history={eval_history}"
                            )

                        # Special case for 14B*. We want to retry if the model is competitive in either 14B or 14B*.
                        if competition.id == CompetitionId.B14_MODEL:
                            # Check that 14B* is currently running.
                            competition_14b = (
                                competition_utils.get_competition_for_block(
                                    CompetitionId.B14_MODEL_MULTI_DATASET,
                                    curr_block,
                                    constants.COMPETITION_SCHEDULE_BY_BLOCK,
                                )
                            )
                            if competition_14b is not None:
                                eval_history_14b_star = self.model_tracker.get_eval_results_for_miner_hotkey(
                                    hotkey, CompetitionId.B14_MODEL_MULTI_DATASET
                                )
                                force_sync_14b_star = should_retry_model(
                                    competition_14b.constraints.epsilon_func,
                                    curr_block,
                                    eval_history_14b_star,
                                )
                                if force_sync_14b_star:
                                    # Even if it is already logging for 14B, also log for 14B*.
                                    logging.debug(
                                        f"Force downloading model for UID {next_uid} because it should be retried for 14B*. Eval_history={eval_history_14b_star}"
                                    )
                                    force_sync = True

                # Compare metadata and tracker, syncing new model from remote store to local if necessary.
                try:
                    updated = asyncio.run(
                        self.model_updater.sync_model(
                            hotkey=hotkey,
                            curr_block=curr_block,
                            schedule_by_block=constants.COMPETITION_SCHEDULE_BY_BLOCK,
                            force=force_sync,
                        )
                    )
                except MinerMisconfiguredError as e:
                    self.model_tracker.on_model_evaluated(
                        hotkey,
                        0,  # Technically this is B7 but that is unused.
                        EvalResult(
                            block=curr_block,
                            score=math.inf,
                            # We don't care about the winning model for this check since we just need to log the model eval failure.
                            winning_model_block=0,
                            winning_model_score=0,
                        ),
                    )
                    raise e

                if updated:
                    metadata = self.model_tracker.get_model_metadata_for_miner_hotkey(
                        hotkey
                    )
                    if metadata is not None:
                        with self.pending_uids_to_eval_lock:
                            self.pending_uids_to_eval[metadata.id.competition_id].add(
                                next_uid
                            )
                            logging.debug(
                                f"Found a new model for UID={next_uid} for competition {metadata.id.competition_id}. It will be evaluated on the next loop."
                            )
                    else:
                        logging.warning(
                            f"Failed to find metadata for uid {next_uid} with hotkey {hotkey}"
                        )
            except InvalidStatus as e:
                logging.info(
                    f"Websocket exception in update loop: {e}. Waiting 3 minutes."
                )
                time.sleep(180)
            except (RepositoryNotFoundError, RevisionNotFoundError) as e:
                logging.trace(e)
            except MinerMisconfiguredError as e:
                logging.trace(e)
            except Exception as e:
                logging.error(f"Error in update loop: {e} \n {traceback.format_exc()}")

        logging.info("Exiting update models loop.")

    def _wait_for_open_eval_slot(self) -> None:
        """Waits until there is at least one slot open to download and evaluate a model."""
        pending_uid_count, current_uid_count = self.get_pending_and_current_uid_counts()

        while pending_uid_count + current_uid_count >= self.config.updated_models_limit:
            # Wait 5 minutes for the eval loop to process them.
            logging.info(
                f"Update loop: There are already {pending_uid_count + current_uid_count} synced models pending eval. Checking again in 5 minutes."
            )
            time.sleep(300)
            # Check to see if the pending uids have been cleared yet.
            pending_uid_count, current_uid_count = (
                self.get_pending_and_current_uid_counts()
            )

    def _queue_top_models_for_eval(self) -> None:
        # Take a deep copy of the metagraph for use in the top uid retry check.
        # The regular loop below will use self.metagraph which may be updated as we go.
        with self.metagraph_lock:
            metagraph = copy.deepcopy(self.metagraph)

        # Find any miner UIDs which top valis are assigning weight and aren't currently scheduled for an eval.
        # This is competition agnostic, as anything with weight is 'winning' a competition for some vali.
        top_miner_uids = metagraph_utils.get_top_miners(
            metagraph,
            constants.WEIGHT_SYNC_VALI_MIN_STAKE,
            constants.WEIGHT_SYNC_MINER_MIN_PERCENT,
        )

        with self.pending_uids_to_eval_lock:
            all_uids_to_eval = set()
            all_pending_uids_to_eval = set()
            # Loop through the uids across all competitions.
            for uids in self.uids_to_eval.values():
                all_uids_to_eval.update(uids)
            for uids in self.pending_uids_to_eval.values():
                all_pending_uids_to_eval.update(uids)

            # Reduce down to top models that are not in any competition yet.
            uids_to_add = top_miner_uids - all_uids_to_eval - all_pending_uids_to_eval

        for uid in uids_to_add:
            # Check when we last evaluated this model.
            hotkey = metagraph.hotkeys[uid]
            last_eval_block = self.model_tracker.get_block_last_evaluated(hotkey) or 0
            curr_block = self._get_current_block()
            if curr_block - last_eval_block >= constants.model_retry_cadence:
                try:
                    # It's been long enough - redownload this model and schedule it for eval.
                    # This still respects the eval block delay so that previously top uids can't bypass it.
                    try:
                        should_retry = asyncio.run(
                            self.model_updater.sync_model(
                                hotkey=hotkey,
                                curr_block=curr_block,
                                schedule_by_block=constants.COMPETITION_SCHEDULE_BY_BLOCK,
                                force=True,
                            )
                        )
                    except MinerMisconfiguredError as e:
                        self.model_tracker.on_model_evaluated(
                            hotkey,
                            0,  # Technically this is B7 but that is unused.
                            EvalResult(
                                block=curr_block,
                                score=math.inf,
                                # We don't care about the winning model for this check since we just need to log the model eval failure.
                                winning_model_block=0,
                                winning_model_score=0,
                            ),
                        )
                        raise e

                    if not should_retry:
                        continue

                    # Since this is a top model (as determined by other valis),
                    # we don't worry if self.pending_uids is already "full". At most
                    # there can be 10 * comps top models that we'd add here and that would be
                    # a wildy exceptional case. It would require every vali to have a
                    # different top model.
                    # Validators should only have ~1 winner per competition and we only check bigger valis
                    # so there should not be many simultaneous top models not already being evaluated.
                    top_model_metadata = (
                        self.model_tracker.get_model_metadata_for_miner_hotkey(hotkey)
                    )
                    if top_model_metadata is not None:
                        logging.trace(
                            f"Shortcutting to top model or retrying evaluation for previously discarded top model with incentive for UID={uid}"
                        )
                        with self.pending_uids_to_eval_lock:
                            self.pending_uids_to_eval[
                                top_model_metadata.id.competition_id
                            ].add(uid)
                    else:
                        logging.warning(
                            f"Failed to find metadata for uid {uid} with hotkey {hotkey}"
                        )

                except Exception:
                    logging.debug(
                        f"Failure in update loop for UID={uid} during top model check. {traceback.format_exc()}"
                    )

    def clean_models(self):
        """Cleans up models that are no longer referenced."""

        # Delay the clean-up thread until the update loop has had time to run one full pass after an upgrade.
        # This helps prevent unnecessarily deleting a model which is on disk, but hasn't yet been re-added to the
        # model tracker by the update loop.
        time.sleep(dt.timedelta(hours=1).total_seconds())

        # The below loop checks to clear out all models in local storage that are no longer referenced.
        while not self.stop_event.is_set():
            try:
                logging.trace("Starting cleanup of stale models.")

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
                    for pending_uids in self.pending_uids_to_eval.values():
                        uids_to_keep.update(pending_uids)
                    for eval_uids in self.uids_to_eval.values():
                        uids_to_keep.update(eval_uids)

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
                    grace_period_seconds=600,
                )
            except Exception as e:
                logging.error(f"Error in clean loop: {e}")

            # Only check every 5 minutes.
            time.sleep(dt.timedelta(minutes=5).total_seconds())

        logging.info("Exiting clean models loop.")

    def set_weights(self):
        """Set weights on the chain regularly."""

        # Check that we have some weights internally for startup situations.
        all_zero_weights = True
        while all_zero_weights is True:
            # Technically returns a tensor but it evaluates to true.
            with self.weight_lock:
                all_zero_weights = torch.all(self.weights == 0)
            logging.trace(
                "Waiting 60 seconds for internal weights before continuing to try set weights."
            )
            time.sleep(60)

        while not self.stop_event.is_set():
            try:
                set_weights_success = False
                while not set_weights_success:
                    set_weights_success, _ = asyncio.run(self.try_set_weights(ttl=60))
                    # Wait for 120 seconds before we try to set weights again.
                    if set_weights_success:
                        logging.info("Successfully set weights.")
                    else:
                        time.sleep(120)
            except Exception as e:
                logging.error(f"Error in set weights: {e}")

            # Only set weights once every hour
            time.sleep(60 * 60)

        logging.info("Exiting set weights loop.")

    async def try_set_weights(self, ttl: int) -> typing.Tuple[bool, str]:
        """Sets the weights on the chain with ttl, without raising exceptions if it times out."""

        async def _try_set_weights() -> typing.Tuple[bool, str]:
            with self.metagraph_lock:
                uids = self.metagraph.uids
            try:
                with self.weight_lock:
                    self.weights.nan_to_num(0.0)
                    weights_to_set = self.weights

                return self.weights_subtensor.set_weights(
                    netuid=self.config.netuid,
                    wallet=self.wallet,
                    uids=uids,
                    weights=weights_to_set.numpy(),
                    wait_for_inclusion=True,
                    version_key=constants.weights_version_key,
                    max_retries=1,
                )
            except Exception as e:
                logging.warning(
                    f"Failed to set weights due to {e}. Trying again later."
                )
                return (False, str(e))

        try:
            logging.debug(f"Setting weights.")
            status = await asyncio.wait_for(_try_set_weights(), ttl)
            logging.debug(f"Finished setting weights with status: {status}.")
            return status
        except asyncio.TimeoutError:
            logging.error(f"Failed to set weights after {ttl} seconds")
            return (False, f"Timeout after {ttl} seconds")

    def _get_current_block(self) -> int:
        """Returns the current block."""

        @retry(tries=5, delay=1, backoff=2)
        def _get_block_with_retry():
            return self.subtensor.block

        try:
            return _get_block_with_retry()
        except:
            logging.debug(
                "Failed to get the latest block from the chain. Using the block from the cached metagraph."
            )
            # Network call failed. Fallback to using the block from the metagraph,
            # even though it'll be a little stale.
            with self.metagraph_lock:
                return self.metagraph.block.item()

    def _on_subnet_metagraph_updated(
        self, metagraph: bt.metagraph, netuid: int
    ) -> None:
        """Processes an update to the metagraph for the subnet."""
        if netuid != self.config.netuid:
            logging.error(f"Unexpected subnet uid in subnet metagraph syncer: {netuid}")
            return

        with self.metagraph_lock:
            logging.info("Synced metagraph")
            self.metagraph = copy.deepcopy(metagraph)
            self.miner_iterator.set_miner_uids(self.metagraph.uids.tolist())
            self.model_tracker.on_hotkeys_updated(set(self.metagraph.hotkeys))

    def _get_seed(self):
        # Synchronize the random seed used by validators.
        try:

            @retry(tries=3, delay=1, backoff=2)
            def _get_seed_with_retry():
                return metagraph_utils.get_hash_of_sync_block(
                    self.subtensor, constants.SYNC_BLOCK_CADENCE
                )

            return _get_seed_with_retry()
        except:
            logging.trace(f"Failed to get hash of sync block. Using fallback seed.")
            return None

    async def try_run_step(self, ttl: int):
        """Runs a step with ttl in a background process, without raising exceptions if it times out."""

        async def _try_run_step():
            await self.run_step()

        try:
            logging.trace("Running step.")
            await asyncio.wait_for(_try_run_step(), ttl)
            logging.trace("Finished running step.")
        except asyncio.TimeoutError:
            logging.error(f"Failed to run step after {ttl} seconds")

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

        cur_block = self._get_current_block()

        # Get the competition schedule for the current block.
        # This is a list of competitions
        competition_schedule: typing.List[Competition] = (
            competition_utils.get_competition_schedule_for_block(
                block=cur_block,
                schedule_by_block=constants.COMPETITION_SCHEDULE_BY_BLOCK,
            )
        )

        # Every validator step should pick a single competition in a round-robin fashion
        competition = competition_schedule[self.global_step % len(competition_schedule)]
        # If the competition is 14b* we skip it since we run it concurrently with 14b instead.
        if competition.id == CompetitionId.B14_MODEL_MULTI_DATASET:
            logging.info(
                "Skipping step for B14* competition. It will be evaluated as part of the regular B14 competition."
            )
            return

        logging.info("Starting evaluation for competition: " + str(competition.id))

        running_14b_star = (
            competition.id == CompetitionId.B14_MODEL
            and cur_block < constants.BLOCK_MULTI_DATASETS
            and any(
                comp.id == CompetitionId.B14_MODEL_MULTI_DATASET
                for comp in competition_schedule
            )
        )

        if running_14b_star:
            competition_14b_star = competition_utils.get_competition_for_block(
                comp_id=CompetitionId.B14_MODEL_MULTI_DATASET,
                block=cur_block,
                schedule_by_block=constants.COMPETITION_SCHEDULE_BY_BLOCK,
            )
            logging.info("Additionally running competition 14B* in parallel to 14B.")

        # Add uids with newly updated models to the upcoming batch of evaluations.
        with self.pending_uids_to_eval_lock:
            self.uids_to_eval[competition.id].update(
                self.pending_uids_to_eval[competition.id]
            )
            self.pending_uids_to_eval[competition.id].clear()

        # Pull relevant uids for step. If they aren't found in the model tracker on eval they will be skipped.
        uids = list(self.uids_to_eval[competition.id])

        if not uids:
            logging.debug(f"No uids to eval for competition {competition.id}.")
            # Check if no competitions have uids, if so wait 5 minutes to download.
            pending_uid_count, current_uid_count = (
                self.get_pending_and_current_uid_counts()
            )
            if pending_uid_count + current_uid_count == 0:
                logging.debug(
                    "No uids to eval for any competition. Waiting 5 minutes to download models."
                )
                time.sleep(300)
            return

        uid_to_state = defaultdict(PerUIDEvalState)

        logging.trace(f"Current block: {cur_block}")

        if running_14b_star:
            # This will be set to a copy of 14b later with the additional eval losses appended.
            uid_to_state_14b_star = defaultdict(PerUIDEvalState)

        # Get the tokenizer
        tokenizer = pt.model.load_tokenizer(
            competition.constraints, cache_dir=self.config.model_dir
        )

        # Pull the latest sample data based on the competition.
        load_data_perf = PerfMonitor("Eval: Load data")

        # Try to synchronize the data used by validators.
        seed = self._get_seed()
        eval_tasks: typing.List[EvalTask] = []
        data_loaders: typing.List[SubsetLoader] = []
        samples: typing.List[typing.List[EvalSample]] = []

        logging.debug(f"Seed used for loading data is: {seed}.")

        # Load data based on the competition.
        with load_data_perf.sample():
            for eval_task in competition.eval_tasks:
                data_loader = DatasetLoaderFactory.get_loader(
                    dataset_id=eval_task.dataset_id,
                    dataset_kwargs=eval_task.dataset_kwargs,
                    seed=seed,
                    sequence_length=competition.constraints.sequence_length,
                    tokenizer=tokenizer,
                )
                batches = list(data_loader)

                # Shuffle before trancating the list
                random.Random(seed).shuffle(batches)

                if cur_block < constants.BLOCK_MULTI_DATASETS:
                    max_batches_per_dataset = len(batches)
                else:
                    max_batches_per_dataset = constants.MAX_BATCHES_PER_DATASET

                if batches:
                    eval_tasks.append(eval_task)
                    data_loaders.append(data_loader)

                    logging.debug(
                        f"Found {len(batches)} batches of size: {len(batches[0])} for data_loader: {data_loader.name}:{data_loader.config} over pages {data_loader.get_page_names()}. Up to {max_batches_per_dataset} batches were randomly chosen for evaluation."
                    )

                    samples.append(batches[:max_batches_per_dataset])

                else:
                    raise ValueError(
                        f"Did not find any data for data loader: {data_loader.name}"
                    )

        logging.debug(f"Competition {competition.id} | Computing losses on {uids}")

        if running_14b_star:
            # Pull the latest sample data based on the competition.
            load_data_perf_14b_star = PerfMonitor("Eval: Load data 14b star")

            # This is a temporary hack until 14b star is itself a full competition.
            eval_task_14b_star: EvalTask = None
            data_loaders_14b_star: typing.List[SubsetLoader] = []
            samples_14b_star: typing.List[typing.List[EvalSample]] = []

            with load_data_perf_14b_star.sample():
                # We only want the stack v2 here.
                for task_14b_star in competition_14b_star.eval_tasks:
                    if task_14b_star.name == "STACKV2_DEDUP":
                        data_loader_14b_star = DatasetLoaderFactory.get_loader(
                            dataset_id=task_14b_star.dataset_id,
                            dataset_kwargs=task_14b_star.dataset_kwargs,
                            seed=seed,
                            sequence_length=competition_14b_star.constraints.sequence_length,
                            tokenizer=tokenizer,
                        )
                        batches_14_star = list(data_loader_14b_star)
                        if batches_14_star:
                            eval_task_14b_star = task_14b_star
                            # We overwrite the weight here to 1 as we are handling the weighting by sample ratio later.
                            eval_task_14b_star.weight = 1
                            data_loaders_14b_star.append(data_loader_14b_star)

                            logging.debug(
                                f"For 14b*: found {len(batches_14_star)} batches of size: {len(batches_14_star[0])} for data_loader: {data_loader_14b_star.name} over pages {data_loader_14b_star.get_page_names()}"
                            )

                            samples_14b_star.append(batches_14_star)
                        else:
                            raise ValueError(
                                f"For 14b*: did not find any data for data loader: {data_loader_14b_star.name}"
                            )

            compute_loss_perf_14b_star = PerfMonitor("Eval: Compute loss 14b star")

        # Prepare evaluation.
        kwargs = competition.constraints.kwargs.copy()
        kwargs["use_cache"] = True

        load_model_perf = PerfMonitor("Eval: Load model")
        compute_loss_perf = PerfMonitor("Eval: Compute loss")

        for uid_i in uids:
            score: float = math.inf
            score_details = {task.name: ScoreDetails() for task in eval_tasks}
            if running_14b_star:
                score_14b_star: float = math.inf
                score_details_14b_star = {eval_task_14b_star.name: ScoreDetails()}

            logging.trace(f"Getting metadata for uid: {uid_i}.")

            # Check that the model is in the tracker.
            with self.metagraph_lock:
                hotkey = self.metagraph.hotkeys[uid_i]
                uid_to_state[uid_i].hotkey = hotkey

            model_i_metadata = self.model_tracker.get_model_metadata_for_miner_hotkey(
                hotkey
            )

            if (
                model_i_metadata is not None
                and model_i_metadata.id.competition_id == competition.id
            ):
                try:
                    logging.info(
                        f"Evaluating uid: {uid_i} / hotkey: {hotkey} with metadata: {model_i_metadata} and hf_url: {model_utils.get_hf_url(model_i_metadata)}."
                    )

                    # Update the block this uid last updated their model.
                    uid_to_state[uid_i].block = model_i_metadata.block
                    # Update the hf repo for this model.
                    uid_to_state[uid_i].repo_name = model_utils.get_hf_repo_name(
                        model_i_metadata
                    )

                    # Get the model locally and evaluate its loss.
                    model_i = None
                    with load_model_perf.sample():
                        model_i = self.local_store.retrieve_model(
                            hotkey, model_i_metadata.id, kwargs
                        )
                    # Currently all competitions define a default tokenizer, so we set it here.
                    model_i.tokenizer = tokenizer

                    with compute_loss_perf.sample():
                        # Run each computation in a subprocess so that the GPU is reset between each model.
                        score, score_details = utils.run_in_subprocess(
                            functools.partial(
                                pt.validation.score_model,
                                model_i,
                                eval_tasks,
                                samples,
                                self.config.device,
                            ),
                            ttl=430,
                            mode="spawn",
                        )
                    if running_14b_star:
                        with compute_loss_perf_14b_star.sample():
                            try:
                                score_14b_star, score_details_14b_star = (
                                    utils.run_in_subprocess(
                                        functools.partial(
                                            pt.validation.score_model,
                                            model_i,
                                            [eval_task_14b_star],
                                            samples_14b_star,
                                            self.config.device,
                                        ),
                                        ttl=430,
                                        mode="spawn",
                                    )
                                )
                            except Exception:
                                logging.error(
                                    f"Error in eval loop: {traceback.format_exc()}. Setting 14b* score for uid: {uid_i} to infinity."
                                )
                    del model_i

                except Exception:
                    logging.error(
                        f"Error in eval loop: {traceback.format_exc()}. Setting score for uid: {uid_i} to infinity."
                    )
            else:
                logging.debug(
                    f"Unable to load the model for {uid_i} or it belongs to another competition. Setting loss to inifinity for this competition."
                )

            uid_to_state[uid_i].score = score
            uid_to_state[uid_i].score_details = score_details
            logging.info(
                f"Computed model score for uid: {uid_i} with score: {score}. Details: {score_details}"
            )

            if running_14b_star:
                # Make a deep copy of the current uid_to_state, average out the scores, and append new details.
                uid_to_state_14b_star[uid_i] = copy.deepcopy(uid_to_state[uid_i])
                # For scores we know there are only two tasks, so we weight it by the additional 14b task.
                num_fineweb_samples = score_details["FINEWEB_EDU2"].num_samples
                num_stack_samples = score_details_14b_star["STACKV2_DEDUP"].num_samples
                total_samples = num_fineweb_samples + num_stack_samples
                fineweb_weighted_score = score * num_fineweb_samples / total_samples
                stack_weighted_score = (
                    score_14b_star * num_stack_samples / total_samples
                )

                uid_to_state_14b_star[uid_i].score = (
                    fineweb_weighted_score + stack_weighted_score
                )

                # We also need to adjust the weighted score details as each thinks it was 100%.
                # 14B only runs FINEWEB so we can just adjust this one score details.
                uid_to_state_14b_star[uid_i].score_details[
                    "FINEWEB_EDU2"
                ].weighted_norm_score = fineweb_weighted_score
                # Likewise we know 14B* only adds Stack V2
                score_details_14b_star["STACKV2_DEDUP"].weighted_norm_score = (
                    stack_weighted_score
                )
                # And copy over into the new score details dictionary.
                uid_to_state_14b_star[uid_i].score_details["STACKV2_DEDUP"] = (
                    score_details_14b_star["STACKV2_DEDUP"]
                )

                logging.info(
                    f"Computed 14b* model score for uid: {uid_i} with score: {uid_to_state_14b_star[uid_i].score}. Details: {uid_to_state_14b_star[uid_i].score_details}"
                )

        # Calculate new wins and win_rate with only the competitive uids considered.
        wins, win_rate = self._compute_and_set_competition_weights(
            cur_block=cur_block,
            uids=uids,
            uid_to_state=uid_to_state,
            competition=competition,
        )

        if running_14b_star:
            wins_14b_star, win_rate_14b_star = (
                self._compute_and_set_competition_weights(
                    cur_block=cur_block,
                    uids=uids,
                    uid_to_state=uid_to_state_14b_star,
                    competition=competition_14b_star,
                )
            )
            # Since we choose the models to keep based on weight only if they are in the win_rate dict,
            # We need to add anything only winning in 14B* to it. We use a wr of 0 so it is only based on weight.
            win_rate.update(
                {key: 0.0 for key in win_rate_14b_star if key not in win_rate}
            )

        # Get ids for all competitions in the schedule.
        active_competition_ids = set([comp.id for comp in competition_schedule])
        # Align competition_tracker to only track active competitions.
        self.competition_tracker.reset_competitions(active_competition_ids)
        # Update self.weights to the merged values across active competitions.
        with self.weight_lock:
            self.weights = self.competition_tracker.get_subnet_weights(
                competition_schedule
            )

        # Prioritize models for keeping up to the sample_min for the next eval loop.
        # If the model has any significant weight, prioritize by weight with greater weights being kept first.
        # Then for the unweighted models, prioritize by win_rate.
        # Use the competition weights from the tracker which also handles moving averages.
        tracker_competition_weights = self.competition_tracker.get_competition_weights(
            competition.id
        )
        if running_14b_star:
            # Get the weights as if only the 14b and 14b* competition existed for purposes of model prioritization.
            tracker_competition_weights = self.competition_tracker.get_subnet_weights(
                [competition, competition_14b_star]
            )

        model_prioritization = {
            uid: (
                # Add 1 to ensure it is always greater than a win rate.
                1 + tracker_competition_weights[uid].item()
                if tracker_competition_weights[uid].item() >= 0.001
                else wr
            )
            for uid, wr in win_rate.items()
        }

        models_to_keep = set(
            sorted(model_prioritization, key=model_prioritization.get, reverse=True)[
                : self.config.sample_min
            ]
        )

        # Note when breaking ties of 0 weight models we use the primary dataset in all cases.
        uid_to_average_score = {uid: state.score for uid, state in uid_to_state.items()}

        # Make sure we always keep around sample_min number of models to maintain previous behavior.
        if len(models_to_keep) < self.config.sample_min:
            for uid in sorted(uid_to_average_score, key=uid_to_average_score.get):
                if len(models_to_keep) >= self.config.sample_min:
                    break
                models_to_keep.add(uid)

        self._update_uids_to_eval(
            competition.id, models_to_keep, active_competition_ids
        )

        # Save state
        self.save_state()

        # Log the performance of the eval loop.
        logging.debug(load_data_perf.summary_str())
        logging.debug(load_model_perf.summary_str())
        logging.debug(compute_loss_perf.summary_str())

        # Log to screen and wandb.
        self.log_step(
            competition.id,
            competition.constraints.epsilon_func,
            eval_tasks,
            cur_block,
            uids,
            uid_to_state,
            self._get_uids_to_competition_ids(),
            seed,
            data_loaders,
            wins,
            win_rate,
            load_model_perf,
            compute_loss_perf,
            load_data_perf,
        )

        if running_14b_star:
            logging.debug(load_data_perf_14b_star.summary_str())
            logging.debug(compute_loss_perf_14b_star.summary_str())

            uids_to_competition_ids_14b_star = {
                k: (
                    CompetitionId.B14_MODEL_MULTI_DATASET.value
                    if v == CompetitionId.B14_MODEL
                    else v
                )
                for k, v in self._get_uids_to_competition_ids().items()
            }

            self.log_step(
                CompetitionId.B14_MODEL_MULTI_DATASET,
                competition.constraints.epsilon_func,
                [eval_task_14b_star],
                cur_block,
                uids,
                uid_to_state_14b_star,
                uids_to_competition_ids_14b_star,
                seed,
                data_loaders + data_loaders_14b_star,
                wins_14b_star,
                win_rate_14b_star,
                load_model_perf,
                compute_loss_perf_14b_star,
                load_data_perf_14b_star,
            )

    def _compute_and_set_competition_weights(
        self,
        cur_block: int,
        uids: typing.List[int],
        uid_to_state: typing.Dict[int, PerUIDEvalState],
        competition: Competition,
    ) -> typing.Tuple[typing.Dict[int, int], typing.Dict[int, float]]:
        """Computes competition weights including checks for competitiveness and records them internally.

        Args:
            cur_block (int): The current block.
            uids (typing.List[int]): All uids being considered during the current evaluation.
            uid_to_state (typing.Dict[int, PerUIDEvalState]): Evaluation information for each uid.
            competition (Competition): The current competition being evaluated.

        Returns:
            tuple: A tuple containing two dictionaries, one for wins and one for win rates.
        """
        uid_to_score = {uid: state.score for uid, state in uid_to_state.items()}
        uid_to_block = {uid: state.block for uid, state in uid_to_state.items()}

        # Filter to the list of uids that may at one point be a top model.
        competitive_uids = pt.validation.compute_competitive_uids(
            uid_to_score, uid_to_block, competition.constraints.epsilon_func
        )

        # Log which models got dropped for the second pass.
        dropped_uids = [uid for uid in uids if uid not in competitive_uids]
        if dropped_uids:
            logging.info(
                f"The following uids were not included in the win rate calculation because they did not beat the fully decayed score of any previously submitted model in this eval batch: {dropped_uids}."
            )

        # Calculate new wins and win_rate with only the competitive uids considered.
        wins, win_rate = pt.validation.compute_wins(
            competitive_uids,
            uid_to_score,
            uid_to_block,
            competition.constraints.epsilon_func,
            cur_block,
        )

        top_uid = max(win_rate, key=win_rate.get)
        self._record_eval_results(top_uid, cur_block, uid_to_state, competition.id)

        # Compute softmaxed weights based on win rate.
        model_weights = torch.tensor(
            [win_rate.get(uid, 0) for uid in uids], dtype=torch.float32
        )
        step_weights = torch.softmax(model_weights / constants.temperature, dim=0)

        # Fill in metagraph sized tensor with the step weights of the evaluated models.
        with self.metagraph_lock:
            competition_weights = torch.zeros_like(torch.from_numpy(self.metagraph.S))

        for i, uid_i in enumerate(uids):
            competition_weights[uid_i] = step_weights[i]

        # Record weights for the current competition.
        self.competition_tracker.record_competition_weights(
            competition.id, competition_weights
        )

        return wins, win_rate

    def _update_uids_to_eval(
        self,
        competition_id: CompetitionId,
        uids: typing.Set[int],
        active_competitions: typing.Set[int],
    ):
        """Updates the uids to evaluate and clears out any sunset competitions.

        Args:
            competition_id (CompetitionId): The competition id to update.
            uids (typing.Set[int]): The set of uids to evaluate in this competition on the next eval loop.
        """
        with self.pending_uids_to_eval_lock:
            self.uids_to_eval[competition_id] = uids

            # Clean up sunset competitions.
            # This works as expected even though the keys are CompetitionIds and active_competitions are ints.
            comps_to_delete = (
                set(self.uids_to_eval.keys()) | set(self.pending_uids_to_eval.keys())
            ) - active_competitions
            for comp in comps_to_delete:
                logging.debug(
                    f"Cleaning up uids to eval from sunset competition {comp}."
                )
                if comp in self.uids_to_eval:
                    del self.uids_to_eval[comp]
                if comp in self.pending_uids_to_eval:
                    del self.pending_uids_to_eval[comp]

    def _record_eval_results(
        self,
        top_uid: int,
        curr_block: int,
        uid_to_state: typing.Dict[int, PerUIDEvalState],
        competition_id: CompetitionId,
    ) -> None:
        """Records the results of the evaluation step to the model tracker.

        Args:
            top_uid (int): The uid of the model with the higest win rate.
            curr_block (int): The current block.
            uid_to_state (typing.Dict[int, PerUIDEvalState]): A dictionary mapping uids to their eval state.
        """
        top_model_loss = uid_to_state[top_uid].score
        for _, state in uid_to_state.items():
            self.model_tracker.on_model_evaluated(
                state.hotkey,
                competition_id,
                EvalResult(
                    block=curr_block,
                    score=state.score,
                    winning_model_block=uid_to_state[top_uid].block,
                    winning_model_score=top_model_loss,
                ),
            )

    def log_step(
        self,
        competition_id: CompetitionId,
        competition_epsilon_func: EpsilonFunc,
        eval_tasks: typing.List[EvalTask],
        current_block: int,
        uids: typing.List[int],
        uid_to_state: typing.Dict[int, PerUIDEvalState],
        uid_to_competition_id: typing.Dict[int, typing.Optional[int]],
        seed: int,
        data_loaders: typing.List[SubsetLoader],
        wins: typing.Dict[int, int],
        win_rate: typing.Dict[int, float],
        load_model_perf: PerfMonitor,
        compute_loss_perf: PerfMonitor,
        load_data_perf: PerfMonitor,
    ):
        """Logs the results of the step to the console and wandb (if enabled)."""
        # Get pages from each data loader
        pages = []
        for loader in data_loaders:
            for page_name in loader.get_page_names():
                pages.append(f"{loader.name}:{loader.config}:{page_name}")

        # Build step log
        step_log = {
            "timestamp": time.time(),
            "competition_id": competition_id,
            "seed": seed,
            "pages": pages,
            "uids": uids,
            "uid_data": {},
        }

        # The sub-competition weights
        sub_competition_weights = self.competition_tracker.get_competition_weights(
            competition_id
        )

        # Get a copy of weights to print.
        with self.weight_lock:
            log_weights = self.weights

        # All uids in the competition step log are from the same competition.
        for uid in uids:
            step_log["uid_data"][str(uid)] = {
                "uid": uid,
                "block": uid_to_state[uid].block,
                "hf": uid_to_state[uid].repo_name,
                "competition_id": int(competition_id),
                "average_loss": uid_to_state[
                    uid
                ].score,  # Keep the log here as loss to avoid breaking the leaderboard.
                "epsilon_adv": competition_epsilon_func.compute_epsilon(
                    current_block, uid_to_state[uid].block
                ),
                # We use 0 in the case where a uid was not competitive and therefore not used in win rate calcs.
                "win_rate": win_rate[uid] if uid in win_rate else 0,
                "win_total": wins[uid] if uid in wins else 0,
                "weight": log_weights[uid].item(),
                "norm_weight": sub_competition_weights[uid].item(),
                "dataset_perf": {},
            }

            for task in eval_tasks:
                step_log["uid_data"][str(uid)][f"{task.name}.raw_score"] = (
                    uid_to_state[uid].score_details[task.name].raw_score
                )
                step_log["uid_data"][str(uid)][f"{task.name}.norm_score"] = (
                    uid_to_state[uid].score_details[task.name].norm_score
                )
                step_log["uid_data"][str(uid)][f"{task.name}.weighted_norm_score"] = (
                    uid_to_state[uid].score_details[task.name].weighted_norm_score
                )

            # Also log in this older format to avoid breaking the leaderboards.
            for task_name, score_detail in uid_to_state[uid].score_details.items():
                # Hack to get the 'right' name back here.
                task_to_dataset_name = {
                    "FALCON": "tiiuae/falcon-refinedweb",
                    "FINEWEB": "HuggingFaceFW/fineweb",
                    "FINEWEB_EDU2": "HuggingFaceFW/fineweb-edu-score-2",
                    "STACKV2_DEDUP": "bigcode/the-stack-v2-dedup",
                    "PES2OX": "laion/Pes2oX-fulltext",
                    "FINEMATH_3P": "HuggingFaceTB/finemath:finemath-3p",
                    "INFIWEBMATH_3P": "HuggingFaceTB/finemath:infiwebmath-3p",
                }
                dataset_name = (
                    task_to_dataset_name[task_name]
                    if task_name in task_to_dataset_name
                    else "DatasetNameNotFound"
                )
                step_log["uid_data"][str(uid)]["dataset_perf"][f"{dataset_name}"] = {
                    "average_loss": score_detail.raw_score
                }

        table = Table(title="Step", expand=True)
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("hf", style="magenta", overflow="fold")
        table.add_column("avg_loss", style="magenta", overflow="fold")
        table.add_column("epsilon_adv", style="magenta", overflow="fold")
        table.add_column("win_rate", style="magenta", overflow="fold")
        table.add_column("win_total", style="magenta", overflow="fold")
        table.add_column("total_weight", style="magenta", overflow="fold")
        table.add_column("comp_weight", style="magenta", overflow="fold")
        table.add_column("block", style="magenta", overflow="fold")
        table.add_column("comp", style="magenta", overflow="fold")
        for uid in uids:
            try:
                table.add_row(
                    str(uid),
                    str(step_log["uid_data"][str(uid)]["hf"]),
                    str(round(step_log["uid_data"][str(uid)]["average_loss"], 4)),
                    str(round(step_log["uid_data"][str(uid)]["epsilon_adv"], 4)),
                    str(round(step_log["uid_data"][str(uid)]["win_rate"], 4)),
                    str(step_log["uid_data"][str(uid)]["win_total"]),
                    str(round(log_weights[uid].item(), 4)),
                    str(round(sub_competition_weights[uid].item(), 4)),
                    str(step_log["uid_data"][str(uid)]["block"]),
                    str(step_log["uid_data"][str(uid)]["competition_id"]),
                )
            except:
                pass
        console = Console()
        console.print(table)

        ws, ui = self.weights.topk(len(log_weights))
        table = Table(title="Weights > 0.001")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("weight", style="magenta")
        table.add_column("comp", style="magenta")
        for index, weight in list(zip(ui.tolist(), ws.tolist())):
            if weight > 0.001:
                table.add_row(
                    str(index),
                    str(round(weight, 4)),
                    str(uid_to_competition_id[index]),
                )
        console = Console()
        console.print(table)

        # Sink step log.
        logging.trace(f"Step results: {step_log}")

        if self.config.wandb.on and not self.config.offline:
            # If we have already completed X steps then we will complete the current wandb run and make a new one.
            if (
                self.run_step_count
                and self.run_step_count % constants.MAX_RUN_STEPS_PER_WANDB_RUN == 0
            ):
                logging.trace(
                    f"Validator has completed {self.run_step_count} run steps. Creating a new wandb run."
                )
                self.wandb_run.finish()
                self._new_wandb_run()

            original_format_json = json.dumps(step_log)
            uids = step_log["uids"]
            uid_data = step_log["uid_data"]

            # Create a new dictionary with the required format
            graphed_data = {
                "time": time.time(),
                "step_competition_id": competition_id,
                "block": current_block,
                "uid_data": {
                    str(uid): uid_data[str(uid)]["average_loss"] for uid in uids
                },
                "uid_epsilon_adv": {
                    str(uid): uid_data[str(uid)]["epsilon_adv"] for uid in uids
                },
                "win_rate_data": {
                    str(uid): uid_data[str(uid)]["win_rate"] for uid in uids
                },
                "win_total_data": {
                    str(uid): uid_data[str(uid)]["win_total"] for uid in uids
                },
                "weight_data": {str(uid): log_weights[uid].item() for uid in uids},
                "competition_weight_data": {
                    str(uid): sub_competition_weights[uid].item() for uid in uids
                },
                "competition_id": {str(uid): int(competition_id)},
                "load_model_perf": {
                    "min": load_model_perf.min(),
                    "median": load_model_perf.median(),
                    "max": load_model_perf.max(),
                    "P90": load_model_perf.percentile(90),
                },
                "compute_model_perf": {
                    "min": compute_loss_perf.min(),
                    "median": compute_loss_perf.median(),
                    "max": compute_loss_perf.max(),
                    "P90": compute_loss_perf.percentile(90),
                },
                "load_data_perf": {
                    "min": load_data_perf.min(),
                    "median": load_data_perf.median(),
                    "max": load_data_perf.max(),
                    "P90": load_data_perf.percentile(90),
                },
            }
            # Add the score details to the graphed data.
            for task in eval_tasks:
                graphed_data[f"{task.name}.raw_score"] = {
                    str(uid): uid_data[str(uid)][f"{task.name}.raw_score"]
                    for uid in uids
                }
                graphed_data[f"{task.name}.norm_score"] = {
                    str(uid): uid_data[str(uid)][f"{task.name}.norm_score"]
                    for uid in uids
                }
                graphed_data[f"{task.name}.weighted_norm_score"] = {
                    str(uid): uid_data[str(uid)][f"{task.name}.weighted_norm_score"]
                    for uid in uids
                }
            logging.trace("Logging to Wandb")
            self.wandb_run.log(
                {**graphed_data, "original_format_json": original_format_json}
            )

            # Increment the number of completed run steps by 1
            self.run_step_count += 1

    def _get_uids_to_competition_ids(
        self,
    ) -> typing.Dict[int, typing.Optional[int]]:
        """Returns a mapping of uids to competition id ints, based on the validator's current state"""
        hotkey_to_metadata = (
            self.model_tracker.get_miner_hotkey_to_model_metadata_dict()
        )
        with self.metagraph_lock:
            uids_to_competition_ids = {}
            # Check all uids currently registered as we default to None if they don't have metadata.
            for uid in range(len(self.metagraph.uids)):
                hotkey = self.metagraph.hotkeys[uid]
                metadata = hotkey_to_metadata.get(hotkey, None)
                uids_to_competition_ids[uid] = (
                    metadata.id.competition_id if metadata else None
                )

            return uids_to_competition_ids

    async def run(self):
        """Runs the validator loop, which continuously evaluates models and sets weights."""
        while True:
            try:
                self._configure_logging(self.config)
                await self.try_run_step(ttl=120 * 60)
                self.global_step += 1

            except KeyboardInterrupt:
                logging.info(
                    "KeyboardInterrupt caught, gracefully closing the wandb run..."
                )
                if self.wandb_run:
                    self.wandb_run.finish()
                exit()

            except Exception as e:
                logging.error(
                    f"Error in validator loop \n {e} \n {traceback.format_exc()}"
                )


if __name__ == "__main__":
    # Set an output width explicitly for rich table output (affects the pm2 tables that we use).
    try:
        width = os.get_terminal_size().columns
    except:
        width = 0
    os.environ["COLUMNS"] = str(max(200, width))

    asyncio.run(Validator().run())
