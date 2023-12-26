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

import os
import tqdm
import json
import math
import time
import torch
import typing
import random
import asyncio
import argparse
import pretrain
import traceback
import threading
import multiprocessing
from rich.table import Table
from rich.console import Console
from multiprocessing import Value

import bittensor as bt
import pretrain as pt

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class Validator:
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
            default=3,
            help="Number of pages used to eval each step.",
        )
        parser.add_argument(
            "--sample_min",
            type=int,
            default=30,
            help="Number of uids to eval each step.",
        )
        parser.add_argument(
            "--reset_wandb",
            action="store_true",
            help="Creates a new wandb run instead of using an older on.",
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
            "--test",
            action="store_true",
            help="Runs steps with max 3 uids to eval for faster testing.",
        )
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        bt.axon.add_args(parser)
        config = bt.config(parser)
        return config

    def __init__(self):
        self.config = Validator.config()
        bt.logging(config=self.config)

        # === Bittensor objects ====
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = self.subtensor.metagraph(pt.NETUID)
        torch.backends.cudnn.benchmark = True

        # Dont check registration status if offline.
        if not self.config.offline:
            if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
                raise Exception(
                    f"You are not registered. Use `btcli s register --netuid {pt.NETUID}` to register."
                )
            self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            bt.logging.success(
                f"You are registered with address: {self.wallet.hotkey.ss58_address} and uid: {self.uid}"
            )

        # Dont log to wandb if offline.
        if not self.config.offline:
            self.wandb_run = pt.mining.init_validator(
                self.wallet, metagraph=self.metagraph
            )

        # === Running args ===
        self.weights = torch.zeros_like(self.metagraph.S)
        self.epoch_step = 0
        self.global_step = 0
        self.last_epoch = self.metagraph.block.item()
        self.last_update_check = {}
        self.metadata = {
            uid: pt.graph.metadata(uid) for uid in self.metagraph.uids.tolist()
        }

        # === Build initial uids to eval ===
        self.uids_to_eval = []
        for uid in self.metagraph.uids.tolist():
            if self.metadata[uid] != None:
                self.uids_to_eval.append(uid)
        random.shuffle(self.uids_to_eval)
        # If test, only samples 3 initial uids.
        if self.config.test:
            self.uids_to_eval = self.uids_to_eval[: self.config.sample_min + 1]
        self.uids_to_eval = set(self.uids_to_eval)
        self.pending_uids_to_eval = set()

        # == Initialize the update thread ==
        self.stop_event = threading.Event()
        self.update_thread = threading.Thread(target=self.update_models, daemon=True)
        self.update_thread.start()

    def __del__(self):
        if hasattr(self, "stop_event"):
            self.stop_event.set()
            self.update_thread.join()

    def update_models(self):
        # The below loop iterates across all miner uids and checks to see
        # if they should be updated.
        last_uid_update = -1
        while not self.stop_event.is_set():
            try:
                if self.stop_event.is_set():
                    return
                block = self.try_get_block(10)
                if block == None:
                    continue
                uid = block % 256
                if uid == last_uid_update:
                    time.sleep(1)
                    continue
                last_uid_update = uid
                bt.logging.success(f"Syncing miner for uid: {uid} and block: {block}")
                if pt.graph.sync(uid, self.metagraph):
                    bt.logging.success(f"Pulled new model for uid: {uid}")
                else:
                    bt.logging.success(f"Model up to date for uid: {uid}")
                bt.logging.trace(f"adding {uid} to pending uids to eval.")
                self.pending_uids_to_eval.add(uid)
            except Exception as e:
                bt.logging.error(f"Error in update loop: {e}")

    def try_get_block(self, ttl: int):
        def get_block(endpoint, queue):
            try:
                block_number = bt.subtensor(endpoint).block
                queue.put(block_number)
            except Exception as e:
                queue.put(None)
                bt.logging.error(f"Error getting block: {e}")

        queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=get_block, args=(self.subtensor.chain_endpoint, queue)
        )
        process.start()
        process.join(timeout=ttl)
        if process.is_alive():
            process.terminate()
            process.join()
            bt.logging.error(f"Failed to get_block after {ttl} seconds")
            return None
        block_number = queue.get()
        return block_number

    async def try_set_weights(self, ttl: int):
        async def _try_set_weights():
            try:
                self.weights.nan_to_num(0.0)
                self.subtensor.set_weights(
                    netuid=pt.NETUID,
                    wallet=self.wallet,
                    uids=self.metagraph.uids,
                    weights=self.weights,
                    wait_for_inclusion=False,
                    version_key=pt.weights_version_key,
                )
            except:
                pass
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
            metagraph = bt.subtensor(endpoint).metagraph(pt.NETUID)
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
        self.metagraph.load()

    async def try_run_step(self, ttl: int):
        async def _try_run_step():
            await self.run_step()

        try:
            bt.logging.debug(f"Running step.")
            await asyncio.wait_for(_try_run_step(), ttl)
            bt.logging.debug(f"Finished running step.")
        except asyncio.TimeoutError:
            bt.logging.error(f"Failed to run step after {ttl} seconds")

    async def run_step(self):
        """
        Executes a step in the evaluation process of models. This function performs several key tasks:
        1. Identifies valid models for evaluation based on metadata and synchronization status.
        2. Generates random pages for evaluation and prepares batches for each page from the dataset.
        3. Computes the scoring for each model based on the losses incurred on the evaluation batches.
        4. Calculates wins and win rates for each model to determine their performance relative to others.
        5. Updates the weights of each model based on their performance and applies a softmax normalization.
        6. Implements a blacklist mechanism to remove underperforming models from the evaluation set.
        7. Logs all relevant data for the step, including model IDs, pages, batches, wins, win rates, and losses.
        """

        # Pull relevant uids, timestamps and metadata for step.
        # Remove uids without valid runs on wandb or which are not synced.
        uids = []
        timestamps = []
        for uid in list(self.uids_to_eval):
            meta = pt.graph.metadata(uid)
            if meta == None:
                continue
            # Check that the uid has a valid wandb run and the model is synced locally.
            if pt.graph.has_valid_run(uid, self.metagraph):
                uids.append(uid)
                timestamps.append(pt.graph.last_download(uid))
            else:
                bt.logging.debug(
                    f"uid:{uid} run does not exist or is not valid, removing from uids to eval."
                )
        bt.logging.success(
            f"Runnning step with uids: {uids} and timestamps: {timestamps}"
        )

        # Generate random pages for evaluation and prepare batches for each page
        # the dataset contains >900 million pages to eval over.
        pages = [
            random.randint(1, pt.dataset.SubsetFalconLoader.max_pages)
            for _ in range(self.config.pages_per_eval)
        ]
        batches = list(
            pt.dataset.SubsetFalconLoader(
                batch_size=pt.batch_size,
                sequence_length=pt.sequence_length,
                pages=pages,
            )
        )

        # Compute model losses on batches.
        bt.logging.debug(f"computing losses on {uids}")
        losses_per_uid = {muid: None for muid in uids}
        for uid_i in uids:
            # get model or uid or None if we have not synced the model.
            model_i = pt.graph.model(uid_i, device=self.config.device)
            if model_i == None:
                losses = [math.inf for _ in batches]
            else:
                losses = pt.validation.compute_losses(
                    model_i, batches, device=self.config.device
                )
            losses_per_uid[uid_i] = losses
            average_model_loss = sum(losses) / len(losses)
            bt.logging.debug(
                f"Compute model losses for uid:{uid_i} with average loss: {average_model_loss}"
            )
            del model_i

        # Compute wins per uid.
        wins = {uid: 0 for uid in uids}
        win_rate = {uid: 0 for uid in uids}
        for i, uid_i in enumerate(uids):
            total_matches = 0
            time_i = timestamps[i]
            for j, uid_j in enumerate(uids):
                if i == j:
                    continue
                time_j = timestamps[j]
                for batch_idx, _ in enumerate(batches):
                    loss_i = losses_per_uid[uid_i][batch_idx]
                    loss_j = losses_per_uid[uid_j][batch_idx]
                    wins[uid_i] += (
                        1 if pt.validation.iswin(loss_i, loss_j, time_i, time_j) else 0
                    )
                    total_matches += 1
            # Calculate win rate for uid i
            win_rate[uid_i] = wins[uid_i] / total_matches if total_matches > 0 else 0

        # Compute softmaxed weights based on win rate.
        model_weights = torch.tensor(
            [win_rate[uid] for uid in uids], dtype=torch.float32
        )
        step_weights = torch.softmax(model_weights / pt.temperature, dim=0)
        bt.logging.success(f"Computed model wins: {wins}")

        # Update weights based on moving average.
        new_weights = torch.zeros_like(self.weights)
        for i, uid_i in enumerate(uids):
            new_weights[uid_i] = step_weights[i]
        new_weights /= new_weights.sum()
        self.weights = pt.alpha * self.weights + (1 - pt.alpha) * new_weights
        self.weights = self.weights.nan_to_num(0.0)

        # Filter based on win rate removing all by the sample_min best models for evaluation.
        self.uids_to_eval = set(
            sorted(win_rate, key=win_rate.get, reverse=True)[: self.config.sample_min]
        )
        self.uids_to_eval.update(self.pending_uids_to_eval)
        self.pending_uids_to_eval.clear()

        # Log to screen and wandb.
        self.log_step(
            uids,
            timestamps,
            pages,
            batches,
            wins,
            win_rate,
            losses_per_uid,
        )
        bt.logging.debug("Finished run step.")

    def log_step(
        self, uids, timestamps, pages, batches, wins, win_rate, losses_per_uid
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
                "runid": pt.graph.runid(uid),
                "timestamp": timestamps[i],
                "last_update": pt.graph.last_update(uid),
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
        table.add_column("last_update", style="magenta")
        table.add_column("timestamp", style="magenta")
        for uid in uids:
            try:
                table.add_row(
                    str(uid),
                    str(round(step_log["uid_data"][str(uid)]["average_loss"], 4)),
                    str(round(step_log["uid_data"][str(uid)]["win_rate"], 4)),
                    str(step_log["uid_data"][str(uid)]["win_total"]),
                    str(round(self.weights[uid].item(), 4)),
                    str(round(step_log["uid_data"][str(uid)]["last_update"], 0)),
                    str(step_log["uid_data"][str(uid)]["timestamp"]),
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
        original_format_json = json.dumps(step_log)
        uids = step_log["uids"]
        uid_data = step_log["uid_data"]

        # Create a new dictionary with the required format
        graphed_data = {
            "time": time.time(),
            "block": self.metagraph.block.item(),
            "uid_data": {str(uid): uid_data[str(uid)]["average_loss"] for uid in uids},
            "weight_data": {str(uid): self.weights[uid].item() for uid in uids},
        }
        if self.config.wandb.on and not self.config.offline:
            bt.logging.trace("Logging to Wandb")
            self.wandb_run.log(
                {**graphed_data, "original_format_json": original_format_json},
                step=self.global_step,
            )
            bt.logging.trace("finished log to Wandb")

    async def run(self):
        while True:
            try:
                while (
                    self.metagraph.block.item() - self.last_epoch
                    < self.config.blocks_per_epoch
                ):
                    await self.try_run_step(ttl=60 * 20)
                    await self.try_sync_metagraph(ttl=60)
                    bt.logging.debug(
                        f"{self.metagraph.block.item() - self.last_epoch } / {self.config.blocks_per_epoch} blocks until next epoch."
                    )
                    self.global_step += 1

                if (
                    not self.config.dont_set_weights
                    and not self.config.offline
                    and not self.config.test
                ):
                    await self.try_set_weights(ttl=60)
                self.last_epoch = self.metagraph.block.item()
                self.epoch_step += 1

            except KeyboardInterrupt:
                bt.logging.info(
                    "KeyboardInterrupt caught, gracefully closing the wandb run..."
                )
                if self.config.wandb.on and not self.config.offline:
                    self.wandb_run.finish()
                exit()

            except Exception as e:
                bt.logging.error(
                    f"Error in validator loop \n {e} \n {traceback.format_exc()}"
                )


if __name__ == "__main__":
    asyncio.run(Validator().run())
