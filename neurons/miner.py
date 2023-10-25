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

# Step 1: Import necessary libraries and modules
import os
import time
import torch
import argparse
import traceback
import bittensor as bt

# import this repo
import pretrain

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--device', type = str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the miner on.' )
    parser.add_argument( '--batch_size', type=int, default=8, help='Training batch size' )
    parser.add_argument( '--sequence_length', type=int, default=512, help='Training sequence length' )
    # Adds subtensor specific arguments i.e. --subtensor.chain_endpoint ... --subtensor.network ...
    bt.subtensor.add_args(parser)
    # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
    bt.logging.add_args(parser)
    # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
    bt.wallet.add_args(parser)
    # Adds axon specific arguments i.e. --axon.port ...
    bt.axon.add_args(parser)
    # Activating the parser to read any command-line inputs.
    # To print help message, run python3 template/miner.py --help
    config = bt.config(parser)
    # Set up logging directory
    # Logging captures events for diagnosis or understanding miner's behavior.
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            pretrain.NETUID,
            "miner",
        )
    )
    # Ensure the directory for logging exists, else create one.
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)
    return config

# Main takes the config and starts the miner.
def main(config):
    # Activating Bittensor's logging with the set configurations.
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(
        f"Running miner for subnet: { pretrain.NETUID } on network: {config.subtensor.chain_endpoint} with config:"
    )

    # This logs the active configuration to the specified logging directory for review.
    bt.logging.info(config)

    # Step 4: Initialize Bittensor miner objects
    # These classes are vital to interact and function within the Bittensor network.
    bt.logging.info("Setting up bittensor objects.")

    # Wallet holds cryptographic information, ensuring secure transactions and communication.
    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    # subtensor manages the blockchain connection, facilitating interaction with the Bittensor blockchain.
    subtensor = bt.subtensor(config=config)
    bt.logging.info(f"Subtensor: {subtensor}")

    # metagraph provides the network's current state, holding state about other participants in a subnet.
    metagraph = subtensor.metagraph( pretrain.NETUID )
    bt.logging.info(f"Metagraph: {metagraph}")

    # Each miner gets a unique identity (UID) in the network for differentiation.
    my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    bt.logging.info(f"Running miner on uid: {my_subnet_uid}")

    # Step 5: Initialize miner specific objects
    model = pretrain.model.get_model().to( config.device )

    # --- Define dataloader.
    dataloader = pretrain.dataset.get_dataloader( config.batch_size, config.sequence_length )

    # --- Define forward function.
    # Accepts the model state, loads the state into the model and computes a set of 
    # aggregated gradients. The gradients are then packed into the response and returned.
    def compute_gradients( synapse: pretrain.protocol.ComputeGradients ) -> pretrain.protocol.ComputeGradients:
        bt.logging.info(f'Start forward')
        # Clear previous gradients.
        model.zero_grad()
        model.load_state_dict( synapse.deserialize_state() )

        # Apply n_steps gradients aggregations.
        step = 0
        while True:
            batch = next( dataloader )
            inputs = batch.to( config.device )            
            outputs = model( inputs, labels = inputs )
            loss = outputs.loss/synapse.n_steps      
            loss.backward()
            bt.logging.info(f'Step: {step}, Loss: {loss.item() * synapse.n_steps}')
            if step >= synapse.n_steps: break
            else: step += 1

        # Serialize the gradients onto the model state.
        synapse.serialize_state( state_dict = { k: v.grad for k, v in model.named_parameters() } ) 
        synapse.loss = loss.item()
        bt.logging.info(f'End forward')
        return synapse

    # Step 6: Build and link miner functions to the axon.
    # The axon handles request processing, allowing validators to send this process requests.
    axon = bt.axon(wallet=wallet, config=config)
    bt.logging.info(f"Axon {axon}")

    # Attach determiners which functions are called when servicing a request.
    bt.logging.info(f"Attaching forward function to axon.")
    axon.attach( forward_fn = compute_gradients )

    # Serve passes the axon information to the network + netuid we are hosting on.
    # This will auto-update if the axon port of external ip have changed.
    bt.logging.info(
        f"Serving axon with synapse: {compute_gradients} on network: {config.subtensor.chain_endpoint} with netuid: {pretrain.NETUID}"
    )
    axon.serve(netuid=pretrain.NETUID, subtensor=subtensor)

    # Start  starts the miner's axon, making it active on the network.
    bt.logging.info(f"Starting axon server on port: {config.axon.port}")
    axon.start()

    # Step 7: Keep the miner alive
    # This loop maintains the miner's operations until intentionally stopped.
    bt.logging.info(f"Starting main loop")
    step = 0
    while True:
        try:
            time.sleep(1)

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            break
        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as e:
            bt.logging.error(traceback.format_exc())
            continue


# This is the main function, which runs the miner.
if __name__ == "__main__":
    main(get_config())