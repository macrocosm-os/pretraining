import cProfile
import pstats
import asyncio
import torch
import sys
from neurons.validator import Validator


def profile_validator():
    """Profiles the Validator class's run method."""
    # Simulate passing CLI arguments
    sys.argv = [
        "neurons/validator.py",
        "--wallet.name",
        "roel-s9v",
        "--wallet.hotkey",
        "roel-s9v-hot",
        "--logging.debug",
        "--offline",
    ]

    # Initialize the Validator
    validator = Validator()

    # Define a wrapper function to run the validator
    async def run_validator():
        await validator.run()

    # Profile the wrapper function
    def profile_run():
        # Synchronize GPU before starting the profiling
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        cProfile.run("asyncio.run(run_validator())", "validator_profile.prof")

        # Synchronize GPU after finishing the profiling
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    profile_run()

    # Print the profiling results
    p = pstats.Stats("validator_profile.prof")
    p.sort_stats("cumulative").print_stats(
        50
    )  # Print the top 50 functions by cumulative time


if __name__ == "__main__":
    profile_validator()
