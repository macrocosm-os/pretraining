#!/usr/bin/env python3
"""
This script runs a validator process and automatically updates it when a new version is released.

Command-line arguments will be forwarded to validator (`neurons/validator.py`), so you can pass
them like this:

    python3 scripts/start_validator.py --wallet.name=my-wallet

Auto-updates are enabled by default and will make sure that the latest version is always running
by pulling the latest version from git and upgrading python packages. This is done periodically.
Local changes may prevent the update, but they will be preserved.

To disable auto-updates, set `AUTO_UPDATE` environment variable to `0`:

    AUTO_UPDATE=0 python3 scripts/start_validator.py

The script will use the same virtual environment as the one used to run it. If you want to run
validator within virtual environment, run this auto-update script from the virtual environment.

If pm2 process manager is installed and accessible (`npm install pm2`), then it will be used to
start the validator process. However, one needs to monitor app state on his own by checking
`pm2 list` or `pm2 monit`. If pm2 is not available, then the script will use `subprocess.Popen`
to start the validator process.
"""
import logging
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from shlex import split
from shutil import which

from envparse import env

log = logging.getLogger(__name__)
UPDATES_CHECK_TIME = timedelta(minutes=5)
ROOT_DIR = Path(__file__).parent.parent
PM2_ENABLED = which('pm2') is not None

def get_version() -> str:
    """ Extract the version as current git commit hash """
    result = subprocess.run(
        split("git rev-parse HEAD"),
        check=True, capture_output=True, cwd=ROOT_DIR,
    )
    commit = result.stdout.decode().strip()
    assert len(commit) == 40, f'Invalid commit hash: {commit}'  # noqa: PLR2004
    return commit[:8]

def start_validator_process() -> subprocess.Popen:
    """
    Spawn a new python process running neurons.validator.

    `sys.executable` ensures thet the same python interpreter is used as the one
    used to run this auto-updater.
    """
    assert sys.executable, 'Failed to get python executable'
    if PM2_ENABLED:
        name = f"validator-{get_version()}"
        log.info('Starting validator process with pm2, name: %s', name)
        process = subprocess.Popen(
            (
                "pm2", "start",
                sys.executable,
                "--name", name,
                "--no-autorestart",
                "--", "-m", "neurons.validator", *sys.argv,
            ),
            cwd=ROOT_DIR,
        )
        process.pm2_name = name
    else:
        log.info('Starting validator process')
        process = subprocess.Popen(
            (sys.executable, *split("-m neurons.validator"), *sys.argv),
            cwd=ROOT_DIR,
            preexec_fn=os.setsid,
        )
        time.sleep(1)
        if (return_code := process.poll()) is not None:
            raise RuntimeError('Failed to start validator process, return code: %s', return_code)
    return process


def stop_validator_process(process: subprocess.Popen) -> None:
    """ Stop the validator process """
    if PM2_ENABLED:
        subprocess.run(("pm2", "delete", process.pm2_name), cwd=ROOT_DIR, check=True)
    elif process.poll() is None:
        process.terminate()


def pull_latest_version() -> None:
    """
    Pull the latest version from git.

    This uses `git pull --rebase`, so if any changes were made to the local repository,
    this will try to apply them on top of origin's changes. This is intentional, as we
    don't want to overwrite any local changes. However, if there are any conflicts,
    this will abort the rebase and return to the original state.

    The conflicts are expected to happen rarely since validator is expected
    to be used as-is.
    """
    try:
        subprocess.run(split("git pull --rebase --autostash"), check=True, cwd=ROOT_DIR)
    except subprocess.CalledProcessError as exc:
        log.error('Failed to pull, reverting: %s', exc)
        subprocess.run(split("git rebase --abort"), check=True, cwd=ROOT_DIR)


def upgrade_packages() -> None:
    """
    Upgrade python packages by running `pip install --upgrade -r requirements.txt`.

    Notice: this won't work if some package in `requirements.txt` is downgraded.
    Ignored as this is unlikely to happen.
    """

    log.info('Upgrading packages')
    try:
        subprocess.run(
            split("pip install --upgrade --disable-pip-version-check -r requirements.txt"),
            check=True, cwd=ROOT_DIR,
        )
    except subprocess.CalledProcessError as exc:
        log.error('Failed to upgrade packages, proceeding anyway. %s', exc)


def main(auto_update: bool = True) -> None:
    """
    Run the validator process and automatically update it when a new version is released.

    This will check for updates every `UPDATES_CHECK_TIME` and update the validator
    if a new version is available. Update is performed as simple `git pull --rebase`.
    """

    validator = start_validator_process()
    current_version = latest_version = get_version()
    log.info('Current version: %s', current_version)

    try:
        while True:
            if auto_update:
                pull_latest_version()
                latest_version = get_version()
                log.info('Latest version: %s', latest_version)

            if latest_version != current_version:
                log.info('Upgraded to latest version: %s -> %s', current_version, latest_version)
                upgrade_packages()

                stop_validator_process(validator)
                validator = start_validator_process()
                current_version = latest_version

            time.sleep(UPDATES_CHECK_TIME.total_seconds())

    finally:
        stop_validator_process(validator)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main(
        auto_update=env.bool('AUTO_UPDATE', default=True),
    )
