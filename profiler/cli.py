from __future__ import annotations

import click

from calibration_cli import calibrate
from dataset_writer_cli import dataset


@click.group()
def cli() -> None:
    """
    Command-line tools for FLIR beam profiling workflows.
    """


cli.add_command(dataset)
cli.add_command(calibrate)


if __name__ == "__main__":
    cli()
