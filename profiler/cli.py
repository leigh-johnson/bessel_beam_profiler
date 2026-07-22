from __future__ import annotations

import click

from auto_scan_cli import auto_scan
from calibration_cli import calibrate
from dataset_writer_cli import dataset
from fluidnc_stage_cli import gantry


@click.group()
def cli() -> None:
    """
    Command-line tools for FLIR beam profiling workflows.
    """


dataset.add_command(auto_scan)

cli.add_command(dataset)
cli.add_command(calibrate)
cli.add_command(gantry)


if __name__ == "__main__":
    cli()
