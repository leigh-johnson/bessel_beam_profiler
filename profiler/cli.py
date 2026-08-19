from __future__ import annotations

import click

from align_cli import align
from auto_scan_cli import auto_scan
from calibration_cli import calibrate
from composite_cli import composite
from dataset_writer_cli import dataset
from fluidnc_stage_cli import gantry
from log_utils import configure_cli_logging
from preflight_cli import preflight
from scan_preview_cli import watch_command

@click.group()
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False
    ),
    help="Console log level. Dataset subcommands also write the log to a "
    "scan.log file inside the run directory.",
)
def cli(log_level: str) -> None:
    """
    Command-line tools for FLIR beam profiling workflows.
    """

    configure_cli_logging(log_level)


dataset.add_command(auto_scan)
dataset.add_command(composite)
dataset.add_command(watch_command)

cli.add_command(align)
cli.add_command(dataset)
cli.add_command(calibrate)
cli.add_command(gantry)
cli.add_command(preflight)


if __name__ == "__main__":
    cli()
