"""`dataset watch` — live viewer for a running (or finished) auto scan."""

from __future__ import annotations

from pathlib import Path

import click


@click.command(name="watch")
@click.argument("run_dir", required=False, type=click.Path(path_type=Path))
@click.option(
    "--dataset-root",
    default="data",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Where to look for the newest auto_scan-* run when RUN_DIR is "
    "not given.",
)
@click.option(
    "--interval",
    default=1.0,
    show_default=True,
    type=click.FloatRange(min=0.1),
    help="Seconds between directory checks.",
)
def watch_command(run_dir, dataset_root: Path, interval: float) -> None:
    """
    Show the newest frame of an auto-scan run as it is written, in a
    matplotlib window that refreshes live.

    Runs completely independently of the scan (reads files only — never
    the camera), so it can be started and closed at any time without
    touching data capture. `dataset auto --preview` starts one
    automatically. With no RUN_DIR, watches the newest auto_scan-* run
    under --dataset-root.
    """

    from scan_preview import newest_run_dir, watch

    if run_dir is None:
        run_dir = newest_run_dir(dataset_root)
        if run_dir is None:
            raise click.ClickException(
                f"No auto_scan-* runs under {dataset_root} — pass a run "
                "directory explicitly."
            )
        click.echo(f"Watching the newest run: {run_dir}")

    if not run_dir.is_dir():
        raise click.ClickException(f"Not a directory: {run_dir}")

    watch(run_dir, interval_s=interval, display=True)
