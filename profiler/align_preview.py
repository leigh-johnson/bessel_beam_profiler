"""
Live matplotlib preview for the axicon alignment patrol.

One window, refreshed after every station and every cycle:

    left   — the patrolled ring stitched in machine coordinates
             (compose_canvas), with the fitted ellipse, fitted center,
             reference cross, and station markers overlaid
    right  — big-number metrics panel + center-offset history plot

Keys:  r = set the alignment reference to the CURRENT fitted center
       o = run a full orbit lap now (stream mode: refreshes radius,
           roundness, and azimuthal uniformity)
       f = re-run the find-beam bootstrap (stream mode)
       q = quit (also just close the window)

The preview owns no hardware: align_cli feeds it CycleResults through
AxiconAlignSession.run's callbacks. display=False renders on the Agg
backend (nothing shown) so the same code path is unit-testable and
save_png still works — useful for headless/remote runs too.
"""

from __future__ import annotations

from typing import Optional
import logging

import numpy as np

from align_axicon import AlignConfig, CycleResult, StreamSample, compose_canvas

logger = logging.getLogger(__name__)


class AlignPreview:
    def __init__(self, config: AlignConfig, display: bool = True):
        import matplotlib

        if not display:
            matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        self.plt = plt
        self.config = config
        self.display = display
        self.quit_requested = False
        self.reference_requested = False
        self.orbit_requested = False
        self.refind_requested = False
        self._offset_history: list[tuple[int, float, float]] = []
        # Last orbit's full-ring numbers, kept on screen between laps.
        self._last_cycle: Optional[CycleResult] = None

        if display:
            plt.ion()

        self.figure = plt.figure("Axicon alignment", figsize=(12.5, 7.0))
        grid = self.figure.add_gridspec(
            2, 2, width_ratios=[1.9, 1.0], height_ratios=[1.6, 1.0]
        )
        self.ax_ring = self.figure.add_subplot(grid[:, 0])
        self.ax_text = self.figure.add_subplot(grid[0, 1])
        self.ax_history = self.figure.add_subplot(grid[1, 1])
        self.ax_text.set_axis_off()

        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.figure.canvas.mpl_connect("close_event", self._on_close)

        self._status = self.figure.suptitle(
            "Axicon alignment — bootstrapping (find-beam sweep)...",
            fontsize=11,
        )

    # -- events ---------------------------------------------------------

    def _on_key(self, event) -> None:
        if event.key == "q":
            self.quit_requested = True
        elif event.key == "r":
            self.reference_requested = True
        elif event.key == "o":
            self.orbit_requested = True
        elif event.key == "f":
            self.refind_requested = True

    def _on_close(self, _event) -> None:
        self.quit_requested = True

    def _pump(self) -> None:
        if self.display:
            self.plt.pause(0.001)

    # -- station-level updates (cheap: title only) ----------------------

    def on_station(self, sample, _frame) -> None:
        state = "signal" if sample.HasSignal else "no signal"
        self._status.set_text(
            f"Axicon alignment — station at X {sample.X_mm:.1f}, "
            f"Z {sample.Z_mm:.1f} mm ({state}, peak {sample.Peak:.0f})   "
            "[r = set reference, q = quit]"
        )
        self._pump()

    # -- stream-mode redraw (single parked frame, a few Hz) --------------

    def update_stream(
        self, sample: StreamSample, frame: Optional[np.ndarray]
    ) -> None:
        ax = self.ax_ring
        ax.clear()
        ax.set_xlabel("machine X (mm)")
        ax.set_ylabel("machine Z (mm)")
        ax.set_aspect("equal")

        if frame is not None:
            mm_per_px = self.config.mm_per_px()
            rows, cols = frame.shape
            half_w = cols * mm_per_px / 2.0
            half_h = rows * mm_per_px / 2.0
            lit = frame[frame > 0]
            vmax = float(np.percentile(lit, 99.5)) if lit.size else 1.0
            ax.imshow(
                frame,
                extent=(
                    sample.ParkX_mm - half_w,
                    sample.ParkX_mm + half_w,
                    sample.ParkZ_mm - half_h,
                    sample.ParkZ_mm + half_h,
                ),
                origin="upper",
                cmap="inferno",
                vmin=0.0,
                vmax=max(vmax, 1.0),
            )

        if sample.CenterX_mm is not None:
            from matplotlib.patches import Circle as CirclePatch

            ax.add_patch(
                CirclePatch(
                    (sample.CenterX_mm, sample.CenterZ_mm),
                    sample.Radius_mm,
                    fill=False,
                    color="cyan",
                    linewidth=1.0,
                    linestyle="--",
                )
            )
            ax.plot(
                sample.CenterX_mm,
                sample.CenterZ_mm,
                "+",
                color="cyan",
                markersize=12,
                markeredgewidth=2,
            )

        reference = None
        if sample.Offset_mm is not None and sample.CenterX_mm is not None:
            reference = (
                sample.CenterX_mm - sample.Offset_mm[0],
                sample.CenterZ_mm - sample.Offset_mm[1],
            )
        if reference is not None:
            ax.plot(
                reference[0],
                reference[1],
                "+",
                color="limegreen",
                markersize=16,
                markeredgewidth=1.5,
            )

        self._draw_stream_metrics(sample)

        if sample.Offset_mm is not None:
            self._offset_history.append(
                (sample.Index, sample.Offset_mm[0], sample.Offset_mm[1])
            )
        self._redraw_history()

        state = "ARC LOST" if sample.Lost else "streaming"
        self._status.set_text(
            f"Axicon alignment — frame {sample.Index} @ machine Y "
            f"{sample.MachineY_mm:g} mm: {state} "
            f"({sample.Elapsed_s:.2f} s/frame)   "
            "[r = reference, o = orbit, f = re-find, q = quit]"
        )
        self.figure.canvas.draw_idle()
        self._pump()

    def _draw_stream_metrics(self, sample: StreamSample) -> None:
        ax = self.ax_text
        ax.clear()
        ax.set_axis_off()

        lines: list[str] = []

        if sample.CenterX_mm is not None:
            lines.append(
                f"center   X {sample.CenterX_mm:8.3f}   "
                f"Z {sample.CenterZ_mm:8.3f} mm"
            )
            lines.append(
                f"         (arc fit @ fixed r, rms "
                f"{sample.FitRMS_mm * 1000:.0f} um)"
            )
        else:
            lines.append("center   —  (no arc in view)")

        if sample.Offset_mm is not None:
            dx, dz = sample.Offset_mm
            lines.append(
                f"offset   dX {dx * 1000:+7.0f} um   dZ {dz * 1000:+7.0f} um"
            )
        else:
            lines.append("offset   —  (press r to set the reference)")

        lines.append(f"radius   {sample.Radius_mm:6.3f} mm  (from last orbit)")

        if sample.WidthFWHM_mm is not None:
            lines.append(f"width    {sample.WidthFWHM_mm * 1000:6.0f} um FWHM")

        cycle = self._last_cycle
        if cycle is not None:
            if cycle.Ellipse is not None:
                lines.append(
                    f"round    {cycle.Ellipse.axis_ratio:6.4f}  "
                    f"(major @ {cycle.Ellipse.MajorAxisAngle_deg:5.1f} deg)"
                )
            if cycle.Uniformity is not None:
                lines.append(
                    f"azimuth  min/max {cycle.Uniformity.MinMaxRatio:5.3f}   "
                    f"dim @ {cycle.Uniformity.DimmestAngle_deg:5.1f} deg"
                )
            lines.append(f"         (orbit {cycle.Index} — press o to refresh)")

        lines.append(f"peak     {sample.ArcPeak:6.0f} counts")

        ax.text(
            0.02,
            0.98,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=10,
            fontfamily="monospace",
            verticalalignment="top",
        )

    # -- cycle-level redraw ---------------------------------------------

    def update(
        self,
        result: CycleResult,
        frames: list[tuple[np.ndarray, float, float]],
    ) -> None:
        self._last_cycle = result
        self._draw_ring(result, frames)
        self._draw_metrics(result)
        self._draw_history(result)

        y_label = f"machine Y {result.MachineY_mm:g} mm"
        state = "RING LOST" if result.Lost else "tracking"
        self._status.set_text(
            f"Axicon alignment — cycle {result.Index} @ {y_label}: {state} "
            f"({result.Elapsed_s:.1f} s/cycle)   [r = set reference, q = quit]"
        )

        self.figure.canvas.draw_idle()
        self._pump()

    def _draw_ring(self, result: CycleResult, frames) -> None:
        ax = self.ax_ring
        ax.clear()
        ax.set_xlabel("machine X (mm)")
        ax.set_ylabel("machine Z (mm)")
        ax.set_aspect("equal")

        composed = compose_canvas(frames, self.config.mm_per_px())
        if composed is not None:
            canvas, (x_min, x_max, z_min, z_max) = composed
            # Un-imaged pixels are NaN in the canvas: render them as a
            # neutral gray so coverage gaps between station frames are
            # visually distinct from genuinely dark beam regions.
            import matplotlib.pyplot as plt

            cmap = plt.get_cmap("inferno").copy()
            cmap.set_bad("0.35")

            lit = canvas[np.isfinite(canvas) & (canvas > 0)]
            vmax = float(np.percentile(lit, 99.5)) if lit.size else 1.0
            ax.imshow(
                canvas,
                extent=(x_min, x_max, z_min, z_max),
                origin="upper",
                cmap=cmap,
                vmin=0.0,
                vmax=max(vmax, 1.0),
            )

        # Ring-locus points used by the fits.
        if result.Points_mm.shape[0]:
            ax.plot(
                result.Points_mm[:, 0],
                result.Points_mm[:, 1],
                ".",
                markersize=2,
                color="deepskyblue",
                alpha=0.6,
            )

        # Patrol stations: filled = saw signal, open = dark, x = skipped;
        # gray = interior fill stations (disk cover mode, not fitted).
        for sample in result.Stations:
            if sample.Skipped:
                ax.plot(sample.X_mm, sample.Z_mm, "x", color="gray")
            else:
                color = (
                    "white" if getattr(sample, "Role", "ring") == "ring"
                    else "darkgray"
                )
                ax.plot(
                    sample.X_mm,
                    sample.Z_mm,
                    "o",
                    markersize=5,
                    markerfacecolor=color if sample.HasSignal else "none",
                    markeredgecolor=color,
                    alpha=0.7,
                )

        if result.Ellipse is not None:
            from matplotlib.patches import Ellipse as EllipsePatch

            ellipse = result.Ellipse
            ax.add_patch(
                EllipsePatch(
                    (ellipse.CenterX_mm, ellipse.CenterZ_mm),
                    2.0 * ellipse.SemiMajor_mm,
                    2.0 * ellipse.SemiMinor_mm,
                    angle=ellipse.MajorAxisAngle_deg,
                    fill=False,
                    color="cyan",
                    linewidth=1.2,
                )
            )

        if result.Circle is not None:
            ax.plot(
                result.Circle.CenterX_mm,
                result.Circle.CenterZ_mm,
                "+",
                color="cyan",
                markersize=12,
                markeredgewidth=2,
            )

        if result.Reference is not None:
            ax.plot(
                result.Reference[0],
                result.Reference[1],
                "+",
                color="limegreen",
                markersize=16,
                markeredgewidth=1.5,
            )

    def _draw_metrics(self, result: CycleResult) -> None:
        ax = self.ax_text
        ax.clear()
        ax.set_axis_off()

        lines: list[str] = []

        if result.Circle is not None:
            circle = result.Circle
            lines.append(
                f"center   X {circle.CenterX_mm:8.3f}   Z {circle.CenterZ_mm:8.3f} mm"
            )
            lines.append(
                f"radius   {circle.Radius_mm:6.3f} mm   "
                f"(fit rms {circle.RMS_mm * 1000:.0f} um, n={circle.NPoints})"
            )
        else:
            lines.append("center   —  (no ring fit this cycle)")

        if result.Offset_mm is not None:
            dx, dz = result.Offset_mm
            lines.append(
                f"offset   dX {dx * 1000:+7.0f} um   dZ {dz * 1000:+7.0f} um"
            )
        else:
            lines.append("offset   —  (press r to set the reference)")

        if result.Ellipse is not None:
            ellipse = result.Ellipse
            lines.append(
                f"round    {ellipse.axis_ratio:6.4f}  "
                f"(major axis @ {ellipse.MajorAxisAngle_deg:5.1f} deg)"
            )
        else:
            lines.append("round    —")

        if result.Uniformity is not None:
            uniformity = result.Uniformity
            lines.append(
                f"azimuth  min/max {uniformity.MinMaxRatio:5.3f}   "
                f"cv {uniformity.CoefficientOfVariation:5.3f}"
            )
            lines.append(
                f"         dim @ {uniformity.DimmestAngle_deg:5.1f} deg   "
                f"bright @ {uniformity.BrightestAngle_deg:5.1f} deg"
            )
            lines.append(
                f"coverage {uniformity.CoverageFraction * 100:4.0f} % of ring"
            )
        else:
            lines.append("azimuth  —")

        if result.Tilt is not None:
            tilt = result.Tilt
            lines.append(
                f"tilt     X {tilt.TiltX_mrad:+6.2f}   "
                f"Z {tilt.TiltZ_mrad:+6.2f} mrad "
                f"(dY {tilt.DeltaBeamPath_mm:+.0f} mm)"
            )
            if getattr(tilt, "Cone_mrad", None) is not None:
                lines.append(
                    f"cone     {tilt.Cone_mrad:+6.2f} mrad "
                    "(+ = ring expands downstream)"
                )
        elif self.config.MachineY2_mm is not None:
            lines.append("tilt     —  (waiting for both Y planes)")

        if result.Exposure_us is not None:
            lines.append(f"exposure {result.Exposure_us:8.0f} us")

        ax.text(
            0.02,
            0.98,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=10,
            fontfamily="monospace",
            verticalalignment="top",
        )

    def _draw_history(self, result: CycleResult) -> None:
        if result.Offset_mm is not None:
            self._offset_history.append(
                (result.Index, result.Offset_mm[0], result.Offset_mm[1])
            )
        self._redraw_history()

    def _redraw_history(self) -> None:
        ax = self.ax_history
        ax.clear()
        ax.set_xlabel("cycle / frame")
        ax.set_ylabel("center offset (mm)")
        ax.axhline(0.0, color="gray", linewidth=0.6)

        if self._offset_history:
            history = np.array(self._offset_history)
            ax.plot(history[:, 0], history[:, 1], "-o", markersize=3, label="dX")
            ax.plot(history[:, 0], history[:, 2], "-o", markersize=3, label="dZ")
            ax.legend(loc="upper right", fontsize=8)

    # -- persistence ----------------------------------------------------

    def save_png(self, path) -> Optional[str]:
        try:
            self.figure.savefig(path, dpi=110, facecolor="white")
            return str(path)
        except Exception as ex:  # noqa: BLE001 - snapshot is best-effort
            logger.warning(f"Could not save the preview snapshot: {ex}")
            return None

    def close(self) -> None:
        self.plt.close(self.figure)


class CorePreview:
    """
    Core-mode window (axicon 3): live image crop + X/Z chords + radial
    mean + J0^2 fit — the interactive version of the core-frame gallery
    panels.

    Keys:  r         save a snapshot (figure PNG + raw crop .npy)
           up/down   jog machine Y by +/- the configured step
           f         re-run find-beam + exposure calibration here
           q         quit (closing the window also quits)
    """

    def __init__(self, config: AlignConfig, display: bool = True):
        import matplotlib

        if not display:
            matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        self.plt = plt
        self.config = config
        self.display = display
        self.quit_requested = False
        self.save_requested = False
        self.refind_requested = False
        self.jog_up_requested = False
        self.jog_down_requested = False

        if display:
            plt.ion()

        self.figure = plt.figure("Axicon 3 core alignment", figsize=(12.5, 5.6))
        grid = self.figure.add_gridspec(1, 2, width_ratios=[1.0, 1.6])
        self.ax_image = self.figure.add_subplot(grid[0, 0])
        self.ax_profile = self.figure.add_subplot(grid[0, 1])

        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.figure.canvas.mpl_connect("close_event", self._on_close)

        self._status = self.figure.suptitle(
            "Axicon 3 core alignment — bootstrapping (find-beam sweep)...",
            fontsize=11,
        )

    def _on_key(self, event) -> None:
        if event.key == "q":
            self.quit_requested = True
        elif event.key == "r":
            self.save_requested = True
        elif event.key == "f":
            self.refind_requested = True
        elif event.key == "up":
            self.jog_up_requested = True
        elif event.key == "down":
            self.jog_down_requested = True

    def _on_close(self, _event) -> None:
        self.quit_requested = True

    def _pump(self) -> None:
        if self.display:
            self.plt.pause(0.001)

    def update_core(self, sample, frame) -> None:
        from matplotlib.colors import PowerNorm

        px_um = self.config.PixelSize_um
        ax = self.ax_image
        ax.clear()

        if frame is not None and not sample.Lost:
            crop_px = max(
                8, int(round(self.config.CoreCropRadius_um / px_um))
            )
            ir = int(round(sample.CentroidRow_px))
            ic = int(round(sample.CentroidCol_px))
            r0, r1 = max(ir - crop_px, 0), min(ir + crop_px + 1, frame.shape[0])
            c0, c1 = max(ic - crop_px, 0), min(ic + crop_px + 1, frame.shape[1])
            crop = frame[r0:r1, c0:c1].astype(float)
            extent = (
                (c0 - sample.CentroidCol_px) * px_um,
                (c1 - sample.CentroidCol_px) * px_um,
                (r1 - sample.CentroidRow_px) * px_um,
                (r0 - sample.CentroidRow_px) * px_um,
            )
            ax.imshow(
                crop,
                extent=extent,
                cmap="inferno",
                norm=PowerNorm(0.4, vmax=max(crop.max(), 1.0)),
            )
            ax.set_xlabel("x (um)")
            ax.set_ylabel("z (um)")
        else:
            ax.text(0.5, 0.5, "no core found", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_xticks([]), ax.set_yticks([])

        ax = self.ax_profile
        ax.clear()

        title = f"frame {sample.Index} — no fit"
        if not sample.Lost and sample.XChord is not None:
            ax.plot(*sample.XChord, ".", ms=3, color="tab:blue",
                    label="X chord")
            ax.plot(*sample.ZChord, ".", ms=3, color="tab:green",
                    label="Z chord")
            r_um, radial = sample.Radial
            ax.plot(r_um, radial, "-", color="black", lw=1.0,
                    label="radial (mean)")
            ax.plot(-r_um, radial, "-", color="black", lw=1.0)
            if sample.FitCurve is not None:
                rf, cf = sample.FitCurve
                ax.plot(rf, cf, "-", color="tab:red", lw=1.2,
                        label="J0^2 fit to radial")
                ax.plot(-rf, cf, "-", color="tab:red", lw=1.2)

            ideal = sample.KrIdeal_per_m

            def pct(k):
                return (
                    f"{(k / ideal - 1) * 100:+.1f}%"
                    if (k and ideal) else "—"
                )

            title = (
                f"kx {pct(sample.Kx_per_m)}, kz {pct(sample.Kz_per_m)}, "
                f"kr {pct(sample.Kr_per_m)}"
            )
            if sample.Kr_per_m and sample.KrStd_per_m and ideal:
                title += f" ± {sample.KrStd_per_m / ideal * 100:.1f}%"
            title += " vs. ideal"
            if sample.RmsOverA is not None:
                title += f".  rms/A_fit = {sample.RmsOverA:.3f}"
            if sample.Saturated:
                title += "  [SATURATED]"

        ax.set_title(title, fontsize=9)
        ax.set_xlabel("position (um) — X/Z chords and +/- r")
        ax.set_ylabel("(I - I_bg)/T (counts/us)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

        state = "CORE LOST" if sample.Lost else "streaming"
        self._status.set_text(
            f"Axicon 3 core — machine Y {sample.MachineY_mm:g} mm, "
            f"park (X {sample.ParkX_mm:g}, Z {sample.ParkZ_mm:g}), "
            f"T {sample.Exposure_us:.0f} us: {state} "
            f"({sample.Elapsed_s:.2f} s/frame)   "
            "[r = save, up/down = jog Y, f = re-find, q = quit]"
        )
        self.figure.canvas.draw_idle()
        self._pump()

    def save_png(self, path) -> Optional[str]:
        try:
            self.figure.savefig(path, dpi=110, facecolor="white")
            return str(path)
        except Exception as ex:  # noqa: BLE001 - snapshot is best-effort
            logger.warning(f"Could not save the core snapshot: {ex}")
            return None

    def close(self) -> None:
        self.plt.close(self.figure)


class FreePreview:
    """
    Free-stream window: the WHOLE downsampled frame in machine
    coordinates (axes = machine X / Z, so the view pans as you jog) —
    no fits, no compositing, just live pixels + a position readout.

    Keys (same bindings work typed into the launching TERMINAL, so the
    window never needs focus — see term_keys.py):

        arrows      jog X (left/right) and Z (up/down)
        , / .       jog Y upstream / downstream (also < / >)
        - / =       halve / double the jog step
        e / E       exposure down / up (switches auto-exposure OFF)
        a           toggle the auto-exposure servo
        r           save a snapshot (figure PNG + raw frame .npy)
        q           quit (closing the window also quits)
    """

    def __init__(self, config: AlignConfig, display: bool = True):
        import matplotlib

        if not display:
            matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        self.plt = plt
        self.config = config
        self.display = display
        self.quit_requested = False
        self.save_requested = False
        self.pending_actions: list[str] = []

        if display:
            plt.ion()

        self.figure = plt.figure("Live camera stream", figsize=(9.0, 7.2))
        self.ax = self.figure.add_subplot(1, 1, 1)

        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.figure.canvas.mpl_connect("close_event", self._on_close)

        self._status = self.figure.suptitle(
            "Free stream — waiting for the first frame...", fontsize=11
        )

    def _on_key(self, event) -> None:
        from align_axicon import FREE_KEY_ACTIONS

        if event.key == "q":
            self.quit_requested = True
        elif event.key == "r":
            self.save_requested = True
        else:
            action = FREE_KEY_ACTIONS.get(event.key)
            if action is not None:
                self.pending_actions.append(action)

    def _on_close(self, _event) -> None:
        self.quit_requested = True

    def pop_action(self) -> Optional[str]:
        """Oldest queued key action, or None (one applied per frame)."""

        return self.pending_actions.pop(0) if self.pending_actions else None

    def _pump(self) -> None:
        if self.display:
            self.plt.pause(0.001)

    def update_free(self, sample, frame) -> None:
        """Draw one FreeSample + its NATIVE oriented frame (or None)."""

        from matplotlib.colors import PowerNorm

        from composite import mean_pool

        ax = self.ax
        ax.clear()

        if frame is not None:
            view = mean_pool(frame, self.config.Downsample)
            mm = self.config.mm_per_px()
            height_mm = view.shape[0] * mm
            width_mm = view.shape[1] * mm
            extent = (
                sample.MachineX_mm - width_mm / 2.0,
                sample.MachineX_mm + width_mm / 2.0,
                sample.MachineZ_mm - height_mm / 2.0,
                sample.MachineZ_mm + height_mm / 2.0,
            )
            ax.imshow(
                view,
                extent=extent,
                cmap="inferno",
                norm=PowerNorm(0.4, vmax=max(float(view.max()), 1.0)),
                origin="upper",
            )
            ax.set_xlabel("machine X (mm)")
            ax.set_ylabel("machine Z (mm)")
            title = f"frame {sample.Index} — peak {sample.Peak:g}"
            if sample.Saturated:
                title += "  [SATURATED]"
        else:
            ax.text(0.5, 0.5, "no frame", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_xticks([]), ax.set_yticks([])
            title = f"frame {sample.Index} — NO FRAME"
        ax.set_title(title, fontsize=10)

        auto = "auto" if sample.AutoExposure else "MANUAL"
        self._status.set_text(
            f"X {sample.MachineX_mm:g}  Y {sample.MachineY_mm:g}  "
            f"Z {sample.MachineZ_mm:g} mm — step {sample.JogStep_mm:g} mm, "
            f"T {sample.Exposure_us:.0f} us ({auto}), "
            f"{sample.Elapsed_s:.2f} s/frame   "
            "[arrows = X/Z, ,/. = Y, -/= step, e/E exp, a auto, "
            "r snap, q quit]"
        )
        self.figure.canvas.draw_idle()
        self._pump()

    def save_png(self, path) -> Optional[str]:
        try:
            self.figure.savefig(path, dpi=110, facecolor="white")
            return str(path)
        except Exception as ex:  # noqa: BLE001 - snapshot is best-effort
            logger.warning(f"Could not save the free-stream snapshot: {ex}")
            return None

    def close(self) -> None:
        self.plt.close(self.figure)
