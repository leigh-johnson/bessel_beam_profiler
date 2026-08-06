"""
Minimal Mode-B reproducer — pure PySpin, ZERO project imports.
 
Mimics the auto-scan acquisition rhythm exactly:
 
    BeginAcquisition -> arm delay -> TriggerSoftware -> GetNextImage(timeout)
    -> EndAcquisition -> idle pause (stands in for the gantry move)
 
and classifies every iteration by FrameID:
 
    OK      FrameID 1 on the first trigger        (healthy)
    MODE-B  timeout, retry delivers FrameID >= 2  (camera exposed frame 1,
                                                   host never received it)
    MODE-A  timeout, retry delivers FrameID 1     (trigger genuinely dropped)
    DEAD    first trigger AND retry both time out (nothing arriving at all)
 
If MODE-B appears here, the scan code is exonerated by construction —
there is nothing here but PySpin.
 
Suggested protocol:
    1. Run with the gantry parked:      python mode_b_repro.py --serial 24520699
    2. Run again while jogging the gantry around from the FluidNC web UI.
       If (2) fails and (1) doesn't, the moving cable is the culprit.
 
Ctrl-C exits cleanly and still prints the summary.
"""
 
import argparse
import datetime
import sys
import time
 
import PySpin
 
 
def log(msg: str) -> None:
    print(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}  {msg}", flush=True)
 
 
def open_camera(system, serial):
    cam_list = system.GetCameras()
    if cam_list.GetSize() == 0:
        raise RuntimeError("No FLIR cameras detected.")
    cam = None
    if serial:
        for i in range(cam_list.GetSize()):
            c = cam_list.GetByIndex(i)
            if c.TLDevice.DeviceSerialNumber.GetValue() == serial:
                cam = c
                break
        if cam is None:
            raise RuntimeError(f"No camera with serial {serial!r} found.")
    else:
        cam = cam_list.GetByIndex(0)
    cam.Init()
    log(f"Opened camera serial {cam.TLDevice.DeviceSerialNumber.GetValue()}")
    return cam_list, cam
 
 
def configure(cam, exposure_us):
    """Mirror the scan's camera_settings.json (the acquisition-relevant part)."""
    cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
    cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)
    cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
    cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    cam.ExposureMode.SetValue(PySpin.ExposureMode_Timed)
    cam.ExposureTime.SetValue(float(exposure_us))
    cam.GainAuto.SetValue(PySpin.GainAuto_Off)
    cam.Gain.SetValue(0.0)
    cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)
    cam.GevSCPSPacketSize.SetValue(1500)
    # Same throughput limit the scan ends up with (10 MB/s clamped to camera min).
    node = cam.DeviceLinkThroughputLimit
    node.SetValue(max(node.GetMin(), min(10_000_000, node.GetMax())))
    # Host-side stream buffers: NewestOnly, 10 manual buffers (as in the scan).
    stream = cam.TLStream
    stream.StreamBufferCountMode.SetValue(PySpin.StreamBufferCountMode_Manual)
    stream.StreamBufferCountManual.SetValue(10)
    stream.StreamBufferHandlingMode.SetValue(
        PySpin.StreamBufferHandlingMode_NewestOnly
    )
 
 
def one_iteration(cam, timeout_ms, arm_delay_s):
    """Returns (category, detail). Mirrors dataset_writer's trigger/retry shape."""
    cam.BeginAcquisition()
    try:
        time.sleep(arm_delay_s)
 
        cam.TriggerSoftware.Execute()
        t0 = time.monotonic()
        try:
            img = cam.GetNextImage(timeout_ms)
            fid = img.GetFrameID()
            inc = img.IsIncomplete()
            img.Release()
            dt = time.monotonic() - t0
            if inc:
                return "INCOMPLETE", f"FrameID {fid} after {dt:.2f}s"
            return ("OK" if fid == 1 else "WEIRD"), f"FrameID {fid} after {dt:.2f}s"
        except PySpin.SpinnakerException as ex:
            if "-1011" not in str(ex):
                raise
 
        # First trigger timed out — retry once, classify by FrameID.
        cam.TriggerSoftware.Execute()
        t1 = time.monotonic()
        try:
            img = cam.GetNextImage(timeout_ms)
            fid = img.GetFrameID()
            img.Release()
            dt = time.monotonic() - t1
            if fid >= 2:
                return "MODE-B", (
                    f"frame 1 never arrived; retry gave FrameID {fid} "
                    f"after {dt:.2f}s"
                )
            return "MODE-A", f"retry gave FrameID {fid} after {dt:.2f}s"
        except PySpin.SpinnakerException as ex:
            if "-1011" not in str(ex):
                raise
            return "DEAD", "first trigger AND retry both timed out"
    finally:
        try:
            cam.EndAcquisition()
        except PySpin.SpinnakerException as ex:
            log(f"EndAcquisition failed: {ex}")
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serial", default=None, help="camera serial (e.g. 24520699)")
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--exposure-us", type=float, default=7000.0)
    ap.add_argument("--timeout-ms", type=int, default=2000)
    ap.add_argument("--arm-delay-s", type=float, default=0.1)
    ap.add_argument(
        "--idle-s", type=float, default=1.5,
        help="pause between iterations; stands in for the gantry move",
    )
    args = ap.parse_args()
 
    counts = {}
    system = PySpin.System.GetInstance()
    cam_list = cam = None
    try:
        cam_list, cam = open_camera(system, args.serial)
        configure(cam, args.exposure_us)
        log(
            f"Config: exposure {args.exposure_us:g} us, timeout "
            f"{args.timeout_ms} ms, arm {args.arm_delay_s:g} s, idle "
            f"{args.idle_s:g} s, {args.iterations} iterations"
        )
 
        for i in range(1, args.iterations + 1):
            try:
                cat, detail = one_iteration(cam, args.timeout_ms, args.arm_delay_s)
            except KeyboardInterrupt:
                raise
            except PySpin.SpinnakerException as ex:
                cat, detail = "ERROR", str(ex)
            counts[cat] = counts.get(cat, 0) + 1
            if cat != "OK" or i % 25 == 0:
                log(f"[{i:4d}] {cat}: {detail}   totals={counts}")
            time.sleep(args.idle_s)
 
    except KeyboardInterrupt:
        log("Interrupted.")
    finally:
        log(f"SUMMARY: {counts}")
        n = sum(counts.values())
        if n:
            bad = n - counts.get("OK", 0)
            log(f"{bad}/{n} iterations abnormal ({100.0 * bad / n:.1f}%)")
        try:
            if cam is not None:
                if cam.IsStreaming():
                    cam.EndAcquisition()
                cam.DeInit()
            del cam
            if cam_list is not None:
                cam_list.Clear()
            system.ReleaseInstance()
        except Exception as ex:  # noqa: BLE001
            log(f"Cleanup warning: {ex}")
        sys.exit(0)
 
 
if __name__ == "__main__":
    main()
