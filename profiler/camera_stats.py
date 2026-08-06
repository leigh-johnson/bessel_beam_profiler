"""
Camera-side diagnostics snapshot for the FLIR BFS (pure PySpin).

Reads the camera's own counters — the closest thing a Blackfly S has to
"camera logs" — plus the host driver's stream statistics:

    DeviceUptime            seconds since the CAMERA last booted. Survives
                            host reconnects; only resets on a camera
                            reboot/power loss. If this is ever smaller than
                            the time since you last checked, the camera
                            rebooted (PoE brownout smoking gun).
    TransmitFailureCount    frames the camera FAILED TO TRANSMIT. If this
                            climbs during a Mode-B episode, frames are dying
                            inside the camera, not on the wire.
    DeviceLinkSpeed         negotiated link speed (renegotiation detector).
    DeviceTemperature       thermal suspect tracking.
    GevSCPSPacketSize       live negotiated packet size (config says 1500;
                            ~576-590 = large-packet fallback engaged).
    TLStream nodemap        every readable integer/float node — includes
                            delivered/lost/dropped frame counts and resend
                            counters (the driver's delivery accounting).

Usage:
    python camera_stats.py --serial 24520699             # one snapshot
    python camera_stats.py --serial 24520699 --watch 5   # repeat every 5 s,
                                                         # printing deltas

NOTE: needs exclusive camera access — run BETWEEN scans, not during one.
Uptime/temperature/TransmitFailureCount live on the camera, so a snapshot
taken right AFTER a failing run still tells you what happened during it.
"""

import argparse
import datetime
import time

import PySpin


def stamp() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def read_quickspin(cam, name):
    try:
        node = getattr(cam, name)
        return node.GetValue()
    except Exception:  # noqa: BLE001 - node absent/unreadable
        return None


def read_stream_nodes(cam) -> dict:
    """Dump every readable integer/float node from the TLStream nodemap."""
    out = {}
    try:
        nodemap = cam.GetTLStreamNodeMap()
        nodes = nodemap.GetNodes()
    except Exception:  # noqa: BLE001
        return out
    for node in nodes:
        try:
            name = node.GetName()
            intf = node.GetPrincipalInterfaceType()
            if intf == PySpin.intfIInteger:
                out[name] = PySpin.CIntegerPtr(node).GetValue()
            elif intf == PySpin.intfIFloat:
                out[name] = PySpin.CFloatPtr(node).GetValue()
        except Exception:  # noqa: BLE001 - skip unreadable nodes
            continue
    return out


def snapshot(cam) -> dict:
    snap = {}
    for name in (
        "DeviceUptime",
        "DeviceTemperature",
        "DeviceLinkSpeed",
        "DeviceLinkThroughputLimit",
        "GevSCPSPacketSize",
        "TransmitFailureCount",
        "ExposureTime",
    ):
        val = read_quickspin(cam, name)
        if val is not None:
            snap[name] = val
    snap.update(read_stream_nodes(cam))
    return snap


def print_snapshot(snap: dict, prev: dict | None) -> None:
    print(f"--- {stamp()} ---")
    for key in sorted(snap):
        val = snap[key]
        line = f"  {key:38s} {val}"
        if prev is not None and key in prev and prev[key] != val:
            try:
                line += f"   (delta {val - prev[key]:+g})"
            except TypeError:
                line += f"   (was {prev[key]})"
        print(line)
    if prev is not None and "DeviceUptime" in snap and "DeviceUptime" in prev:
        if snap["DeviceUptime"] < prev["DeviceUptime"]:
            print(
                "  *** DeviceUptime DECREASED — the camera REBOOTED since "
                "the previous snapshot (power loss / brownout?) ***"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serial", default=None)
    ap.add_argument(
        "--watch", type=float, default=None,
        help="repeat every N seconds, printing deltas (Ctrl-C to stop)",
    )
    args = ap.parse_args()

    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    cam = None
    if args.serial:
        for i in range(cam_list.GetSize()):
            c = cam_list.GetByIndex(i)
            if c.TLDevice.DeviceSerialNumber.GetValue() == args.serial:
                cam = c
                break
        if cam is None:
            raise SystemExit(f"No camera with serial {args.serial!r} found.")
    else:
        if cam_list.GetSize() == 0:
            raise SystemExit("No FLIR cameras detected.")
        cam = cam_list.GetByIndex(0)

    cam.Init()
    print(f"Camera serial {cam.TLDevice.DeviceSerialNumber.GetValue()}")

    prev = None
    try:
        while True:
            snap = snapshot(cam)
            print_snapshot(snap, prev)
            prev = snap
            if args.watch is None:
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cam.DeInit()
            del cam
            cam_list.Clear()
            system.ReleaseInstance()
        except Exception as ex:  # noqa: BLE001
            print(f"Cleanup warning: {ex}")


if __name__ == "__main__":
    main()
