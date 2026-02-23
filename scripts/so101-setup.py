#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import serial


HDR = b"\xff\xff"
INST_PING = 0x01
INST_WRITE = 0x03

# Feetech STS/SCS control-table defaults used by SO-101 stacks.
REG_ID = 0x05
REG_BAUD = 0x06
REG_TORQUE_ENABLE = 0x28

# Common Feetech baud code map. 0 -> 1_000_000.
BAUD_CODE_1M = 0


@dataclass
class Link:
    port: str
    baudrate: int
    ser: serial.Serial


def checksum(dev_id: int, length: int, instruction: int, params: bytes) -> int:
    value = (dev_id + length + instruction + sum(params)) & 0xFF
    return (~value) & 0xFF


def packet(dev_id: int, instruction: int, params: bytes = b"") -> bytes:
    length = len(params) + 2
    csum = checksum(dev_id, length, instruction, params)
    return HDR + bytes([dev_id, length, instruction]) + params + bytes([csum])


def read_status(ser: serial.Serial, timeout_s: float = 0.02) -> bytes | None:
    deadline = time.monotonic() + timeout_s
    buf = bytearray()
    while time.monotonic() < deadline:
        chunk = ser.read(1)
        if chunk:
            buf.extend(chunk)
            if len(buf) >= 4 and buf[:2] == HDR:
                length = buf[3]
                full = 2 + 1 + 1 + length
                while len(buf) < full and time.monotonic() < deadline:
                    more = ser.read(full - len(buf))
                    if more:
                        buf.extend(more)
                return bytes(buf) if len(buf) >= full else None
    return None


def ping(ser: serial.Serial, dev_id: int) -> bool:
    ser.reset_input_buffer()
    ser.write(packet(dev_id, INST_PING))
    resp = read_status(ser)
    return bool(resp and len(resp) >= 6 and resp[2] == dev_id and resp[4] == 0x00)


def write_reg(ser: serial.Serial, dev_id: int, reg: int, value: int) -> bool:
    params = bytes([reg, value & 0xFF])
    ser.reset_input_buffer()
    ser.write(packet(dev_id, INST_WRITE, params))
    resp = read_status(ser)
    return bool(resp and len(resp) >= 6 and resp[2] == dev_id and resp[4] == 0x00)


def open_link(port: str, baudrate: int) -> Link:
    ser = serial.Serial(port=port, baudrate=baudrate, timeout=0.01, write_timeout=0.05)
    return Link(port=port, baudrate=baudrate, ser=ser)


def discover_port(explicit: str | None) -> str:
    if explicit:
        return explicit
    try:
        from rfx import node

        ports = node.discover_so101_ports()
    except Exception:
        ports = []
    if not ports:
        raise RuntimeError("No SO-101 serial adapter found. Pass --port explicitly.")
    return ports[0]


def scan_ids(ser: serial.Serial, start: int, end: int) -> list[int]:
    found: list[int] = []
    for dev_id in range(start, end + 1):
        if ping(ser, dev_id):
            found.append(dev_id)
    return found


def maybe_confirm(force: bool, prompt: str) -> None:
    if force:
        return
    if not sys.stdin.isatty():
        raise RuntimeError(f"{prompt} (non-interactive: pass --yes)")
    answer = input(f"{prompt} [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        raise RuntimeError("Aborted by user")


def normalize_ids(ser: serial.Serial, ids: list[int], yes: bool) -> None:
    if ids == [1, 2, 3, 4, 5, 6]:
        print("IDs already normalized to 1..6")
        return
    if len(ids) != 6:
        raise RuntimeError(
            f"Need exactly 6 responsive motors to normalize IDs, found {len(ids)}: {ids}"
        )

    maybe_confirm(
        yes,
        f"Normalize motor IDs {ids} -> [1, 2, 3, 4, 5, 6] on current serial bus?",
    )

    targets = [1, 2, 3, 4, 5, 6]
    temporary_start = 200
    temp_map: dict[int, int] = {}

    for old in ids:
        if old in targets:
            continue
        temp = temporary_start
        temporary_start += 1
        if not write_reg(ser, old, REG_ID, temp):
            raise RuntimeError(f"Failed re-ID step old={old} -> temp={temp}")
        temp_map[old] = temp
        time.sleep(0.03)

    source_now = [temp_map.get(old, old) for old in ids]
    source_now.sort()
    for src, dst in zip(source_now, targets, strict=True):
        if src == dst:
            continue
        if not write_reg(ser, src, REG_ID, dst):
            raise RuntimeError(f"Failed re-ID step src={src} -> dst={dst}")
        time.sleep(0.03)

    verified = scan_ids(ser, 1, 6)
    if verified != targets:
        raise RuntimeError(
            f"ID normalization verification failed. Found IDs 1..6 subset: {verified}"
        )
    print("ID normalization complete: [1, 2, 3, 4, 5, 6]")


def set_baud_1m(ser: serial.Serial, ids: list[int], yes: bool) -> None:
    maybe_confirm(
        yes,
        f"Write baud code {BAUD_CODE_1M} (1Mbps) to motors {ids}?",
    )
    for dev_id in ids:
        if not write_reg(ser, dev_id, REG_BAUD, BAUD_CODE_1M):
            raise RuntimeError(f"Failed baud write on motor ID {dev_id}")
        time.sleep(0.02)
    print("Baud register updated. Reopen at 1,000,000 baud.")


def set_torque(ser: serial.Serial, ids: list[int], enabled: bool) -> None:
    value = 1 if enabled else 0
    for dev_id in ids:
        if not write_reg(ser, dev_id, REG_TORQUE_ENABLE, value):
            raise RuntimeError(f"Failed torque write on motor ID {dev_id}")
        time.sleep(0.01)


def main() -> int:
    parser = argparse.ArgumentParser(description="SO-101 motor setup (ID/baud) via direct serial")
    parser.add_argument("--port", default=None, help="Serial port (auto-detect if omitted)")
    parser.add_argument(
        "--baud",
        type=int,
        default=None,
        help="Fixed serial baud to use for scan (default: try common bauds)",
    )
    parser.add_argument("--scan-start", type=int, default=1)
    parser.add_argument("--scan-end", type=int, default=20)
    parser.add_argument("--normalize-ids", action="store_true", help="Normalize IDs to 1..6")
    parser.add_argument(
        "--set-baud-1m", action="store_true", help="Write motor baud register to 1Mbps"
    )
    parser.add_argument("--torque-on", action="store_true", help="Enable torque after setup")
    parser.add_argument("--torque-off", action="store_true", help="Disable torque after setup")
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmations")
    args = parser.parse_args()

    if args.torque_on and args.torque_off:
        raise RuntimeError("Use only one of --torque-on or --torque-off")

    port = discover_port(args.port)
    bauds = [args.baud] if args.baud else [1_000_000, 500_000, 115_200]

    chosen: Link | None = None
    found_ids: list[int] = []
    for baud in bauds:
        link = open_link(port, baud)
        try:
            ids = scan_ids(link.ser, args.scan_start, args.scan_end)
            if ids:
                chosen = link
                found_ids = ids
                break
        finally:
            if chosen is not link:
                link.ser.close()

    if chosen is None:
        raise RuntimeError(
            f"No motors responded on {port} at bauds {bauds}. "
            "Check USB adapter, power, and try wider --scan-start/--scan-end."
        )

    print(f"Connected: port={chosen.port} baud={chosen.baudrate}")
    print(f"Detected motor IDs: {found_ids}")

    try:
        if args.normalize_ids:
            normalize_ids(chosen.ser, sorted(found_ids), args.yes)
            found_ids = scan_ids(chosen.ser, 1, 6)
            print(f"Post-normalize IDs: {found_ids}")

        if args.set_baud_1m:
            set_baud_1m(chosen.ser, sorted(found_ids), args.yes)

        if args.torque_on:
            set_torque(chosen.ser, sorted(found_ids), enabled=True)
            print("Torque enabled")
        elif args.torque_off:
            set_torque(chosen.ser, sorted(found_ids), enabled=False)
            print("Torque disabled")

        print("SO-101 setup complete")
    finally:
        chosen.ser.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
