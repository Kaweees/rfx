from __future__ import annotations

from pathlib import Path

import rfx.real.so101 as so101_mod


def test_auto_pair_persists_identity_across_port_renumber(monkeypatch, tmp_path: Path) -> None:
    map_path = tmp_path / "so101_port_map.json"
    monkeypatch.setenv("RFX_SO101_PORT_MAP", str(map_path))

    calls = {"n": 0}

    def _discover_ports():
        calls["n"] += 1
        if calls["n"] == 1:
            return [
                {
                    "port": "/dev/ttyACM0",
                    "description": "usb serial",
                    "robot_type": "so101",
                    "hwid": "A",
                    "serial_number": "SER-LEFT",
                    "location": "1-1",
                },
                {
                    "port": "/dev/ttyACM1",
                    "description": "usb serial",
                    "robot_type": "so101",
                    "hwid": "B",
                    "serial_number": "SER-RIGHT",
                    "location": "1-2",
                },
            ]
        return [
            {
                "port": "/dev/ttyACM7",
                "description": "usb serial",
                "robot_type": "so101",
                "hwid": "A",
                "serial_number": "SER-LEFT",
                "location": "1-1",
            },
            {
                "port": "/dev/ttyACM3",
                "description": "usb serial",
                "robot_type": "so101",
                "hwid": "B",
                "serial_number": "SER-RIGHT",
                "location": "1-2",
            },
        ]

    monkeypatch.setattr("rfx.node.discover_ports", _discover_ports)

    leader1, follower1 = so101_mod._auto_pair()
    assert (leader1, follower1) == ("/dev/ttyACM0", "/dev/ttyACM1")

    leader2, follower2 = so101_mod._auto_pair()
    assert (leader2, follower2) == ("/dev/ttyACM7", "/dev/ttyACM3")


def test_auto_pair_uses_deterministic_fallback_without_mapping(
    monkeypatch, tmp_path: Path
) -> None:
    map_path = tmp_path / "so101_port_map.json"
    monkeypatch.setenv("RFX_SO101_PORT_MAP", str(map_path))

    monkeypatch.setattr(
        "rfx.node.discover_ports",
        lambda: [
            {
                "port": "/dev/ttyACM9",
                "description": "usb serial",
                "robot_type": "so101",
                "hwid": "X",
            },
            {
                "port": "/dev/ttyACM2",
                "description": "usb serial",
                "robot_type": "so101",
                "hwid": "Y",
            },
        ],
    )

    leader, follower = so101_mod._auto_pair()
    assert (leader, follower) == ("/dev/ttyACM2", "/dev/ttyACM9")
