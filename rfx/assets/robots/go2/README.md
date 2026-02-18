# Go2 Asset Integration

Go2 assets are intentionally not bundled in rfx.
Drop your own model files here.

Expected files:

- URDF: `rfx/assets/robots/go2/urdf/go2.urdf`
- MJCF (future MuJoCo path): `rfx/assets/robots/go2/mjcf/go2.xml`

Directory layout:

```text
rfx/assets/robots/go2/
├── urdf/
│   ├── go2.urdf
│   └── meshes/...
└── mjcf/
    ├── go2.xml
    └── assets/...
```

Reference meshes source:

- https://github.com/unitreerobotics/unitree_ros/tree/master/robots/go2_description/meshes

Then run:

```bash
uv run --python 3.13 rfx/examples/genesis_viewer.py --config rfx/configs/go2.yaml --auto-install
```
