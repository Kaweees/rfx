# Robot Assets

This directory stores local simulation assets that are not bundled in the core package.

Layout:

- `rfx/assets/robots/<robot_name>/urdf/` for URDF assets
- `rfx/assets/robots/<robot_name>/mjcf/` for MJCF assets

Example:

- `rfx/assets/robots/go2/urdf/go2.urdf`
- `rfx/assets/robots/go2/mjcf/go2.xml`

Notes:

- Keep referenced meshes/textures near the URDF/MJCF files (or under a sibling `meshes/` directory).
- Go2 assets are intentionally not bundled; provide your own files under `rfx/assets/robots/go2/`.
- `rfx/configs/go2.yaml` points to `rfx/assets/robots/go2/urdf/go2.urdf` by default.
