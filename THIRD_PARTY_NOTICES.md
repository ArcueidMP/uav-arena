# Third-party notices

UAV Arena is licensed under the GNU General Public License v3.0 only ([LICENSE](LICENSE)). It
contains the third-party files below, unchanged, under their own licenses. At run time the package
checks every vendored file against the SHA-256 recorded in its `UPSTREAM.json`.

## MuJoCo Menagerie: Bitcraze Crazyflie 2 model

- Source: [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie),
  revision `f3b6013629c7b94f8ee93fd977b161fa8350f9bc`, directory `bitcraze_crazyflie_2`.
- Location: `src/uav_arena/third_party/menagerie/bitcraze_crazyflie_2/` (`cf2.xml`, meshes, `LICENSE`).
- License: MIT. The license text is in that directory's `LICENSE`.

## IMRCLab crazyflie-firmware: Lee position and attitude controller

- Source: [IMRCLab/crazyflie-firmware](https://github.com/IMRCLab/crazyflie-firmware/tree/b08d9f3cbcc581d8e7f988cabad22c26d35f4a58),
  commit `b08d9f3cbcc581d8e7f988cabad22c26d35f4a58`, the firmware revision pinned by
  [IMRCLab/coltrans-planning](https://github.com/IMRCLab/coltrans-planning).
- Location: `src/uav_arena/third_party/crazyflie_firmware_lee/`. `UPSTREAM.json` records each file's
  original path, Git blob and SHA-256.

| File | Notice |
|---|---|
| `controller_lee.c` | MIT, Copyright (c) 2019 Wolfgang Hoenig |
| `math3d.h` | MIT, Copyright (c) 2016-2018 James Alan Preiss |
| `controller_lee.h`, `stabilizer_types.h`, `imu_types.h` | GPL-3.0, Copyright (C) Bitcraze AB |
| `lighthouse_types.h` | no file-level notice; part of the GPL-3.0 firmware repository |
| `LICENSE.txt` | the firmware repository's GPL-3.0 license |

The controller is compiled unchanged. The arena's wrapper (`src/uav_arena/lowlevel.py`) only sets
the scheduler rate and adds the batched interface, the bounded rotor mixer and the cable-load
feedforward.

## Runtime dependencies (not vendored)

| Package | License |
|---|---|
| MuJoCo | Apache-2.0 |
| NumPy, SciPy | BSD-3-Clause |
| Gymnasium (`gym` extra) | MIT |
| PyTorch (`plan` extra) | BSD-3-Clause |
| stable-worldmodel 0.1.1 (`plan` extra) | MIT |
| OpenCV (`video` extra) | Apache-2.0 |
