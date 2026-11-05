# Pinned upstream Lee position and attitude controller

The seven source and license files listed in [UPSTREAM.json](UPSTREAM.json) are copied byte for byte
from [IMRCLab/crazyflie-firmware](https://github.com/IMRCLab/crazyflie-firmware/tree/b08d9f3cbcc581d8e7f988cabad22c26d35f4a58),
commit `b08d9f3cbcc581d8e7f988cabad22c26d35f4a58`. This is the firmware revision pinned by
[coltrans-planning](https://github.com/IMRCLab/coltrans-planning/tree/146b205e33e20ac26975993d32ec89b140754550).
For each file, UPSTREAM.json records its original path, Git blob, SHA-256 and file-level license
notice. `uav_arena.lowlevel` checks every hash before it builds the controller.

`controller_lee.c` implements the geometric position and attitude controller of Lee, Leok and
McClamroch (CDC 2010). `uav_arena/lowlevel.py` compiles it unchanged together with a small wrapper,
which:
- sets the controller's scheduler to the arena's 500 Hz physics rate;
- runs one complete `controllerLee_t` instance per aircraft, including its position and attitude
  integrals;
- adds the bounded rotor mixer and the nominal cable-load feedforward.

All published gains are retained.

## License notices

`controller_lee.c` and `math3d.h` carry MIT notices. The firmware interface headers carry GPL-3.0
notices, and the firmware's [LICENSE.txt](LICENSE.txt) is retained. All copyright and license text
is intact. This bundle is not distributed as MIT-only code; UAV Arena as a whole is GPL-3.0-only.
