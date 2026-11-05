# Roadmap

The release keeps the team at three aircraft and opens the task through payload, wind, camera,
obstacle and episode settings. These are the directions planned next.

## Variable team size: a scalable multi-agent benchmark

**The goal.** N aircraft carry one payload, with N from 2 to 6 or more. With N as a knob, the arena
becomes a scaling benchmark for multi-agent world models:
- how prediction error, counterfactual accuracy (layer 2) and decision quality (layer 3) change as
  more agents couple through one payload;
- whether permutation-invariant or equivariant models generalize to team sizes they were not
  trained on;
- how centralized and decentralized decision makers compare when each aircraft sees the team with
  its own delay;
- heterogeneous teams, where aircraft differ in thrust, cable length or latency.

**The design.**
1. **Plant.** The formation becomes a regular N-gon, with one cable and one onboard controller per
   aircraft. The rotor-level controller is already batched over any count. The onboard cable-load
   feedforward divides the nominal load by N; at N = 3 it is today's.
2. **Physics.** N ≥ 3 holds the payload from all sides. N = 2 leaves it free to swing about the line
   through the two aircraft, an unactuated mode that makes the task harder in a qualitatively
   different way. The team's radius, and with it the corridor the scenario generator keeps free,
   grows with N. Feasibility checks cover thrust headroom and cable angles.
3. **Interfaces.** Actions become (N, 3) setpoints. Observations become aircraft-major blocks of a
   declared width, so models can share weights across aircraft. Costs run over all N cables and
   all aircraft pairs, and the layer-2 counterfactuals are defined per aircraft.
4. **Reference rows and data.** The oracle, physics and adaptive rows carry over directly. Each team
   size gets its own standard datasets and manifests.
5. **Compatibility.** N = 3 stays arena-v0.3 bit for bit, checked by the same parity tests.

**Milestones.**
1. An N-generic plant and controller, with N = 3 parity.
2. Scenario generation and feasibility for N from 2 to 6.
3. N-generic costs, Gymnasium spaces and harness.
4. Datasets and reference rows for N = 4 and 6.
5. A scaling protocol across team sizes.

## Wind and aerodynamics

- **Spatially varying wind:** shear layers, gridded wind fields, and wakes behind obstacles, applied
  as per-body drag forces instead of MuJoCo's global air velocity.
- **Rotor drag:** linear in airspeed, which dominates for small quadrotors at low speed.
- **Cable drag.**
- **Downwash:** between aircraft, and onto the payload.

## Sensing

- Sensor noise and IMU-level observations.
- Per-aircraft observation delays and dropouts.
- Event-camera and depth-noise models for the virtual cameras.

## Tasks

- **Payloads:** rigid bars and boxes with several attachment points, and payloads that change mass
  in flight (pick-up and drop).
- **Airframes:** other airframes from MuJoCo Menagerie.

## Tooling

- A PyPI release.
- Prebuilt controller libraries, so no C compiler is needed.
- Windows support (the build lock uses `fcntl` today).
