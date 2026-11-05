"""Render the camera presets mid-flight and save them as one labelled image.

    MUJOCO_GL=egl python examples/cameras.py --out runs/cameras.png

Needs an image writer: imageio (installed with the `plan` extra) or OpenCV (`video` extra).
"""
import argparse
from pathlib import Path

import numpy as np

from uav_arena import CONDITIONS, ArenaEnv, make_scenario
from uav_arena.cameras import chase_cameras, downward_cameras, onboard_cameras, overview_camera, payload_camera


def save_png(path, image):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio
        iio.imwrite(path, image)
    except ImportError:
        import cv2
        cv2.imwrite(str(path), image[:, :, ::-1])


def label(image, text):
    try:
        import cv2
    except ImportError:
        return image
    image = np.ascontiguousarray(image)
    cv2.putText(image, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("runs/cameras.png"))
    parser.add_argument("--size", type=int, default=192, help="tile size (pixels)")
    args = parser.parse_args()
    size = dict(width=args.size, height=args.size)
    tiles = (onboard_cameras(**size)[0], downward_cameras(**size)[0], chase_cameras(**size)[0], payload_camera(**size))
    overview = overview_camera(width=4*args.size, height=3*args.size)
    env = ArenaEnv(make_scenario(0, CONDITIONS[5]), cameras=(*tiles, overview))
    try:
        obs = env.reset()
        for _ in range(15):          # fly toward the goal for 1.5 s
            obs, *_ = env.step(obs["recent_setpoints"][-1]+np.array([0.08, 0.0, 0.0]))
        images = obs["images"]
        row = np.concatenate([label(images[c.name], c.name) for c in tiles], axis=1)
        sheet = np.concatenate([label(images["overview"], "overview"), row], axis=0)
        save_png(args.out, sheet)
        print(f"{len(images)} cameras at t = {obs['time']:.1f} s -> {args.out}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
