"""Save representative images for each detection step.

Example: python explain_detection.py video1.mp4 8
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from detect_circular_signs import circular_candidates


def read_frame_at_second(path, second):
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    target = round(second * fps)
    frame = None
    for _ in range(target + 1):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read {path} at {second:g}s")
    capture.release()
    return frame


def make_steps(path, second, output_dir):
    frame = read_frame_at_second(path, second)
    name = f"{path.stem}_{second:g}s"
    output_dir.mkdir(parents=True, exist_ok=True)
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(cv2.inRange(hsv, (0, 85, 65), (11, 255, 255)),
                         cv2.inRange(hsv, (168, 85, 65), (179, 255, 255)))
    blue = cv2.inRange(hsv, (94, 75, 50), (132, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_clean = cv2.morphologyEx(cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel),
                                 cv2.MORPH_OPEN, kernel)
    blue_clean = cv2.morphologyEx(cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel),
                                  cv2.MORPH_OPEN, kernel)
    mask_display = np.zeros_like(frame)
    mask_display[:, :, 2] = red
    mask_display[:, :, 0] = blue
    clean_display = np.zeros_like(frame)
    clean_display[:, :, 2] = red_clean
    clean_display[:, :, 0] = blue_clean
    overlay = frame.copy()
    overlay[red_clean > 0] = (0, 0, 255)
    overlay[blue_clean > 0] = (255, 0, 0)
    overlay = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
    final = frame.copy()
    for x1, y1, x2, y2 in circular_candidates(frame):
        cv2.rectangle(final, (x1, y1), (x2, y2), (0, 255, 0), 3)
    stages = [
        ("01_original", frame),
        ("02_hsv_color_masks", mask_display),
        ("03_clean_masks", clean_display),
        ("04_masks_on_frame", overlay),
        ("05_candidate_boxes", final),
    ]
    for stage, image in stages:
        destination = output_dir / f"{name}_{stage}.png"
        cv2.imwrite(str(destination), image)
        print(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("second", type=float)
    parser.add_argument("--output-dir", type=Path, default=Path("output/steps"))
    args = parser.parse_args()
    make_steps(args.video, args.second, args.output_dir)


if __name__ == "__main__":
    main()
