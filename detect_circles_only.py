"""Geometry-only circular-object detector for inspecting the first video seconds.

This diagnostic intentionally ignores sign color and temporal tracking.
Example: python detect_circles_only.py --seconds 20 video1.mp4 video2.mp4
"""

import argparse
from pathlib import Path

import cv2


def detect_circles(frame, scale=0.5):
    height, width = frame.shape[:2]
    small = cv2.resize(frame, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1.3,
        minDist=max(20, round(width * scale * 0.026)),
        param1=120,
        param2=40,
        minRadius=max(8, round(width * scale * 0.008)),
        maxRadius=round(width * scale * 0.042),
    )
    if circles is None:
        return []

    result = []
    for cx, cy, radius in circles[0]:
        x, y, r = round(cx / scale), round(cy / scale), round(radius / scale)
        if y > height * 0.42:
            continue
        result.append((x, y, r))
    return result


def process(source, destination, seconds):
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {source}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    limit = round(seconds * fps)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height), True)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create {destination}")
    count = 0
    total_circles = 0
    while count < limit:
        ok, frame = capture.read()
        if not ok:
            writer.release()
            capture.release()
            destination.unlink(missing_ok=True)
            raise RuntimeError(f"Input stopped before {seconds:g}s: {source}")
        circles = detect_circles(frame)
        for x, y, r in circles:
            pad = max(2, round(r * 0.08))
            cv2.rectangle(frame, (max(0, x - r - pad), max(0, y - r - pad)),
                          (min(width - 1, x + r + pad),
                           min(height - 1, y + r + pad)), (0, 255, 0), 2)
        total_circles += len(circles)
        writer.write(frame)
        count += 1
    writer.release()
    capture.release()
    print(f"{source} -> {destination}: {count} frames, {fps:.3f} fps, "
          f"{width}x{height}; {total_circles} circle boxes across frames")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("videos", nargs="+", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source in args.videos:
        destination = args.output_dir / f"{source.stem}_circles_only_{args.seconds:g}s.mp4"
        process(source, destination, args.seconds)


if __name__ == "__main__":
    main()
