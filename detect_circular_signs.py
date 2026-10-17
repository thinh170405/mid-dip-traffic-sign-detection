
import math
from pathlib import Path

import cv2
import numpy as np



STUDENT_ID = "TEST"              
MAX_SECONDS = None                   
INPUT_VIDEOS = ("video1.mp4", "video2.mp4")
OUTPUT_FOLDER = "output"
MIN_SIGN_SCORE = 0.75
MIN_CENTER_Y_RATIO = 0.18  # Ignore circles in the top 18% of the frame.
MIN_CIRCLE_RADIUS_RATIO = 0.0045  # About 9 px at 1920 px width.


def intersection_over_union(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    area = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return area / max(1, aa + bb - area)


def circular_candidates(frame, return_scores=False):
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    colors = {
        "red": cv2.bitwise_or(
            cv2.inRange(hsv, (0, 85, 65), (11, 255, 255)),
            cv2.inRange(hsv, (168, 85, 65), (179, 255, 255)),
        ),
        "blue": cv2.inRange(hsv, (94, 75, 50), (132, 255, 255)),
        "blue_strong": cv2.inRange(hsv, (94, 145, 50), (132, 255, 255)),
        "red_faded": cv2.inRange(hsv, (125, 20, 55), (179, 180, 220)),
    }
    broad_red = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 70, 50), (15, 255, 255)),
        cv2.inRange(hsv, (145, 70, 50), (179, 255, 255)),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    candidates = []

    def color_evidence(mask, color, cx, cy, radius):
        x0 = max(0, round(cx - radius))
        y0 = max(0, round(cy - radius))
        x1 = min(width, round(cx + radius + 1))
        y1 = min(height, round(cy + radius + 1))
        local = mask[y0:y1, x0:x1]
        ring = np.zeros(local.shape, np.uint8)
        cv2.circle(ring, (round(cx) - x0, round(cy) - y0),
                   max(1, round(radius * 0.92)), 255, -1)
        cv2.circle(ring, (round(cx) - x0, round(cy) - y0),
                   max(1, round(radius * 0.55)), 0, -1)
        coverage = cv2.countNonZero(cv2.bitwise_and(local, ring)) / max(1, cv2.countNonZero(ring))
        if coverage < (0.25 if color.startswith("red") else 0.32):
            return None
        x0 = max(0, round(cx - radius * 0.52))
        y0 = max(0, round(cy - radius * 0.52))
        x1 = min(width, round(cx + radius * 0.52))
        y1 = min(height, round(cy + radius * 0.52))
        center_hsv = hsv[y0:y1, x0:x1]
        if color.startswith("blue") and np.median(center_hsv[:, :, 1]) < 125:
            return None
        pale_fraction = np.mean((center_hsv[:, :, 1] < (100 if color.startswith("red") else 140)) &
                                (center_hsv[:, :, 2] > (115 if color.startswith("red") else 100)))
        if color.startswith("red") and pale_fraction < (0.65 if color == "red_faded" else 0.08):
            return None
        if color == "red_faded" and np.median(center_hsv[:, :, 1]) > 65:
            return None
        if color == "red_faded":
            if radius < 15 or cx / width < 0.20:
                return None
            sx0, sy0 = max(0, round(cx - radius)), max(0, round(cy - radius))
            sx1, sy1 = min(width, round(cx + radius + 1)), min(height, round(cy + radius + 1))
            ring_saturation = hsv[sy0:sy1, sx0:sx1, 1]
            yy, xx = np.ogrid[sy0:sy1, sx0:sx1]
            ring_distance = ((xx - cx) ** 2 + (yy - cy) ** 2) / max(1, radius * radius)
            color_ring = (ring_distance >= 0.60 ** 2) & (ring_distance <= 1.0)
            if np.median(ring_saturation[color_ring]) > 80:
                return None
        red_fraction = 0.0
        if color.startswith("blue") and pale_fraction < (0.02 if color == "blue_strong" else 0.035):
            margin = round(radius * 1.35)
            bx0, by0 = max(0, round(cx) - margin), max(0, round(cy) - margin)
            bx1, by1 = min(width, round(cx) + margin + 1), min(height, round(cy) + margin + 1)
            red_local = broad_red[by0:by1, bx0:bx1]
            yy, xx = np.ogrid[by0:by1, bx0:bx1]
            ring_distance = ((xx - cx) ** 2 + (yy - cy) ** 2) / max(1, radius * radius)
            outer_ring = (ring_distance >= 0.9 ** 2) & (ring_distance <= 1.35 ** 2)
            red_fraction = np.mean(red_local[outer_ring] > 0)
            if red_fraction < 0.16:
                return None
        # A colored patch on a billboard has color outside the alleged circle too.
        margin = round(radius * 1.45)
        bx0, by0 = max(0, round(cx) - margin), max(0, round(cy) - margin)
        bx1, by1 = min(width, round(cx) + margin + 1), min(height, round(cy) + margin + 1)
        nearby = mask[by0:by1, bx0:bx1]
        yy, xx = np.ogrid[by0:by1, bx0:bx1]
        distance = ((xx - cx) ** 2 + (yy - cy) ** 2) / max(1, radius * radius)
        outside = (distance >= 1.12 ** 2) & (distance <= 1.45 ** 2)
        outside_fraction = float(np.mean(nearby[outside] > 0)) if np.any(outside) else 0.0
        return coverage, pale_fraction, red_fraction, outside_fraction

    def add_circle(mask, color, cx, cy, radius, shape_score, recovered=False):
        if radius < max(8, width * MIN_CIRCLE_RADIUS_RATIO) or radius > width * 0.18:
            return
        if cy < height * MIN_CENTER_Y_RATIO:
            return
        if cy > height * 0.48:
            return
        if cx < width * 0.12:
            return
        evidence = color_evidence(mask, color, cx, cy, radius)
        if evidence is None:
            return
        coverage, pale_fraction, red_fraction, outside_fraction = evidence
        if recovered and coverage + pale_fraction < 0.88:
            return
        if recovered:
            inner, outer = [], []
            for angle in np.linspace(0, 2 * math.pi, 36, endpoint=False):
                direction_x, direction_y = math.cos(angle), math.sin(angle)
                inner_samples = []
                outer_samples = []
                for factor in (0.72, 0.82, 0.92):
                    px = min(width - 1, max(0, round(cx + radius * factor * direction_x)))
                    py = min(height - 1, max(0, round(cy + radius * factor * direction_y)))
                    inner_samples.append(mask[py, px] > 0)
                for factor in (1.08, 1.18, 1.28):
                    px = min(width - 1, max(0, round(cx + radius * factor * direction_x)))
                    py = min(height - 1, max(0, round(cy + radius * factor * direction_y)))
                    outer_samples.append(mask[py, px] > 0)
                inner.append(np.mean(inner_samples))
                outer.append(np.mean(outer_samples))
            if np.mean(inner) < 0.60 or np.mean(inner) - np.mean(outer) < 0.22:
                return
        shape = min(1.0, shape_score / 1.7)
        rim = min(1.0, coverage / (0.70 if color.startswith("red") else 0.75))
        contrast = min(1.0, max(0.0, coverage - outside_fraction) / 0.55)
        symbol = min(1.0, pale_fraction / (0.23 if color.startswith("red") else 0.12))
        if color.startswith("blue"):
            symbol = max(symbol, min(1.0, red_fraction / 0.25))
        center_x = cx / width
        position = min(1.0, max(0.0, (center_x - 0.12) / 0.13),
                       max(0.0, (0.88 - center_x) / 0.13))
        score = 0.15 * shape + 0.20 * rim + 0.20 * contrast + 0.20 * symbol + 0.25 * position
        if score < MIN_SIGN_SCORE:
            return
        pad = max(2, round(radius * 0.12))
        box = (max(0, round(cx - radius - pad)),
               max(0, round(cy - radius - pad)),
               min(width - 1, round(cx + radius + pad)),
               min(height - 1, round(cy + radius + pad)))
        candidates.append((box, score))

    for color, mask in colors.items():
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if area < max(18, width * height * 0.000008) or perimeter <= 0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            ratio = w / h
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            fill = area / max(1.0, math.pi * radius * radius)
            if 0.68 <= ratio <= 1.45 and circularity >= (0.25 if color.startswith("red") else 0.52) and fill >= 0.45:
                add_circle(mask, color, cx, cy, radius, circularity + fill)
            elif (color.startswith("blue") and 400 <= area <= 30000
                  and 24 <= w <= 250 and 24 <= h <= 300
                  and 0.50 <= ratio <= 1.7 and cy < height * 0.50):
                gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
                circles = cv2.HoughCircles(
                    gray, cv2.HOUGH_GRADIENT, 1.2, 30,
                    param1=120, param2=30,
                    minRadius=10, maxRadius=min(w, h) // 2,
                )
                if circles is not None:
                    for local_x, local_y, local_radius in circles[0]:
                        add_circle(mask, color, x + local_x, y + local_y,
                                   local_radius, 0.9, recovered=True)

    # A distant no-entry sign often splits into two red lobes around its white bar.
    red_lobes = cv2.inRange(hsv, (145, 80, 55), (179, 255, 255))
    red_lobes = cv2.morphologyEx(red_lobes, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(red_lobes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lobes = [cv2.boundingRect(contour) for contour in contours
             if cv2.contourArea(contour) >= 35]
    for x, y, w, h in lobes:
        if not (7 <= w <= 80 and 4 <= h <= 35):
            continue
        for bx, by, bw, bh in lobes:
            if by <= y or not (7 <= bw <= 80 and 4 <= bh <= 35):
                continue
            if abs((x + w / 2) - (bx + bw / 2)) > 0.25 * max(w, bw):
                continue
            if not (0 <= by - (y + h) <= 0.55 * max(w, bw)):
                continue
            combined_width = max(x + w, bx + bw) - min(x, bx)
            combined_height = by + bh - y
            if not (0.85 <= combined_height / combined_width <= 1.65):
                continue
            cx = (x + w / 2 + bx + bw / 2) / 2
            cy = (y + by + bh) / 2
            radius = max(combined_width, combined_height) / 2
            if cy < height * MIN_CENTER_Y_RATIO or cy > height * 0.48 or cx < width * 0.12:
                continue
            strip = hsv[y + h:by, max(x, bx):min(x + w, bx + bw)]
            if strip.size == 0 or np.mean((strip[:, :, 1] < 120) &
                                          (strip[:, :, 2] > 110)) < 0.4:
                continue
            add_circle(red_lobes, "red", cx, cy, radius, 1.0)

    candidates.sort(key=lambda item: item[1], reverse=True)
    selected = []
    for candidate in candidates:
        if all(intersection_over_union(candidate[0], other[0]) < 0.35
               for other in selected):
            selected.append(candidate)
    if return_scores:
        return selected
    return [box for box, _ in selected]


class SignTracker:
    def __init__(self, fps):
        self.tracks = []
        self.min_hits = max(2, min(4, round(fps * 0.1)))
        self.display_misses = max(4, round(fps * 0.3))
        self.max_misses = self.display_misses

    def update(self, detections):
        matches = []
        for ti, track in enumerate(self.tracks):
            old = track["box"]
            ox, oy = (old[0] + old[2]) / 2, (old[1] + old[3]) / 2
            osize = max(old[2] - old[0], old[3] - old[1])
            for di, box in enumerate(detections):
                nx, ny = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                nsize = max(box[2] - box[0], box[3] - box[1])
                distance = math.hypot(nx - ox, ny - oy)
                size_ratio = max(osize, nsize) / max(1, min(osize, nsize))
                if distance < max(12, 0.75 * max(osize, nsize)) and size_ratio < 1.9:
                    matches.append((distance / max(osize, nsize), ti, di))
        used_tracks, used_detections = set(), set()
        for _, ti, di in sorted(matches):
            if ti in used_tracks or di in used_detections:
                continue
            track = self.tracks[ti]
            track["box"] = tuple(0.70 * a + 0.30 * b
                                 for a, b in zip(track["box"], detections[di]))
            track["hits"] = min(self.min_hits, track["hits"] + 1)
            track["misses"] = 0
            used_tracks.add(ti)
            used_detections.add(di)
        for ti, track in enumerate(self.tracks):
            if ti not in used_tracks:
                track["misses"] += 1
                if track["hits"] < self.min_hits:
                    track["hits"] = 0
        for di, box in enumerate(detections):
            if di not in used_detections:
                self.tracks.append({"box": tuple(map(float, box)), "hits": 1, "misses": 0})
        self.tracks = [track for track in self.tracks
                       if track["misses"] <= self.max_misses]
        return [tuple(map(round, track["box"])) for track in self.tracks
                if track["hits"] >= self.min_hits and track["misses"] <= self.display_misses]


def process_video(source, destination, student_id, max_seconds=None):
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open input video: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    expected_frames = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata: {source}")
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height), True)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {destination}")
    tracker = SignTracker(fps)
    count = 0
    while True:
        if max_seconds is not None and count >= round(max_seconds * fps):
            break
        ok, frame = capture.read()
        if not ok:
            break
        boxes = tracker.update(circular_candidates(frame))
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, student_id, (18, max(35, height - 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, student_id, (18, max(35, height - 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        count += 1
    writer.release()
    capture.release()
    if max_seconds is None and expected_frames > 0 and count < expected_frames * 0.98:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"Input decoding stopped early at {count}/{expected_frames} frames: {source}. "
            "Replace the damaged input file before making a final video."
        )
    print(f"{source} -> {destination} ({count} frames, {fps:g} fps, {width}x{height})")


def main():
    folder = Path(__file__).resolve().parent
    output_dir = folder / OUTPUT_FOLDER
    output_dir.mkdir(parents=True, exist_ok=True)
    for video_name in INPUT_VIDEOS:
        source = folder / video_name
        if not source.is_file():
            raise FileNotFoundError(f"Input video not found: {source}")
        suffix = (f"_preview_{MAX_SECONDS:g}s" if MAX_SECONDS is not None else "")
        destination = output_dir / f"{source.stem}_{STUDENT_ID}{suffix}.mp4"
        process_video(source, destination, STUDENT_ID, MAX_SECONDS)


if __name__ == "__main__":
    main()
