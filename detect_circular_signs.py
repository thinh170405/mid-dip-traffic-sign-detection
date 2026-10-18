"""Circular traffic-sign detection using OpenCV, NumPy and temporal tracking."""

import math

import cv2
import numpy as np


# Input/output: run this file directly. The output folder must already exist.
STUDENT_ID = "TEST"
INPUT_VIDEOS = ("video1.mp4", "video2.mp4")
OUTPUT_FOLDER = "output"          # "" saves beside this Python file
MAX_SECONDS = None                # None: full videos; e.g. 20: first 20 seconds
OPENCV_THREADS = 2

# One configuration for every video. Ratios refer to the ORIGINAL frame width.
CENTER_ROI = (0.12, 0.18, 1.00, 0.48)  # left, top, right, bottom; approximate road ROI
CONTEXT_MARGIN_RATIO = 0.10        # retain complete rims around the center ROI
RADIUS_RANGE = (0.0042, 0.10)      # smaller minimum sees farther, admits more noise
MIN_AXIS_RATIO = 0.60              # minor/major ellipse axis; smaller permits more tilt
MAX_ELLIPSE_ERROR = 0.12           # normalized boundary error; larger is less selective
MIN_BOUNDARY_SUPPORT = 0.60       # real contour near ellipse; allow inward symbol notches
MIN_LIGHT_NOTCH_RATIO = 0.70     # recovered inward gaps must mostly be light symbols
MIN_LIGHT_NOTCH_AREA_RATIO = 0.03  # geometry recovery must not rely on a tiny aliasing gap
RECTANGLE_FILL_LIMIT = 0.87
MIN_ANGULAR_SUPPORT = 0.58         # fraction of 12 rim sectors with color
MIN_RADIAL_SUPPORT = 0.65          # colored rim must end near ellipse in at least 8/12 sectors
MIN_PATTERN_SCORE = 0.28          # weak safeguard against solid patches and flags
MIN_SIGN_SCORE = 0.78
SMALL_SIGN_SCORE_BONUS = 0.03      # smoothly raise the threshold near minimum radius
SCORE_WEIGHTS = (0.40, 0.25, 0.25, 0.10)  # shape, angular support, pattern, isolation
NMS_IOU = 0.35
NMS_CONTAINMENT = 0.85             # remove small boxes mostly inside a larger valid box
BOX_PADDING_RATIO = 0.08

# HSV uses H=0..179, S/V=0..255. These are common color ranges, not sign identities.
RED_HUE_RANGES = ((0, 15), (140, 179))
BLUE_HUE_RANGE = (94, 132)
COLOR_MIN_SATURATION = (35, 65)    # red, blue; lower values admit faded colors + noise
STRONG_SATURATION_FACTOR = 1.8     # extra proposals when weak colors bridge to background
COLOR_MIN_VALUE = 45
ENABLE_LIGHT_NORMALIZATION = True
CLAHE_CLIP_LIMIT = 2.0
LIGHT_NORMALIZATION_BLEND = 0.40
MORPH_WIDTH_RATIO = 0.0015        # kernel scales with width; retain un-opened rims too

# Shared appearance definitions, measured on a normalized ellipse patch.
PATCH_SIZE = 64
WHITE_MAX_SATURATION = 75
SYMBOL_SATURATION_RATIO = 0.75    # light symbol must be less saturated than blue ground
PATTERN_TARGETS = (0.55, 0.03, 0.16, 0.16)  # white ground, dark ink, white symbol, red rim

# Tracker policy is in seconds, so it scales with source FPS.
TRACK_CONFIRM_SECONDS = 4 / 30
TRACK_HISTORY_SECONDS = 6 / 30
TRACK_HOLD_SECONDS = 0.25          # prediction after last verified image observation
TRACK_RETAIN_SECONDS = 0.80        # delete after this long without detector OR valid flow
TRACK_MATCH_DISTANCE_RATIO = 0.75
TRACK_MATCH_SIZE_RATIO = 1.9
TRACK_OBSERVATION_WEIGHT = 0.65
FLOW_MIN_SIMILARITY = 0.70


# Sampling geometry defines the features, rather than extra per-video filters.
_axis = np.linspace(-1.4, 1.4, PATCH_SIZE, dtype=np.float32)
_GRID_X, _GRID_Y = np.meshgrid(_axis, _axis)
_DISTANCE = _GRID_X**2 + _GRID_Y**2
_INNER = _DISTANCE <= 0.65**2
_RING = (_DISTANCE >= 0.70**2) & (_DISTANCE <= 1.05**2)
_BORDER = (_DISTANCE >= 0.90**2) & (_DISTANCE <= 1.40**2)
_OUTSIDE = (_DISTANCE >= 1.12**2) & (_DISTANCE <= 1.40**2)
_ANGLES = np.arange(72) * (2 * math.pi / 72)
_RIM_X = np.cos(_ANGLES)[None, :] * np.array([0.72, 0.84, 0.96, 1.04])[:, None]
_RIM_Y = np.sin(_ANGLES)[None, :] * np.array([0.72, 0.84, 0.96, 1.04])[:, None]
_RIM_COLS = np.rint((_RIM_X + 1.4) / 2.8 * (PATCH_SIZE - 1)).astype(int)
_RIM_ROWS = np.rint((_RIM_Y + 1.4) / 2.8 * (PATCH_SIZE - 1)).astype(int)
_OUTER_X = np.cos(_ANGLES)[None, :] * np.array([1.12, 1.24, 1.36])[:, None]
_OUTER_Y = np.sin(_ANGLES)[None, :] * np.array([1.12, 1.24, 1.36])[:, None]
_OUTER_COLS = np.rint((_OUTER_X + 1.4) / 2.8 * (PATCH_SIZE - 1)).astype(int)
_OUTER_ROWS = np.rint((_OUTER_Y + 1.4) / 2.8 * (PATCH_SIZE - 1)).astype(int)


def clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def count_event(diagnostics, name, amount=1):
    if diagnostics is not None:
        diagnostics[name] = diagnostics.get(name, 0) + amount


def color_masks(hsv):
    """Use the same two color definitions in proposals, scoring and tracking."""
    h, s, v = cv2.split(hsv)
    red_hue = np.zeros(h.shape, bool)
    for low, high in RED_HUE_RANGES:
        red_hue |= (h >= low) & (h <= high)
    red = red_hue & (s >= COLOR_MIN_SATURATION[0]) & (v >= COLOR_MIN_VALUE)
    blue = ((h >= BLUE_HUE_RANGE[0]) & (h <= BLUE_HUE_RANGE[1])
            & (s >= COLOR_MIN_SATURATION[1]) & (v >= COLOR_MIN_VALUE))
    return red.astype(np.uint8) * 255, blue.astype(np.uint8) * 255


def ellipse_patch(image, ellipse):
    """Map a tilted ellipse + nearby context onto one circular coordinate system."""
    (cx, cy), (diameter_x, diameter_y), angle = ellipse
    co, si = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    local_x = _GRID_X * (diameter_x / 2)
    local_y = _GRID_Y * (diameter_y / 2)
    map_x = (cx + co * local_x - si * local_y).astype(np.float32)
    map_y = (cy + si * local_x + co * local_y).astype(np.float32)
    return cv2.remap(image, map_x, map_y, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT)


def split_color_components(mask, saturation, minimum_area):
    """Split saturated sign faces from weaker, touching background using local Otsu."""
    result = np.zeros_like(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < minimum_area:
            continue
        component = labels[y:y+h, x:x+w] == label
        local_s = saturation[y:y+h, x:x+w]
        threshold, _ = cv2.threshold(local_s[component].reshape(-1, 1),
                                     0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result[y:y+h, x:x+w][component & (local_s > threshold)] = 255
    return result


def light_masks(patch, blue):
    """Shared light-symbol evidence, including blue-tinted compressed arrows."""
    s, v = patch[:, :, 1], patch[:, :, 2]
    upper_value = float(np.percentile(v[_INNER], 90))
    red, _ = color_masks(patch)
    # On a red-rimmed sign, the pale center can itself have a reddish tint.
    # Estimate the red reference from the rim, not from that white center.
    red_s = s[_RING & (red > 0)]
    neutral_s = min(WHITE_MAX_SATURATION, SYMBOL_SATURATION_RATIO * float(np.percentile(red_s, 90))) if red_s.size else WHITE_MAX_SATURATION
    light = (s < neutral_s) & (v >= max(COLOR_MIN_VALUE, 0.60 * upper_value))
    # Compression may tint white arrows blue. Compare with the actual blue ground;
    # an absolute S<130 alone also counts an entire pale-blue patch as its symbol.
    ground_s = s[_INNER & blue]
    symbol_s = SYMBOL_SATURATION_RATIO * float(np.median(ground_s)) if ground_s.size else 0
    blue_light = (s < symbol_s) & (v >= max(COLOR_MIN_VALUE, 0.60 * upper_value))
    return light, blue_light, upper_value


def boundary_features(patch, mask):
    """Measure a color boundary and isolation using the same coherent rim color."""
    saturation = patch[:, :, 1]
    rim_s = saturation[_RING & mask]
    if rim_s.size == 0:
        return 0.0, 0.0
    coherent = mask & (saturation >= SYMBOL_SATURATION_RATIO * np.median(rim_s))
    inside = coherent[_RIM_ROWS, _RIM_COLS].reshape(4, 12, 6).mean(axis=(0, 2))
    outside = coherent[_OUTER_ROWS, _OUTER_COLS].reshape(3, 12, 6).mean(axis=(0, 2))
    support = float(np.mean((inside >= .20) & (inside-outside >= .20)))
    isolation = 1-clip01(coherent[_OUTSIDE].mean()/.40)
    return support, isolation


def bar_evidence(light, red):
    """Measure a light bar along its principal direction, independent of rotation."""
    points = np.column_stack((_GRID_X[light & _INNER], _GRID_Y[light & _INNER]))
    if len(points) < 2:
        return 0.0
    covariance = np.cov(points, rowvar=False)
    angle = .5 * math.atan2(2*covariance[0, 1], covariance[0, 0]-covariance[1, 1])
    u = math.cos(angle)*_GRID_X + math.sin(angle)*_GRID_Y
    v = -math.sin(angle)*_GRID_X + math.cos(angle)*_GRID_Y
    stripe = (np.abs(u) <= .55) & (np.abs(v) <= .15)
    top = (np.abs(u) <= .55) & (v >= -.55) & (v <= -.25)
    bottom = (np.abs(u) <= .55) & (v >= .25) & (v <= .55)
    contrast = light[stripe].mean()-max(light[top].mean(), light[bottom].mean())
    return clip01(contrast/.70) * clip01(min(red[top].mean(), red[bottom].mean())/.65)


def symbol_evidence(light):
    """A coherent symbol is stronger evidence than the same area of scattered specks."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (light & _INNER).astype(np.uint8), connectivity=8)
    fraction = float(stats[1:, cv2.CC_STAT_AREA].max()) / np.count_nonzero(_INNER) if count > 1 else 0.0
    return clip01(fraction/PATTERN_TARGETS[2])**2


def appearance_features(hsv, ellipse):
    """Continuous evidence for red/light, red/blue and blue/light sign patterns."""
    patch = ellipse_patch(hsv, ellipse)
    red, blue = color_masks(patch)
    red, blue = red > 0, blue > 0
    light, blue_light, upper_value = light_masks(patch, blue)
    v = patch[:, :, 2]
    dark = v < 0.65 * upper_value
    # A white ground must occupy a substantial part of the face. Sparse lettering
    # on a helmet/advert is not equivalent to the ground of a prohibition sign.
    white_ground = clip01(2 * light[_INNER].mean() / PATTERN_TARGETS[0] - 1)
    dark_ink = clip01(dark[_INNER].mean() / PATTERN_TARGETS[1])
    white_bar = bar_evidence(light, red)
    blue_ground = clip01(blue[_INNER].mean() / 0.45)
    red_border = clip01(red[_BORDER].mean() / PATTERN_TARGETS[3])
    red_pattern = max(white_ground * dark_ink, white_bar, blue_ground * red_border)
    blue_s = patch[:, :, 1][_INNER & blue]
    blue_color_confidence = float(np.median(blue_s))/255 if blue_s.size else 0.0
    blue_pattern = max(symbol_evidence(blue_light)*blue_color_confidence, red_border)
    red_angles = red[_RIM_ROWS, _RIM_COLS].reshape(4, 12, 6).mean(axis=(0, 2))
    blue_angles = blue[_RIM_ROWS, _RIM_COLS].reshape(4, 12, 6).mean(axis=(0, 2))
    red_support = float(np.mean(red_angles >= 0.20))
    blue_support = float(np.mean(blue_angles >= 0.20))
    # Evaluate both colors with identical scoring; keep the stronger explanation.
    options = []
    for support, pattern, mask in ((red_support, red_pattern, red),
                                   (blue_support, blue_pattern*blue_ground, blue)):
        boundary, isolation = boundary_features(patch, mask)
        options.append((support, pattern if boundary >= MIN_RADIAL_SUPPORT else 0.0, isolation))
    eligible = [f for f in options if f[0] >= MIN_ANGULAR_SUPPORT and f[1] >= MIN_PATTERN_SCORE]
    return max(eligible or options, key=lambda f: SCORE_WEIGHTS[1]*f[0]
               + SCORE_WEIGHTS[2]*f[1] + SCORE_WEIGHTS[3]*f[2])


def contour_geometry(contour, hsv=None):
    """Reject rectangular hulls, fit an ellipse, and check its actual contour."""
    hull = cv2.convexHull(contour)
    if len(hull) < 5:
        return None
    area = cv2.contourArea(hull)
    perimeter = cv2.arcLength(hull, True)
    if area <= 0 or perimeter <= 0:
        return None
    polygon = cv2.approxPolyDP(hull, 0.025 * perimeter, True)
    rw, rh = cv2.minAreaRect(hull)[1]
    if len(polygon) <= 4 and area / max(1.0, rw * rh) > RECTANGLE_FILL_LIMIT:
        return None
    ellipse = cv2.fitEllipse(contour)
    (cx, cy), (a, b), angle = ellipse
    if min(a, b) <= 0 or min(a, b) / max(a, b) < MIN_AXIS_RATIO:
        return None
    def residuals(fitted):
        (cx, cy), (a, b), angle = fitted
        points = contour[:, 0, :].astype(float) - (cx, cy)
        co, si = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        u = (points[:, 0]*co + points[:, 1]*si) / (a/2)
        v = (-points[:, 0]*si + points[:, 1]*co) / (b/2)
        return np.hypot(u, v)-1

    error = float(np.quantile(np.abs(residuals(ellipse)), 0.9))
    if not np.isfinite(error):
        return None
    if error > MAX_ELLIPSE_ERROR:
        # Recover a rim cut by a light symbol, not arbitrary ragged boundaries.
        if hsv is None:
            return None
        ellipse = cv2.fitEllipse(hull)
        (cx, cy), (a, b), angle = ellipse
        if min(a, b) <= 0 or min(a, b)/max(a, b) < MIN_AXIS_RATIO:
            return None
        residual = residuals(ellipse)
        error = float(np.quantile(np.abs(residual), MIN_BOUNDARY_SUPPORT))
        if (not np.isfinite(error) or error > MAX_ELLIPSE_ERROR
                or np.quantile(np.maximum(residual, 0), .9) > MAX_ELLIPSE_ERROR):
            return None
        x, y, w, h = cv2.boundingRect(contour)
        filled = np.zeros((h, w), np.uint8)
        cv2.drawContours(filled, [contour-(x, y)], -1, 255, cv2.FILLED)
        local_ellipse = ((cx-x, cy-y), (a, b), angle)
        missing = _INNER & (ellipse_patch(filled, local_ellipse) == 0)
        patch = ellipse_patch(hsv, ellipse)
        _, blue = color_masks(patch)
        light, blue_light, _ = light_masks(patch, blue > 0)
        if (missing[_INNER].mean() < MIN_LIGHT_NOTCH_AREA_RATIO
                or (light | blue_light)[missing].mean() < MIN_LIGHT_NOTCH_RATIO):
            return None
    # Hull circularity is less sensitive to aliasing on tiny rims; the raw contour
    # still must pass the ellipse-error guard above.
    circularity = 4 * math.pi * area / perimeter**2
    # A broken rim may cover little area yet still fit an ellipse. Its full rim
    # and interior must be supported by the image in appearance_features().
    shape = 0.50*clip01(circularity/0.90) + 0.50*clip01(1-error)
    return ellipse, shape


def intersection_over_union(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    aa = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    bb = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return intersection / max(1.0, aa+bb-intersection)


def suppress_duplicates(candidates):
    """Remove contained fragments first, then suppress overlapping duplicate rims."""
    outer_first = sorted(candidates, key=lambda item:
                         (item[0][2]-item[0][0])*(item[0][3]-item[0][1]), reverse=True)
    unnested = []
    for box, score in outer_first:
        area = max(1, (box[2]-box[0])*(box[3]-box[1]))
        for other, _ in unnested:
            intersection = (max(0, min(box[2], other[2])-max(box[0], other[0]))
                            * max(0, min(box[3], other[3])-max(box[1], other[1])))
            if intersection/area >= NMS_CONTAINMENT:
                break
        else:
            unnested.append((box, score))
    selected = []
    for box, score in sorted(unnested, key=lambda item: item[1], reverse=True):
        for index, (other, other_score) in enumerate(selected):
            if intersection_over_union(box, other) < NMS_IOU:
                continue
            area = (box[2]-box[0]) * (box[3]-box[1])
            other_area = (other[2]-other[0]) * (other[3]-other[1])
            distance = math.hypot((box[0]+box[2]-other[0]-other[2])/2,
                                  (box[1]+box[3]-other[1]-other[3])/2)
            if (other_area < area <= 2*other_area and score >= other_score-0.10
                    and distance < 0.25 * math.sqrt(other_area)):
                selected[index] = (box, other_score)
            break
        else:
            selected.append((box, score))
    return selected


class SignDetector:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))

    def detect(self, frame, diagnostics=None):
        height, width = frame.shape[:2]
        left, top, right, bottom = CENTER_ROI
        margin = round(width * CONTEXT_MARGIN_RATIO)
        x0, y0 = max(0, round(left*width)-margin), max(0, round(top*height)-margin)
        x1, y1 = min(width, round(right*width)+margin), min(height, round(bottom*height)+margin)
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return []
        # Both variants use one validator: OPEN removes bridges, raw CLOSE keeps rims.
        blur_size = max(1, round(width/1920)*2+1)
        hsv = cv2.cvtColor(cv2.GaussianBlur(roi, (blur_size, blur_size), 0), cv2.COLOR_BGR2HSV)
        if ENABLE_LIGHT_NORMALIZATION:
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            light = lab[:, :, 0]
            enhanced = cv2.addWeighted(light, 1-LIGHT_NORMALIZATION_BLEND,
                                      self.clahe.apply(light), LIGHT_NORMALIZATION_BLEND, 0)
            lab[:, :, 0] = enhanced
            corrected = cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = np.maximum(hsv[:, :, 2], corrected[:, :, 2])
        red, blue = color_masks(hsv)
        kernel_size = max(1, round(width*MORPH_WIDTH_RATIO))
        kernel_size += 1-kernel_size % 2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        minimum_radius, maximum_radius = width*RADIUS_RANGE[0], width*RADIUS_RANGE[1]
        proposals = []
        # Fixed masks retain faded rims. Local Otsu splits saturated faces from
        # connected background; every proposal still uses the same validator.
        masks = []
        strong_red = np.where(hsv[:, :, 1] >= COLOR_MIN_SATURATION[0]*STRONG_SATURATION_FACTOR, red, 0).astype(np.uint8)
        strong_blue = np.where(hsv[:, :, 1] >= COLOR_MIN_SATURATION[1]*STRONG_SATURATION_FACTOR, blue, 0).astype(np.uint8)
        minimum_area = math.pi * minimum_radius**2 * 0.15
        split_red = split_color_components(strong_red, hsv[:, :, 1], minimum_area)
        split_blue = split_color_components(strong_blue, hsv[:, :, 1], minimum_area)
        for mask in (red, blue, strong_red, strong_blue, split_red, split_blue,
                     cv2.bitwise_or(red, blue)):
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            masks.extend((closed, cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)))
        for cleaned in masks:
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                if cv2.contourArea(contour) < math.pi * minimum_radius**2 * 0.15:
                    continue
                bx, by, bw, bh = cv2.boundingRect(contour)
                if (max(bw, bh) > 2.5*maximum_radius or max(bw, bh) < 1.5*minimum_radius
                        or min(bw, bh) < minimum_radius*MIN_AXIS_RATIO):
                    continue
                if not (left*width-bw*.1 <= bx+x0+bw/2 <= right*width+bw*.1
                        and top*height-bh*.1 <= by+y0+bh/2 <= bottom*height+bh*.1):
                    continue
                geometry = contour_geometry(contour, hsv)
                if geometry is None:
                    count_event(diagnostics, 'reject_geometry')
                    continue
                ellipse, shape = geometry
                (cx, cy), axes, angle = ellipse
                radius = max(axes)/2
                gx, gy = cx+x0, cy+y0
                if not (minimum_radius <= radius <= maximum_radius
                        and left*width <= gx <= right*width
                        and top*height <= gy <= bottom*height):
                    count_event(diagnostics, 'reject_size_position')
                    continue
                support, pattern, isolation = appearance_features(hsv, ellipse)
                if support < MIN_ANGULAR_SUPPORT or pattern < MIN_PATTERN_SCORE:
                    count_event(diagnostics, 'reject_appearance')
                    continue
                score = sum(weight*value for weight, value in zip(
                    SCORE_WEIGHTS, (shape, support, pattern, isolation)))
                small = 1-clip01((radius/minimum_radius-1)/2)
                required_score = MIN_SIGN_SCORE + SMALL_SIGN_SCORE_BONUS*small
                if score < required_score:
                    count_event(diagnostics, 'reject_score')
                    continue
                # Bounding rectangle of the fitted rotated ellipse, not its square hull.
                co, si = math.cos(math.radians(angle)), math.sin(math.radians(angle))
                ex = math.hypot(axes[0]/2*co, axes[1]/2*si)
                ey = math.hypot(axes[0]/2*si, axes[1]/2*co)
                pad = radius*BOX_PADDING_RATIO
                box = (max(0, round(gx-ex-pad)), max(0, round(gy-ey-pad)),
                       min(width-1, round(gx+ex+pad)), min(height-1, round(gy+ey+pad)))
                proposals.append((box, score))
        selected = suppress_duplicates(proposals)
        count_event(diagnostics, 'frames')
        count_event(diagnostics, 'proposals', len(proposals))
        count_event(diagnostics, 'selected', len(selected))
        return selected


def circular_candidates(frame, return_scores=False, diagnostics=None):
    """Convenience for single-frame experiments; videos reuse one SignDetector."""
    selected = SignDetector().detect(frame, diagnostics)
    return selected if return_scores else [box for box, _ in selected]


class SignTracker:
    @staticmethod
    def state(box):
        return ((box[0]+box[2])/2, (box[1]+box[3])/2, box[2]-box[0], box[3]-box[1])

    @staticmethod
    def box(state):
        cx, cy, width, height = state
        width, height = max(2.0, width), max(2.0, height)
        return (cx-width/2, cy-height/2, cx+width/2, cy+height/2)

    def __init__(self, fps):
        self.tracks = []
        self.min_hits = max(2, round(fps*TRACK_CONFIRM_SECONDS))
        self.history_size = max(self.min_hits, round(fps*TRACK_HISTORY_SECONDS))
        self.display_misses = max(1, round(fps*TRACK_HOLD_SECONDS))
        self.max_misses = max(self.display_misses, round(fps*TRACK_RETAIN_SECONDS))
        self.previous_gray = None

    @staticmethod
    def appearance(gray, box):
        x1, y1, x2, y2 = map(round, box)
        if (x1 < 0 or y1 < 0 or x2 > gray.shape[1] or y2 > gray.shape[0]
                or min(x2-x1, y2-y1) < 4):
            return None
        patch = cv2.resize(gray[y1:y2, x1:x2], (32, 32))
        return patch if patch.std() >= 8 else None

    @staticmethod
    def flow_color_support(frame, box):
        cx, cy, w, h = SignTracker.state(box)
        margin = max(w, h)
        x0, y0 = max(0, math.floor(cx-margin)), max(0, math.floor(cy-margin))
        x1, y1 = min(frame.shape[1], math.ceil(cx+margin)), min(frame.shape[0], math.ceil(cy+margin))
        if min(x1-x0, y1-y0) < 4:
            return False
        ellipse = ((cx-x0, cy-y0), (w/(1+BOX_PADDING_RATIO), h/(1+BOX_PADDING_RATIO)), 0)
        hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        support, pattern, _ = appearance_features(hsv, ellipse)
        return support >= MIN_ANGULAR_SUPPORT and pattern >= MIN_PATTERN_SCORE

    def flow_prediction(self, track, gray, frame):
        if (not track['confirmed'] or track['appearance'] is None
                or self.previous_gray is None):
            return None
        cx, cy, width, height = self.state(track['box'])
        # Build LK pyramids on a local crop, rather than on the whole frame per track.
        margin = max(4, round(max(width, height)*0.60))
        x0, y0 = max(0, math.floor(cx-width/2)-margin), max(0, math.floor(cy-height/2)-margin)
        x1, y1 = min(gray.shape[1], math.ceil(cx+width/2)+margin), min(gray.shape[0], math.ceil(cy+height/2)+margin)
        if min(x1-x0, y1-y0) < 6:
            return None
        previous, current = self.previous_gray[y0:y1, x0:x1], gray[y0:y1, x0:x1]
        mask = np.zeros(previous.shape, np.uint8)
        cv2.ellipse(mask, (round(cx)-x0, round(cy)-y0),
                    (max(2, round(width*.40)), max(2, round(height*.40))), 0, 0, 360, 255, -1)
        points = cv2.goodFeaturesToTrack(previous, 20, .02, 2, mask=mask, blockSize=3)
        if points is None or len(points) < 4:
            return None
        options = dict(winSize=(15, 15), maxLevel=2,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .03))
        moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, **options)
        if moved is None or not np.isfinite(moved).all():
            return None
        returned, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, moved, None, **options)
        if returned is None:
            return None
        original, destination = points[:, 0], moved[:, 0]
        tolerance = max(.75, gray.shape[1]/1920*1.5)
        good = ((status.ravel() == 1) & (back_status.ravel() == 1)
                & (np.linalg.norm(returned[:, 0]-original, axis=1) < tolerance))
        if np.count_nonzero(good) < max(4, math.ceil(len(points)*.60)):
            return None
        a, b = original[good], destination[good]
        shift = np.median(b-a, axis=0)
        consistent = np.linalg.norm((b-a)-shift, axis=1) < max(tolerance, max(width, height)*.12)
        if np.count_nonzero(consistent) < 4 or np.linalg.norm(shift) > max(gray.shape[1]*.003, max(width, height)*.35):
            return None
        a, b = a[consistent], b[consistent]
        shift = np.median(b-a, axis=0)
        old_dist = np.linalg.norm(a-np.median(a, axis=0), axis=1)
        new_dist = np.linalg.norm(b-np.median(b, axis=0), axis=1)
        valid = old_dist > 2
        scale = float(np.median(new_dist[valid]/old_dist[valid])) if np.any(valid) else 1.0
        if not .85 <= scale <= 1.15:
            return None
        # LK points can follow the inner symbol while underestimating rim growth.
        # Refine its estimate against the fixed detector reference at three nearby
        # scales. This also works when the sign has moved outside the detection ROI.
        nx, ny, nw, nh = cx+shift[0], cy+shift[1], width*scale, height*scale
        search_margin = max(2, round(max(nw, nh)*.15))
        best = None
        for refinement in (.94, 1.0, 1.06):
            tw, th = max(4, round(nw*refinement)), max(4, round(nh*refinement))
            sx0, sy0 = max(0, round(nx-tw/2)-search_margin), max(0, round(ny-th/2)-search_margin)
            sx1, sy1 = min(gray.shape[1], round(nx+tw/2)+search_margin), min(gray.shape[0], round(ny+th/2)+search_margin)
            if sx1-sx0 < tw or sy1-sy0 < th:
                continue
            reference = cv2.resize(track['appearance'], (tw, th))
            scores = cv2.matchTemplate(gray[sy0:sy1, sx0:sx1], reference, cv2.TM_CCOEFF_NORMED)
            _, similarity, _, location = cv2.minMaxLoc(scores)
            box = (sx0+location[0], sy0+location[1], sx0+location[0]+tw, sy0+location[1]+th)
            if (similarity >= FLOW_MIN_SIMILARITY and (best is None or similarity > best[0])
                    and self.flow_color_support(frame, box)):
                best = (similarity, box)
        return best[1] if best is not None else None

    def update(self, detections, scores=None, frame=None):
        scores = [1.0]*len(detections) if scores is None else scores
        if len(scores) != len(detections):
            raise ValueError('Each detection must have one score')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame is not None else None
        predictions = [self.box(tuple(a+b for a, b in zip(self.state(t['box']), t['velocity'])))
                       for t in self.tracks]
        matches = []
        for ti, old in enumerate(predictions):
            ox, oy, ow, oh = self.state(old)
            for di, box in enumerate(detections):
                nx, ny, nw, nh = self.state(box)
                size = max(ow, oh, nw, nh)
                distance = math.hypot(nx-ox, ny-oy)
                ratio = max(ow, oh, nw, nh)/max(1, min(max(ow, oh), max(nw, nh)))
                if distance < max((frame.shape[1]*.006 if frame is not None else 12), TRACK_MATCH_DISTANCE_RATIO*size) and ratio < TRACK_MATCH_SIZE_RATIO:
                    matches.append((distance/size, ti, di))
        used_tracks, used_detections = set(), set()
        for _, ti, di in sorted(matches):
            if ti in used_tracks or di in used_detections:
                continue
            track = self.tracks[ti]
            measured, predicted = self.state(detections[di]), self.state(predictions[ti])
            weight = TRACK_OBSERVATION_WEIGHT
            # Smooth the center, but use the observed size to avoid lag as a sign
            # rapidly grows nearby. A stale size can leave the box inside its rim.
            state = tuple((1-weight)*a+weight*b for a, b in zip(predicted[:2], measured[:2])) + measured[2:]
            track['box'] = self.box(state)
            gap, size = track['misses']+1, max(measured[2:])
            velocity = []
            for axis, (value, previous, speed) in enumerate(zip(measured, track['measurement'], track['velocity'])):
                limit = size*(.15 if axis < 2 else .10)
                velocity.append(float(np.clip(.65*speed+.35*(value-previous)/gap, -limit, limit)))
            track.update(velocity=tuple(velocity), measurement=measured, misses=0,
                         unsupported=0, flow_valid=False)
            track['history'].append(float(scores[di]))
            used_tracks.add(ti)
            used_detections.add(di)
        for ti, track in enumerate(self.tracks):
            if ti not in used_tracks:
                flow = self.flow_prediction(track, gray, frame) if gray is not None else None
                track['flow_valid'] = flow is not None
                track['unsupported'] = 0 if flow is not None else track['unsupported']+1
                track['box'] = flow if flow is not None else predictions[ti]
                track['velocity'] = tuple(v*.85 for v in track['velocity'])
                track['misses'] += 1
                track['history'].append(0.0)
        for di, box in enumerate(detections):
            if di not in used_detections:
                self.tracks.append(dict(box=tuple(map(float, box)), measurement=self.state(box),
                                        velocity=(0., 0., 0., 0.), history=[float(scores[di])],
                                        confirmed=False, misses=0, unsupported=0,
                                        appearance=None, flow_valid=False))
        for track in self.tracks:
            if gray is not None and track['misses'] == 0:
                track['appearance'] = self.appearance(gray, self.box(track['measurement']))
            track['history'] = track['history'][-self.history_size:]
            hits = [s for s in track['history'] if s > 0]
            if len(hits) >= self.min_hits and sum(hits)/len(hits) >= MIN_SIGN_SCORE:
                track['confirmed'] = True
        self.tracks = [t for t in self.tracks if t['unsupported'] <=
                       (self.max_misses if t['confirmed'] else self.history_size)]
        self.previous_gray = gray
        visible = [(tuple(map(round, t['box'])), max(t['history'])) for t in self.tracks
                   if t['confirmed'] and t['unsupported'] <= self.display_misses]
        return [box for box, _ in suppress_duplicates(visible)]


def process_video(source, destination, student_id, max_seconds=None, diagnostics=None):
    source, destination = str(source), str(destination)
    if source == destination:
        raise ValueError('Input and output paths must differ')
    capture = cv2.VideoCapture(source)
    writer = None
    try:
        if not capture.isOpened():
            raise RuntimeError('Cannot open input video: '+source)
        fps = capture.get(cv2.CAP_PROP_FPS)
        expected = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = round(capture.get(3)), round(capture.get(4))
        if not np.isfinite(fps) or fps <= 0 or min(width, height) <= 0:
            raise RuntimeError('Invalid input video metadata: '+source)
        if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds <= 0):
            raise ValueError('MAX_SECONDS must be None or a positive number')
        limit = round(max_seconds*fps) if max_seconds is not None else None
        if limit == 0:
            raise ValueError('MAX_SECONDS is shorter than one frame')
        target = min(expected, limit) if expected > 0 and limit is not None else expected
        if max_seconds is not None and expected <= 0:
            target = round(max_seconds*fps)
        writer = cv2.VideoWriter(destination, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), True)
        if not writer.isOpened():
            raise RuntimeError('Cannot write output. Check the existing folder: '+destination)
        detector, tracker, total = SignDetector(), SignTracker(fps), 0
        while limit is None or total < limit:
            ok, frame = capture.read()
            if not ok:
                break
            candidates = detector.detect(frame, diagnostics)
            boxes = tracker.update([b for b, _ in candidates], [s for _, s in candidates], frame)
            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(frame, (max(0,x1), max(0,y1)), (min(width-1,x2), min(height-1,y2)), (0,255,0), max(1,round(width/640)))
            text_origin = (round(width*.01), height-max(12,round(height*.02)))
            font_scale = max(.35, width/2133)
            cv2.putText(frame, student_id, text_origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), max(2,round(width/480)), cv2.LINE_AA)
            cv2.putText(frame, student_id, text_origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), max(1,round(width/960)), cv2.LINE_AA)
            writer.write(frame)
            total += 1
        if target > 0 and total != target:
            raise RuntimeError(f'Incomplete decoding: {total}/{target} frames: {source}. Output is incomplete.')
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    print(f'{source} -> {destination} ({total} frames, {fps:g} fps, {width}x{height})', flush=True)
    return total


def main():
    if (len(SCORE_WEIGHTS) != 4 or min(SCORE_WEIGHTS) < 0
            or abs(sum(SCORE_WEIGHTS)-1) > 1e-6):
        raise ValueError('SCORE_WEIGHTS must contain four weights adding to 1')
    cv2.setNumThreads(OPENCV_THREADS)
    script_path = __file__.replace('\\', '/')
    folder = script_path.rsplit('/', 1)[0] if '/' in script_path else '.'
    output = folder+'/'+OUTPUT_FOLDER.strip('/\\') if OUTPUT_FOLDER else folder
    for name in INPUT_VIDEOS:
        stem = name.replace('\\', '/').rsplit('/', 1)[-1].rsplit('.', 1)[0]
        suffix = f'_preview_{MAX_SECONDS:g}s' if MAX_SECONDS is not None else ''
        destination = output+'/'+stem+'_'+STUDENT_ID+suffix+'.mp4'
        process_video(folder+'/'+name, destination, STUDENT_ID, MAX_SECONDS)


if __name__ == '__main__':
    main()
