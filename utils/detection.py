import cv2
import numpy as np


def detect_candidates(img, min_area=50, max_area=5000):
    """Return bounding boxes of potential pictograms."""
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)

    # Simple color segmentation for typical metro colors
    color_ranges = {
        1: ((20, 80, 80), (40, 255, 255)),    # yellow
        2: ((100, 80, 80), (130, 255, 255)),   # blue
        3: ((50, 50, 50), (80, 255, 255)),     # green
        4: ((150, 80, 80), (170, 255, 255)),   # magenta
        5: ((5, 80, 80), (20, 255, 255)),      # orange/red
        6: ((10, 30, 30), (25, 255, 200)),     # brown/orange
        7: ((160, 80, 50), (180, 255, 255)),   # pink/red
        8: ((120, 80, 80), (160, 255, 255)),   # purple
        9: ((70, 50, 50), (100, 255, 255)),    # teal
        10: ((0, 0, 100), (180, 60, 255)),     # gray
        11: ((20, 50, 50), (35, 255, 255)),    # light yellow
        12: ((0, 80, 80), (10, 255, 255)),     # red
        13: ((40, 0, 80), (70, 60, 255)),      # turquoise/gray
        14: ((130, 50, 50), (160, 255, 255)),  # violet
    }

    mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in color_ranges.values():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask_total = cv2.bitwise_or(mask_total, mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel)
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(mask_total, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    if hierarchy is None:
        return boxes

    for idx, cnt in enumerate(contours):
        if hierarchy[0][idx][3] != -1:
            # ignore child contours
            continue
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.5:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, x + w, y + h))
    return boxes
