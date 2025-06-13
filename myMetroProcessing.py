# -*- coding: utf-8 -*-
"""Main processing pipeline for metro pictogram detection."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from utils.preprocessing import preprocess_image
from utils.detection import detect_candidates
from utils.classification import classify_pictogram
from utils.io_tools import extract_rois, build_output, visual_debug


def processOneMetroImage(nom, im, n, resizeFactor):
    """Detect and recognise pictograms in one image."""

    # --- preprocessing ----------------------------------------------------
    im_resized = preprocess_image(im, resizeFactor)

    # --- detection --------------------------------------------------------
    boxes = detect_candidates(im_resized)
    rois = extract_rois(im_resized, boxes)

    # --- classification ---------------------------------------------------
    labels = [classify_pictogram(patch) for patch in rois]

    bd = build_output(n, boxes, labels)

    # --- debug visualisation ---------------------------------------------
    try:
        visual_debug(im_resized.copy(), bd, f"outputs/debug_images/{nom}.png")
    except Exception:
        pass

    return im_resized, np.array(bd)


# =======================================================================

def draw_rectangle(x1, x2, y1, y2, color):
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                     edgecolor=color, facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)
