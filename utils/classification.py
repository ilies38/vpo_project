import os
import pickle
import cv2
import numpy as np

_classifier = None


def load_classifier(model_path='metro_classifier.pkl'):
    global _classifier
    if _classifier is None and os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            _classifier = pickle.load(f)
    return _classifier


def color_histogram(patch, bins=(8, 8, 8)):
    hsv = cv2.cvtColor((patch * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def heuristic_label(patch):
    hsv = cv2.cvtColor((patch * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    mean_h = np.mean(hsv[:, :, 0])
    if 20 <= mean_h <= 40:
        return 1
    if 100 <= mean_h <= 130:
        return 2
    if 50 <= mean_h <= 80:
        return 3
    if 150 <= mean_h <= 170:
        return 4
    if mean_h < 10 or mean_h > 170:
        return 12
    return 14


def classify_pictogram(patch):
    clf = load_classifier()
    feat = color_histogram(patch)
    if clf is not None:
        try:
            label = int(clf.predict([feat])[0])
            return label
        except Exception:
            pass
    return heuristic_label(patch)
