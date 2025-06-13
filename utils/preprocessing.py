import cv2
import numpy as np


def preprocess_image(img, resize_factor=1.0):
    """Resize and apply basic preprocessing steps to improve detection."""
    if resize_factor != 1:
        img = cv2.resize(img, None, fx=resize_factor, fy=resize_factor,
                         interpolation=cv2.INTER_AREA)
    img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # CLAHE on the L channel to improve contrast
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Gaussian blur to remove small noise
    img_blur = cv2.GaussianBlur(img_clahe, (5, 5), 0)
    img_rgb = cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB)

    return img_rgb
