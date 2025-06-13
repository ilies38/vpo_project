import os
import cv2


def extract_rois(img, boxes):
    rois = []
    for x1, y1, x2, y2 in boxes:
        patch = img[y1:y2, x1:x2]
        rois.append(patch)
    return rois


def build_output(n, boxes, labels):
    bd = []
    for (x1, y1, x2, y2), label in zip(boxes, labels):
        bd.append([n, x1, x2, y1, y2, label])
    return bd


def visual_debug(img, bd, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    for entry in bd:
        _, x1, x2, y1, y2, label = entry
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, str(label), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
