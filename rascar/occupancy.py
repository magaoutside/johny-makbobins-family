import numpy as np
import cv2
from .core import CLASSES, CROP, BBOX

def occupancy(record, manual):
    c = record[CROP].to_numpy(float)
    cell = (c[2:] - c[:2]) / 32
    xx = c[0] + np.arange(32) * cell[0]
    yy = c[1] + np.arange(32) * cell[1]
    result = np.zeros((4, 32, 32), np.float32)
    for k, cls in enumerate(CLASSES):
        boxes = manual[manual.frame_id.eq(record.frame_id) & manual.class_name.eq(cls)]
        for box in boxes[BBOX].to_numpy(float):
            wx = np.maximum(0, np.minimum(xx + cell[0], box[2]) - np.maximum(xx, box[0])) / cell[0]
            wy = np.maximum(0, np.minimum(yy + cell[1], box[3]) - np.maximum(yy, box[1])) / cell[1]
            result[k] = np.maximum(result[k], wy[:, None] * wx[None, :])
    return result

def overlap(maps, rows):
    result = []
    for p, (_, r) in zip(maps, rows.iterrows()):
        c, box = r[CROP].to_numpy(float), r[BBOX].to_numpy(float)
        mask = (p[CLASSES.index(r.class_name)] >= .5).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        boxes = [np.array([x, y, x+w, y+h])/32*np.tile(c[2:]-c[:2], 2)+np.tile(c[:2], 2)
                 for x, y, w, h, area in stats[1:] if area >= 4]
        if not boxes:
            result.append(np.nan)
            continue
        b = np.array(boxes)
        inter = np.maximum(0, np.minimum(box[2:], b[:, 2:])-np.maximum(box[:2], b[:, :2])).prod(1)
        union = (box[2:]-box[:2]).prod() + (b[:, 2:]-b[:, :2]).prod(1)-inter
        result.append(float((inter/np.maximum(union, 1e-9)).max()))
    return np.array(result)
