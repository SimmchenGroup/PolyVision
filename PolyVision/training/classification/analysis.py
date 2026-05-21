import numpy as np
import cv2

def compute_particle_area(filepaths, image_size):
    areas = []

    for path in filepaths:
        img = cv2.imread(path)
        img = cv2.resize(img, image_size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        area = np.sum(thresh > 0)
        areas.append(area)

    return np.array(areas)

def compute_confidence(predictions):
    return np.max(predictions, axis=1)

def compute_correlation(x, y):
    return np.corrcoef(x, y)[0, 1]