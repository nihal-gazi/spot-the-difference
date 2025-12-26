import cv2
import numpy as np

def align_images(img1, img2):
    orb = cv2.ORB_create()
    k1, d1 = orb.detectAndCompute(img1, None)
    k2, d2 = orb.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher()
    matches = matcher.match(d1, d2)

    src = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    H, _ = cv2.findHomography(dst, src, cv2.RANSAC)
    h, w = img1.shape[:2]

    aligned = cv2.warpPerspective(img2, H, (w,h))
    return img1, aligned
