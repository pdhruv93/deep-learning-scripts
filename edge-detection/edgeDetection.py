from makeImageDataGenCompatibleDir import ImgGenCompatibleDir
import cv2
import numpy as np


def applyBinaryMask(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (7,7), 3)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, new = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return new


ImgGenCompatibleDir("C://deep learning//fingers-count-scratch//dataset//train",
                    "C://deep learning//fingers-count-scratch//dataset//test",
                    "C://deep learning//fingers-count-scratch//dataset//train_custom",
                    "C://deep learning//fingers-count-scratch//dataset//test_custom",
                    applyBinaryMask
                    ).make()