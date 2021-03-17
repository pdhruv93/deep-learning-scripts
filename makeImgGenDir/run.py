from makeImageDataGenCompatibleDir import ImgGenCompatibleDir
import cv2
import numpy as np


def applyHSV(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = cv2.inRange(hsv, np.array([-30, -30, 100]), np.array([50, 50, 250]))
    return image


ImgGenCompatibleDir("C://deep learning//fingers-count-scratch//dataset//train",
                    "C://deep learning//fingers-count-scratch//dataset//test",
                    "C://deep learning//fingers-count-scratch//dataset//train_custom",
                    "C://deep learning//fingers-count-scratch//dataset//test_custom",
                    applyHSV
                    ).make()