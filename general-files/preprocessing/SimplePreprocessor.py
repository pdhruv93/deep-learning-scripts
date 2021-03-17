import cv2;

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        #store the target width and height and interpolation
        # to which the image will be resized
        self.width=width
        self.height=height
        self.inter=inter

    def preprocess(self, image):
        #resize the image to targetted dimensions without
        #No Aspect aware resizing here
        print('Resizing the Image to {0}X{1}'.format(self.width, self.height))
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)