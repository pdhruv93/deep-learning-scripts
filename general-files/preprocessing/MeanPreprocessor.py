import cv2

class MeanPreprocessor:
    def __init__(self, rMean, gMean, bMean):
        #the mean values for each R,G,B computed over entire dataset
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean

    def preprocess(self, image):
        #split the image into its R, G, B channels
        (B, G, R) = cv2.split(image.astype("float32"))

        #subtract the means for each channeö
        R = R-self.rMean
        G = G - self.rMean
        B = B - self.rMean

        #merge the channel back together and return as image
        print("Returning image after applying MeanPreprocessing")
        return cv2.merge([B, G, R])