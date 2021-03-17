import eli5
from keras.preprocessing.image import load_img, img_to_array


IMG_URL = "to_predict_images/1.jpg"

im = load_img(IMG_URL, target_size=(96, 96))
doc = img_to_array(im) # -> numpy array


eli5.show_prediction("finger_count_model.h5", doc, image=IMG_URL)