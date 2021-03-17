#This class creates directory structure compatible to be used with ImageDataGenerator.flow_from_directory()

import os
from imutils import paths
from shutil import copyfile

class ImgGenCompatibleDir():
    def __init__(self, train_dir, test_dir, new_train_dir, new_test_dir):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.new_train_dir = new_train_dir
        self.new_test_dir = new_test_dir

    def make(self):
        print("Working on Train Dir....")
        for imagePath in sorted(list(paths.list_images(self.train_dir))):
            img_name = os.path.basename(imagePath)
            img_label = os.path.splitext(imagePath)[0][-2:]

            if not os.path.exists(os.path.join(self.new_train_dir, img_label)):
                os.makedirs(os.path.join(self.new_train_dir, img_label))    #create dir if not exists

            if not os.path.exists(os.path.join(self.new_train_dir, img_label, img_name)):
                copyfile(imagePath, os.path.join(self.new_train_dir, img_label, img_name)) #copy file if not exists in new folder
        print("New Training Directory compatible with flow_from_directory() created")



        print("Working on Test Dir....")
        for imagePath in sorted(list(paths.list_images(self.test_dir))):
            img_name = os.path.basename(imagePath)
            img_label = os.path.splitext(imagePath)[0][-2:]

            if not os.path.exists(os.path.join(self.new_test_dir, img_label)):
                os.makedirs(os.path.join(self.new_test_dir, img_label))  # create dir if not exists

            if not os.path.exists(os.path.join(self.new_test_dir, img_label, img_name)):
                copyfile(imagePath, os.path.join(self.new_test_dir, img_label, img_name))  # copy file if not exists in new folder
        print("New Test Directory compatible with flow_from_directory() created")