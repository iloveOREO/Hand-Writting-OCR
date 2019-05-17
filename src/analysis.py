import glob
import shutil
import cv2
import os
from PIL import Image


def move_files():
    destination = "./data/"
    for filename in glob.iglob('./data/**/**/*.png', recursive=True):
        image_name = filename.split("/")[-1]
        shutil.move(filename, destination + image_name)


def inscept_image():
    base_dir = "./data/"
    for image in os.listdir("./data"):
        im = Image.open(base_dir + image)
        print(im.size)


if __name__ == '__main__':
    inscept_image()
