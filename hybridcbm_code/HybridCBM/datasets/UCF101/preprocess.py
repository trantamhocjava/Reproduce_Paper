import os
import pandas as pd
import pickle


def mv():
    import glob
    import shutil
    images = glob.glob('images/*/*.jpg')
    for image in images:
        shutil.move(image, 'images/')


if __name__ == '__main__':
    mv()
