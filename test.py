import pandas as pd
import os
from PIL import Image
from skimage import color
from skimage import io
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from datetime import datetime
from procesImage import procesImage
import pickle
ovr =  pickle.load(open("ensemble.mod", 'rb'))


path =r'C:\Users\brent\OneDrive\Bureaublad\classificatie'
for word in ("buiten", "aanwezig"):
    images = os.listdir(rf"{path}\{word}")
    
    for im in images:
        print("PROCESSING IMAGE")
        img = io.imread(rf"{path}\{word}\{im}")
        res = procesImage(img, im, ovr)
        if(res != word):
            print(f"label : {word}    res : {res}" )
            imgplot = plt.imshow(img)
            plt.show()
        else:
            print(f"label : {word}    res : {res}" )
            print("OK ")

