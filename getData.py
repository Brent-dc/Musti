import pandas as pd
import os
from PIL import Image
from skimage import color
from skimage import io
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from datetime import datetime

target = []
data = []
dfData = pd.DataFrame()
dfTarget = pd.DataFrame(columns=['target'])

path =r'C:\Users\brent\OneDrive\Bureaublad\classificatie'
for word in ("niets", "aanwezig","buiten"):
    images = os.listdir(rf"{path}\{word}")
    
    for im in images:
        print("PROCESSING IMAGE")
        img = io.imread(rf"{path}\{word}\{im}")
        imgGray = color.rgb2gray(img)
        print(word)
        timestamp =datetime.strptime( im.split("_")[1].split(".")[0].split(" ")[0]  , '%H%M%S'   )

        pix_val = imgGray 
        pix_val_flat = [x for sets in pix_val for x in sets]
        print(im.split("_")[1])
        print(timestamp.hour)
        pix_val_flat.append(timestamp.hour)
        data.append(pix_val_flat)
        dfDataTemp =pd.DataFrame(data)
        dfData = pd.concat([dfData, dfDataTemp])
        print(dfData.shape)
        dfTarget = dfTarget.append({'target':word }, ignore_index=True)
        data = []
       

print(dfData)


print(dfTarget)
dfData.to_pickle("data_file_w_ts.pkl")

dfTarget.to_pickle("target_file_w_ts.pkl")
print("pickle saved")
