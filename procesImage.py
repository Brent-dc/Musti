import pandas as pd
import os
from PIL import Image
from skimage import color
from skimage import io
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from datetime import datetime
def procesImage(img, timestamp, model):
        data = []
        imgGray = color.rgb2gray(img)
        timestamp =datetime.strptime( timestamp.split("_")[1].split(".")[0].split(" ")[0]  , '%H%M%S'   )

        pix_val = imgGray 
        pix_val_flat = [x for sets in pix_val for x in sets]
        
        print(timestamp.hour)
        pix_val_flat.append(timestamp.hour)
        data.append(pix_val_flat)
        dfDataTemp =pd.DataFrame(data)
        res = model.predict_proba(dfDataTemp)
        print(res)
        res = model.predict(dfDataTemp)
        return res