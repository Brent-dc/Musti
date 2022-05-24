import pickle
import pandas as pd
from PIL import Image
from skimage import color
from skimage import io
import io
from ensembleModel import train_voteclass
from getData import getData
def fit(path):
        getData(path)

        svm_1 = pickle.load(open("supportVectorMachine_1.mod", 'rb'))
        svm_2 = pickle.load(open("supportVectorMachine_2.mod", 'rb'))
        ovr_1 =  pickle.load(open("oneVsRest_1.mod", 'rb'))
        ovr_2 =  pickle.load(open("oneVsRest_2.mod", 'rb'))
        


        models = [ovr_1, ovr_2, svm_1,svm_2]


        
        data = pd.read_pickle("data_file_w_ts.pkl")
        target = pd.read_pickle("target_file_w_ts.pkl") 
        for model in models: 
            model.fit(data, target.values.ravel())
        train_voteclass()


