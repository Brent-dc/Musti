from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd
import os
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import pickle


data = pd.read_pickle("../data_file_w_ts_kat_al.pkl")
target = pd.read_pickle("../target_file_w_ts_kat_al.pkl")
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(data[[225280]])
data[225280] = pd.DataFrame( min_max_scaler.transform(data[[225280]])   )
for d in data[225280]:
    print(d)
print(target)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2, random_state = 42)
print(X_train.values.ravel())
print(y_train.values.ravel())
log = LogisticRegression(max_iter= 500)
ovr = OneVsRestClassifier(log)

ovr.fit(X_train, y_train.values.ravel())
pred = ovr.predict(X_test)
res = confusion_matrix(y_test, pred, labels=["aanwezig", "buiten", "niets"])
print(res)

pickle.dump(ovr, open("oneVsRest_1.mod", 'wb'))

