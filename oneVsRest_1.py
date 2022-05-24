from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd
import os
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pickle


data = pd.read_pickle("../data_file_w_ts_kat_al.pkl")
target = pd.read_pickle("../target_file_w_ts_kat_al.pkl")
print(data.iloc[ :, -1:])
print(target)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2, random_state = 42)
print(X_train.values.ravel())
print(y_train.values.ravel())
log = LogisticRegression(max_iter= 4000)
param = {'penalty': ['none','l2' ] }
grid = GridSearchCV(log, param)
grid.fit(X_train, y_train.values.ravel())
ovr = OneVsRestClassifier(grid)

ovr.fit(X_train, y_train.values.ravel())
pred = ovr.predict(X_test)
res = confusion_matrix(y_test, pred, labels=["aanwezig", "buiten", "niets"])
print(res)

pickle.dump(ovr, open("oneVsRest_1.mod", 'wb'))

