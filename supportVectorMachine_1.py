from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd 
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import pickle


data = pd.read_pickle("../data_file_w_ts_kat_al.pkl")
target = pd.read_pickle("../target_file_w_ts_kat_al.pkl")
min_max_scaler = preprocessing.MinMaxScaler()

data[225280] = pd.DataFrame(min_max_scaler.fit_transform(data[225280].values.reshape(-1, 1)))
print(target)
print(data[225280])
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2, random_state = 42)
print(X_train.values.ravel())
print(y_train.values.ravel())

ovr = svm.SVC(decision_function_shape='ovo', max_iter = 4000)
param = {'kernel': ['rbf','poly', 'sigmoid', 'linear'] }
grid = GridSearchCV(ovr, param)
print("fit")
grid.fit(X_train, y_train.values.ravel())
print("fit done")
pred = grid.predict(X_test)
res = confusion_matrix(y_test, pred, labels=["aanwezig", "buiten", "niets"])
print(res)

pickle.dump(grid, open("supportVectorMachine_1.mod", 'wb'))

