from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import pandas as pd 
from sklearn import svm
from sklearn.metrics import confusion_matrix
import pickle


data = pd.read_pickle("data_file_w_ts.pkl")
target = pd.read_pickle("target_file_w_ts.pkl")
print(data.iloc[ :, -1:])
print(target)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2, random_state = 42)
print(X_train.values.ravel())
print(y_train.values.ravel())

ovr = svm.SVC(decision_function_shape='ovo')

ovr.fit(X_train, y_train.values.ravel())
pred = ovr.predict(X_test)
res = confusion_matrix(y_test, pred, labels=["aanwezig", "buiten", "niets"])
print(res)

pickle.dump(ovr, open("supportVectorMachine.mod", 'wb'))

