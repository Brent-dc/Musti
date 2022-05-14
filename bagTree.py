from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import ExtraTreeClassifier
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

extra_tree = ExtraTreeClassifier(random_state=0, max_depth = 25 )

cls = BaggingClassifier(extra_tree, random_state=0).fit(X_train, y_train.values.ravel())
pred = cls.predict(X_test)
res = confusion_matrix(y_test, pred, labels=["aanwezig", "buiten", "niets"])
print(res)

pickle.dump(cls, open("bagTree.mod", 'wb'))

