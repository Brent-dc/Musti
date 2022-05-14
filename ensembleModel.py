from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import pandas as pd
import os
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pickle

tree= pickle.load(open("bagTree.mod", 'rb'))

svm = pickle.load(open("supportVectorMachine.mod", 'rb'))
ovr =  pickle.load(open("oneVsRest.mod", 'rb'))
voting_clf = VotingClassifier(
    estimators=[("svm", svm),("ovr", ovr),("tree", tree)], voting='hard',
    n_jobs= -1
)
data = pd.read_pickle("data_file_w_ts.pkl")
target = pd.read_pickle("target_file_w_ts.pkl")
print(data.iloc[ :, -1:])
print(target)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2, random_state = 42)
print(X_train.values.ravel())
print(y_train.values.ravel())

print("fitting")
voting_clf.fit(X_train, y_train.values.ravel())
print('fit done')
pred = voting_clf.predict(X_test)
print("conf start")
res = confusion_matrix(y_test, pred, labels=["aanwezig", "buiten", "niets"])
print(res)

pickle.dump(voting_clf, open("ensemble.mod", 'wb'))

