from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import pandas as pd
import os
from sklearn import preprocessing

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pickle
def train_voteclass():
    svm_1 = pickle.load(open("ml_gui\supportVectorMachine_1.mod", 'rb'))
    svm_2 = pickle.load(open("ml_gui\supportVectorMachine_2.mod", 'rb'))
    ovr_1 =  pickle.load(open("ml_gui\oneVsRest_1.mod", 'rb'))
    ovr_2 =  pickle.load(open("ml_gui\oneVsRest_2.mod", 'rb'))

    voting_clf = VotingClassifier(
        estimators=[("svm_1", svm_1),("svm_2", svm_2),("ovr_1", ovr_1),("ovr_2", ovr_2)], voting='hard',
        n_jobs= -1
    )
    data = pd.read_pickle("data_file_w_ts_kat_al.pkl")
    target = pd.read_pickle("target_file_w_ts_kat_al.pkl")
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(data[[225280]])
    data[225280] = pd.DataFrame( min_max_scaler.transform(data[[225280]])   )
    print(target)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.1, random_state = 42)


    print("fitting")
    voting_clf.fit(X_train, y_train.values.ravel())
    print('fit done')
    pred = voting_clf.predict(X_test)
    print("conf start")
    res = confusion_matrix(y_test, pred, labels=["aanwezig", "buiten", "niets"])
    print(res)

    pickle.dump(voting_clf, open("ensemble.mod", 'wb'))

train_voteclass()