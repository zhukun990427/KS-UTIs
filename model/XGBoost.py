import pandas as pd
import numpy as np
from collections import Counter
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report,RocCurveDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import cm

dataPath = '------------.xlsx'
data = pd.read_excel(dataPath)
data.describe()

y = data['Label']
data_a = data[y == 0]
data_b = data[y == 1]
X = data.iloc[:,2:]

###Split the training set and the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 21, stratify = y)

X_train_a = X_train[y_train == 0]
X_train_b = X_train[y_train == 1]
X_test_a = X_test[y_test == 0]
X_test_b = X_test[y_test == 1]


###Data standardization
scaler = StandardScaler()
X_train_scal = scaler.fit_transform(X_train)
X_test_scal = scaler.transform(X_test)

X_train_scal = pd.DataFrame(X_train_scal,columns = colNamesSel_mwU)
X_test_scal = pd.DataFrame(X_test_scal,columns = colNamesSel_mwU)

### XGBoost
scoreTrainList, scoreTestList = [], []
maxTreeNum = 100
for i in range(1,maxTreeNum):
    XGB = XGBClassifier( random_state = 21
                        ,n_estimators = i
                        ,objective = 'binary:hinge'
                         ,use_label_encoder = False
                        )
    XGB.fit(X_train_mul_scal,y_train)
    score_test =  XGB.score(X_test_mul,y_test)
    score_train =  XGB.score(X_train_mul,y_train)
    scoreTestList.append(score_test)
    scoreTrainList.append(score_train)
plt.plot(range(1,maxTreeNum), scoreTestList, label = 'Test')
plt.plot(range(1,maxTreeNum), scoreTrainList, label = 'Train')
plt.legend() 
plt.show()


y_train_prob = XGB.predict_proba(X_train_scal)[:,1]
for i in range(len(y_train_prob)):
    print(y_train_prob[i], end=',')
y_train_pred = XGB.predict(X_train_scal)
y_test_prob= XGB.predict_proba(X_test_scal)[:,1]
for i in range(len(y_test_prob)):
    print(y_test_prob[i], end=',')
y_test_pred = XGB.predict(X_test_scal)

### Calculate the training set confusion matrix and classification report
classes = ['0', '1']
cm_train = confusion_matrix(y_train, y_train_pred)
cr_train = classification_report(y_train, y_train_pred)


### Calculate each performance indicator
tn_train, fp_train, fn_train, tp_train = cm_train.ravel()
accuracy_train = (tp_train + tn_train) / (tp_train + tn_train + fp_train + fn_train)
precision_train = tp_train / (tp_train + fp_train)
recall_train = tp_train / (tp_train + fn_train)
f1_score_train = 2 * precision_train * recall_train / (precision_train + recall_train)
sensitivity_train = tp_train / (tp_train + fn_train)
specificity_train = tn_train / (tn_train + fp_train)
ppv_train = tp_train / (tp_train + fp_train)
npv_train = tn_train / (tn_train + fn_train)
dsc_train = 2 * tp_train / (fp_train + 2 * tp_train + fn_train)
ji_train = tp_train / (tp_train + fp_train + fn_train)


### Output performance index
print("Accuracy_train: {:.2f}".format(accuracy_train))
print("Precision_train: {:.2f}".format(precision_train))
print("Recall_train (Sensitivity): {:.2f}".format(sensitivity_train))
print("F1-score_train: {:.2f}".format(f1_score_train))
print("Specificity_train: {:.2f}".format(specificity_train))
print("Positive Predictive Value _train (PPV_train): {:.2f}".format(ppv_train))
print("Negative Predictive Value _train (NPV_train): {:.2f}".format(npv_train))
print("Dice Similarity Coefficient _train (DSC_train): {:.2f}".format(dsc_train))
print("Jaccard Index _train (JI_train): {:.2f}".format(ji_train))