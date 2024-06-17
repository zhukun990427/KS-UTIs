import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from keras.models import Sequential  
from keras.layers import Embedding, Bidirectional, SimpleRNN, Dense  
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools


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


###RNN
max_features = 20000
model = Sequential() 
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32, return_sequences=True)) 
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid')) 
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']) 


###Cross verification
kfold = KFold(n_splits=5, shuffle=True)  # 假设我们使用5折交叉验证  
fold_scores = []  
fold_losses = []

for train_idx, val_idx in kfold.split(x_train, y_train):  

    train_data, val_data = x_train[train_idx], x_train[val_idx]  
    train_labels, val_labels = y_train[train_idx], y_train[val_idx]  

    model.fit(train_data, train_labels, epochs=10, batch_size=32)  

    val_loss, val_acc = model.evaluate(val_data, val_labels)  
    fold_scores.append(val_acc)  
    fold_losses.append(val_loss)
    
### Output validation accuracy and loss values for each fold 
print('Fold scores:', fold_scores)
print('Fold losses:', fold_losses)

# Calculate the average validation accuracy and loss values 
print('Average validation accuracy:', np.mean(fold_scores))
print('Average validation loss:', np.mean(fold_losses)) 
model.fit(x_train, y_train, epochs=10, batch_size=32)  

# Evaluate the performance of the model on the test set 
test_loss, test_acc = model.evaluate(x_test, y_test)  
print('Test loss:', test_loss)  
print('Test accuracy:', test_acc)

y_train_prob = model.predict(x_train)
y_test_prob = model.predict(x_test)
y_train_pred = (y_train_prob > 0.5).astype(int)
y_test_pred = (y_test_prob > 0.5).astype(int)


#Calculate the training set confusion matrix
cm = confusion_matrix(y_train, y_train_pred)  


classes = ['0', '1']

plt.imshow(cm, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.show()

### Calculate each performance indicator
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * precision * recall / (precision + recall)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
ppv = tp / (tp + fp)
npv = tn / (tn + fn)
dsc = 2 * tp / (fp + 2 * tp + fn)
ji = tp / (tp + fp + fn)

###  Output performance index
print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall (Sensitivity): {:.2f}".format(sensitivity))
print("F1-score: {:.2f}".format(f1_score))
print("Specificity: {:.2f}".format(specificity))
print("Positive Predictive Value (PPV): {:.2f}".format(ppv))
print("Negative Predictive Value (NPV): {:.2f}".format(npv))
print("Dice Similarity Coefficient (DSC): {:.2f}".format(dsc))
print("Jaccard Index (JI): {:.2f}".format(ji))