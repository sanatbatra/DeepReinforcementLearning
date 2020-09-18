import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


def plotCf(a,b,t):
    plt.clf()
    cf =confusion_matrix(a,b)
    plt.imshow(cf,cmap=plt.cm.Blues,interpolation='nearest')
    plt.colorbar()
    plt.title(t)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    tick_marks = np.arange(len(set(a))) # length of classes
    class_labels = ['2','4']
    plt.xticks(tick_marks,class_labels)
    plt.yticks(tick_marks,class_labels)
    thresh = cf.max() / 2.
    for i,j in itertools.product(range(cf.shape[0]),range(cf.shape[1])):
        plt.text(j,i,format(cf[i,j],'d'),horizontalalignment='center',color='white' if cf[i,j] >thresh else 'black')
    plt.savefig('ConfusionMatrixTestSet.png')

dataset = pd.read_csv('breast-cancer-wisconsin.data', header=None)
dataset = dataset[~dataset[6].isin(['?'])]
print(dataset.head(10))
X = dataset.iloc[:,1:10].values
y = dataset.iloc[:,10:].values

sc = StandardScaler()
X = sc.fit_transform(X)
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.2)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size = 0.5)
model = Sequential()
model.add(Dense(16, input_dim=9, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(12, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data = (X_val,y_val), epochs=100, batch_size=64)
y_pred = model.predict(X_val)

pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))

test = list()
for i in range(len(y_val)):
    test.append(np.argmax(y_val[i]))
a = accuracy_score(pred,test)

print('Accuracy on validation set:', a*100)

y_pred = model.predict(X_test)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('LossPlot.png')

pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))

test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))
a = accuracy_score(pred,test)
plotCf(test,pred,'Confusion matrix Test Set')
print('Accuracy on test set:', a*100)

