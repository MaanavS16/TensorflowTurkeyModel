import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

data = pd.read_csv("ProductionData.csv", sep=',')
data.head()

d1 = data.iloc[2:698]
dates = d1.iloc[:,0]

turkey = d1.iloc[:,13]
pd.to_numeric(turkey)
dataset = pd.DataFrame({'Date': dates, 'Weight': turkey})
dataset = dataset.iloc[::-1]
dataset.index = np.arange(0, len(dataset))
dataset.index.name = "Month number"

arrdata = dataset.to_numpy()
arrMonth = []
arrDate = []
arrWeight = []

for i in range(len(arrdata)):
    arrMonth.append(i)
    arrDate.append(arrdata[i][0])
    arrWeight.append(float(arrdata[i][1]))
arrWeightyr = []
arrYear = []

for i in range(58):
    sum = 0
    for j in range(0,11):
        sum += float(arrWeight[12*i + j])
    arrWeightyr.append(sum)
    arrYear.append(i)
#arrYear
plt.scatter(arrYear, arrWeightyr)
plt.show()
#Data Parse Complete
#-------------------------------------------------------------#
#Prepare Training and validation data

x_train = []
y_train = []
x_val = []
y_val = []

for i in arrMonth:
    if i % 2 == 0:
        x_train.append(i)
        y_train.append(arrWeight[i])
    else:
        x_val.append(i)
        y_val.append(arrWeight[i])


model = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(10, input_shape=[1], activation='relu'),
    tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001),
    tf.keras.layers.Dense(90, activation='relu'),
    tf.keras.layers.Dense(1)
    ])

#sgd = tf.optimizers.SGD(lr=.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=["accuracy"])

model.fit(x_train, y_train, epochs=4000, validation_data=(x_val,y_val))


print("enter in a year to predict")
#inval = (int(input()) - 1960) * 12
#prediction = model.predict([inval])
#print("prediction: ",float(prediction[0][0]) * 12)
#print("actual: ", arrWeightyr[inval])
#def ytm(x):
#    return((x-1960) * 12)
#print(model.predict([ytm(2018), ytm(1950), ytm(1980)]))

arrx = []
arry = []

for i in range(0,700):
    arrx.append(i)
    prediction = model.predict([i])
    arry.append(prediction[0][0])
plt.plot(arrx,arry)
plt.show()
