import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation

df = pd.read_csv('dataset/data.csv')
print(df.head())
print(df.info())

# Let's create a heatmap that will show the correlation between features
# of data
# sns.heatmap(df.iloc[:, 3:-1].corr(), annot=True, square=False)
# plt.show()


# Let’s create a pairplot that will show us the complete relationship
# between radius mean, texture mean, perimeter mean, area mean and
# smoothness mean on the basis of diagnosis type.
# sns.pairplot(df, hue='diagnosis', palette='coolwarm', vars=['radius_mean', 'texture_mean', 'perimeter_mean',
# 'area_mean', 'smoothness_mean'])
# plt.show()


# count the number of empty values in each column:
print(df.isna().sum())

# drop the columns with all the missing values:
df.dropna(axis=1, inplace=True)

# Get the count of the number of Malignant(M) or Benign(B) cells
print(df['diagnosis'].value_counts())

# Let's get a countplot of those values for better visualisation
# sns.countplot(df['diagnosis'], label='count')
# plt.show()

# Let's remove and add again column that we need to predict, diagnosis, so that it
# will be last columns
df['diagnosis'] = df.pop('diagnosis')
# Let's create X and y from our dataset
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Now we will convert our text (B and M) to integers (0 and 1)
# using LabelEncoder provided by sklearn library.
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Now we will standardize test set and train set with StandardScaler
# ss = StandardScaler()
# X_train = ss.fit_transform(X_train)
# X_test = ss.transform(X_test)
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

# In this step, we will be creating a Sequential model with the help of
# TensorFlow and Keras. In the model we have created three Dense layers in
# which one is the input layer with 128 hidden layers and the activation
# function is relu, and the second layer consists of 64 hidden layers
# with activation function relu and the third layer is final output
# layer and activation function is sigmoid.
model = Sequential()
model.add(layer=Dense(units=128, activation='relu'))
model.add(layer=Dropout(rate=0.5))
model.add(layer=Dense(units=64, activation='relu'))
model.add(layer=Dropout(rate=0.5))
model.add(layer=Dense(units=1))
model.add(layer=Activation('sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, verbose=1, steps_per_epoch=7, epochs=100, batch_size=64,
                    validation_data=(X_test, y_test))
model.summary()

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = np.arange(1, len(loss) + 1)
fig, ax = plt.subplots(1, 2, figsize=(8, 8))
ax[0].set_title('loss')
ax[0].plot(epochs, loss, color='red', label='Training Loss')
ax[0].plot(epochs, val_loss, color='blue', label='Validation Loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
ax[1].set_title('loss')
ax[1].plot(epochs, acc, color='g', label='Training Acc')
ax[1].plot(epochs, val_acc, color='b', label='Validation Acc')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')
fig.legend()
plt.show()

y_pred = model.predict(X_test) > 0.5
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
plt.show()
