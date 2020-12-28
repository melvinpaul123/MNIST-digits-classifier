import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score

(X_train1, Y_train1), (X_test, Y_test) = mnist.load_data() #shape is 28*28
X_train, X_val, Y_train, Y_val = train_test_split(X_train1, Y_train1, test_size=0.1, random_state=42)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_val = X_val.astype('float32')
X_train = X_train/255
X_test = X_test/255
X_val = X_val/255


print("Total no. of Classes = 10")
print("Loading completed")
print("No. of Training Images = %d, No. of Training Labels= %d" %(X_train.shape[0], Y_train.shape[0]))
print("No. of Testing Images = %d, No. of Testing Labels= %d" %(X_test.shape[0], Y_test.shape[0]))
print("No. of Validation Images = %d, No. of Validation Labels= %d" %(X_val.shape[0], Y_val.shape[0]))

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 10
mc = ModelCheckpoint('model_best.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=10)
history = model.fit(X_train, Y_train, batch_size=32, epochs=epochs, validation_data=(X_test, Y_test),callbacks=[early_stop, mc])
best_model= load_model('model_best.h5')
print("--> Saving model as .h5")
best_model.save("mnist_classifier.h5")

#plotting graphs for accuracy 
plt.figure(0)
plt.plot(history.history['accuracy'], label='training acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

#testing accuracy on test dataset
pred = best_model.predict_classes(X_val)
#Accuracy with the test data
print("Accuracy Score = %f" %((accuracy_score(Y_val, pred))*100)+"%")