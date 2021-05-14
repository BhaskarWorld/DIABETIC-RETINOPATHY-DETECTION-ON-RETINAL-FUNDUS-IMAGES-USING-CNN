from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from matplotlib import pyplot
from keras.optimizers import SGD
from keras.regularizers import l2

# Configuration
img_width, img_height = 256, 256

train_data_dir = 'H:/project/prog_data/training/'
validation_data_dir = 'H:/project/prog_data/validation/'
test_data_dir = 'H:/project/prog_data/testing/'
nb_train_samples = 350
nb_validation_samples = 5265
epochs = 15
batch_size = 70

# Setting image Shape
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Building the vanilla CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('softmax'))

# Learning Algorithm
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Compiling the model
model.compile(loss="categorical_crossentropy",
              optimizer=sgd,
              metrics=['categorical_accuracy', 'accuracy'])
model.summary()  # summary of the model

# Preparing pipeline that allows data to flow for training process
datagen = ImageDataGenerator()


train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    classes=['lvl0', 'lvl1', 'lvl2', 'lvl3', 'lvl4']
)

validation_generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    classes=['lvl0', 'lvl1', 'lvl2', 'lvl3', 'lvl4']
)
test_generator = datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    classes=['lvl0', 'lvl1', 'lvl2', 'lvl3', 'lvl4']
)

# Starting the trainning Process
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size
)


# Give Graphical visualization of the training phase
f1 = pyplot.figure()
pyplot.plot(history.history['accuracy'], label='train_acc')
pyplot.plot(history.history['val_accuracy'], label='val_acc')
f2 = pyplot.figure()
pyplot.plot(history.history['loss'], label='train_loss')
pyplot.plot(history.history['val_loss'], label='val_loss')
pyplot.legend()
pyplot.show()


# Evaluating the result with the Training set
print('\n# Evaluate on test data')
results = model.evaluate(test_generator)
print('test loss, test acc:', results)


# saving the weight
model.save_weights('H:/model_saved_full.h5', overwrite=True)
