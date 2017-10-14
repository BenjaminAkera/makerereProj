from ENet import enet, train_utils
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def train_val_generators(original_dir, annotated_dir, target_size=(256,256), classes=2, batch_size=16, train_size=.8):
	random_state = 42
	X,y,masks = train_utils.get_samples(original_dir, annotated_dir, target_size, classes=2)

	traingen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0.,
        width_shift_range=0.,
        height_shift_range=0.,
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=True)

	valgen = ImageDataGenerator(featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0.,
        width_shift_range=0.,
        height_shift_range=0.,
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=True)

        if train_size < 1:
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)
                return traingen.flow(X_train,y_train, batch_size=batch_size, shuffle=True), valgen.flow(X_test,y_test, batch_size=batch_size, shuffle=True)
        else:
                return traingen.flow(X,y, batch_size=batch_size, shuffle=True),  None
