"""
Defines ResNet architectures and train on Cifar-10

Block/Stack decomposition is heavily inspired from official TF/Keras implementation of ResNet
"""
import tensorflow as tf

from scripts.resnet import ResnetBuilder, BATCH_NORM_NAME, EVONORM_S0_NAME, EVONORM_B0_NAME


INPUT_SHAPE = (32, 32, 3)


kwargs = {
    "backend": tf.keras.backend,
    "layers": tf.keras.layers,
    "utils": tf.keras.utils,
    "models": tf.keras.models,
}


if __name__ == "__main__":
    dataset_name = "CIFAR10"
    is_dataset_cifar_10 = dataset_name == "CIFAR10"

    dataset = tf.keras.datasets.cifar10 if is_dataset_cifar_10 else tf.keras.datasets.cifar100
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    num_classes = 10 if is_dataset_cifar_10 else 100


    x_train = x_train.astype("float32") / 255.
    x_test = x_test.astype("float32") / 255.
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

    batch_size = 64
    epochs = 45

    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        horizontal_flip=True,
        zoom_range=0.1,
    )

    evonorm_b0_model = ResnetBuilder.build_resnet_18(INPUT_SHAPE, num_classes, block_fn_name=EVONORM_B0_NAME)
    evonorm_b0_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    evonorm_b0_model.fit(
        data_generator.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(x_train) // batch_size,
        validation_data=(x_test, y_test),
        epochs=epochs,
        callbacks=[tf.keras.callbacks.TensorBoard(f"logs/{dataset_name}/resnet_evonorm_b0")],
        shuffle=True,
    )

    evonorm_model = ResnetBuilder.build_resnet_18(INPUT_SHAPE, num_classes, block_fn_name=EVONORM_S0_NAME)
    evonorm_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    evonorm_model.fit(
        data_generator.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(x_train) // batch_size,
        validation_data=(x_test, y_test),
        epochs=epochs,
        callbacks=[tf.keras.callbacks.TensorBoard(f"logs/{dataset_name}/resnet_evonorm_s0")],
        shuffle=True,
    )

    model = ResnetBuilder.build_resnet_18(INPUT_SHAPE, num_classes, block_fn_name=BATCH_NORM_NAME)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(
        data_generator.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(x_train) // batch_size,
        validation_data=(x_test, y_test),
        epochs=epochs,
        callbacks=[tf.keras.callbacks.TensorBoard(f"logs/{dataset_name}/resnet")],
        shuffle=True,
    )

