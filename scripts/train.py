"""
Defines ResNet architectures and train on Cifar-10

Block/Stack decomposition is heavily inspired from official TF/Keras implementation of ResNet
"""
import tensorflow as tf

from scripts.resnet import ResnetBuilder, BATCH_NORM_NAME, EVONORM_S0_NAME


INPUT_SHAPE = (32, 32, 3)


kwargs = {
    "backend": tf.keras.backend,
    "layers": tf.keras.layers,
    "utils": tf.keras.utils,
    "models": tf.keras.models,
}


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.
    x_test = x_test.astype("float32") / 255.
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    num_classes = 10
    batch_size = 64
    epochs = 30

    evonorm_model = ResnetBuilder.build_resnet_18(INPUT_SHAPE, num_classes, block_fn_name=EVONORM_S0_NAME)
    evonorm_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    evonorm_model.fit(
        x_train,
        y=y_train,
        validation_data=(x_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.TensorBoard("logs/resnet_evonorm"),
            tf.keras.callbacks.ModelCheckpoint("models/resnet_evonorm", monitor="val_loss", save_best_only=True)
        ],
    )

    evonorm_b0_model = ResnetBuilder.build_resnet_18(INPUT_SHAPE, num_classes, block_fn_name=EVONORM_B0_NAME)
    evonorm_b0_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    evonorm_b0_model.fit(
        x_train,
        y=y_train,
        validation_data=(x_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.TensorBoard("logs/resnet_evonorm"),
            tf.keras.callbacks.ModelCheckpoint("models/resnet_evonorm", monitor="val_loss", save_best_only=True)
        ],
    )

    model = ResnetBuilder.build_resnet_18(INPUT_SHAPE, num_classes, block_fn_name=BATCH_NORM_NAME)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(
        x_train,
        y=y_train,
        validation_data=(x_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.TensorBoard("logs/resnet"),
            tf.keras.callbacks.ModelCheckpoint("models/resnet", monitor="val_loss", save_best_only=True)
        ],
    )

