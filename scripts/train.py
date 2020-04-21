"""
Defines ResNet architectures and train on Cifar-10

Block/Stack decomposition is heavily inspired from official TF/Keras implementation of ResNet
"""
import pickle
from pathlib import Path

import click
import tensorflow as tf
from loguru import logger

from scripts.resnet import ResnetBuilder, BATCH_NORM_NAME, EVONORM_S0_NAME, EVONORM_B0_NAME


kwargs = {
    "backend": tf.keras.backend,
    "layers": tf.keras.layers,
    "utils": tf.keras.utils,
    "models": tf.keras.models,
}

@click.command()
@click.option("--dataset_name", default="cifar10", help="tf.keras dataset to use")
@click.option("--number_of_experiments", default=1, help="Number of experiments to run")
@click.option("--logdir", default="logs", help="Logs directory")
def launch_training(dataset_name, number_of_experiments, logdir):
    dataset = getattr(tf.keras.datasets, dataset_name)
    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    x_train = x_train.astype("float32") / 255.
    x_test = x_test.astype("float32") / 255.
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    input_shape = x_train.shape[1:]
    num_classes = y_train.shape[1]
    batch_size = 64
    epochs = 45

    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        horizontal_flip=True,
        zoom_range=0.1,
    )

    resnet_bn_base_path = Path(f"{logdir}/{dataset_name}/resnet")
    resnet_evonorm_b0_base_path = Path(f"{logdir}/{dataset_name}/resnet_evonorm_b0")
    resnet_evonorm_s0_base_path = Path(f"{logdir}/{dataset_name}/resnet_evonorm_s0")

    for experiment_index in range(number_of_experiments):
        resnet_bn_experiment_path = resnet_bn_base_path / str(experiment_index)
        resnet_evonorm_b0_experiment_path = resnet_evonorm_b0_base_path / str(experiment_index)
        resnet_evonorm_s0_experiment_path = resnet_evonorm_s0_base_path / str(experiment_index)

        evonorm_b0_model = ResnetBuilder.build_resnet_18(input_shape, num_classes, block_fn_name=EVONORM_B0_NAME)
        evonorm_b0_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        history = evonorm_b0_model.fit(
            data_generator.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=1, #len(x_train) // batch_size,
            validation_data=(x_test, y_test),
            epochs=epochs,
            callbacks=[tf.keras.callbacks.TensorBoard(resnet_evonorm_b0_experiment_path)],
            shuffle=True,
        )
        with open(resnet_bn_experiment_path / "history", 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        evonorm_model = ResnetBuilder.build_resnet_18(input_shape, num_classes, block_fn_name=EVONORM_S0_NAME)
        evonorm_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        evonorm_model.fit(
            data_generator.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=1, #len(x_train) // batch_size,
            validation_data=(x_test, y_test),
            epochs=epochs,
            callbacks=[tf.keras.callbacks.TensorBoard(resnet_evonorm_s0_experiment_path)],
            shuffle=True,
        )
        with open(resnet_evonorm_s0_experiment_path / "history", 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        model = ResnetBuilder.build_resnet_18(input_shape, num_classes, block_fn_name=BATCH_NORM_NAME)
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        model.fit(
            data_generator.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=1, #len(x_train) // batch_size,
            validation_data=(x_test, y_test),
            epochs=epochs,
            callbacks=[tf.keras.callbacks.TensorBoard(resnet_bn_experiment_path)],
            shuffle=True,
        )
        with open(resnet_evonorm_b0_experiment_path / "history", 'wb') as file_pi:
            pickle.dump(history.history, file_pi)


if __name__ == "__main__":
    launch_training()
