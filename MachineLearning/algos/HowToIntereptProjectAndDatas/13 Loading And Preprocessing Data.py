# loading and preprocessing data with tensorflow
# in large data sets, tensorflow cares it with multithreading, queuing, batching
# and prefetching with data api

# Data Api
# ingesting a large dataset and preprocessing it efficiently can be complex
# this api makes it simply

# splitting a large dataset into multiple files makes it possible to shuffle
# it at a coarse level before shuffling it at a finer level using shuffling buffer


import os

import keras
import numpy as np
import pandas as pd
# it is efficient based on protocol buffers.

# Gradient Descent works best when the instances in the training sets are independent and identically distributed
#

import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from person_pb2 import Person
from sklearn.datasets import load_sample_images


class TrainingExample:
    def __init__(self):
        print("ok")
        # tf transform



    def tf_record(self):
        # tfRecord format
        with tf.io.TFRecordWriter("my_data.tfrecord") as f:
            f.write(b"This is the first record")
            f.write(b"And this is the second record")

        filepaths = ["my_data.tfrecord"]
        dataset = tf.data.TFRecordDataset(filepaths)
        for item in dataset:
            print(item)

        # compress them
        options = tf.io.TFRecordOptions(compression_type="GZIP")
        with tf.io.TFRecordWriter("my_compressed.tfrecord", options) as f:
            f.write(b"This is the first record")
            f.write(b"And this is the second record")
        dataset = tf.data.TFRecordDataset(["my_compressed.tfrecord"],
                                          compression_type="GZIP")
        for item in dataset:
            print(item)

        person = Person(name="Al", id=123, email=["a@b.com"])  # create a Person
        print(person)  # display the Person
        person.name = "fuc"
        print(person.name)
        print(person.email[0])
        person.email.append("c@d.com")
        print(person.email)
        serializedToString = person.SerializeToString()
        print(serializedToString)
        person2 = Person()  # create a new Person
        print(person2.ParseFromString(serializedToString))

        person_tf = tf.io.decode_proto(
            bytes=serializedToString,
            message_type="Person",
            field_names=["name", "id", "email"],
            output_types=[tf.string, tf.int32, tf.string],
            descriptor_source="person.desc")

        print(person_tf.values)

    def dataApi(self):
        housing = fetch_california_housing()
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            housing.data, housing.target.reshape(-1, 1), random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full, y_train_full, random_state=42)

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_mean = scaler.mean_
        X_std = scaler.scale_

        train_data = np.c_[X_train, y_train]
        valid_data = np.c_[X_valid, y_valid]
        test_data = np.c_[X_test, y_test]
        header_cols = housing.feature_names + ["MedianHouseValue"]
        header = ",".join(header_cols)

        train_filepaths = self.save_to_multiple_csv_files(train_data, "train", header, n_parts=20)
        valid_filepaths = self.save_to_multiple_csv_files(valid_data, "valid", header, n_parts=10)
        test_filepaths = self.save_to_multiple_csv_files(test_data, "test", header, n_parts=10)

        # print(pd.read_csv(train_filepaths[0]).head())
        # print(train_filepaths)

        # building an input pipeline
        filepath_dataset = tf.data.Dataset.list_files(train_filepaths, seed=42)
        # for filepath in filepath_dataset:
        #     print(filepath)

        n_readers = 5
        dataset = filepath_dataset.interleave(
            lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
            cycle_length=n_readers)
        # for line in dataset.take(5):
        #     print(line.numpy())

        n_inputs = 8  # X_train.shape[-1]

        @tf.function
        def preprocess(line):
            defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
            fields = tf.io.decode_csv(line, record_defaults=defs)
            x = tf.stack(fields[:-1])
            y = tf.stack(fields[-1:])
            return (x - X_mean) / X_std, y

        # print(preprocess(b'4.2083,44.0,5.3232,0.9171,846.0,2.3370,37.47,-122.2,2.782'))

        def csv_reader_dataset(filepaths, repeat=1, n_readers=5,
                               n_read_threads=None, shuffle_buffer_size=10000,
                               n_parse_threads=5, batch_size=32):
            dataset = tf.data.Dataset.list_files(filepaths).repeat(repeat)
            dataset = dataset.interleave(
                lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
                cycle_length=n_readers, num_parallel_calls=n_read_threads)
            dataset = dataset.shuffle(shuffle_buffer_size)
            dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
            dataset = dataset.batch(batch_size)
            return dataset.prefetch(1)

        tf.random.set_seed(42)

        train_set = csv_reader_dataset(train_filepaths, batch_size=3)
        # for X_batch, y_batch in train_set.take(2):
        #     print("X =", X_batch)
        #     print("y =", y_batch)
        #     print()

        # using the dataset with tf.keras
        train_set = csv_reader_dataset(train_filepaths, repeat=None)
        valid_set = csv_reader_dataset(valid_filepaths)
        test_set = csv_reader_dataset(test_filepaths)

        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)

        model = keras.models.Sequential([
            keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
            keras.layers.Dense(1),
        ])
        model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1e-3))
        batch_size = 32
        model.fit(train_set, steps_per_epoch=len(X_train) // batch_size, epochs=5,
                  validation_data=valid_set)
        print(model.evaluate(test_set, steps=len(X_test) // batch_size))

        new_set = test_set.map(lambda X, y: X)  # we could instead just pass test_set, Keras would ignore the labels
        X_new = X_test
        print(model.predict(new_set, steps=len(X_new) // batch_size))

        optimizer = keras.optimizers.Nadam(learning_rate=0.01)
        loss_fn = keras.losses.mean_squared_error

        n_epochs = 5
        batch_size = 32
        n_steps_per_epoch = len(X_train) // batch_size
        total_steps = n_epochs * n_steps_per_epoch
        global_step = 0
        for X_batch, y_batch in train_set.take(total_steps):
            global_step += 1
            print("\rGlobal step {}/{}".format(global_step, total_steps), end="")
            with tf.GradientTape() as tape:
                y_pred = model(X_batch)
                main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
                loss = tf.add_n([main_loss] + model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)
        optimizer = keras.optimizers.Nadam(learning_rate=0.01)
        loss_fn = keras.losses.mean_squared_error

        @tf.function
        def train(model, n_epochs, batch_size=32,
                  n_readers=5, n_read_threads=5, shuffle_buffer_size=10000, n_parse_threads=5):
            train_set = csv_reader_dataset(train_filepaths, repeat=n_epochs, n_readers=n_readers,
                                           n_read_threads=n_read_threads, shuffle_buffer_size=shuffle_buffer_size,
                                           n_parse_threads=n_parse_threads, batch_size=batch_size)
            for X_batch, y_batch in train_set:
                with tf.GradientTape() as tape:
                    y_pred = model(X_batch)
                    main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
                    loss = tf.add_n([main_loss] + model.losses)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train(model, 5)

        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)
        optimizer = keras.optimizers.Nadam(learning_rate=0.01)
        loss_fn = keras.losses.mean_squared_error

        @tf.function
        def train(model, n_epochs, batch_size=32,
                  n_readers=5, n_read_threads=5, shuffle_buffer_size=10000, n_parse_threads=5):
            train_set = csv_reader_dataset(train_filepaths, repeat=n_epochs, n_readers=n_readers,
                                           n_read_threads=n_read_threads, shuffle_buffer_size=shuffle_buffer_size,
                                           n_parse_threads=n_parse_threads, batch_size=batch_size)
            n_steps_per_epoch = len(X_train) // batch_size
            total_steps = n_epochs * n_steps_per_epoch
            global_step = 0
            for X_batch, y_batch in train_set.take(total_steps):
                global_step += 1
                if tf.equal(global_step % 100, 0):
                    tf.print("\rGlobal step", global_step, "/", total_steps)
                with tf.GradientTape() as tape:
                    y_pred = model(X_batch)
                    main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
                    loss = tf.add_n([main_loss] + model.losses)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train(model, 5)

        # short description of each method
        for m in dir(tf.data.Dataset):
            if not (m.startswith("_") or m.endswith("_")):
                func = getattr(tf.data.Dataset, m)
                if hasattr(func, "__doc__"):
                    print("● {:21s}{}".format(m + "()", func.__doc__.split("\n")[0]))

    def ChainingTransformations(self):
        X = tf.range(10)
        datasets = tf.data.Dataset.from_tensor_slices(X)

        # print(datasets)

        # for _ in datasets:
        #     print(_)

        # datasets = datasets.repeat(3).batch(7)
        # for _ in datasets:
        #     print(_)

        datasets = datasets.map(lambda x: x * 2)
        datasets = datasets.filter(lambda x: x < 10)

        for _ in datasets.take(3):
            print(_)

    def save_to_multiple_csv_files(self, data, name_prefix, header=None, n_parts=10):
        housing_dir = os.path.join("datasets", "housing")
        os.makedirs(housing_dir, exist_ok=True)
        path_format = os.path.join(housing_dir, "my_{}_{:02d}.csv")

        filepaths = []
        m = len(data)
        for file_idx, row_indices in enumerate(np.array_split(np.arange(m), n_parts)):
            part_csv = path_format.format(name_prefix, file_idx)
            filepaths.append(part_csv)
            with open(part_csv, "wt", encoding="utf-8") as f:
                if header is not None:
                    f.write(header)
                    f.write("\n")
                for row_idx in row_indices:
                    f.write(",".join([repr(col) for col in data[row_idx]]))
                    f.write("\n")
        return filepaths


def program1():
    TrainingExample()


if __name__ == '__main__':
    program1()
