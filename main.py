# natural language processing with rnn and attention
# a common approach for nlp tasks are to use rnn, first we will use stateless rnn
import os.path

import keras

# stateful rnn
# preserves the hidden state between training iterations and continues reading where it left off,
# allowing it to learn longer patterns
# capable of performing neural machine translation
# seq2seq api provided by tensorflow addons project

# we will look at attention mechanisms,
# and we will see how to boost the performance of an rnn based encoder and decoder

# Generating shakespearean text using a character RNN
#
import numpy as np
import tensorflow as tf
import joblib

train_model_cache_download = joblib.Memory('./tmp/nlp/train_model_cache_download')


@train_model_cache_download.cache
def getDataImdbFrom():
    return keras.datasets.imdb.load_data()


class TrainingExample:
    def __init__(self):
        print("ok")

        # shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        # filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)
        # with open(filepath) as f:
        #     shakespeare_text = f.read()
        # print(shakespeare_text[:148])
        # print("".join(sorted(set(shakespeare_text.lower()))))

        # encoding every character as an integer
        # one options is to create a custom preprocessing layer
        # keras's tokenizer
        # char level  true, to get character-level encoding rather than the default word-level encoding.
        # print("".join(sorted(set(shakespeare_text.lower()))))
        # tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
        # tokenizer.fit_on_texts(shakespeare_text)
        # encode a sentence to list of char ids and back and it tells use how many distinct char
        # print(tokenizer.texts_to_sequences(["First"]))
        # print(tokenizer.sequences_to_texts([[20, 6, 9, 8, 3]]))

        # max_id = len(tokenizer.word_index)  # number of distinct characters
        # print(max_id)  # number of distinct chars
        # dataset_size = tokenizer.document_count  # total number of characters
        # print(dataset_size)  # total number of chars

        # it is crucial to avoid any overlap between the training set.
        # [encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1
        # train_size = dataset_size * 90 // 100
        # dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])

        # print(train_size)
        # print(dataset)

        # chopping the sequential dataset into a multiple windows.
        # training set has million chars, instead of training it in a single layer, we can use
        # window method to convert it into many smaller windows of a text
        # n_steps = 100
        # window_length = n_steps + 1  # target = input shifted 1 character ahead
        # dataset = dataset.window(window_length, shift=1, drop_remainder=True)
        # to get the largest training set we use shift = 1 , first windows will have 0 to 100
        # second one, 1, to 101 and so on.
        # flatting the nested dataset with flat_map()
        # dataset = dataset.flat_map(lambda window: window.batch(window_length))

        # each windows has same length, dataset contains consecutive windows of 101 each.
        # stateless rnn
        # np.random.seed(42)
        # tf.random.set_seed(42)
        # batch_size = 32
        # dataset = dataset.shuffle(10000).batch(batch_size)
        # dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
        # dataset = dataset.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
        # dataset = dataset.prefetch(1)
        # for X_batch, Y_batch in dataset.take(1):
        #     print(X_batch.shape, Y_batch.shape)
        #
        # # building and training the char-rnn model
        # # to predict previous 100 chars, we can use an rnn with 2 gru of 128 units each and
        # # %20 drop-out and hidden states
        # #  dense layer has to have 39 units because there are 39 units
        #
        # model = None
        # if not os.path.isfile("models/nlp_first_model.h5"):
        #     model = keras.models.Sequential([
        #         keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id],
        #                          dropout=0.2, recurrent_dropout=0.2),
        #         keras.layers.GRU(128, return_sequences=True,
        #                          dropout=0.2, recurrent_dropout=0.2),
        #         keras.layers.TimeDistributed(keras.layers.Dense(max_id,
        #                                                         activation="softmax"))
        #     ])
        #     model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
        #     history = model.fit(dataset, epochs=1)
        #     model.save("models/nlp_first_model.h5")
        # else:
        #     model = keras.models.load_model("models/nlp_first_model.h5")
        # print(model.summary())
        #
        # X_new = self.preprocess( max_id, tokenizer, ["how are yo"])
        # Y_pred = model.predict_classes(X_new)
        # print(tokenizer.sequences_to_texts(Y_pred + 1)[0][-1]) # u
        #
        # # Generating fake shakespearean text
        # # to generate new text using the char-rnn model,
        # print(self.complete_text(tokenizer, model, max_id, "t", temperature=0.2))
        # print(self.complete_text(tokenizer, model, max_id, "w", temperature=1.))
        # print(self.complete_text(tokenizer, model, max_id, "w", temperature=2.))

        # stateful rnn
        # model can learn long-term patterns despite only backpropagation through short
        # sequences.

        # tf.random.set_seed(42)
        # dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])
        # dataset = dataset.window(window_length, shift=n_steps, drop_remainder=True)
        # dataset = dataset.flat_map(lambda window: window.batch(window_length))
        # dataset = dataset.batch(1)
        # dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
        # dataset = dataset.map(
        #     lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
        # dataset = dataset.prefetch(1)
        # batch_size = 32
        # encoded_parts = np.array_split(encoded[:train_size], batch_size)
        # datasets = []
        # for encoded_part in encoded_parts:
        #     dataset = tf.data.Dataset.from_tensor_slices(encoded_part)
        #     dataset = dataset.window(window_length, shift=n_steps, drop_remainder=True)
        #     dataset = dataset.flat_map(lambda window: window.batch(window_length))
        #     datasets.append(dataset)
        # dataset = tf.data.Dataset.zip(tuple(datasets)).map(lambda *windows: tf.stack(windows))
        # dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
        # dataset = dataset.map(
        #     lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
        # dataset = dataset.prefetch(1)
        #
        # model = keras.models.Sequential([
        #     keras.layers.GRU(128, return_sequences=True, stateful=True,
        #                      # dropout=0.2, recurrent_dropout=0.2,
        #                      dropout=0.2,
        #                      batch_input_shape=[batch_size, None, max_id]),
        #     keras.layers.GRU(128, return_sequences=True, stateful=True,
        #                      # dropout=0.2, recurrent_dropout=0.2),
        #                      dropout=0.2),
        #     keras.layers.TimeDistributed(keras.layers.Dense(max_id,
        #                                                     activation="softmax"))
        # ])
        #
        # class ResetStatesCallback(keras.callbacks.Callback):
        #     def on_epoch_begin(self, epoch, logs):
        #         self.model.reset_states()
        #
        # model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
        # history = model.fit(dataset, epochs=5,
        #                     callbacks=[ResetStatesCallback()])
        #
        # stateless_model = keras.models.Sequential([
        #     keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id]),
        #     keras.layers.GRU(128, return_sequences=True),
        #     keras.layers.TimeDistributed(keras.layers.Dense(max_id,
        #                                                     activation="softmax"))
        # ])
        # stateless_model.build(tf.TensorShape([None, None, max_id]))
        # stateless_model.set_weights(model.get_weights())
        # model = stateless_model
        # tf.random.set_seed(42)
        #
        # print(self.complete_text(tokenizer, model, max_id, "t"))

        # sentiment analysis
        (X_train, y_train), (X_test, y_test) = getDataImdbFrom()
        print(X_train[0][:10])

        # 0 1 2 are special; they represent the padding token, start of a sequence
        # to decode our integers
        # word_index = keras.datasets.imdb.get_word_index()
        # id_to_word = {id_ + 3: word for word, id_ in word_index.items()}
        # for id_, token in enumerate(("<pad>", "<sos>", "<unk>")):
        #    id_to_word[id_] = token
        # print(" ".join([id_to_word[id_] for id_ in X_train[0][:]]))
        s


    def complete_text(self, tokenizer, model, max_id, text, n_chars=50, temperature=1.):
        for _ in range(n_chars):
            text += self.next_char(tokenizer, model, max_id, text, temperature)
        return text

    def next_char(self, tokenizer, model, max_id, text, temperature=1.):
        X_new = self.preprocess(max_id, tokenizer, [text])
        y_proba = model.predict(X_new)[0, -1:, :]
        rescaled_logits = tf.math.log(y_proba) / temperature
        char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
        return tokenizer.sequences_to_texts(char_id.numpy())[0]

    def preprocess(self, max_id, tokenizer, texts):
        X = np.array(tokenizer.texts_to_sequences(texts)) - 1
        return tf.one_hot(X, max_id)


def program1():
    TrainingExample()


if __name__ == '__main__':
    program1()
