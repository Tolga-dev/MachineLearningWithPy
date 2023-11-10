# natural language processing with rnn and attention
# a common approach for nlp tasks are to use rnn, first we will use stateless rnn


# stateful rnn
# preserves the hidden state between training iterations and continues reading where it left off,
# allowing it to learn longer patterns
# capable of performing neural machine translation
# seq2seq api provided by tensorflow addons project

# we will look at attention mechanisms,
# and we will see how to boost the performance of an rnn based encoder and decoder

# Generating shakespearean text using a character RNN

# Exercises
# stateful rnn and stateless rnn;
# stateless rnn can ony capture patterns whose length is less thar or equal to, size of windows
# conversely, stateful rnn, can capture longer-term patterns but its harder

# why do people use encoder-decoder rnn rather than plain sequence to sequence rnns for automatic translation
# it is much better to read the whole sentence first and then translate it


import numpy as np
import tensorflow as tf
import joblib
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from collections import Counter
import tensorflow_addons as tfa
import os.path
from collections import Counter
import keras

train_model_cache_download = joblib.Memory('./tmp/nlp/train_model_cache_download')


@train_model_cache_download.cache
def getDataImdbFrom():
    return keras.datasets.imdb.load_data()


class TrainingExample:
    def __init__(self):
        print("ok")

        shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)
        with open(filepath) as f:
            shakespeare_text = f.read()
        # print(shakespeare_text[:148])
        # print("".join(sorted(set(shakespeare_text.lower()))))

        # encoding every character as an integer
        # one options is to create a custom preprocessing layer
        # keras's tokenizer
        # char level  true, to get character-level encoding rather than the default word-level encoding.
        # print("".join(sorted(set(shakespeare_text.lower()))))
        tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
        tokenizer.fit_on_texts(shakespeare_text)
        # encode a sentence to list of char ids and back and it tells use how many distinct char
        # print(tokenizer.texts_to_sequences(["First"]))
        # print(tokenizer.sequences_to_texts([[20, 6, 9, 8, 3]]))

        max_id = len(tokenizer.word_index)  # number of distinct characters
        # print(max_id)  # number of distinct chars
        dataset_size = tokenizer.document_count  # total number of characters
        # print(dataset_size)  # total number of chars

        # it is crucial to avoid any overlap between the training set.
        [encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1
        train_size = dataset_size * 90 // 100
        dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])

        # print(train_size)
        # print(dataset)

        # chopping the sequential dataset into a multiple windows.
        # training set has million chars, instead of training it in a single layer, we can use
        # window method to convert it into many smaller windows of a text
        n_steps = 100
        window_length = n_steps + 1  # target = input shifted 1 character ahead
        dataset = dataset.window(window_length, shift=1, drop_remainder=True)
        # to get the largest training set we use shift = 1 , first windows will have 0 to 100
        # second one, 1, to 101 and so on.
        # flatting the nested dataset with flat_map()
        dataset = dataset.flat_map(lambda window: window.batch(window_length))

        # each windows has same length, dataset contains consecutive windows of 101 each.
        # stateless rnn
        np.random.seed(42)
        tf.random.set_seed(42)
        batch_size = 32
        dataset = dataset.shuffle(10000).batch(batch_size)
        dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
        dataset = dataset.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
        dataset = dataset.prefetch(1)
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

        tf.random.set_seed(42)
        dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])
        dataset = dataset.window(window_length, shift=n_steps, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(window_length))
        dataset = dataset.batch(1)
        dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
        dataset = dataset.map(
            lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
        dataset = dataset.prefetch(1)
        batch_size = 32
        encoded_parts = np.array_split(encoded[:train_size], batch_size)
        datasets = []
        for encoded_part in encoded_parts:
            dataset = tf.data.Dataset.from_tensor_slices(encoded_part)
            dataset = dataset.window(window_length, shift=n_steps, drop_remainder=True)
            dataset = dataset.flat_map(lambda window: window.batch(window_length))
            datasets.append(dataset)
        dataset = tf.data.Dataset.zip(tuple(datasets)).map(lambda *windows: tf.stack(windows))
        dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
        dataset = dataset.map(
            lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
        dataset = dataset.prefetch(1)

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

        # (X_train, y_train), (X_test, y_test) = getDataImdbFrom()
        # print(X_train[0][:10])

        # 0 1 2 are special; they represent the padding token, start of a sequence
        # to decode our integers
        #
        # word_index = keras.datasets.imdb.get_word_index()
        # id_to_word = {id_ + 3: word for word, id_ in word_index.items()}
        # for id_, token in enumerate(("<pad>", "<sos>", "<unk>")):
        #    id_to_word[id_] = token
        # print(" ".join([id_to_word[id_] for id_ in X_train[0][:]]))

        # Sub-word regularization, improving neural network translation
        # using byte pair encoding.
        datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
        train_size = info.splits["train"].num_examples
        test_size = info.splits["test"].num_examples
        # print(train_size)
        # print(info)
        # print(datasets)

        # preprocessing function, we truncated that end of reviews
        for X_batch, y_batch in datasets["train"].batch(2).take(1):
            for review, label in zip(X_batch.numpy(), y_batch.numpy()):
                print("Review:", review.decode("utf-8")[:200], "...")
                print("Label:", label, "= Positive" if label else "= Negative")
                print()

        def preprocess(X_batch, y_batch):
            X_batch = tf.strings.substr(X_batch, 0, 300)
            X_batch = tf.strings.regex_replace(X_batch, rb"<br\s*/?>", b" ")
            X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
            X_batch = tf.strings.split(X_batch)
            return X_batch.to_tensor(default_value=b"<pad>"), y_batch

        # print(preprocess(X_batch, y_batch))

        vocabulary = Counter()
        for X_batch, y_batch in datasets["train"].batch(32).map(preprocess):
            for review in X_batch:
                vocabulary.update(list(review.numpy()))

        # print(vocabulary.most_common()[:3])
        # print(len(vocabulary))
        vocab_size = 10000
        truncated_vocabulary = [word for word, count in vocabulary.most_common()[:vocab_size]]
        #
        word_to_id = {word: index for index, word in enumerate(truncated_vocabulary)}
        # for word in b"This movie was faaaaaantastic".split():
        #     print(word_to_id.get(word) or vocab_size)

        words = tf.constant(truncated_vocabulary)
        word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
        vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
        num_oov_buckets = 1000
        table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)

        # print(table.lookup(tf.constant([b"This movie was faaaaaantastic".split()])))

        def encode_words(X_batch, y_batch):
            return table.lookup(X_batch), y_batch

        train_set = datasets["train"].batch(32).map(preprocess)
        train_set = train_set.map(encode_words).prefetch(1)
        # for X_batch, y_batch in train_set.take(1):
        #     print(X_batch)
        #     print(y_batch)
        embed_size = 128
        # model = keras.models.Sequential([
        #     keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size,
        #                            input_shape=[None]),
        #     keras.layers.GRU(128, return_sequences=True),
        #     keras.layers.GRU(128),
        #     keras.layers.Dense(1, activation="sigmoid")
        # ])
        # model.compile(loss="binary_crossentropy", optimizer="adam",
        #               metrics=["accuracy"])
        # history = model.fit(train_set, epochs=5)

        # reusing pretrained embeddings
        # tensorflow hub projects makes it easy to reuse pretrained model components

        # TFHUB_CACHE_DIR = os.path.join(os.curdir, "my_tfhub_cache")
        # os.environ["TFHUB_CACHE_DIR"] = TFHUB_CACHE_DIR
        #
        # model = keras.Sequential([
        #     hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1",
        #                    dtype=tf.string, input_shape=[], output_shape=[50]),
        #     keras.layers.Dense(128, activation="relu"),
        #     keras.layers.Dense(1, activation="sigmoid")
        # ])
        # model.compile(loss="binary_crossentropy", optimizer="adam",
        #               metrics=["accuracy"])
        # # for dirpath, dirnames, filenames in os.walk(TFHUB_CACHE_DIR):
        # #     for filename in filenames:
        # #         print(os.path.join(dirpath, filename))
        # datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
        # train_size = info.splits["train"].num_examples
        # batch_size = 32
        # train_set = datasets["train"].batch(batch_size).prefetch(1)
        # history = model.fit(train_set, epochs=5)

        # encoder decoder network for neural machine translation
        # tf.random.set_seed(42)
        # vocab_size = 100
        # embed_size = 10
        #
        # encoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
        # decoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
        # sequence_lengths = keras.layers.Input(shape=[], dtype=np.int32)
        #
        # embeddings = keras.layers.Embedding(vocab_size, embed_size)
        # encoder_embeddings = embeddings(encoder_inputs)
        # decoder_embeddings = embeddings(decoder_inputs)
        #
        # encoder = keras.layers.LSTM(512, return_state=True)
        # encoder_outputs, state_h, state_c = encoder(encoder_embeddings)
        # encoder_state = [state_h, state_c]
        #
        # sampler = tfa.seq2seq.sampler.TrainingSampler()
        #
        # decoder_cell = keras.layers.LSTMCell(512)
        # output_layer = keras.layers.Dense(vocab_size)
        # decoder = tfa.seq2seq.basic_decoder.BasicDecoder(decoder_cell, sampler,
        #                                                  output_layer=output_layer)
        # final_outputs, final_state, final_sequence_lengths = decoder(
        #     decoder_embeddings, initial_state=encoder_state,
        #     sequence_length=sequence_lengths)
        # Y_proba = tf.nn.softmax(final_outputs.rnn_output)
        #
        # model = keras.models.Model(
        #     inputs=[encoder_inputs, decoder_inputs, sequence_lengths],
        #     outputs=[Y_proba])
        # model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
        # X = np.random.randint(100, size=10 * 1000).reshape(1000, 10)
        # Y = np.random.randint(100, size=15 * 1000).reshape(1000, 15)
        # X_decoder = np.c_[np.zeros((1000, 1)), Y[:, :-1]]
        # seq_lengths = np.full([1000], 15)
        #
        # history = model.fit([X, X_decoder, seq_lengths], Y, epochs=2)

        # bidirectional rnn
        # this run two recurrent layers on the same inputs,
        # one for reading the words from left to right and other reading them from r to l
        # then combine their outputs at each time step.
        # model = keras.models.Sequential([
        #     keras.layers.GRU(10, return_sequences=True, input_shape=[None, 10]),
        #     keras.layers.Bidirectional(keras.layers.GRU(10, return_sequences=True))
        # ])
        # model.summary()

        # beam search.
        # Encoder-decoder moder, and we use it to translate the sentence.
        # when there is a mistake and the model could not back and fix word, it just fill the sentence.
        # but this method will be trash at translating long sentences,
        # because of this short memory of runs
        # beam_width = 10
        # decoder = tfa.seq2seq.beam_search_decoder.BeamSearchDecoder(
        #     cell=decoder_cell, beam_width=beam_width, output_layer=output_layer)
        # decoder_initial_state = tfa.seq2seq.beam_search_decoder.tile_batch(
        #     encoder_state, multiplier=beam_width)
        # outputs, _, _ = decoder(
        #     embedding_decoder, start_tokens=start_tokens, end_token=end_token,
        #     initial_state=decoder_initial_state)

        # attention mechanisms

        # positional embeddings
        # it is a dense vector that encodes the position of a word within a sentence

        # recent innovations in language models
        # image Net moment for nlp process

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


class TrainingExercises:
    def __init__(self):
        print("k")

    def Ex1(self):
        default_reber_grammar = [
            [("B", 1)],  # (state 0) =B=>(state 1)
            [("T", 2), ("P", 3)],  # (state 1) =T=>(state 2) or =P=>(state 3)
            [("S", 2), ("X", 4)],  # (state 2) =S=>(state 2) or =X=>(state 4)
            [("T", 3), ("V", 5)],  # and so on...
            [("X", 3), ("S", 6)],
            [("P", 4), ("V", 6)],
            [("E", None)]]  # (state 6) =E=>(terminal state)

        embedded_reber_grammar = [
            [("B", 1)],
            [("T", 2), ("P", 3)],
            [(default_reber_grammar, 4)],
            [(default_reber_grammar, 5)],
            [("T", 6)],
            [("P", 6)],
            [("E", None)]]

        def generate_string(grammar):
            state = 0
            output = []
            while state is not None:
                index = np.random.randint(len(grammar[state]))
                production, state = grammar[state][index]
                if isinstance(production, list):
                    production = generate_string(grammar=production)
                output.append(production)
            return "".join(output)

        POSSIBLE_CHARS = "BEPSTVX"

        def generate_corrupted_string(grammar, chars=POSSIBLE_CHARS):
            good_string = generate_string(grammar)
            index = np.random.randint(len(good_string))
            good_char = good_string[index]
            bad_char = np.random.choice(sorted(set(chars) - set(good_char)))
            return good_string[:index] + bad_char + good_string[index + 1:]

        np.random.seed(42)

        # for _ in range(42):
        #     print(generate_string(default_reber_grammar), end=" ")
        # print()
        # for _ in range(42):
        #     print(generate_string(embedded_reber_grammar), end=" ")
        # for _ in range(25):
        #     print(generate_corrupted_string(embedded_reber_grammar), end=" ")

        # we cannot feed strings directly to an rnn, we need to encode them somehow.
        # one option, one-hot encode each char and another option using embeddings,
        # since there are just a handful of chars, one-hot encoding would probably be a good option as well.
        # for embeddings, we need to convert each string into a sequence of chars ids
        def string_to_ids(s, chars=POSSIBLE_CHARS):
            return [chars.index(c) for c in s]

        # print(string_to_ids("BTTTXXVVETE"))

        def generate_dataset(size):
            good_strings = [string_to_ids(generate_string(embedded_reber_grammar))
                            for _ in range(size // 2)]
            bad_strings = [string_to_ids(generate_corrupted_string(embedded_reber_grammar))
                           for _ in range(size - size // 2)]
            all_strings = good_strings + bad_strings
            X = tf.ragged.constant(all_strings, ragged_rank=1)
            y = np.array([[1.] for _ in range(len(good_strings))] +
                         [[0.] for _ in range(len(bad_strings))])
            return X, y

        np.random.seed(42)

        X_train, y_train = generate_dataset(10000)
        X_valid, y_valid = generate_dataset(2000)
        np.random.seed(42)
        tf.random.set_seed(42)

        embedding_size = 5

        model = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=[None], dtype=tf.int32, ragged=True),
            keras.layers.Embedding(input_dim=len(POSSIBLE_CHARS), output_dim=embedding_size),
            keras.layers.GRU(30),
            keras.layers.Dense(1, activation="sigmoid")
        ])
        optimizer = keras.optimizers.SGD(learning_rate=0.02, momentum=0.95, nesterov=True)
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))

        test_strings = ["BPBTSSSSSSSXXTTVPXVPXTTTTTVVETE",
                        "BPBTSSSSSSSXXTTVPXVPXTTTTTVVEPE"]
        X_test = tf.ragged.constant([string_to_ids(s) for s in test_strings], ragged_rank=1)

        y_proba = model.predict(X_test)
        print()
        print("Estimated probability that these are Reber strings:")
        for index, string in enumerate(test_strings):
            print("{}: {:.2f}%".format(string, 100 * y_proba[index][0]))




def program1():
    TrainingExercises()


if __name__ == '__main__':
    program1()
