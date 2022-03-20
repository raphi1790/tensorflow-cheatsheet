# Tensorflow Cheat Sheet
## Utilities
### Load Data from JSON
    import json

    # Load the JSON file
    with open("./sarcasm.json", 'r') as f:
        datastore = json.load(f)

    # Initialize the lists
    sentences = []
    labels = []

    # Collect sentences and labels into the lists
    for item in datastore:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])
### Plot loss and accuracy
    import matplotlib.pyplot as plt

    # Plot utility
    def plot_graphs(history, string):
        """
        history: returned element of Keras model.fit()-method
        """
        plt.plot(history.history[string])
        plt.plot(history.history['val_'+string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_'+string])
        plt.show()

    # Plot the accuracy and results 
    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")

### One-Hot-Encoding using Keras
    y_ohe = tf.keras.utils.to_categorical(labels, num_classes=total_words)

## Image Classification
## Natural Language Processing
### Import Dataset using Tensorflow Dataset (tfds)
tfds stores some common datasets. This can be loaded using the following code. Some Datasets are already tokenized.

    import tensorflow_datasets as tfds

    imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

### Transform loaded Datasets into lists of Python-objects
    # Get the train and test sets
    train_data, test_data = imdb['train'], imdb['test']

    # Initialize sentences and labels lists
    training_sentences = []
    training_labels = []

    testing_sentences = []
    testing_labels = []

    # Loop over all training examples and save the sentences and labels
    for s,l in train_data:
        training_sentences.append(s.numpy().decode('utf8'))
        training_labels.append(l.numpy())

    # Loop over all test examples and save the sentences and labels
    for s,l in test_data:
        testing_sentences.append(s.numpy().decode('utf8'))
        testing_labels.append(l.numpy())

    # Convert labels lists to numpy array
    training_labels_final = np.array(training_labels)
    testing_labels_final = np.array(testing_labels)

### Tokeinzing and Padding from Scratch
    # Parameters
    vocab_size = 10000
    max_length = 120
    trunc_type='post'
    oov_tok = "<OOV>"

    # Initialize the Tokenizer class
    tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)

    # Generate the word index dictionary for the training sentences
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index

    # Generate and pad the training sequences
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

    # Generate and pad the test sequences
    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences,maxlen=max_length)
### Train Model using LSTM
    # Parameters
    embedding_dim = 16
    lstm_dim = 32
    dense_dim = 6

    # Model Definition with LSTM
    model_lstm = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        # make sure you use return_sequences if we use multiple LSTM
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm1_dim, return_sequences=True)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm2_dim)),
        tf.keras.layers.Dense(dense_dim, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Set the training parameters
    model_lstm.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    # Print the model summary
    model_lstm.summary()

    NUM_EPOCHS = 10
    BATCH_SIZE = 128

    # Train the model
    history_lstm = model_lstm.fit(padded, training_labels_final, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(testing_padded, testing_labels_final))

### Train Model using GRU
    import tensorflow as tf

    # Parameters
    embedding_dim = 16
    gru_dim = 32
    dense_dim = 6

    # Model Definition with GRU
    model_gru = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_dim)),
        tf.keras.layers.Dense(dense_dim, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Set the training parameters
    model_gru.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    # Print the model summary
    model_gru.summary()
    NUM_EPOCHS = 10
    BATCH_SIZE = 128

    # Train the model
    history_gru = model_gru.fit(padded, training_labels_final, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(testing_padded, testing_labels_final))

### Train Model using Convolution
    # Parameters
    embedding_dim = 16
    filters = 128
    kernel_size = 5
    dense_dim = 6

    # Model Definition with Conv1D
    model_conv = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(filters, kernel_size, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(dense_dim, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Set the training parameters
    model_conv.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    # Print the model summary
    model_conv.summary()

    NUM_EPOCHS = 10
    BATCH_SIZE = 128

    # Train the model
    history_conv = model_conv.fit(padded, training_labels_final, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(testing_padded, testing_labels_final))

### Create n_gram-sequences for Text Generation
    input_sequences = []
    # corpus is equal to a list of sentences
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    # pad sequences 
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    # create predictors and label
    xs, labels = input_sequences[:,:-1],input_sequences[:,-1]

    # One-Hot-Encoding
    ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

### Generate Text based on a model
    seed_text = "Laurence went to dublin"
    next_words = 100
    
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
       	predict_x=model.predict(token_list)
	    predicted=np.argmax(predict_x,axis=1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    print(seed_text)




