import numpy as np
from preprocess import preprocess_data
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, GRU, Input, Masking
from sklearn.metrics import confusion_matrix, classification_report


def create_new_model(df):
    df, x, y, label_encoder = preprocess_data(df)
    model_trained = model(df, x, y, label_encoder)
    return model_trained


def create_ltsm(label_encoder, dropout_rates=[0.2, 0.3, 0.4, 0.5]):
    # LSTM model
    lstm_model = Sequential()
    lstm_model.add(Embedding(input_dim=5000, output_dim=128, input_length=200))
    lstm_model.add(Bidirectional(LSTM(128, return_sequences=True)))  # Return sequences for stacking
    lstm_model.add(Dropout(dropout_rates[0]))
    lstm_model.add(Bidirectional(LSTM(64, return_sequences=True)))
    lstm_model.add(Dropout(dropout_rates[1]))
    lstm_model.add(Bidirectional(LSTM(32)))  # Does not return sequences
    lstm_model.add(Dropout(dropout_rates[2]))
    lstm_model.add(Dense(64, activation='relu'))
    lstm_model.add(Dropout(dropout_rates[3]))
    lstm_model.add(Dense(len(label_encoder.classes_), activation='softmax'))
    return lstm_model


def create_gru(label_encoder, dropout_rates=[0.2, 0.3, 0.4, 0.5]):
    # GFU model
    gru_model = Sequential()
    gru_model.add(Embedding(input_dim=5000, output_dim=128, input_length=200))
    gru_model.add(Bidirectional(GRU(128, return_sequences=True)))
    gru_model.add(Dropout(dropout_rates[0]))
    gru_model.add(Bidirectional(GRU(64, return_sequences=True)))
    gru_model.add(Dropout(dropout_rates[1]))
    gru_model.add(Bidirectional(GRU(32)))  # Does not return sequences
    gru_model.add(Dropout(dropout_rates[2]))
    gru_model.add(Dense(64, activation='relu'))
    gru_model.add(Dropout(dropout_rates[3]))
    gru_model.add(Dense(len(label_encoder.classes_), activation='softmax'))
    return gru_model


def train_model(df, x, y, model):
    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss',  # Monitor the model's validation loss
                                   patience=5,  # Stop after 3 epochs if the validation loss hasn't improved
                                   restore_best_weights=True)  # Restore model weights from the epoch with the best validation loss
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # First, split into training and temporary data (the latter will be split into validation and test sets)
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)

    # Now split the temporary data into validation and test sets
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Initial accuracy: {accuracy * 100:.2f}%")

    # Train the model - Adjust the batch size and epochs to do some 'fine-tuning'
    model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_val, y_val), callbacks=[early_stopping])
    # model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))

    # Evaluate
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    # Predict the labels on test dataset
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    print("Confusion Matrix:")
    print(cm)

    # Generate a classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred_classes))

    # Summarize the model to check layer configurations
    model.summary()
    return model


def train_further(df, model):
    df, x, y, label_encoder = preprocess_data(df)

    # First, split into training and temporary data (the latter will be split into validation and test sets)
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)

    # Now split the temporary data into validation and test sets
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    # Evaluate
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    # additional_epochs = input(print('How many epochs for further training?'))
    additional_epochs = 10
    # If you decide to train further
    model.fit(x_train, y_train, epochs=additional_epochs, validation_data=(x_val, y_val))

    # Evaluate
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    # Summarize the model to check layer configurations
    # model.summary()
    return model


def model(df, x, y, label_encoder):

    print("New Model Creation:")
    print("What type of model?")
    print("1: LTSM")
    print("2: GRU")
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        model = create_ltsm(label_encoder)
    elif choice == '2':
        model = create_gru(label_encoder)
    else:
        print("Invalid choice. Please select 1 or 2.")

    trained_model = train_model(df, x, y, model)
    return trained_model
