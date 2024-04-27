import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
    # Remove stopwords
    stop_words = stopwords.words('english')
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    return text


# Load data
df = pd.read_csv('support_tickets.csv', names=['document', 'topic_group'])

df['cleaned_document'] = df['document'].apply(clean_text)

# Tokenization
tokenizer = Tokenizer(num_words=5000, lower=True, split=' ')
tokenizer.fit_on_texts(df['cleaned_document'].values)  # Tokenizer learns the vocabulary from the cleaned text

x = tokenizer.texts_to_sequences(df['cleaned_document'].values)  # Convert text to sequences of integer IDs
x = pad_sequences(x, maxlen=200)  # Pad or truncate the sequences to a uniform length


# Label encoding
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(df['topic_group'])
y = to_categorical(integer_encoded)

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss',  # Monitor the model's validation loss
                               patience=3,  # Stop after 3 epochs if the validation loss hasn't improved
                               restore_best_weights=True)  # Restore model weights from the epoch with the best validation loss

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=200))
model.add(Bidirectional(LSTM(128, return_sequences=True)))  # Return sequences for stacking
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(64)))  # Second LSTM layer
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# First, split into training and temporary data (the latter will be split into validation and test sets)
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)

# Now split the temporary data into validation and test sets
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# Train
# model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), callbacks=[early_stopping])
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val), callbacks=[early_stopping])


# Evaluate
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))



