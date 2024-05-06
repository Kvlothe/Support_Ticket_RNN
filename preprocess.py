import pandas as pd
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from spellchecker import SpellChecker


# This module is for preprocessing the dataset/text for a NLP utilizing RNN for either LSTM or GRU
stop_words = set(stopwords.words('english'))
# Define your custom stopwords
custom_stopwords = {'icon', 'image', 'enter'}
stop_words.update(custom_stopwords)


def correct_spelling(text):
    spell = SpellChecker()
    # Ensure that we always return the original word if correction returns None (which shouldn't normally happen)
    corrected_text = ' '.join([spell.correction(word) if spell.correction(word) is not None else word for word in text.split()])
    return corrected_text


# Function for cleaning the text in a datset - converts all letters to lowercase, removes punctuation and numbers,
# removes stopwords and lemmatizes the text
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
    text = correct_spelling(text)  # Correct spelling before removing stopwords
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    return text


def preprocess_data(df, save_path='cleaned_data.csv', force=False):
    if os.path.exists(save_path) and not force:
        print("Loading cleaned data from file...")
        cleaned_df = pd.read_csv(save_path)
    else:
        print("Cleaning data...")
        df['cleaned_document'] = df['document'].apply(clean_text)
        df.to_csv(save_path, index=False)  # Save the cleaned data
        cleaned_df = df

    # Tokenization
    tokenizer = Tokenizer(num_words=5000, lower=True, split=' ')
    tokenizer.fit_on_texts(cleaned_df['cleaned_document'].values)

    x = tokenizer.texts_to_sequences(cleaned_df['cleaned_document'].values)
    x = pad_sequences(x, maxlen=200)

    # Label encoding
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(cleaned_df['topic_group'])
    y = to_categorical(integer_encoded)
    return cleaned_df, x, y, label_encoder
