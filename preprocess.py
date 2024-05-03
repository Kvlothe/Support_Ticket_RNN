from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')


# This module is for preprocessing the dataset/text for a NLP utilizing RNN for either LTSM or GRU

# Function for cleaning the text in a datset - converts all letters to lowercase, removes punctuation and numbers,
# removes stopwords and lemmatizes the text
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


# Fucntion called when a new model is being trained. Cleans the text, tokenizes it, pads the data and isolated the
# count of the dependant variables (topic groups) which are used for one hot encoding
def preprocess_data(df):
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
    return df, x, y, label_encoder
