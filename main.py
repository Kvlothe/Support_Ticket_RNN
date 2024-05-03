import pandas as pd
import numpy as np
import tensorrt
import tensorflow as tf
from tensorflow.keras.models import load_model
from model import create_new_model
print("Numpy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("TensorRT version:", tensorrt.__version__)
print("TensorFlow version:", tf.__version__)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def main():
    print("What would you like to do?")
    print("1: Train a new model")
    print("2: Load an existing model")
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        df = pd.read_csv('support_tickets.csv', names=['document', 'topic_group'])
        model = create_new_model(df)
        model.save('new_model.keras')
        print("Model trained and saved.")
    elif choice == '2':
        model_path = "RNN_model.keras"
        model = load_model(model_path)
        print("Model loaded.")
    else:
        print("Invalid choice. Please select 1 or 2.")


if __name__ == "__main__":
    main()
