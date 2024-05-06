import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from model import create_new_model
from model import train_further

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def list_models(directory="."):
    """List .keras model files in a specified directory."""
    models = [f for f in os.listdir(directory) if f.endswith('.keras')]
    return models


def select_and_load_model(directory="."):
    """Display a list of models and load the selected model."""
    models = list_models(directory)
    if not models:
        print("No models found in the directory.")
        return None

    print("Available models:")
    for idx, model_name in enumerate(models, 1):
        print(f"{idx}: {model_name}")

    while True:
        try:
            choice = int(input("Enter the number of the model you want to load: "))
            if 1 <= choice <= len(models):
                model_path = os.path.join(directory, models[choice - 1])
                model = load_model(model_path)
                print(f"Model {models[choice - 1]} loaded successfully.")
                return model
            else:
                print("Invalid number, please select a valid model number.")
        except ValueError:
            print("Please enter a valid integer.")


def main():
    df = pd.read_csv('support_tickets.csv', names=['document', 'topic_group'])
    print("What would you like to do?")
    print("1: Train a new model")
    print("2: Load an existing model")
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        model = create_new_model(df)
        model.save('new_model.keras')
        print("Model trained and saved.")
    elif choice == '2':
        model = select_and_load_model()
        print("Model loaded.")
        train_further(df, model)
        model.save('further_trained.keras')
        print("Model trained and saved.")
    else:
        print("Invalid choice. Please select 1 or 2.")


if __name__ == "__main__":
    main()
