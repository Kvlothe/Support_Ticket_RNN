from lime.lime_text import LimeTextExplainer


def predict_proba(texts):
    # Preprocess the texts
    seqs = tokenizer.texts_to_sequences(texts)
    padded_seqs = pad_sequences(seqs, maxlen=200)
    # Get predictions
    preds = model.predict(padded_seqs)
    return preds


# Function to get explanations and format them
def get_lime_explanation(text):
    exp = explainer.explain_instance(text, predict_proba, num_features=5)
    explanation = exp.as_list()
    return {word: weight for word, weight in explanation}


# This function is supposed to go through the 'cleaned text' and highlight what the model finds important
# and store the words and weights in an excel file.
# Function does not work and needs some attention but from what I understand this function will be memory intense.
# There will be another .py for running the lime analysis using Jupyter Notebook, I already have code and script for
# that and it works but right now I can only analyze one line at a time.
def lime_analysis(df, label_encoder):

    class_names = [str(label) for label in label_encoder.classes_]
    explainer = LimeTextExplainer(class_names=class_names)

    # Choose a specific example to explain
    idx = 124  # Example index in the dataset
    text_instance = df['cleaned_document'].iloc[idx]

    exp = explainer.explain_instance(text_instance, predict_proba, num_features=10)
    # exp.show_in_notebook(text=True)

    # Apply to a subset or full DataFrame
    df['LIME_Explanation'] = df['cleaned_document'].apply(get_lime_explanation)

    # Optionally, convert explanations to a string if easier for Excel view
    df['LIME_Explanation_Str'] = df['LIME_Explanation'].apply(lambda x: ', '.join([f"{k}: {v:.2f}" for k, v in x.items()]))

    # Save to Excel
    df.to_excel('LIME_Explanations.xlsx', index=False)
