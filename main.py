import os
import re
from functools import lru_cache

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "emotion_dataset.csv")


@lru_cache(maxsize=1)
def _text_resources():
    resources = {
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
        "punkt": "tokenizers/punkt",
    }

    for package, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
        except (LookupError, OSError):
            nltk.download(package, quiet=True)

    return set(stopwords.words("english")), WordNetLemmatizer()


def clean_text(text):
    stop_words, lemmatizer = _text_resources()
    text = re.sub(r"[^a-zA-Z\s]", "", str(text).lower())
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(lemmatizer.lemmatize(word) for word in tokens)


def train_model():
    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(
            f"Training data not found: {DATASET_PATH}. "
            "Add the semicolon-separated emotion dataset at that path."
        )

    df = pd.read_csv(
        DATASET_PATH,
        sep=";",
        header=None,
        names=["text", "emotion"],
    )
    df = df.dropna(subset=["text", "emotion"])
    df["cleaned_text"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_text"],
        df["emotion"],
        test_size=0.20,
        random_state=42,
        stratify=df["emotion"],
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_tfidf))
    return model, vectorizer, accuracy


def predict_emotion(text, model, vectorizer):
    text_tfidf = vectorizer.transform([clean_text(text)])
    probabilities = model.predict_proba(text_tfidf)[0]
    scores = [
        {"emotion": str(emotion), "score": float(probability * 100)}
        for emotion, probability in zip(model.classes_, probabilities)
    ]
    best_score = max(scores, key=lambda score: score["score"])

    return {
        "emotion": best_score["emotion"],
        "confidence": best_score["score"],
        "scores": scores,
    }


if __name__ == "__main__":
    trained_model, trained_vectorizer, accuracy = train_model()
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print("Type 'exit' to stop.")

    while True:
        user_text = input("\nEnter a sentence: ")
        if user_text.strip().lower() == "exit":
            break
        if not user_text.strip():
            print("Please enter some text.")
            continue

        result = predict_emotion(user_text, trained_model, trained_vectorizer)
        print(f"Detected Emotion: {result['emotion']}")
        print(f"Confidence: {result['confidence']:.2f}%")