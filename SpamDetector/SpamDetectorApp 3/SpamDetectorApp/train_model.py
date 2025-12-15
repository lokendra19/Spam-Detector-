import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ✅ Load improved dataset
df = pd.read_csv("spam_improved.csv")
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# ✅ Split the data
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.25, random_state=42)

# ✅ Vectorize text
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ✅ Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ✅ Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test_vec))
print("✅ Accuracy:", accuracy)

# ✅ Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model trained and saved successfully.")
