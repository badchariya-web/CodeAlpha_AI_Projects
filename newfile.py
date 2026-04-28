from flask import Flask, request, jsonify
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load FAQs
with open("faqs.json") as f:
    data = json.load(f)

questions = [item["question"] for item in data]
answers = [item["answer"] for item in data]

# Convert text to vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]

    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)

    index = similarity.argmax()
    response = answers[index]

    return jsonify({"reply": response})

if __name__ == "__main__":
    app.run(debug=True)