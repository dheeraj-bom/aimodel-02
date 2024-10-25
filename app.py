from flask import Flask, request, render_template, jsonify
import pickle
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import string
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

# Initialize Flask app
app = Flask(__name__)

# Load stopwords
stop_words = set(stopwords.words('english'))

# Load saved Naive Bayes model and TF-IDF vectorizer
with open("model_nb.pkl", "rb") as model_file:
    model_nb = pickle.load(model_file)
with open("tfidf_vectorizer.pkl", "rb") as tfidf_file:
    tfidf_vectorizer = pickle.load(tfidf_file)

# Load label encoder
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Initialize BERT model and tokenizer for tag extraction
class TagExtractionModel(torch.nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased'):
        super(TagExtractionModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 3)  # Adjust based on label count

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        return self.classifier(pooled_output)

bert_model = TagExtractionModel()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Clean text function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join([word for word in text.split() if word not in stop_words])

# Tag extraction function
def extract_dynamic_tags(text, top_n=3):
    text = clean_text(text)
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    word_counts = vectorizer.fit_transform([text])
    words = vectorizer.get_feature_names_out()
    word_frequencies = Counter(words)
    common_words = [word for word, _ in word_frequencies.most_common(top_n)]
    return common_words

# Define route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    title = data['title']
    body = data['body']
    full_text = title + " " + body

    # TF-IDF and Naive Bayes for Category Prediction
    text_tfidf = tfidf_vectorizer.transform([full_text])
    predicted_category = model_nb.predict(text_tfidf)
    category = label_encoder.inverse_transform(predicted_category)[0]

    # Tag extraction
    tags = extract_dynamic_tags(full_text)

    # Breaking news determination
    news_type = "Breaking" if "breaking" in title.lower() else "Normal"

    # Return JSON response
    return jsonify({
        "category": category,
        "tags": tags,
        "type": news_type
    })

if __name__ == "__main__":
    app.run(debug=True)
