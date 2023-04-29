from flask import Flask, request, jsonify
from flask_cors import CORS
from utilities import get_movie_recommendations

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Hello Project Movie Recommendation"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        sample = data['genres']
    except KeyError:
        return jsonify({'error': 'No text sent'})
    prediction = get_movie_recommendations(sample)
    return prediction

if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 8000, debug=True)