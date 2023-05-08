from flask import Flask, request, jsonify
from flask_cors import CORS
from utilities import get_movie_recommendations_new_user
from utilities import get_movie_recommendations_user_has_rating

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Hello Project Movie Recommendation"

@app.route('/predict_new_user', methods=['POST'])
def predict_new_user():
    data = request.get_json()
    try:
        sample = data['genres']
    except KeyError:
        return jsonify({'error': 'No text sent'})
    prediction = get_movie_recommendations_new_user(sample)
    return prediction

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        sample = data['userId']
    except KeyError:
        return jsonify({'error': 'No text sent'})
    prediction = get_movie_recommendations_user_has_rating(sample)
    return prediction

if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 8000, debug=True)