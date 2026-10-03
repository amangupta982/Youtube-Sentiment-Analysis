import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import mlflow
import numpy as np
import re
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from mlflow.tracking import MlflowClient

import matplotlib.dates as mdates
import pickle


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""

    try:
        comment = comment.lower()
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters except punctuation
        comment = re.sub(
            r'[^A-Za-z0-9\s!?.,]',
            '',
            comment
        )

        # Stopwords
        stop_words = set(stopwords.words('english')) - {
            'not',
            'but',
            'however',
            'no',
            'yet'
        }

        comment = ' '.join(
            word
            for word in comment.split()
            if word not in stop_words
        )

        # Lemmatization
        lemmatizer = WordNetLemmatizer()

        comment = ' '.join(
            lemmatizer.lemmatize(word)
            for word in comment.split()
        )

        return comment

    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment


# ============================================================
# LOAD MODEL + VECTORIZER
# ============================================================

def load_model_and_vectorizer(
    model_name,
    model_version,
    vectorizer_path
):

    try:

        # MLflow server
        mlflow.set_tracking_uri(
            "http://ec2-100-53-30-206.compute-1.amazonaws.com:5000/"
        )

        print("Connecting to MLflow...")

        client = MlflowClient()

        # Model Registry URI
        model_uri = f"models:/{model_name}/{model_version}"

        print(f"Loading model: {model_uri}")

        # Load MLflow model
        model = mlflow.pyfunc.load_model(model_uri)

        print("MLflow model loaded successfully.")

        # Load TF-IDF vectorizer
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)

        print("TF-IDF vectorizer loaded successfully.")

        return model, vectorizer

    except Exception as e:

        print(f"Error loading model/vectorizer: {e}")
        raise


# ============================================================
# ALTERNATIVE LOCAL MODEL LOADER
# ============================================================

def load_local_model(model_path, vectorizer_path):

    try:

        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)

        return model, vectorizer

    except Exception as e:

        print(f"Error loading local model: {e}")
        raise


# ============================================================
# LOAD MODEL
# ============================================================

model, vectorizer = load_local_model(
    "./lgbm_model.pkl",
    "./tfidf_vectorizer.pkl"
)

print("Model and vectorizer are ready.")


# ============================================================
# HELPER FUNCTION
# ============================================================

def transform_comments_for_model(comments):
    """
    Preprocess comments, apply TF-IDF,
    and convert the result into a DataFrame.

    IMPORTANT:
    MLflow model signature expects named TF-IDF columns.
    """

    # Preprocess
    preprocessed_comments = [
        preprocess_comment(comment)
        for comment in comments
    ]

    # TF-IDF transformation
    transformed_comments = vectorizer.transform(
        preprocessed_comments
    )

    # Get exact feature names
    feature_names = vectorizer.get_feature_names_out()

    # Convert sparse matrix to DataFrame
    transformed_df = pd.DataFrame(
        transformed_comments.toarray(),
        columns=feature_names
    )

    return transformed_df


# ============================================================
# HOME
# ============================================================

@app.route('/')
def home():

    return "Welcome to our Flask API"


# ============================================================
# PREDICT
# ============================================================

@app.route('/predict', methods=['POST'])
def predict():

    try:

        data = request.get_json()

        comments = data.get('comments')

        print("Received comments:")
        print(comments)

        print("Type:", type(comments))

        if not comments:

            return jsonify({
                "error": "No comments provided"
            }), 400

        # Make sure comments are strings
        comments = [
            str(comment)
            for comment in comments
        ]

        # ====================================================
        # TRANSFORM INPUT
        # ====================================================

        transformed_df = transform_comments_for_model(
            comments
        )

        print(
            "TF-IDF shape:",
            transformed_df.shape
        )

        print(
            "Number of features:",
            len(transformed_df.columns)
        )

        # ====================================================
        # PREDICTION
        # ====================================================

        predictions = model.predict(
            transformed_df
        )

        # Convert numpy values to normal Python values
        predictions = predictions.tolist()

        print(
            "Predictions:",
            predictions
        )

        # ====================================================
        # RESPONSE
        # ====================================================

        response = [
            {
                "comment": comment,
                "sentiment": sentiment
            }
            for comment, sentiment
            in zip(comments, predictions)
        ]

        return jsonify(response)

    except Exception as e:

        print(
            f"Prediction failed: {e}"
        )

        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500


# ============================================================
# PREDICT WITH TIMESTAMPS
# ============================================================

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():

    try:

        data = request.get_json()

        comments_data = data.get(
            'comments'
        )

        if not comments_data:

            return jsonify({
                "error": "No comments provided"
            }), 400

        # Extract comments
        comments = [
            item['text']
            for item in comments_data
        ]

        # Extract timestamps
        timestamps = [
            item['timestamp']
            for item in comments_data
        ]

        # ====================================================
        # TRANSFORM
        # ====================================================

        transformed_df = transform_comments_for_model(
            comments
        )

        # ====================================================
        # PREDICT
        # ====================================================

        predictions = model.predict(
            transformed_df
        )

        predictions = predictions.tolist()

        predictions = [
            str(prediction)
            for prediction in predictions
        ]

        # ====================================================
        # RESPONSE
        # ====================================================

        response = [
            {
                "comment": comment,
                "sentiment": sentiment,
                "timestamp": timestamp
            }

            for comment, sentiment, timestamp
            in zip(
                comments,
                predictions,
                timestamps
            )
        ]

        return jsonify(response)

    except Exception as e:

        print(
            f"Prediction failed: {e}"
        )

        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500


# ============================================================
# GENERATE CHART
# ============================================================

@app.route('/generate_chart', methods=['POST'])
def generate_chart():

    try:

        data = request.get_json()

        sentiment_counts = data.get(
            'sentiment_counts'
        )

        if not sentiment_counts:

            return jsonify({
                "error": "No sentiment counts provided"
            }), 400

        # Labels
        labels = [
            'Positive',
            'Neutral',
            'Negative'
        ]

        # Sentiment mapping
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]

        if sum(sizes) == 0:

            raise ValueError(
                "Sentiment counts sum to zero"
            )

        colors = [
            '#36A2EB',
            '#C9CBCF',
            '#FF6384'
        ]

        # ====================================================
        # PIE CHART
        # ====================================================

        plt.figure(figsize=(6, 6))

        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )

        plt.axis('equal')

        # Save to memory
        img_io = io.BytesIO()

        plt.savefig(
            img_io,
            format='PNG',
            transparent=True
        )

        img_io.seek(0)

        plt.close()

        return send_file(
            img_io,
            mimetype='image/png'
        )

    except Exception as e:

        app.logger.error(
            f"Error in /generate_chart: {e}"
        )

        return jsonify({
            "error": f"Chart generation failed: {str(e)}"
        }), 500


# ============================================================
# GENERATE WORD CLOUD
# ============================================================

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():

    try:

        data = request.get_json()

        comments = data.get(
            'comments'
        )

        if not comments:

            return jsonify({
                "error": "No comments provided"
            }), 400

        # Preprocess
        preprocessed_comments = [
            preprocess_comment(comment)
            for comment in comments
        ]

        # Combine comments
        text = ' '.join(
            preprocessed_comments
        )

        # ====================================================
        # WORD CLOUD
        # ====================================================

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(
                stopwords.words('english')
            ),
            collocations=False
        ).generate(text)

        # Save image
        img_io = io.BytesIO()

        wordcloud.to_image().save(
            img_io,
            format='PNG'
        )

        img_io.seek(0)

        return send_file(
            img_io,
            mimetype='image/png'
        )

    except Exception as e:

        app.logger.error(
            f"Error in /generate_wordcloud: {e}"
        )

        return jsonify({
            "error": f"Word cloud generation failed: {str(e)}"
        }), 500


# ============================================================
# GENERATE TREND GRAPH
# ============================================================

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():

    try:

        data = request.get_json()

        sentiment_data = data.get(
            'sentiment_data'
        )

        if not sentiment_data:

            return jsonify({
                "error": "No sentiment data provided"
            }), 400

        # DataFrame
        df = pd.DataFrame(
            sentiment_data
        )

        # Convert timestamp
        df['timestamp'] = pd.to_datetime(
            df['timestamp']
        )

        # Set timestamp index
        df.set_index(
            'timestamp',
            inplace=True
        )

        # Sentiment numeric
        df['sentiment'] = df[
            'sentiment'
        ].astype(int)

        # Sentiment labels
        sentiment_labels = {
            -1: 'Negative',
            0: 'Neutral',
            1: 'Positive'
        }

        # ====================================================
        # MONTHLY COUNTS
        # ====================================================

        monthly_counts = (
            df
            .resample('ME')['sentiment']
            .value_counts()
            .unstack(fill_value=0)
        )

        # Total
        monthly_totals = (
            monthly_counts.sum(axis=1)
        )

        # Percentage
        monthly_percentages = (
            monthly_counts.T
            .div(monthly_totals)
            .T
            * 100
        )

        # Ensure all sentiment columns exist
        for sentiment_value in [-1, 0, 1]:

            if sentiment_value not in monthly_percentages.columns:

                monthly_percentages[
                    sentiment_value
                ] = 0

        # Sort
        monthly_percentages = (
            monthly_percentages[
                [-1, 0, 1]
            ]
        )

        # ====================================================
        # PLOT
        # ====================================================

        plt.figure(
            figsize=(12, 6)
        )

        colors = {
            -1: 'red',
            0: 'gray',
            1: 'green'
        }

        for sentiment_value in [-1, 0, 1]:

            plt.plot(
                monthly_percentages.index,
                monthly_percentages[
                    sentiment_value
                ],
                marker='o',
                linestyle='-',
                label=sentiment_labels[
                    sentiment_value
                ],
                color=colors[
                    sentiment_value
                ]
            )

        plt.title(
            'Monthly Sentiment Percentage Over Time'
        )

        plt.xlabel('Month')

        plt.ylabel(
            'Percentage of Comments (%)'
        )

        plt.grid(True)

        plt.xticks(
            rotation=45
        )

        # X-axis formatting
        plt.gca().xaxis.set_major_formatter(
            mdates.DateFormatter('%Y-%m')
        )

        plt.gca().xaxis.set_major_locator(
            mdates.AutoDateLocator(
                maxticks=12
            )
        )

        plt.legend()

        plt.tight_layout()

        # Save image
        img_io = io.BytesIO()

        plt.savefig(
            img_io,
            format='PNG'
        )

        img_io.seek(0)

        plt.close()

        return send_file(
            img_io,
            mimetype='image/png'
        )

    except Exception as e:

        app.logger.error(
            f"Error in /generate_trend_graph: {e}"
        )

        return jsonify({
            "error": f"Trend graph generation failed: {str(e)}"
        }), 500


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == '__main__':

    print(
        "======================================"
    )

    print(
        "Flask API started"
    )

    print(
        "Running on http://127.0.0.1:5001"
    )

    print(
        "======================================"
    )

    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True
    )