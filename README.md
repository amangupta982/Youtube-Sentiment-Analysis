# 📊 YouTube Sentiment Analysis & Chrome Extension

This project is an end-to-end Machine Learning pipeline for analyzing the sentiment of YouTube comments. It includes a robust data processing and modeling pipeline tracked by **DVC** and **MLflow**, a **Flask API** for serving predictions, and a **Chrome Extension** that provides real-time insights on any YouTube video directly in the browser!

---

## 🌟 Key Features

### 🖥️ Chrome Extension: "YouTube Comment Insights"
The included Chrome Extension interacts with the Flask backend to fetch comments from the current video and display rich analytics instantly:
- **Comment Analysis Summary:** View key metrics like Total Comments, Unique Commenters, Average Comment Length, and an overall Average Sentiment Score (normalized out of 10).
- **Sentiment Distribution:** See a breakdown of Positive, Neutral, and Negative comments.
- **Sentiment Trend Over Time:** A generated trend graph showing how the sentiment percentages have shifted over the lifespan of the video.
- **Comment Wordcloud:** A visually engaging word cloud highlighting the most frequent and impactful words used by viewers.
- **Top Comments with Sentiments:** Quickly browse the top 25 comments ranked by engagement, alongside their individual sentiment scores.

### 🧠 Machine Learning Pipeline (LightGBM)
- Natural Language Processing (NLP) pipeline with **TF-IDF Vectorization** (unigrams to trigrams).
- Model building and hyperparameter tuning utilizing **LightGBM** and other algorithms.
- **Handling Imbalanced Data** techniques like SMOTE and class weights.

### 🛠️ MLOps & Tracking
- **DVC (Data Version Control):** Tracks data pipelines, caching, and reproducibility.
- **MLflow:** Tracks experiments, logs model parameters, metrics (Precision, Recall, F1-Score), confusion matrices, and model artifacts.

---

## 🚀 Setup & Installation

### 1. Environment Setup
Create and activate a new Conda environment, then install dependencies:
```bash
conda create -n youtube python=3.11 -y
conda activate youtube
pip install -r requirements.txt
```

### 2. Configure AWS (For MLflow remote tracking)
If using an AWS EC2 instance for the MLflow tracking server:
```bash
aws configure
```

### 3. DVC (Data Version Control) Pipeline
The project pipeline (data ingestion -> preprocessing -> model building -> evaluation) is managed by DVC.
```bash
# Initialize DVC (if not already initialized)
dvc init

# Reproduce the entire pipeline end-to-end
dvc repro

# Visualize the pipeline DAG
dvc dag
```

---

## 🌐 Flask API Backend

The backend is powered by Flask, serving the trained LightGBM model. Make sure to run the Flask server to power the Chrome Extension!

```bash
python flask_app/app.py
```

### API Usage Example (Postman)

**Endpoint:** `POST http://localhost:5001/predict` *(Note: Check your specific port configuration, usually 5000 or 5001)*

**Request Body (JSON):**
```json
{
    "comments": [
        "This video is awsome! I loved a lot", 
        "Very bad explanation. poor video"
    ]
}
```

---

## 🧩 Installing the Chrome Extension

To use the Chrome Extension locally:
1. Open Google Chrome and navigate to `chrome://extensions/`
2. Enable **Developer mode** (toggle in the top right corner).
3. Click **Load unpacked**.
4. Select the `yt-chrome-plugin-frontend` directory in this project.
5. Open any YouTube video and click the extension icon to see live insights!