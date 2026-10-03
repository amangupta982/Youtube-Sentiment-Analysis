import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
import mlflow.lightgbm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature


# ============================================================
# Logging configuration
# ============================================================

logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Avoid duplicate handlers if the script is imported/reloaded
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ============================================================
# Load data
# ============================================================

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)

        logger.debug(
            'Data loaded and NaNs filled from %s',
            file_path
        )

        return df

    except Exception as e:
        logger.error(
            'Error loading data from %s: %s',
            file_path,
            e
        )
        raise


# ============================================================
# Load model
# ============================================================

def load_model(model_path: str):
    """Load the trained LightGBM model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        logger.debug(
            'Model loaded from %s',
            model_path
        )

        return model

    except Exception as e:
        logger.error(
            'Error loading model from %s: %s',
            model_path,
            e
        )
        raise


# ============================================================
# Load vectorizer
# ============================================================

def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load the saved TF-IDF vectorizer."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)

        logger.debug(
            'TF-IDF vectorizer loaded from %s',
            vectorizer_path
        )

        return vectorizer

    except Exception as e:
        logger.error(
            'Error loading vectorizer from %s: %s',
            vectorizer_path,
            e
        )
        raise


# ============================================================
# Load parameters
# ============================================================

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)

        logger.debug(
            'Parameters loaded from %s',
            params_path
        )

        return params

    except Exception as e:
        logger.error(
            'Error loading parameters from %s: %s',
            params_path,
            e
        )
        raise


# ============================================================
# Evaluate model
# ============================================================

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray
):
    """Evaluate the model and return classification report and confusion matrix."""

    try:
        logger.debug('Starting model prediction...')

        y_pred = model.predict(X_test)

        logger.debug('Prediction completed.')

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True
        )

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        logger.debug('Model evaluation completed.')

        return report, cm

    except Exception as e:
        logger.error(
            'Error during model evaluation: %s',
            e
        )
        raise


# ============================================================
# Log confusion matrix
# ============================================================

def log_confusion_matrix(cm, dataset_name):
    """Log confusion matrix as an MLflow artifact."""

    try:
        logger.debug('Creating confusion matrix...')

        plt.figure(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues'
        )

        plt.title(
            f'Confusion Matrix for {dataset_name}'
        )

        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        cm_file_path = (
            f'confusion_matrix_{dataset_name}.png'
        )

        plt.savefig(cm_file_path)

        logger.debug(
            'Confusion matrix saved locally: %s',
            cm_file_path
        )

        mlflow.log_artifact(cm_file_path)

        logger.debug(
            'Confusion matrix uploaded to MLflow.'
        )

        plt.close()

    except Exception as e:
        logger.error(
            'Error logging confusion matrix: %s',
            e
        )
        raise


# ============================================================
# Save model information
# ============================================================

def save_model_info(
    run_id: str,
    model_path: str,
    file_path: str
) -> None:
    """Save MLflow run ID and model path to JSON."""

    try:

        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }

        with open(file_path, 'w') as file:
            json.dump(
                model_info,
                file,
                indent=4
            )

        logger.debug(
            'Model info saved to %s',
            file_path
        )

    except Exception as e:

        logger.error(
            'Error occurred while saving model info: %s',
            e
        )

        raise


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\n========== MODEL EVALUATION STARTED ==========\n",
        flush=True
    )

    # --------------------------------------------------------
    # MLflow tracking server
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        "http://ec2-100-53-30-206.compute-1.amazonaws.com:5000/"
    )

    print(
        "1. MLflow tracking URI configured.",
        flush=True
    )

    # --------------------------------------------------------
    # MLflow experiment
    # --------------------------------------------------------

    mlflow.set_experiment(
        'dvc-pipeline-runs'
    )

    print(
        "2. MLflow experiment configured.",
        flush=True
    )

    # --------------------------------------------------------
    # Start MLflow run
    # --------------------------------------------------------

    with mlflow.start_run() as run:

        try:

            print(
                f"3. MLflow run started: {run.info.run_id}",
                flush=True
            )

            # ------------------------------------------------
            # Find project root
            # ------------------------------------------------

            root_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    '../../'
                )
            )

            print(
                f"4. Project root: {root_dir}",
                flush=True
            )

            # ------------------------------------------------
            # Load parameters
            # ------------------------------------------------

            params_path = os.path.join(
                root_dir,
                'params.yaml'
            )

            print(
                "5. Loading params.yaml...",
                flush=True
            )

            params = load_params(
                params_path
            )

            print(
                "6. Parameters loaded.",
                flush=True
            )

            # ------------------------------------------------
            # Log parameters
            # ------------------------------------------------

            for key, value in params.items():

                # MLflow parameters must be simple values
                if isinstance(
                    value,
                    (str, int, float, bool)
                ):
                    mlflow.log_param(
                        key,
                        value
                    )
                else:
                    mlflow.log_param(
                        key,
                        str(value)
                    )

            print(
                "7. Parameters logged to MLflow.",
                flush=True
            )

            # ------------------------------------------------
            # Load trained LightGBM model
            # ------------------------------------------------

            model_path_file = os.path.join(
                root_dir,
                'lgbm_model.pkl'
            )

            print(
                "8. Loading trained LightGBM model...",
                flush=True
            )

            model = load_model(
                model_path_file
            )

            print(
                "9. LightGBM model loaded successfully.",
                flush=True
            )

            # ------------------------------------------------
            # Load TF-IDF vectorizer
            # ------------------------------------------------

            vectorizer_path = os.path.join(
                root_dir,
                'tfidf_vectorizer.pkl'
            )

            print(
                "10. Loading TF-IDF vectorizer...",
                flush=True
            )

            vectorizer = load_vectorizer(
                vectorizer_path
            )

            print(
                "11. TF-IDF vectorizer loaded successfully.",
                flush=True
            )

            # ------------------------------------------------
            # Load test data
            # ------------------------------------------------

            test_data_path = os.path.join(
                root_dir,
                'data/interim/test_processed.csv'
            )

            print(
                "12. Loading test data...",
                flush=True
            )

            test_data = load_data(
                test_data_path
            )

            print(
                f"13. Test data loaded: {test_data.shape}",
                flush=True
            )

            # ------------------------------------------------
            # Transform test data using TF-IDF
            # ------------------------------------------------

            print(
                "14. Transforming test data using TF-IDF...",
                flush=True
            )

            X_test_tfidf = vectorizer.transform(
                test_data['clean_comment'].values
            )

            y_test = test_data['category'].values

            print(
                f"15. TF-IDF transformation completed: "
                f"{X_test_tfidf.shape}",
                flush=True
            )

            # ------------------------------------------------
            # Create input example
            # ------------------------------------------------

            print(
                "16. Creating MLflow input example...",
                flush=True
            )

            input_example = pd.DataFrame(
                X_test_tfidf.toarray()[:5],
                columns=vectorizer.get_feature_names_out()
            )

            print(
                f"17. Input example created: "
                f"{input_example.shape}",
                flush=True
            )

            # ------------------------------------------------
            # Generate predictions for signature
            # ------------------------------------------------

            print(
                "18. Generating predictions for signature...",
                flush=True
            )

            signature_predictions = model.predict(
                X_test_tfidf[:5]
            )

            print(
                "19. Signature predictions completed.",
                flush=True
            )

            # ------------------------------------------------
            # Infer MLflow signature
            # ------------------------------------------------

            print(
                "20. Inferring MLflow model signature...",
                flush=True
            )

            signature = infer_signature(
                input_example,
                signature_predictions
            )

            print(
                "21. MLflow signature created.",
                flush=True
            )

            # ------------------------------------------------
            # Log LightGBM model
            # ------------------------------------------------

            print(
                "\n22. Uploading LightGBM model to MLflow/S3...",
                flush=True
            )

            # IMPORTANT:
            # Use LightGBM-specific MLflow flavor.
            # This avoids the previous sklearn/skops
            # untrusted-type problem.

            mlflow.lightgbm.log_model(
                model,
                "lgbm_model",
                signature=signature,
                input_example=input_example
            )

            print(
                "23. LightGBM model uploaded successfully!",
                flush=True
            )

            # ------------------------------------------------
            # Save model information
            # ------------------------------------------------

            model_path = "lgbm_model"

            save_model_info(
                run.info.run_id,
                model_path,
                'experiment_info.json'
            )

            print(
                "24. Model information saved.",
                flush=True
            )

            # ------------------------------------------------
            # Log vectorizer
            # ------------------------------------------------

            print(
                "25. Uploading TF-IDF vectorizer...",
                flush=True
            )

            mlflow.log_artifact(
                vectorizer_path
            )

            print(
                "26. TF-IDF vectorizer uploaded.",
                flush=True
            )

            # ------------------------------------------------
            # Evaluate model
            # ------------------------------------------------

            print(
                "27. Evaluating model on test data...",
                flush=True
            )

            report, cm = evaluate_model(
                model,
                X_test_tfidf,
                y_test
            )

            print(
                "28. Model evaluation completed.",
                flush=True
            )

            # ------------------------------------------------
            # Log classification metrics
            # ------------------------------------------------

            print(
                "29. Logging classification metrics...",
                flush=True
            )

            for label, metrics in report.items():

                if isinstance(metrics, dict):

                    mlflow.log_metrics({
                        f"test_{label}_precision":
                            metrics['precision'],

                        f"test_{label}_recall":
                            metrics['recall'],

                        f"test_{label}_f1-score":
                            metrics['f1-score']
                    })

            print(
                "30. Classification metrics logged.",
                flush=True
            )

            # ------------------------------------------------
            # Log confusion matrix
            # ------------------------------------------------

            print(
                "31. Logging confusion matrix...",
                flush=True
            )

            log_confusion_matrix(
                cm,
                "Test_Data"
            )

            print(
                "32. Confusion matrix logged.",
                flush=True
            )

            # ------------------------------------------------
            # MLflow tags
            # ------------------------------------------------

            mlflow.set_tag(
                "model_type",
                "LightGBM"
            )

            mlflow.set_tag(
                "task",
                "Sentiment Analysis"
            )

            mlflow.set_tag(
                "dataset",
                "YouTube Comments"
            )

            mlflow.set_tag(
                "pipeline",
                "DVC"
            )

            print(
                "33. MLflow tags added.",
                flush=True
            )

            print(
                "\n========== MODEL EVALUATION COMPLETED ==========\n",
                flush=True
            )

        except Exception as e:

            logger.error(
                f"Failed to complete model evaluation: {e}"
            )

            print(
                f"\nERROR: {e}",
                flush=True
            )

            raise


# ============================================================
# Run main
# ============================================================

if __name__ == '__main__':
    main()