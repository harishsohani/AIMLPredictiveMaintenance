
# import os library
import os

# for data manipulation
import numpy as np
import pandas as pd
from pprint import pprint

# import sklearn library
import sklearn

# libraries for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder

# libraries for encoding, used during pre-processing
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, LabelEncoder

# libraries for column transformer
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer

# for data preprocessing and pipeline creation
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

# for model training, tuning, and evaluation
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV

import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    classification_report
)

# for model serialization
import joblib

# import mlflow for expermemintation and logging
import mlflow
import mlflow.sklearn


# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError


# -------------------------
# 1. Configure parameters
# -------------------------
DATA_PATH = "capstone_project3/data/engine_data.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20   # final test split
MLFLOW_EXPERIMENT = "PredictiveMaintenanceOnGitHub"


# initialize mlflow - set tracking uri and experiment name
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment(MLFLOW_EXPERIMENT)

# --------------------------------------------------
# 2. Load Train and test data from Hugging Face Space
# --------------------------------------------------

# Initialize Hiugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))

# update path variables with data paths for train and test data sets
Xtrain_path = "hf://datasets/harishsohani/AIMLPredictiveMaintenance/Xtrain.csv"
Xtest_path  = "hf://datasets/harishsohani/AIMLPredictiveMaintenance/Xtest.csv"
ytrain_path = "hf://datasets/harishsohani/AIMLPredictiveMaintenance/ytrain.csv"
ytest_path  = "hf://datasets/harishsohani/AIMLPredictiveMaintenance/ytest.csv"


# load train and test data
X_train = pd.read_csv(Xtrain_path)
X_test = pd.read_csv(Xtest_path)
y_train = pd.read_csv(ytrain_path)
y_test = pd.read_csv(ytest_path)

# print shape of train and test data (input variables)
print("\nShapes: X_train", X_train.shape, "X_test", X_train.shape)

# print shape of train and test data (target variable)
print("\nShapes: y_train", y_train.shape, "y_test", y_test.shape)


# --------------------------------------------------
# 4. Compute scale_pos_weight for XGBoost
#    Note: data set is imbalanced
# --------------------------------------------------

# convert both to series
y_train = y_train.squeeze()
y_test = y_test.squeeze()


# ----------------------------
# 6. Build model and pipeline
# ----------------------------
xgb_model  = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    reg_alpha=0.1,
    reg_lambda=5.0,
    min_child_weight=1,
    scale_pos_weight=1.2,
    subsample=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# --------------------------------------
# 8. Run RandomizedSearchCV with MLFlow
# --------------------------------------

with mlflow.start_run(run_name="xgb_pipeline_test"):
    xgb_model.fit(X_train, y_train)

    best_params = xgb_model.get_params()

    # log best parameters
    mlflow.log_params(best_params)

    # print best parameters
    print("\nBest parameters found during training:")
    pprint(best_params)

    THRESHOLD = 0.5

    def eval_and_log(X_, y_, dataset_name, threshold=THRESHOLD):
        y_proba = xgb_model.predict_proba(X_)[:,1]
        y_pred = (y_proba >= threshold).astype(int)
        acc = accuracy_score(y_, y_pred)
        auc = roc_auc_score(y_, y_proba)
        report = classification_report(y_, y_pred, output_dict=True)
        cm = confusion_matrix(y_, y_pred)

        mlflow.log_metric(f"{dataset_name}_accuracy", float(acc))
        mlflow.log_metric(f"{dataset_name}_roc_auc", float(auc))
        mlflow.log_metric(f"{dataset_name}_threshold", float(threshold))

        if "1" in report:
            mlflow.log_metric(f"{dataset_name}_precision_pos", float(report["1"]["precision"]))
            mlflow.log_metric(f"{dataset_name}_recall_pos", float(report["1"]["recall"]))
            mlflow.log_metric(f"{dataset_name}_f1_pos", float(report["1"]["f1-score"]))

        print(f"\n{dataset_name} - acc: {acc:.4f} | roc_auc: {auc:.4f} | threshold={threshold}")
        print("confusion_matrix:\n", cm)
        print("classification_report:\n", classification_report(y_, y_pred))
        return {"acc": acc, "auc": auc, "cm": cm, "report": report}

    train_metrics = eval_and_log(X_train, y_train, "train")
    test_metrics = eval_and_log(X_test, y_test, "test")

    # Save the model locally
    model_path = "best_eng_fail_pred_model.joblib"
    joblib.dump(xgb_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")


    # ---------------------------------------
    # Upload file to Back End
    # ---------------------------------------

    # Upload to Hugging Face
    hf_repo_id    = "harishsohani/AIMLPredictiveMaintenanceBackEnd"

    try:
        create_repo (
            repo_id=hf_repo_id,           # Name of the space to be created for back end
            repo_type="space",            # repository type as 'space'
            space_sdk="docker",           # Specify the space SDK as "docker" to create a Docker space
            private=False                 # This is set as False, so that it is accessble to all
        )
        print(f"Backend space created: {hf_repo_id}")
    except Exception as e:
        # Handle potential errors during repository creation
        if "RepositoryAlreadyExistsError" in str(e):
            print("Repository already exists. Skipping creation.")
        else:
            print(f"Error in creating repository: {e}")

    # Upload model
    api.upload_file(
        path_or_fileobj="best_eng_fail_pred_model.joblib",
        path_in_repo="best_eng_fail_pred_model.joblib",
        repo_id=hf_repo_id,
        repo_type="space",
    )


    # display best pipeline
    # Configure scikit-learn to display estimators as diagrams
    sklearn.set_config(display='diagram')

    # Display the best pipeline directly
    xgb_model
