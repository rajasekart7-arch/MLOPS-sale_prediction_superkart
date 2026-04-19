
import os
import joblib
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

from sklearn.compose import make_column_transformer
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlflow.set_tracking_uri("file:./mlruns")

mlflow.set_experiment("mlops-training-experiment")

def adj_r2_score(n_features: int, targets, predictions) -> float:
    """Compute adjusted R-squared."""
    r2 = r2_score(targets, predictions)
    n = len(targets)
    k = n_features
    if n - k - 1 <= 0:
        return r2
    return 1 - ((1 - r2) * (n - 1) / (n - k - 1))


def model_performance_regression(model, predictors: pd.DataFrame, target) -> dict:
    """Compute regression metrics for a fitted pipeline."""
    pred = model.predict(predictors)

    transformed_X = model.named_steps["preprocessor"].transform(predictors)
    n_features = transformed_X.shape[1]

    return {
        "RMSE": float(np.sqrt(mean_squared_error(target, pred))),
        "MAE": float(mean_absolute_error(target, pred)),
        "R-squared": float(r2_score(target, pred)),
        "Adj. R-squared": float(adj_r2_score(n_features, target, pred)),
        "MAPE": float(mean_absolute_percentage_error(target, pred)),
    }


def main():
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found. Add it to your environment or .env file.")

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("mlops-training-experiment")

    api = HfApi(token=hf_token)

    xtrain_path = "hf://datasets/Rajse/Superkart-Dataset/Xtrain.csv"
    xtest_path = "hf://datasets/Rajse/Superkart-Dataset/Xtest.csv"
    ytrain_path = "hf://datasets/Rajse/Superkart-Dataset/ytrain.csv"
    ytest_path = "hf://datasets/Rajse/Superkart-Dataset/ytest.csv"

    Xtrain = pd.read_csv(xtrain_path)
    Xtest = pd.read_csv(xtest_path)
    ytrain = pd.read_csv(ytrain_path)
    ytest = pd.read_csv(ytest_path)

    if isinstance(ytrain, pd.DataFrame) and ytrain.shape[1] == 1:
        ytrain = ytrain.iloc[:, 0]

    if isinstance(ytest, pd.DataFrame) and ytest.shape[1] == 1:
        ytest = ytest.iloc[:, 0]

    numeric_features = Xtrain.select_dtypes(include=np.number).columns.tolist()
    categorical_features = Xtrain.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(handle_unknown="ignore"), categorical_features),
        remainder="drop",
    )

    xgb_regressor = xgb.XGBRegressor(
        random_state=42,
        objective="reg:squarederror",
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", xgb_regressor),
        ]
    )

    param_grid = {
        "model__n_estimators": [100],
        "model__learning_rate": [0.01],
        "model__max_depth": [3],
        "model__subsample": [0.7],
        "model__colsample_bytree": [0.7],
    }

    os.makedirs("model", exist_ok=True)

    with mlflow.start_run():
        grid_search = GridSearchCV(
            estimator=model_pipeline,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            scoring="neg_mean_squared_error",
            verbose=0,
        )
        grid_search.fit(Xtrain, ytrain)

        results = grid_search.cv_results_
        for i in range(len(results["params"])):
            with mlflow.start_run(nested=True):
                mlflow.log_params(results["params"][i])
                mlflow.log_metric("mean_test_score", float(results["mean_test_score"][i]))
                mlflow.log_metric("std_test_score", float(results["std_test_score"][i]))

        mlflow.log_params(grid_search.best_params_)

        best_model = grid_search.best_estimator_

        train_report = model_performance_regression(best_model, Xtrain, ytrain)
        test_report = model_performance_regression(best_model, Xtest, ytest)

        mlflow.log_metrics({
            "train_rmse": train_report["RMSE"],
            "train_mae": train_report["MAE"],
            "train_r2": train_report["R-squared"],
            "train_adj_r2": train_report["Adj. R-squared"],
            "train_mape": train_report["MAPE"],
            "test_rmse": test_report["RMSE"],
            "test_mae": test_report["MAE"],
            "test_r2": test_report["R-squared"],
            "test_adj_r2": test_report["Adj. R-squared"],
            "test_mape": test_report["MAPE"],
        })

        model_path = "model/sales_prediction_model_v1_0.joblib"
        joblib.dump(best_model, model_path)

        mlflow.log_artifact(model_path, artifact_path="model")
        mlflow.sklearn.log_model(best_model, artifact_path="sklearn-model")

        print(f"Model saved locally at: {model_path}")

        repo_id = "Rajse/Superkart-model"
        repo_type = "model"

        try:
            api.repo_info(repo_id=repo_id, repo_type=repo_type)
            print(f"Repo '{repo_id}' already exists. Using it.")
        except RepositoryNotFoundError:
            print(f"Repo '{repo_id}' not found. Creating new repo...")
            create_repo(
                repo_id=repo_id,
                repo_type=repo_type,
                private=False,
                token=hf_token,
            )
            print(f"Repo '{repo_id}' created.")

        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="sales_prediction_model_v1_0.joblib",
            repo_id=repo_id,
            repo_type=repo_type,
        )

        print("Model uploaded to Hugging Face successfully.")


if __name__ == "__main__":
    main()
