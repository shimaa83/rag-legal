import mlflow

MLFLOW_TRACKING_URI = "http://localhost:5000"

DVC_DATA_HASH = "5086ef5f16f15d856278db943e58e948"
GIT_COMMIT = "8b0ea07b622a881195df9f6680de90d01797fe6f"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("legal-rag")

with mlflow.start_run(run_name="dvc-data-version-test"):
    mlflow.set_tag("dvc_data_hash", DVC_DATA_HASH)
    mlflow.set_tag("git_commit", GIT_COMMIT)
    mlflow.set_tag("dataset", "Egyptian Civil Code")
    mlflow.set_tag("dataset_path", "data/raw/egyptian_civil_code.pdf")

    print("MLflow run created successfully.")
    print(f"DVC hash: {DVC_DATA_HASH}")
    print(f"Git commit: {GIT_COMMIT}")