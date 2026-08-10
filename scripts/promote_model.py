import logging
import mlflow
import os

# -------------------------
# Logging configuration
# -------------------------

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("model_evaluation_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# -------------------------
# MLflow / DagsHub setup
# -------------------------

dagshub_token = os.getenv("DAGSHUB_PAT")

if not dagshub_token:
    raise EnvironmentError(
        "DAGSHUB_PAT environment variable is not set"
    )

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "singhsumitt05"
repo_name = "mlops-mini"

mlflow.set_tracking_uri(
    f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
)


# -------------------------
# Get latest model version
# -------------------------

def get_latest_model_version(
    model_name: str,
    stage: str = "Staging"
):
    client = mlflow.MlflowClient()

    latest_versions = client.get_latest_versions(
        model_name,
        stages=[stage]
    )

    return (
        latest_versions[0].version
        if latest_versions
        else None
    )


# -------------------------
# Promote model
# -------------------------

def promote_model(model_name: str, version: int):

    try:
        client = mlflow.MlflowClient()

        logger.debug(
            f"Promoting model {version} "
            f"to Production: {model_name}"
        )

        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )

        logger.debug(
            f"Model {version} promoted to Production: "
            f"{model_name}"
        )

    except Exception as e:
        logger.error(
            f"Error promoting model version "
            f"{version} to Production: {e}"
        )
        raise


# -------------------------
# Main
# -------------------------

if __name__ == "__main__":

    new_model_name = "my_model"

    new_model_version = get_latest_model_version(
        new_model_name,
        stage="Staging"
    )

    if new_model_version is None:
        raise ValueError(
            f"No Staging version found for "
            f"{new_model_name}"
        )

    promote_model(
        new_model_name,
        new_model_version
    )