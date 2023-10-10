import pandas as pd
import argparse
import logging
import wandb
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="test_model")

    logger.info("Downloading and reading the model")
    model_path = run.use_artifact(args.model_artifact).download()
    pipe = mlflow.sklearn.load_model(model_path)

    logger.info("Downloading test artifact")
    test_data_path = run.use_artifact(args.test_data).file()
    df = pd.read_csv(test_data_path, low_memory=False)
    
    logger.info("Extracting target from dataframe")
    X_test = df.copy()
    y_test = X_test.pop("price")

    logger.info("Scoring")
    r_squared = pipe.score(X_test, y_test)

    run.summary['r2'] = r_squared

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="test the input model")

    parser.add_argument(
        "--model_artifact",
        type=str,
        help="Artifact containing the model."
    )

    parser.add_argument(
        "--test_data",
        type=str,
        help="Artifact containing the test data."
    )

    args = parser.parse_args()
    go(args)