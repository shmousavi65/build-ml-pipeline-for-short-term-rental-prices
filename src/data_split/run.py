import os
import logging
import argparse
import tempfile
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):

    run = wandb.init(job_type="data_split")

    wandb_input_data_path = os.path.join(os.environ["WANDB_PROJECT"], args.input)
    logger.info(f"downloading the input artifact {wandb_input_data_path}")
    input_data_path = run.use_artifact(wandb_input_data_path).file()

    data_df = pd.read_csv(input_data_path)

    logger.info(f"splitting data into train and test")
    splits = {}
    splits["trainval"], splits["test"] = train_test_split(
        data_df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=data_df[args.stratify_by] if args.stratify_by != 'null' else None,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        for split, df in splits.items():

            # Make the artifact name from the provided root plus the name of the split
            artifact_name = f"data_{split}.csv"

            # Get the path on disk within the temp directory
            temp_path = os.path.join(temp_dir, artifact_name)

            logger.info(f"Uploading the {split} dataset to {artifact_name}")

            # Save then upload to W&B
            df.to_csv(temp_path)

            artifact = wandb.Artifact(
                name=artifact_name,
                type="raw_data",
                description=f"{split} split of dataset {args.input}",
            )
            artifact.add_file(temp_path)

            logger.info("Logging artifact")
            run.log_artifact(artifact)

            artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a dataset into train and test",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input",
        type=str,
        help="name for the input data artifact to be splitted",
        required=True
    )

    parser.add_argument(
        "--test_size",
        type=float,
        help="Fraction of data to be used for test split",
        required=True
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="integer number used for random generator",
        required=True
    )

    parser.add_argument(
        "--stratify_by",
        help="If set, it is the name of a column to use for stratified splitting",
        type=str,
        required=False,
        default='null'  
    )

    args = parser.parse_args()

    go(args)


