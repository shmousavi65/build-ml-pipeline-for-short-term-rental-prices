import os
import pandas as pd
import wandb
import logging
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):

    run = wandb.init(job_type="basic_cleaning")

    wandb_input_data_path = os.path.join(os.environ["WANDB_PROJECT"], args.input_artifact)
    logger.info(f"downloading the input data from {wandb_input_data_path}")
    input_data_path = run.use_artifact(wandb_input_data_path).file()

    data_df = pd.read_csv(input_data_path)

    logger.info(f"dropping price outliers out of  min {args.min_price} and max {args.max_price} ")
    idx = data_df['price'].between(args.min_price, args.max_price)
    data_df = data_df[idx].copy()

    logger.info(f"dropping longtitude outliers out of  min {-74.25} and max {-73.50} ")
    logger.info(f"dropping latitude outliers out of  min {40.5} and max {41.2} ")
    idx = data_df['longitude'].between(-74.25, -73.50) & data_df['latitude'].between(40.5, 41.2)
    data_df = data_df[idx].copy()


    file_name = "clean_sample.csv"
    data_df.to_csv(file_name, index=False)
    
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )
    artifact.add_file(file_name)

    logger.info(f"stroing the cleaned data as artifact {args.output_artifact}")
    wandb.log_artifact(artifact)

    os.remove(file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Do the basic cleaning",
        fromfile_prefix_chars="@"
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="name for the raw data artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="name for the cleaned output data artifact",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="type of the cleaned output data artifact",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="description of the cleaned output data artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="lower bound for price (output) outlier removal",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="upper bound for price (output) outlier removal",
        required=True
    )

    args = parser.parse_args()

    go(args)


