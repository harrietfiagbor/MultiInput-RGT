import pandas as pd
import numpy as np
import os
import pickle
import logging
import boto3


def load_pickle(remote_filename="repickled.p", force_reload=False):
    """
    Load pickle file from s3 bucket
    Args:
      remote_filename: name of pickle file in s3
    Returns:
      pickle_data:
    """
    pickle_file = "pickle.p"
    if force_reload or not os.path.isfile(pickle_file):
        logging.info("Downloading pickle...")
        bucket = "semler-input-data"
        boto3.client("s3").download_file(bucket, remote_filename, pickle_file)
        logging.info("Pickle downloaded.")

    logging.info("Loading data...")
    pickle_data = {}
    if os.path.isfile(pickle_file):
        with open(pickle_file, "rb") as handle:
            pickle_data = pickle.load(handle)
            logging.info(f"Loaded pickle: {pickle_file}")

    return pickle_data


def load_meta_data():

    bucket = "semler-input-data"
    filename = "metadata.xlsx"
    boto3.client("s3").download_file(bucket, filename, filename)

    meta_data_df = pd.read_excel(
        filename,
        usecols=[
            "Subject ID",
            "Use for Algorithm Training [T/F]",
            "Final EF%",
            "Strain, Global Longitudinal [-%]",
            "Clinic ID",
            "Age [yrs rounded to nearest]",
        ],
    )
    meta_data_df.rename(
        columns={
            "Subject ID": "Subject_ID",
            "Use for Algorithm Training [T/F]": "Use_4_Train",
            "Final EF%": "LVEF",
            "Strain, Global Longitudinal [-%]": "Strain",
            "Clinic ID": "Clinic",
            "Age [yrs rounded to nearest]": "Age",
        },
        inplace=True,
    )
    first_data_row = meta_data_df["Subject_ID"].first_valid_index()
    meta_data_df = meta_data_df.iloc[first_data_row:]
    meta_data_df["Use_4_Train"] = np.where(
        meta_data_df["Use_4_Train"] != "T", False, True
    )

    return meta_data_df
