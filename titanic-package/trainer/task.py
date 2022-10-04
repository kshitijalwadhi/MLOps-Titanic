from google.cloud import bigquery, bigquery_storage
from sklearn.metrics import f1_score
import os, logging, pickle, argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox
from xgboost import XGBClassifier
import pickle
import subprocess


def load_data_from_bq(bq_uri: str) -> pd.DataFrame:
    """
    Loads data from BigQuery table (BQ) to a dataframe

            Parameters:
                    bq_uri (str): bq table uri. i.e: example_project.example_dataset.example_table
            Returns:
                    pandas.DataFrame: a dataframe with the data from GCP loaded
    """
    if not bq_uri.startswith("bq://"):
        raise Exception(
            "uri is not a BQ uri. It should be bq://project_id.dataset.table"
        )
    logging.info("reading bq data: {}".format(bq_uri))
    project, dataset, table = bq_uri.split(".")
    bqclient = bigquery.Client(project=project[5:])
    bqstorageclient = bigquery_storage.BigQueryReadClient()
    query_string = """
    SELECT * from {ds}.{tbl}
    """.format(
        ds=dataset, tbl=table
    )

    return (
        bqclient.query(query_string)
        .result()
        .to_dataframe(bqstorage_client=bqstorageclient)
    )


class FeatureEngineering:
    def __init__(self, data):
        self.df = data

    def get_data(self):
        return self.df

    def perform_feature_engineering(self):
        # Pclass (1>2>3), belonging to higher class increases chances of survival. Hence encoding accordingly.
        self.df["pclass"] = self.df["pclass"].map({1: 1, 2: 0, 3: -1})

        # Title embedded in the second word of the name
        def get_title(name):
            postname = name.split(",")[1]
            title = postname.split(".")[0]
            return title.strip()

        self.df["title"] = self.df["name"].apply(get_title)
        to_keep = ["Mr", "Miss", "Mrs", "Master"]
        self.df["title"] = self.df["title"].apply(
            lambda x: x if x in to_keep else "Other"
        )

        self.df = self.df[self.df["age"].notna()]

        # Feature Engineering from the SibSp and Parch features
        def get_family_size(Parch, SibSp):
            return Parch + SibSp + 1

        def get_family_type(var):
            if var == 1:
                return "alone"
            elif var <= 4:
                return "small"
            else:
                return "big"

        self.df["FamilySize"] = self.df.apply(
            lambda x: get_family_size(x.parch, x.sibsp), axis=1
        )
        self.df["FamilyType"] = self.df.apply(
            lambda x: get_family_type(x.FamilySize), axis=1
        )
        self.df.drop(["FamilySize", "parch", "sibsp"], axis=1, inplace=True)

        # Fill the missing fare values
        self.df = self.df[self.df["fare"].notna()]

        self.df.drop(
            [
                "name",
                "ticket",
                "cabin",
                "boat",
                "body",
                "home_dest",
                "age",
                "fare",
                "embarked",
            ],
            axis=1,
            inplace=True,
        )


class Model:
    def __init__(self):
        self.model = XGBClassifier(
            min_child_weight=3.25, gamma=0.5, subsample=0.5, max_depth=4
        )

    def train(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, Y_test):
        return f1_score(Y_test, self.predict(X_test), average="macro")

    def dump_model(self):
        pickle.dump(self.model, open(f"model.pkl", "wb"))


def data_selection(df):
    df = pd.get_dummies(df, prefix="cat", drop_first=True)
    Y = df["survived"]
    X = df.drop("survived", axis=1)
    return X, Y


# Define all the command line arguments your model can accept for training
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_format",
        choices=["csv", "bigquery"],
        help="format of data uri csv for gs:// paths and bigquery for project.dataset.table formats",
        type=str,
        default=os.environ["AIP_DATA_FORMAT"]
        if "AIP_DATA_FORMAT" in os.environ
        else "csv",
    )
    parser.add_argument(
        "--training_data_uri",
        help="location of training data in either gs:// uri or bigquery uri",
        type=str,
        default=os.environ["AIP_TRAINING_DATA_URI"]
        if "AIP_TRAINING_DATA_URI" in os.environ
        else "",
    )

    parser.add_argument(
        "--testing_data_uri",
        help="location of testing data in either gs:// uri or bigquery uri",
        type=str,
        default=os.environ["AIP_TESTING_DATA_URI"]
        if "AIP_TESTING_DATA_URI" in os.environ
        else "",
    )

    parser.add_argument(
        "--full_dataset_path",
        help="location of testing data in either gs:// uri or bigquery uri",
        type=str,
    )

    parser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )

    args = parser.parse_args()
    arguments = args.__dict__

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    logging.info("Data format: {}".format(arguments["data_format"]))
    logging.info("Training data uri: {}".format(arguments["training_data_uri"]))

    logging.info("Loading {} data".format(arguments["data_format"]))
    if arguments["data_format"] == "bigquery":
        print(arguments["training_data_uri"])
        # df_train = load_data_from_bq(arguments["training_data_uri"])
        # df_test = load_data_from_bq(arguments["testing_data_uri"])
        full_dataset = load_data_from_bq(arguments["full_dataset_path"])
    else:
        raise ValueError("Invalid data type ")

    # split data to train and test
    df_train, df_test = train_test_split(full_dataset, test_size=0.2, random_state=42)

    fe = FeatureEngineering(df_train)
    fe.perform_feature_engineering()
    df_train = fe.get_data()

    fe = FeatureEngineering(df_test)
    fe.perform_feature_engineering()
    df_test = fe.get_data()

    logging.info("Running feature selection")
    X_train, y_train = data_selection(df_train)
    X_test, y_test = data_selection(df_test)

    model = Model()
    model.train(X_train, y_train)

    # Add code for dumping the model to GCS
    model.dump_model()

    gcs_bucket = os.environ["BUCKET_NAME"]
    gcs_model_path = f"gs://{gcs_bucket}/model/model.pkl"

    subprocess.run(
        [
            "gsutil",
            "-m",
            "cp",
            "-r",
            "model.pkl",
            gcs_model_path,
        ]
    )

    logging.info("Evaluating model")
    f1 = model.evaluate(X_test, y_test)

    logging.info("F1 score: {}".format(f1))

    logging.info("Training job completed. Exiting...")
