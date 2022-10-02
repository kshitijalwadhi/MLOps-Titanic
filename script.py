import pandas as pd
import numpy as np
from scipy.stats import boxcox
from xgboost import XGBClassifier
import pickle


class DataIngestion:
    def __init__(self, path):
        self.path = path
        self.data = self.read_data()

    def read_data(self):
        data = pd.read_csv(self.path)
        return data

    def get_data(self):
        return self.data


class FeatureEngineering:
    def __init__(self, data):
        self.df = data

    def get_data(self):
        return self.df

    def perform_feature_engineering(self):
        # Dropping PassengerId since not really relevant
        self.df.drop("PassengerId", axis=1, inplace=True)

        # Pclass (1>2>3), belonging to higher class increases chances of survival. Hence encoding accordingly.
        self.df["Pclass"] = self.df["Pclass"].map({1: 1, 2: 0, 3: -1})

        # Title embedded in the second word of the name
        def get_title(name):
            postname = name.split(",")[1]
            title = postname.split(".")[0]
            return title.strip()

        self.df["Title"] = self.df["Name"].apply(get_title)
        to_keep = ["Mr", "Miss", "Mrs", "Master"]
        self.df["Title"] = self.df["Title"].apply(
            lambda x: x if x in to_keep else "Other"
        )

        # Filling the missing values in age with mean of subset df with same categorical features
        cat = ["Sex", "Pclass", "Title"]
        df_age = self.df[cat + ["Age"]]
        df_age_mean = round(df_age.dropna().groupby(cat, as_index=True).median(), 1)

        def get_age(var, sex, pclass, title):
            if np.isnan(var):
                mean = df_age_mean["Age"][sex][pclass][title]
            else:
                mean = var
            return mean

        df_age["Age2"] = df_age.apply(
            lambda x: get_age(x.Age, x.Sex, x.Pclass, x.Title), axis=1
        )
        self.df["Age"] = self.df.apply(
            lambda x: get_age(x.Age, x.Sex, x.Pclass, x.Title), axis=1
        )

        self.df["Age2"], lam_age = boxcox(self.df["Age"])

        def get_transform(var):
            return (var**lam_age - 1) / lam_age

        self.df["Age"] = self.df.apply(lambda x: get_transform(x.Age), axis=1)
        self.df.drop("Age2", axis=1, inplace=True)

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
            lambda x: get_family_size(x.Parch, x.SibSp), axis=1
        )
        self.df["FamilyType"] = self.df.apply(
            lambda x: get_family_type(x.FamilySize), axis=1
        )
        self.df.drop(["FamilySize", "Parch", "SibSp"], axis=1, inplace=True)

        # Fill the missing fare values
        self.df["Fare"].fillna(self.df["Fare"].mean(), inplace=True)
        self.df["Fare2"], lam_fare = boxcox(self.df["Fare"] + 0.0001)

        def get_transform(var):
            return (var**lam_fare - 1) / lam_fare

        self.df.drop(["Fare2"], axis=1, inplace=True)
        self.df["Fare"] = self.df.apply(lambda x: get_transform(x.Fare), axis=1)

        self.df.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)


class Model:
    def __init__(self, data):
        self.df = data
        self.df = pd.get_dummies(self.df, prefix="cat", drop_first=True)
        self.model = XGBClassifier(
            min_child_weight=3.25, gamma=0.5, subsample=0.5, max_depth=4
        )

    def get_training_data(self):
        Y_train = self.df["Survived"]
        X_train = self.df.drop("Survived", axis=1)
        return X_train, Y_train

    def train(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def dump_model(self):
        pickle.dump(self.model, open("model.pkl", "wb"))


if __name__ == "__main__":
    data = DataIngestion("train.csv")
    train_df = data.get_data()
    fe = FeatureEngineering(train_df)
    fe.perform_feature_engineering()
    train_df = fe.get_data()
    model = Model(train_df)
    X_train, Y_train = model.get_training_data()
    model.train(X_train, Y_train)
    model.dump_model()
