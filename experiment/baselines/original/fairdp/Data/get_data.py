import pandas as pd
from sklearn.preprocessing import LabelEncoder


def get_adult(dmode: str = "none", rat: int = -1, seed: int = 2605):
    header = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]
    label_dict = {
        " <=50K": "<=50K",
        " >50K": ">50K",
        " <=50K.": "<=50K",
        " >50K.": ">50K",
    }
    train_df = pd.read_csv("Data/Adult/adult.data", header=None)
    test_df = pd.read_csv("Data/Adult/adult.test", skiprows=1, header=None)
    all_data = pd.concat([train_df, test_df], axis=0)
    all_data.columns = header

    def hour_per_week(x):
        if x <= 19:
            return "0"
        elif (x > 19) & (x <= 29):
            return "1"
        elif (x > 29) & (x <= 39):
            return "2"
        elif x > 39:
            return "3"

    def age(x):
        if x <= 24:
            return "0"
        elif (x > 24) & (x <= 34):
            return "1"
        elif (x > 34) & (x <= 44):
            return "2"
        elif (x > 44) & (x <= 54):
            return "3"
        elif (x > 54) & (x <= 64):
            return "4"
        else:
            return "5"

    def country(x):
        if x == " United-States":
            return 0
        else:
            return 1

    all_data["hours-per-week"] = all_data["hours-per-week"].map(
        lambda x: hour_per_week(x)
    )
    all_data["age"] = all_data["age"].map(lambda x: age(x))
    all_data["native-country"] = all_data["native-country"].map(lambda x: country(x))
    all_data = all_data.drop(
        [
            "fnlwgt",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "capital-gain",
            "capital-loss",
        ],
        axis=1,
    )
    temp = pd.get_dummies(all_data["age"], prefix="age")
    all_data = pd.concat([all_data, temp], axis=1)
    all_data = all_data.drop("age", axis=1)
    temp = pd.get_dummies(all_data["workclass"], prefix="workclass")
    all_data = pd.concat([all_data, temp], axis=1)
    all_data = all_data.drop("workclass", axis=1)
    temp = pd.get_dummies(all_data["education"], prefix="education")
    all_data = pd.concat([all_data, temp], axis=1)
    all_data = all_data.drop("education", axis=1)
    temp = pd.get_dummies(all_data["race"], prefix="race")
    all_data = pd.concat([all_data, temp], axis=1)
    all_data = all_data.drop("race", axis=1)
    temp = pd.get_dummies(all_data["hours-per-week"], prefix="hour")
    all_data = pd.concat([all_data, temp], axis=1)
    all_data = all_data.drop("hours-per-week", axis=1)
    all_data["income"] = all_data["income"].map(label_dict)
    lb = LabelEncoder()
    all_data["sex"] = lb.fit_transform(all_data["sex"].values)
    lb = LabelEncoder()
    all_data["income"] = lb.fit_transform(all_data["income"].values)

    return all_data.copy()


def get_ccc(dmode: str = "none", rat: int = -1, seed: int = 2605):
    df = pd.read_excel("Data/CCC/ccc.xls")
    header = df.iloc[0]
    df = df[1:]
    df.columns = header
    df["y"] = df["default payment next month"].astype(int)
    df = df.drop(["ID", "default payment next month"], axis=1)
    gender_dict = {1: 0, 2: 1}
    df["SEX"] = df["SEX"].map(gender_dict).astype(int)
    cont_col = ["LIMIT_BAL", "AGE"] + [col for col in df.columns if "AMT" in col]
    cat_col = [col for col in df.columns if col not in cont_col + ["SEX", "y"]]
    df = pd.get_dummies(df, columns=cat_col, prefix_sep="_")
    for col in cont_col:
        df[col] = df[col].astype(float)
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-10)
    return df.copy().reset_index(drop=True)


def get_utk(dmode: str = "none", rat: int = -1, seed: int = 2605):

    utk_data_path = "Data/UTK/age_gender.gz"
    df = pd.read_csv(utk_data_path, compression="gzip")
    df["age"] = df["age"] > 35
    df["age"] = df["age"].astype(int)

    df["ethnicity"] = df["ethnicity"]
    df["ethnicity"] = df["ethnicity"] == 0
    df["ethnicity"] = df["ethnicity"].astype(int)
    return df
