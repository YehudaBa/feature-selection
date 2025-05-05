import os

import pandas as pd

features_path = 'resources/SCANB.csv'
target_path = 'resources/sampleinfo_SCANB_t.csv'
output_path = ("resources/data")


def get_date(features_path, target_path):
    features_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), features_path)
    target_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), target_path)
    df_features = pd.read_csv(features_path)
    df_target = pd.read_csv(target_path)
    return preprocess_input(df_features, df_target)


# the input supposed to contains only one csv and one target col
def preprocess_input(df_features, df_target):
    df_features = df_features.set_index("Unnamed: 0").T.reset_index().rename \
        (columns={"index": "samplename"}).rename_axis(None, axis=1)
    df_target = df_target[["samplename", "ER", "Lympho"]]
    df_target["ER"] = df_target["ER"] - 1
    return df_features.merge(df_target, on="samplename", how="inner")


if __name__ == "__main__":
    df_data = get_date(features_path, target_path)
    df_data.to_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), f"{output_path}.csv"), index=False)
