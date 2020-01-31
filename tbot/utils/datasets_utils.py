import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tweepy.models import User
from ..auth import get_api_object
from ..fetch import fetch_users_metadata
from ..parse import parse_users_metadata
from typing import Dict, List, Tuple, Optional

def encode_labels(df : pd.DataFrame, target_col_name : str = "label") -> Tuple[pd.DataFrame, LabelEncoder]:
    le = LabelEncoder()
    encoded_labels = le.fit_transform(df[target_col_name].tolist())
    df[target_col_name] = encoded_labels
    return df, le

def encode_labels_2(df1 : pd.DataFrame, df2 : pd.DataFrame, target_col_name : str = "label") -> Tuple[pd.DataFrame, LabelEncoder]:
    le = LabelEncoder()
    df1_labels = df1[target_col_name].tolist() if target_col_name in df1.columns else []
    df2_labels = df2[target_col_name].tolist() if target_col_name in df2.columns else []
    labels = df1_labels + df2_labels
    le.fit(labels)
    if df1.size > 0:
        df1[target_col_name] = le.transform(df1[target_col_name].tolist())
    if df2.size > 0:
        df2[target_col_name] = le.transform(df2[target_col_name].tolist())
    return df1, df2, le

def decode_labels(df : pd.DataFrame, le : LabelEncoder, target_col_name : str = "label") -> pd.DataFrame:
    decoded_labels = le.inverse_transform(df[target_col_name].tolist())
    df[target_col_name] = decoded_labels
    return df

def split_dataset(df : pd.DataFrame, test_set_ratio : float = 0.2, target_col_name : str = "label") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if test_set_ratio >= 1 or test_set_ratio < 0:
        raise RuntimeError(f"test_set_ratio should be a float in range [0, 1), passed {test_set_ratio}")
    elif test_set_ratio == 0:
        print(f"test_set_ratio is passed as {test_set_ratio}, test set DataFrames will be emtpy")
        df_train = df
        df_train_X, df_train_y = split_dataset_2(df_train, target_col_name = target_col_name)
        df_test_X = pd.DataFrame()
        df_test_y = pd.DataFrame()
    else:
        df_train, df_test = train_test_split(df, test_size = test_set_ratio, shuffle = True, stratify = df[target_col_name].tolist())
        df_train.reset_index(drop = True, inplace = True)
        df_test.reset_index(drop = True, inplace = True)
        df_train_X, df_train_y, df_test_X, df_test_y = split_dataset_2(df_train, df_test, target_col_name = target_col_name)

    return df_train_X, df_train_y, df_test_X, df_test_y

def split_dataset_2(df_train : pd.DataFrame, df_test : Optional[pd.DataFrame] = None, target_col_name : str = "label") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_train_X = df_train[df_train.columns.difference([target_col_name])]
    df_train_y = df_train[[target_col_name]]

    if df_test is not None:
        if df_test.size > 0:
            df_test_X = df_test[df_test.columns.difference([target_col_name])]
            df_test_y = df_test[[target_col_name]]
        
        else:
            df_test_X = pd.DataFrame()
            df_test_y = pd.DataFrame()

        return df_train_X, df_train_y, df_test_X, df_test_y

    else:
        return df_train_X, df_train_y

def enrich_with_user_metadata(
    df : pd.DataFrame, 
    creds_filepath : str = "",
) -> Dict[str, List[User]]:
    api = tweepy_api_object(filepath = creds_filepath)
    users_metadata = fetch_users_metadata(api, df["user_id"].tolist())
    df_enrichment = pd.DataFrame(parse_users_metadata(users_metadata))
    df_joined = pd.merge(df, df_enrichment, how = "inner", left_on = "user_id", right_on = "user_id_str")

    return df_joined

def get_user_metadata_by_label(
    df : pd.DataFrame, 
    creds_filepath : str = "",
) -> Dict[str, List[User]]:
    api = tweepy_api_object(filepath = creds_filepath)

    labels = df["label"].unique()
    
    users_metadata_by_label = {}
    for i, label in enumerate(labels):
        print(f"Getting users metadata for label = {label}, {i+1}/{len(labels)}")
        users_metadata_by_label[label] = fetch_users_metadata(api, df[df["label"] == label]["user_id"].tolist())

    return users_metadata_by_label