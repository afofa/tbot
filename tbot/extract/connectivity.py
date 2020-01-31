import pandas as pd
import numpy as np
import scipy
from .metadata import user_metadata_features
from typing import Optional, Union, Tuple, List, Dict

def user_connectivity_features(
    df_connectivity : pd.DataFrame,
    df_train_X : pd.DataFrame, 
    df_test_X : Optional[pd.DataFrame] = None,
    is_drop_original_cols : bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:

    def func(df_connectivity_metadata_features: pd.DataFrame, df: pd.DataFrame, df_connectivity_column_names: List[str]) -> pd.DataFrame:
        res = []
        for main_user_id_str in df["user_id_str"].unique():
            user_res = {"user_id_str": main_user_id_str}
            for connection_type in df_connectivity_metadata_features["connection_type"].unique():
                df_tmp = df_connectivity_metadata_features.loc[(df_connectivity_metadata_features["main_user_id_str"] == main_user_id_str) & (df_connectivity_metadata_features["connection_type"] == connection_type)]
                df_tmp.drop(columns = df_connectivity_column_names, inplace = True)
                
                for col in df_tmp.columns:
                    col_res = distribution_feature(df_tmp[col].tolist())
                    col_res = {f"{connection_type}s.{col}.{k}": v for k, v in col_res.items()}
                    user_res = {**user_res, **col_res}
            res.append(user_res)

        df_res = pd.DataFrame(res)
        df = df.merge(df_res)
        return df

    column_names = df_train_X.columns

    df_connectivity_column_names = df_connectivity.columns
    df_connectivity_metadata_features = user_metadata_features(df_connectivity, None, is_drop_original_cols = False)

    if df_train_X.size > 0:
        df_train_X = func(df_connectivity_metadata_features, df_train_X, df_connectivity_column_names)

    if df_test_X is not None and df_test_X.size > 0:
        df_test_X = func(df_connectivity_metadata_features, df_test_X, df_connectivity_column_names)

    if is_drop_original_cols:
        df_train_X.drop(columns = column_names, inplace = True)
        df_test_X.drop(columns = column_names, inplace = True)

    if df_test_X is not None:
        return df_train_X, df_test_X
    else:
        return df_train_X

def distribution_feature(
    nums : Union[List[float], np.ndarray],
) -> Dict[str, float]:
    '''
    min, max, median, mean, std deviation, skewness, kurtosis, entropy
    '''

    if isinstance(nums, list):
        nums = np.array(nums)

    if nums.size <= 0:
        res_dict = {
            "min" : 0.0,
            "max" : 0.0,
            "median" : 0.0,
            "mean" : 0.0,
            "std" : 0.0,
            "skewness" : 0.0,
            "kurtosis" : 0.0,
            "entropy" : 0.0,
        }

    else:
        _, counts = np.unique(nums, return_counts=True)

        res_dict = {
            "min" : np.min(nums),
            "max" : np.max(nums),
            "median" : np.median(nums),
            "mean" : np.mean(nums),
            "std" : np.std(nums),
            "skewness" : scipy.stats.skew(nums),
            "kurtosis" : scipy.stats.kurtosis(nums),
            "entropy" : scipy.stats.entropy(counts),
        }

    return res_dict

### User connectivity features