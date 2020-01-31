import pandas as pd
import numpy as np
import scipy
import os
import pytz
import datetime
from emoji.core import emoji_count
from .ngrams import make_char_ngrams_from_words, calculate_mean_log_prob_of_word, NGramCount, calculate_probability_of_word
from ..utils.io_utils import load_pickle
from typing import Tuple, Optional, Union, List, Dict

def ngram_features(
    df_train_X : pd.DataFrame, 
    df_test_X : Optional[pd.DataFrame] = None, 
    col_names : List[str] = ["name", "screen_name"],
    ns : Union[int, List[int]] = [2, 3],
    is_drop_original_cols : bool = True,
    ngram_dictionaries_path : Optional[str] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:

    if isinstance(ns, int):
        ns = [ns]

    original_col_names = df_train_X.columns

    ngram_dictionaries_path =  os.environ.get("NGRAM_DICTIONARIES_FOLDER", ngram_dictionaries_path)

    if ngram_dictionaries_path is None:
        raise RuntimeError("No path provided for ngram_dictionaries_folder")

    if df_train_X.size > 0:
        for n in ns:
            for col_name in col_names:
                try:
                    ngram_dictionary = load_pickle(os.path.join(ngram_dictionaries_path, f"{col_name}_char_{n}gram_probabilities.pkl"))
                except Exception as e:
                    print(f"{e}")
                    continue
                df_train_X[f"{col_name}_{n}gram_feature"] = df_train_X[col_name].apply(lambda x: calculate_probability_of_word(x, ngram_dictionary, n))

                if df_test_X is not None and df_test_X.size > 0:
                    df_test_X[f"{col_name}_{n}gram_feature"] = df_test_X[col_name].apply(lambda x: calculate_mean_log_prob_of_word(x, ngram_dictionary, n))

    if is_drop_original_cols:
        df_train_X.drop(columns = original_col_names, inplace = True)
        df_test_X.drop(columns = original_col_names, inplace = True)

    if df_test_X is not None:
        return df_train_X, df_test_X
    else:
        return df_train_X

def ngram_features_of_dataset_for_cv(
    splits : List[Tuple[List[int], List[int]]],
    df_train_X : pd.DataFrame, 
    df_test_X : Optional[pd.DataFrame] = None, 
    col_names : List[str] = ["screen_name"],
    ns : Union[int, List[int]] = 2,
    val_type : int = 2,
    is_drop_original_cols : bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:

    if isinstance(ns, int):
        ns = [ns]

    original_col_names = df_train_X.columns

    if df_train_X.size > 0:
        for n in ns:
            for col_name in col_names:
                ngram_counter = NGramCount(df_train_X[col_name].tolist(), n)
                for split in splits:
                    removed_words = df_train_X.loc[split[1], col_name].tolist()
                    ngram_dct = NGramCount.from_count_to_prob(ngram_counter.get_ngram_counts_with_removed_words(removed_words, is_sum = True), is_log = True)
                    df_train_X.loc[split[1], f"{col_name}_{n}gram_feature_of_dataset_for_cv"] = df_train_X.loc[split[1], col_name].apply(lambda x: calculate_mean_log_prob_of_word(x, ngram_dct, n))

                    if df_test_X is not None and df_test_X.size > 0:
                        df_test_X[f"{col_name}_{n}gram_feature_of_dataset_for_cv"] = df_test_X[col_name].apply(lambda x: calculate_mean_log_prob_of_word(x, ngram_dct, n))

    if is_drop_original_cols:
        df_train_X.drop(columns = original_col_names, inplace = True)
        df_test_X.drop(columns = original_col_names, inplace = True)

    if df_test_X is not None:
        return df_train_X, df_test_X
    else:
        return df_train_X


def ngram_features_of_dataset(
    df_train_X : pd.DataFrame, 
    df_test_X : Optional[pd.DataFrame] = None, 
    col_names : List[str] = ["name", "screen_name"],
    ns : Union[int, List[int]] = 2,
    val_type : int = 2,
    is_drop_original_cols : bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:

    if isinstance(ns, int):
        ns = [ns]

    original_col_names = df_train_X.columns

    if df_train_X.size > 0:
        for n in ns:
            for col_name in col_names:
                ngram_dct = make_char_ngrams_from_words(df_train_X[col_name].tolist(), n, val_type)
                df_train_X[f"{col_name}_{n}gram_feature_of_dataset"] = df_train_X[col_name].apply(lambda x: calculate_mean_log_prob_of_word(x, ngram_dct, n))

                if df_test_X is not None and df_test_X.size > 0:
                    df_test_X[f"{col_name}_{n}gram_feature_of_dataset"] = df_test_X[col_name].apply(lambda x: calculate_mean_log_prob_of_word(x, ngram_dct, n))

    if is_drop_original_cols:
        df_train_X.drop(columns = original_col_names, inplace = True)
        df_test_X.drop(columns = original_col_names, inplace = True)

    if df_test_X is not None:
        return df_train_X, df_test_X
    else:
        return df_train_X

def ngram_features_of_dataset_by_label_for_cv(
    splits : List[Tuple[List[int], List[int]]],
    df_train_X : pd.DataFrame, 
    df_train_y : pd.DataFrame,
    df_test_X : Optional[pd.DataFrame] = None, 
    col_names : List[str] = ["name", "screen_name"],
    target_col_name : str = "label",
    ns : Union[int, List[int]] = 2,
    val_type : int = 2,
    is_drop_original_cols : bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:

    if isinstance(ns, int):
        ns = [ns]

    original_col_names = df_train_X.columns

    if df_train_X.size > 0:
        for n in ns:
            for label in df_train_y[target_col_name].unique():
                for col_name in col_names:
                    ngram_counter = NGramCount(df_train_X.loc[df_train_y[target_col_name] == label, col_name], n)
                    for split in splits:
                        removed_words = df_train_X.loc[split[1], col_name].tolist()
                        ngram_dct_by_label = NGramCount.from_count_to_prob(ngram_counter.get_ngram_counts_with_removed_words(removed_words, is_sum = True), is_log = True)
                        df_train_X.loc[split[1], f"{col_name}_{n}gram_feature_of_dataset_by_label_{label}_for_cv"] = df_train_X.loc[split[1], col_name].apply(lambda x: calculate_mean_log_prob_of_word(x, ngram_dct_by_label, n))

                        if df_test_X is not None and df_test_X.size > 0:
                            df_test_X[f"{col_name}_{n}gram_feature_of_dataset_by_label_{label}_for_cv"] = df_test_X[col_name].apply(lambda x: calculate_mean_log_prob_of_word(x, ngram_dct_by_label, n))

    if is_drop_original_cols:
        df_train_X.drop(columns = original_col_names, inplace = True)
        df_test_X.drop(columns = original_col_names, inplace = True)

    if df_test_X is not None:
        return df_train_X, df_test_X
    else:
        return df_train_X

def ngram_features_of_dataset_by_label(
    df_train_X : pd.DataFrame, 
    df_train_y : pd.DataFrame,
    df_test_X : Optional[pd.DataFrame] = None, 
    col_names : List[str] = ["name", "screen_name"],
    target_col_name : str = "label",
    ns : Union[int, List[int]] = 2,
    val_type : int = 2,
    is_drop_original_cols : bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:

    if isinstance(ns, int):
        ns = [ns]

    original_col_names = df_train_X.columns

    if df_train_X.size > 0:
        for n in ns:
            for label in df_train_y[target_col_name].unique():
                for col_name in col_names:
                    ngram_dct_by_label = make_char_ngrams_from_words(df_train_X.loc[df_train_y[target_col_name] == label, col_name].tolist(), n, val_type)
                    df_train_X[f"{col_name}_{n}gram_feature_of_dataset_by_label_{label}"] = df_train_X[col_name].apply(lambda x: calculate_mean_log_prob_of_word(x, ngram_dct_by_label, n))

                    if df_test_X is not None and df_test_X.size > 0:
                        df_test_X[f"{col_name}_{n}gram_feature_of_dataset_by_label_{label}"] = df_test_X[col_name].apply(lambda x: calculate_mean_log_prob_of_word(x, ngram_dct_by_label, n))

    if is_drop_original_cols:
        df_train_X.drop(columns = original_col_names, inplace = True)
        df_test_X.drop(columns = original_col_names, inplace = True)

    if df_test_X is not None:
        return df_train_X, df_test_X
    else:
        return df_train_X