import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from ..datasets import get_raw_dataset, get_fetched_dataset, make_fetched_dataset
from ..utils.datasets_utils import enrich_with_user_metadata, encode_labels, encode_labels_2, decode_labels, split_dataset, split_dataset_2
from ..extract.metadata import user_metadata_features, user_metadata_features_from_yang_2019
from ..extract.likelihood import ngram_features, ngram_features_of_dataset, ngram_features_of_dataset_by_label, ngram_features_of_dataset_for_cv, ngram_features_of_dataset_by_label_for_cv
from ..classification import BotClassifier
from ..dimensonality_reduction import BotDimensionReducer
from typing import Union, List, Optional, Dict

def classification_pipeline(
    dataset_name : str,
    creds_filepath : str = "creds/twitter_credentials_tugrulcan.json",
    test_set_ratio : float = 0.2,
    ngram_features_col_names : List[str] = ["screen_name"],
    ngram_features_ns : Union[int, List[int]] = [2],
    ngram_features_of_dataset_col_names : List[str] = ["screen_name"],
    ngram_features_of_dataset_ns : Union[int, List[int]] = [2],
    ngram_features_of_dataset_by_label_col_names : List[str] = ["screen_name"],
    ngram_features_of_dataset_by_label_ns : Union[int, List[int]] = [2],
    number_of_generations : int = 3,
    population_size : int = 10,
    scoring : str = "accuracy", # "accuracy", "f1", "precision", "recall", "roc_auc"
    number_of_folds_in_cv : int = 5,
    verbosity : int = 0, # 0, 1, 2, 3
    number_of_jobs : int = -1, # -1 = number of cores
    is_feature_importances : bool = False,
    is_export_best_pipeline : bool = False,
    exported_script_name : str = "best_pipeline.py",
) -> BotClassifier:

    # Check if fetched dataset exist
    df = get_fetched_dataset(dataset_name)

    # If not, make fetched dataset
    if df is None:
        df = make_fetched_dataset(dataset_name, creds_filepath)

    # Encode labels
    df, le = encode_labels(df)
    # print(df)

    # # Decode labels
    # df = decode_labels(df, le)
    # print(df)

    # Split dataset
    df_train_X, df_train_y, df_test_X, df_test_y = split_dataset(df, test_set_ratio = test_set_ratio)

    # Cross-validation object
    cv = StratifiedKFold(n_splits = number_of_folds_in_cv, random_state = None, shuffle = True)
    cv_splits = list(cv.split(df_train_X, df_train_y))

    # Save initial column names
    col_names = df_train_X.columns

    # Make user metadata features
    # df_train_X, df_test_X = user_metadata_features(df_train_X, df_test_X, is_drop_original_cols = False)
    df_train_X, df_test_X = user_metadata_features_from_yang_2019(df_train_X, df_test_X, is_drop_original_cols = False)

    # Make ngram features
    df_train_X, df_test_X = ngram_features(df_train_X, df_test_X, is_drop_original_cols = False, col_names = ngram_features_col_names, ns = ngram_features_ns)

    # Make ngram features of dataset
    df_train_X, df_test_X = ngram_features_of_dataset_for_cv(cv_splits, df_train_X, df_test_X, is_drop_original_cols = False, col_names = ngram_features_of_dataset_col_names, ns = ngram_features_of_dataset_ns)
    # df_train_X, df_test_X = ngram_features_of_dataset(df_train_X, df_test_X, is_drop_original_cols = False, col_names = ngram_col_names, ns = ngram_ns)

    # Make ngram features of dataset by label
    df_train_X, df_test_X = ngram_features_of_dataset_by_label_for_cv(cv_splits, df_train_X, df_train_y, df_test_X, is_drop_original_cols = False, col_names = ngram_features_of_dataset_by_label_col_names, ns = ngram_features_of_dataset_by_label_ns)
    # df_train_X, df_test_X = ngram_features_of_dataset_by_label(df_train_X, df_train_y, df_test_X, is_drop_original_cols = False, col_names = ngram_by_label_col_names, ns = ngram_by_label_ns)

    # Drop initial columns
    if df_train_X.size > 0:
        df_train_X.drop(columns = col_names, inplace = True)
    if df_test_X.size > 0:
        df_test_X.drop(columns = col_names, inplace = True)

    # Save as csv file
    featured_dataset_path = os.environ.get("FEATURED_DATASET_FOLDER")
    df_train_X.to_csv(f"{featured_dataset_path}/{dataset_name}_train_X.csv", index = False)
    df_test_X.to_csv(f"{featured_dataset_path}/{dataset_name}_test_X.csv", index = False)
    df_train_y.to_csv(f"{featured_dataset_path}/{dataset_name}_train_y.csv", index = False)
    df_test_y.to_csv(f"{featured_dataset_path}/{dataset_name}_test_y.csv", index = False)

    # Classify
    classifier = BotClassifier(number_of_generations, population_size, scoring, cv_splits, verbosity, number_of_jobs, is_feature_importances)
    classifier.fit(df_train_X, df_train_y)
    
    if is_feature_importances:
        feature_importances = classifier.get_feature_importances()
        print(feature_importances)

    if is_export_best_pipeline:
        classifier.export(exported_script_name)

    if df_test_X.size > 0:
        scores_dict = classifier.scores(df_test_X, df_test_y)
        print(scores_dict)
        return classifier, scores_dict

    else:
        return classifier, {}

def classification_pipeline_with_multiple_datasets(
    training_datasets_names : Union[str, List[str]],
    testing_datasets_names : Union[str, List[str]],
    creds_filepath : str = "creds/twitter_credentials_tugrulcan.json",
    ngram_features_col_names : List[str] = ["screen_name"],
    ngram_features_ns : Union[int, List[int]] = [2],
    ngram_features_of_dataset_col_names : List[str] = ["screen_name"],
    ngram_features_of_dataset_ns : Union[int, List[int]] = [2],
    ngram_features_of_dataset_by_label_col_names : List[str] = ["screen_name"],
    ngram_features_of_dataset_by_label_ns : Union[int, List[int]] = [2],
    number_of_generations : int = 3,
    population_size : int = 10,
    scoring : str = "accuracy", # "accuracy", "f1", "precision", "recall", "roc_auc"
    number_of_folds_in_cv : int = 5,
    verbosity : int = 0, # 0, 1, 2, 3
    number_of_jobs : int = -1, # -1 = number of cores
    is_feature_importances : bool = False,
    is_export_best_pipeline : bool = False,
    exported_script_name : str = "best_pipeline.py",
) -> BotClassifier:

    if isinstance(training_datasets_names, str):
        training_datasets_names = [training_datasets_names]

    if isinstance(testing_datasets_names, str):
        testing_datasets_names = [testing_datasets_names]

    list_df_trains = []

    for training_dataset_name in training_datasets_names:
        # Check if fetched dataset exist
        df = get_fetched_dataset(training_dataset_name)

        # If not, make fetched dataset
        if df is None:
            df = make_fetched_dataset(training_dataset_name, creds_filepath)

        list_df_trains.append(df)

    df_train = pd.concat(list_df_trains, sort = True)
    df_train.reset_index(drop = True, inplace = True)

    list_df_tests = []

    for testing_dataset_name in testing_datasets_names:
        # Check if fetched dataset exist
        df = get_fetched_dataset(testing_dataset_name)

        # If not, make fetched dataset
        if df is None:
            df = make_fetched_dataset(testing_dataset_name, creds_filepath)

        list_df_tests.append(df)

    df_test = pd.DataFrame() if len(list_df_tests) <= 0 else pd.concat(list_df_tests, sort = True)
    df_test.reset_index(drop = True, inplace = True)

    # Encode labels
    df_train, df_test, le = encode_labels_2(df_train, df_test)

    # # Decode labels
    # df_train = decode_labels(df_train, le)
    # df_test = decode_labels(df_test, le)

    # Split dataset
    df_train_X, df_train_y, df_test_X, df_test_y = split_dataset_2(df_train, df_test)

    # Cross-validation object
    cv = StratifiedKFold(n_splits = number_of_folds_in_cv, random_state = None, shuffle = True)
    cv_splits = list(cv.split(df_train_X, df_train_y))

    # Save initial column names
    col_names = df_train_X.columns

    # Make user metadata features
    df_train_X, df_test_X = user_metadata_features(df_train_X, df_test_X, is_drop_original_cols = False)
    # df_train_X, df_test_X = user_metadata_features_from_yang_2019(df_train_X, df_test_X, is_drop_original_cols = False)

    # Make ngram features
    df_train_X, df_test_X = ngram_features(df_train_X, df_test_X, is_drop_original_cols = False, col_names = ngram_features_col_names, ns = ngram_features_ns)

    # # Make ngram features of dataset
    df_train_X, df_test_X = ngram_features_of_dataset_for_cv(cv_splits, df_train_X, df_test_X, is_drop_original_cols = False, col_names = ngram_features_of_dataset_col_names, ns = ngram_features_of_dataset_ns)
    # df_train_X, df_test_X = ngram_features_of_dataset(df_train_X, df_test_X, is_drop_original_cols = False, col_names = ngram_features_of_dataset_col_names, ns = ngram_features_of_dataset_ns)

    # # Make ngram features of dataset by label
    df_train_X, df_test_X = ngram_features_of_dataset_by_label_for_cv(cv_splits, df_train_X, df_train_y, df_test_X, is_drop_original_cols = False, col_names = ngram_features_of_dataset_by_label_col_names, ns = ngram_features_of_dataset_by_label_ns)
    # df_train_X, df_test_X = ngram_features_of_dataset_by_label(df_train_X, df_train_y, df_test_X, is_drop_original_cols = False, col_names = ngram_features_of_dataset_by_label_col_names, ns = ngram_features_of_dataset_by_label_ns)

    # Drop initial columns
    if df_train_X.size > 0:
        df_train_X.drop(columns = col_names, inplace = True)
    if df_test_X.size > 0:
        df_test_X.drop(columns = col_names, inplace = True)

    # Save as csv file
    featured_dataset_path = os.environ.get("FEATURED_DATASET_FOLDER")
    df_train_X.to_csv(f"{featured_dataset_path}/{training_datasets_names}_train_X.csv", index = False)
    df_test_X.to_csv(f"{featured_dataset_path}/{training_datasets_names}_test_X.csv", index = False)
    df_train_y.to_csv(f"{featured_dataset_path}/{testing_datasets_names}_train_y.csv", index = False)
    df_test_y.to_csv(f"{featured_dataset_path}/{testing_datasets_names}_test_y.csv", index = False)

    # Cross-validation object
    cv = StratifiedKFold(n_splits = number_of_folds_in_cv, random_state = None, shuffle = True)
    cv_splits = list(cv.split(df_train_X, df_train_y))

    # Classify
    classifier = BotClassifier(number_of_generations, population_size, scoring, 5, verbosity, number_of_jobs, is_feature_importances)
    classifier.fit(df_train_X, df_train_y)

    if is_feature_importances:
        feature_importances = classifier.get_feature_importances()
        print(feature_importances)

    if is_export_best_pipeline:
        classifier.export(exported_script_name)

    if df_test_X.size > 0:
        scores_dict = classifier.scores(df_test_X, df_test_y)
        print(scores_dict)
        return classifier, scores_dict

    else:
        return classifier, cv_splits, df_train_X, df_train_y

def dimensonality_reduction_pipeline(
    dataset_name : str,
    creds_filepath : str = "creds/twitter_credentials_tugrulcan.json",
    ngram_features_col_names : List[str] = ["screen_name"],
    ngram_features_ns : Union[int, List[int]] = [2],
    ngram_features_of_dataset_col_names : List[str] = ["screen_name"],
    ngram_features_of_dataset_ns : Union[int, List[int]] = [2],
    ngram_features_of_dataset_by_label_col_names : List[str] = ["screen_name"],
    ngram_features_of_dataset_by_label_ns : Union[int, List[int]] = [2],
    number_of_components : int = 2,
    method_config : Dict[str, str] = {}, # "tsne", "pca"
    preprocessing_config : Optional[Dict[str, str]] = {}, # None, "standard_scaler", "normalizer"
) -> None:

    # Check if fetched dataset exist
    df = get_fetched_dataset(dataset_name)

    # If not, make fetched dataset
    if df is None:
        df = make_fetched_dataset(dataset_name, creds_filepath)

    # Encode labels
    df, le = encode_labels(df)

    # Split dataset
    df_X, df_y = split_dataset_2(df)

    # Save initial column names
    col_names = df_X.columns

    # Make user metadata features
    df_X = user_metadata_features(df_X, is_drop_original_cols = False)

     # Make ngram features
    df_X = ngram_features(df_X, is_drop_original_cols = False, col_names = ngram_features_col_names, ns = ngram_features_ns)

    # # Make ngram features of dataset
    df_X = ngram_features_of_dataset(df_X, is_drop_original_cols = False, col_names = ngram_features_of_dataset_col_names, ns = ngram_features_of_dataset_ns)

    # # Make ngram features of dataset by label
    df_X = ngram_features_of_dataset_by_label(df_X, df_y, is_drop_original_cols = False, col_names = ngram_features_of_dataset_by_label_col_names, ns = ngram_features_of_dataset_by_label_ns)

    # Drop initial columns
    if df_X.size > 0:
        df_X.drop(columns = col_names, inplace = True)

    # Save as csv file
    featured_dataset_path = os.environ.get("FEATURED_DATASET_FOLDER")
    df_X.to_csv(f"{featured_dataset_path}/{dataset_name}_X_clustering.csv", index = False)
    df_y.to_csv(f"{featured_dataset_path}/{dataset_name}_y_clustering.csv", index = False)

    # Dimensionality reduction
    dimension_reducer = BotDimensionReducer(method_config, preprocessing_config)
    df_X_reduced = dimension_reducer.fit_transform(df_X)

    # Plot
    colors = df_y["label"].apply(lambda x: "red" if x == 0 else "blue").tolist()
    colors = ["red", "blue"]
    for i, c in enumerate(colors):
        indexes = df_y.loc[df_y["label"] == i, "label"].index
        plt.scatter(df_X_reduced[indexes, 0], df_X_reduced[indexes, 1], s = 2, c = c)

    plt.axis('off')
    plt.legend(["bot", "human"])
    plt.savefig("dim_red.png", bbox_inches = "tight")
    plt.show()