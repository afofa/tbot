import sys
sys.path.append("../tbot")

from tbot.pipelines import classification_pipeline
from dotenv import load_dotenv

load_dotenv(dotenv_path = "main.env")

classification_pipeline(
    dataset_name = "varol-2017",
    test_set_ratio = 0.0,
    ngram_features_col_names = [],
    ngram_features_ns = [2],
    ngram_features_of_dataset_col_names = [],
    ngram_features_of_dataset_ns = [2, 3],
    ngram_features_of_dataset_by_label_col_names = [],
    ngram_features_of_dataset_by_label_ns = [2, 3],
    number_of_generations = 5,
    population_size = 10,
    scoring = "roc_auc",
    number_of_folds_in_cv = 5,
    verbosity = 3,
    number_of_jobs = -1,
    is_feature_importances = True,
)
