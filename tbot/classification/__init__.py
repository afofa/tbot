from tpot import TPOTClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from typing import Dict, Union, Tuple, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class BotClassifier:
    FEATURE_IMPORTANCE_CONFIG = {

        # Classifiers
        'sklearn.tree.DecisionTreeClassifier': {
            'criterion': ["gini", "entropy"],
            'max_depth': range(1, 11),
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21)
        },

        'sklearn.ensemble.ExtraTreesClassifier': {
            'n_estimators': [100],
            'criterion': ["gini", "entropy"],
            'max_features': np.arange(0.05, 1.01, 0.05),
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21),
            'bootstrap': [True, False]
        },

        'sklearn.ensemble.RandomForestClassifier': {
            'n_estimators': [100],
            'criterion': ["gini", "entropy"],
            'max_features': np.arange(0.05, 1.01, 0.05),
            'min_samples_split': range(2, 21),
            'min_samples_leaf':  range(1, 21),
            'bootstrap': [True, False]
        },

        'sklearn.ensemble.GradientBoostingClassifier': {
            'n_estimators': [100],
            'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'max_depth': range(1, 11),
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21),
            'subsample': np.arange(0.05, 1.01, 0.05),
            'max_features': np.arange(0.05, 1.01, 0.05)
        },

        'xgboost.XGBClassifier': {
            'n_estimators': [100],
            'max_depth': range(1, 11),
            'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'subsample': np.arange(0.05, 1.01, 0.05),
            'min_child_weight': range(1, 21),
            'nthread': [1]
        },
    }

    FEATURE_IMPORTANCE_TEMPLATE = "Classifier"

    def __init__(
        self,
        number_of_generations : int = 3,
        population_size : int = 10,
        scoring : str = "accuracy", # "accuracy", "f1", "precision", "recall", "roc_auc"
        cv : Union[int, List[Tuple[List[int], List[int]]]] = 5,
        verbosity : int = 0, # 0, 1, 2, 3
        number_of_jobs : int = -1, # -1 = number of cores
        is_feature_importances : bool = False,
    ) -> None:

        config = self.FEATURE_IMPORTANCE_CONFIG if is_feature_importances else None
        template = self.FEATURE_IMPORTANCE_TEMPLATE if is_feature_importances else None

        self.classifier = TPOTClassifier(
            generations = number_of_generations, 
            population_size = population_size, 
            scoring = scoring,
            cv = cv, 
            verbosity = verbosity, 
            n_jobs = number_of_jobs, 
            config_dict = config,
            template = template,
        )

        self.is_feature_importances = is_feature_importances

    def fit(self, features : pd.DataFrame, classes : pd.DataFrame) -> None:
        self.feature_names = features.columns
        self.classifier.fit(features, classes)

    def predict(self, features : pd.DataFrame) -> np.ndarray:
        return self.classifier.predict(features)

    def predict_proba(self, features : pd.DataFrame) -> np.ndarray:
        return self.classifier.predict_proba(features)

    def score(self, testing_features : pd.DataFrame, testing_classes : pd.DataFrame) -> float:
        return self.classifier.score(testing_features, testing_classes)

    def export(self, output_file_name : str) -> None:
        self.classifier.export(output_file_name)

    def scores(self, testing_features : pd.DataFrame, testing_classes : pd.DataFrame) -> Dict[str, float]:
        # labels = testing_classes["label"].tolist()
        print(1)
        classifier_predictions = self.predict(testing_features)
        print(2)
        classifier_prob_predictions = self.predict_proba(testing_features)[:, 1]
        print(3)

        scores_dict = {
            "accuracy" : accuracy_score(testing_classes, classifier_predictions),
            "precision" : precision_score(testing_classes, classifier_predictions),
            "recall" : recall_score(testing_classes, classifier_predictions),
            "f1" : f1_score(testing_classes, classifier_predictions),
            "roc_auc" : roc_auc_score(testing_classes, classifier_prob_predictions),
        }
        print(4)
        
        return scores_dict

    def get_fitted_pipeline(self) -> Pipeline:
        fitted_pipeline = self.classifier.fitted_pipeline_
        return fitted_pipeline

    def get_pareto_front_fitted_pipelines(self) -> Dict[str, Pipeline]:
        try:
            return self.classifier.pareto_front_fitted_pipelines_
        except Exception as e:
            print(f"cannot get pareto_front_fitted_pipelines_\n{e}")
            return {}

    def get_evaluated_individuals(self) -> Dict[str, Dict[str, Union[int, float, Tuple[str, ...]]]]:
        try:
            return self.classifier.evaluated_individuals_
        except Exception as e:
            print(f"cannot get evaluated_individuals_\n{e}")
            return {}

    def get_feature_importances(self) -> Dict[str, float]:
        if True or self.is_feature_importances:
            try:
                classifier_with_feature_importance = self.get_fitted_pipeline()[-1]
                feature_importances = classifier_with_feature_importance.feature_importances_
                return dict(zip(self.feature_names, feature_importances))
            except Exception as e:
                print(f"cannot get feature importances\n{e}")
                return 
        else:
            print(f"is_feature_importances is passed as False, cannot retrieve feature importances")
            return