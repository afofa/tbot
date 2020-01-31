import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer, FunctionTransformer

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline

from typing import Optional, Dict

class BotDimensionReducer:
    def __init__(
        self,
        method_config : Dict[str, str] = "tsne", # "tsne", "pca"
        preprocessing_config : Optional[Dict[str, str]] = None, # None, "standard_scaler", "normalizer"
    ) -> None:

        pipeline_lst = []

        if preprocessing_config is not None:
            preprocessing_type = preprocessing_config.get("type", None) # "standard_scaler", "normalizer", None
            if preprocessing_type == "standard_scaler":
                with_mean = preprocessing_config.get("with_mean", True) # True, False
                with_std = preprocessing_config.get("with_std", True) # True, False
                pipeline_lst.append(("log", FunctionTransformer(np.log1p, validate=True)))
                pipeline_lst.append((preprocessing_type, StandardScaler(with_mean = with_mean, with_std = with_std)))
            elif preprocessing_type == "normalizer":
                norm = preprocessing_config.get("norm", "l2") # "l1", "l2", "max"
                pipeline_lst.append((preprocessing_type, Normalizer(norm = norm)))
            elif preprocessing_type == "log":
                pipeline_lst.append((preprocessing_type, FunctionTransformer(np.log1p, validate=True)))
            else:
                pass

        method_type = method_config.get("type", "tsne") # "tsne", "pca"
        if method_type == "tsne":
            n_components = method_config.get("n_components", 2)
            perplexity = method_config.get("perplexity", 30.0)
            early_exaggeration = method_config.get("early_exaggeration", 12.0)
            learning_rate = method_config.get("learning_rate", 200.0)
            n_iter = method_config.get("n_iter", 1000)
            n_iter_without_progress = method_config.get("n_iter_without_progress", 300)
            min_grad_norm = method_config.get("min_grad_norm", 1e-07)
            metric = method_config.get("metric", "euclidean")
            init = method_config.get("init", "pca")
            pipeline_lst.append((method_type, TSNE(
                                                    n_components = n_components, 
                                                    perplexity = perplexity, 
                                                    early_exaggeration = early_exaggeration,
                                                    learning_rate = learning_rate,
                                                    n_iter = n_iter,
                                                    n_iter_without_progress = n_iter_without_progress,
                                                    min_grad_norm = min_grad_norm,
                                                    metric = metric,
                                                    init = init,
                                                )))
        elif method_type == "pca":
            n_components = method_config.get("n_components", 2)
            pipeline_lst.append((method_type, PCA(n_components = n_components)))
        else:
            raise RuntimeError(f"Method type is not recognized, got {method_type}")

        self.reducer = Pipeline(pipeline_lst)

    def fit_transform(self, features : pd.DataFrame) -> np.ndarray:
        return self.reducer.fit_transform(features)
