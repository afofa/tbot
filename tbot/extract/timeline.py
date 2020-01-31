import pandas as pd
import numpy as np
import scipy.stats
from emoji.core import emoji_count
from typing import Optional, Union, Tuple, List, Dict

def user_timeline_features(
    df_timeline : pd.DataFrame,
    df_train_X : pd.DataFrame, 
    df_test_X : Optional[pd.DataFrame] = None,
    is_drop_original_cols : bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    funcs = [
        feature_time_between_tweets,
        feature_time_between_retweets,
        feature_time_between_mentions,
        feature_time_between_quotes,
        feature_entropy_of_words_in_tweet,
        feature_number_of_words_in_tweet,
        feature_number_of_hashtags_in_tweet,
        feature_number_of_urls_in_tweet,
        feature_number_of_user_mentions_in_tweet,
        feature_number_of_media_in_tweet,
        feature_number_of_symbols_in_tweet,
        feature_number_of_polls_in_tweet,
        feature_number_of_entities_in_tweet,
        feature_number_of_lower_cases_in_tweet,
        feature_number_of_upper_cases_in_tweet,
        feature_number_of_digits_in_tweet,
        feature_number_of_emojis_in_tweet,
        feature_tweet_length,
        feature_ratio_of_tweets,
        feature_ratio_of_mentions,
        feature_ratio_of_retweets,
        feature_ratio_of_quotes,
        feature_ratio_of_has_coordinates,
        feature_ratio_of_has_place,
        feature_ratio_of_possibly_sensitive,
        feature_number_of_languages,
        feature_entropy_of_languages,
        feature_number_of_sources,
        feature_entropy_of_sources,
    ]

    if df_train_X.size > 0:

        column_names = df_train_X.columns
        
        for func in funcs:
            df_train_X = func(df_train_X, df_timeline)

        if is_drop_original_cols:
            df_train_X.drop(columns = column_names, inplace = True)

    if df_test_X is not None:

        if df_test_X.size > 0:
            column_names = df_test_X.columns
            
            for func in funcs:
                df_test_X = func(df_test_X, df_timeline)

            if is_drop_original_cols:
                df_test_X.drop(columns = column_names, inplace = True)

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

### User timeline features

def feature_time_between_tweets(df : pd.DataFrame, df_timeline : pd.DataFrame, user_id_str_col_name : str = "user_id_str", in_col_name : str = "created_at", out_col_name : str = "time_between_tweets") -> pd.DataFrame:
    ress = []
    for user_id_str in df[user_id_str_col_name].unique():
        nums = df_timeline.loc[df_timeline[user_id_str_col_name] == user_id_str, in_col_name].diff(periods = -1).dropna().apply(lambda x: x.total_seconds()).tolist()
        res = distribution_feature(nums)
        res[user_id_str_col_name] = user_id_str
        ress.append(res)

    df_ress = pd.DataFrame(ress)
    df_ress.columns = [c if c == user_id_str_col_name else f"{out_col_name}.{c}" for c in df_ress.columns]
    df = df.merge(df_ress)

    return df

def feature_time_between_retweets(df : pd.DataFrame, df_timeline : pd.DataFrame, type_col_name : str = "type", user_id_str_col_name : str = "user_id_str", in_col_name : str = "created_at", out_col_name : str = "time_between_retweets") -> pd.DataFrame:
    ress = []
    for user_id_str in df[user_id_str_col_name].unique():
        nums = df_timeline.loc[(df_timeline[user_id_str_col_name] == user_id_str) & (df_timeline[type_col_name] == "retweet"), in_col_name].diff(periods = -1).dropna().apply(lambda x: x.total_seconds()).tolist()
        res = distribution_feature(nums)
        res[user_id_str_col_name] = user_id_str
        ress.append(res)

    df_ress = pd.DataFrame(ress)
    df_ress.columns = [c if c == user_id_str_col_name else f"{out_col_name}.{c}" for c in df_ress.columns]
    df = df.merge(df_ress)

    return df

def feature_time_between_mentions(df : pd.DataFrame, df_timeline : pd.DataFrame, type_col_name : str = "type", user_id_str_col_name : str = "user_id_str", in_col_name : str = "created_at", out_col_name : str = "time_between_mentions") -> pd.DataFrame:
    ress = []
    for user_id_str in df[user_id_str_col_name].unique():
        nums = df_timeline.loc[(df_timeline[user_id_str_col_name] == user_id_str) & (df_timeline[type_col_name] == "mention"), in_col_name].diff(periods = -1).dropna().apply(lambda x: x.total_seconds()).tolist()
        res = distribution_feature(nums)
        res[user_id_str_col_name] = user_id_str
        ress.append(res)

    df_ress = pd.DataFrame(ress)
    df_ress.columns = [c if c == user_id_str_col_name else f"{out_col_name}.{c}" for c in df_ress.columns]
    df = df.merge(df_ress)

    return df

def feature_time_between_quotes(df : pd.DataFrame, df_timeline : pd.DataFrame, type_col_name : str = "type", user_id_str_col_name : str = "user_id_str", in_col_name : str = "created_at", out_col_name : str = "time_between_quotes") -> pd.DataFrame:
    ress = []
    for user_id_str in df[user_id_str_col_name].unique():
        nums = df_timeline.loc[(df_timeline[user_id_str_col_name] == user_id_str) & (df_timeline[type_col_name] == "quote"), in_col_name].diff(periods = -1).dropna().apply(lambda x: x.total_seconds()).tolist()
        res = distribution_feature(nums)
        res[user_id_str_col_name] = user_id_str
        ress.append(res)

    df_ress = pd.DataFrame(ress)
    df_ress.columns = [c if c == user_id_str_col_name else f"{out_col_name}.{c}" for c in df_ress.columns]
    df = df.merge(df_ress)

    return df

def feature_entropy_of_words_in_tweet(df : pd.DataFrame, df_timeline : pd.DataFrame, user_id_str_col_name : str = "user_id_str", in_col_name : str = "text", out_col_name : str = "entropy_of_words") -> pd.DataFrame:
    ress = []
    for user_id_str in df[user_id_str_col_name].unique():
        nums = df_timeline.loc[df_timeline[user_id_str_col_name] == user_id_str, in_col_name].apply(lambda x: scipy.stats.entropy(np.unique(x.split(" "), return_counts=True)[1])).tolist()
        res = distribution_feature(nums)
        res[user_id_str_col_name] = user_id_str
        ress.append(res)

    df_ress = pd.DataFrame(ress)
    df_ress.columns = [c if c == user_id_str_col_name else f"{out_col_name}.{c}" for c in df_ress.columns]
    df = df.merge(df_ress)

    return df

def feature_number_of_words_in_tweet(df : pd.DataFrame, df_timeline : pd.DataFrame, user_id_str_col_name : str = "user_id_str", in_col_name : str = "text", out_col_name : str = "number_of_words") -> pd.DataFrame:
    ress = []
    for user_id_str in df[user_id_str_col_name].unique():
        nums = df_timeline.loc[df_timeline[user_id_str_col_name] == user_id_str, in_col_name].apply(lambda x: len(x.split(" "))).tolist()
        res = distribution_feature(nums)
        res[user_id_str_col_name] = user_id_str
        ress.append(res)

    df_ress = pd.DataFrame(ress)
    df_ress.columns = [c if c == user_id_str_col_name else f"{out_col_name}.{c}" for c in df_ress.columns]
    df = df.merge(df_ress)

    return df

def feature_number_of_hashtags_in_tweet(df : pd.DataFrame, df_timeline : pd.DataFrame, user_id_str_col_name : str = "user_id_str", in_col_name : str = "entities.hashtags", out_col_name : str = "number_of_hashtags") -> pd.DataFrame:
    ress = []
    for user_id_str in df[user_id_str_col_name].unique():
        nums = df_timeline.loc[df_timeline[user_id_str_col_name] == user_id_str, in_col_name].apply(lambda x: len(x)).tolist()
        res = distribution_feature(nums)
        res[user_id_str_col_name] = user_id_str
        ress.append(res)

    df_ress = pd.DataFrame(ress)
    df_ress.columns = [c if c == user_id_str_col_name else f"{out_col_name}.{c}" for c in df_ress.columns]
    df = df.merge(df_ress)

    return df

def feature_number_of_urls_in_tweet(df : pd.DataFrame, df_timeline : pd.DataFrame, user_id_str_col_name : str = "user_id_str", in_col_name : str = "entities.urls", out_col_name : str = "number_of_urls") -> pd.DataFrame:
    ress = []
    for user_id_str in df[user_id_str_col_name].unique():
        nums = df_timeline.loc[df_timeline[user_id_str_col_name] == user_id_str, in_col_name].apply(lambda x: len(x)).tolist()
        res = distribution_feature(nums)
        res[user_id_str_col_name] = user_id_str
        ress.append(res)

    df_ress = pd.DataFrame(ress)
    df_ress.columns = [c if c == user_id_str_col_name else f"{out_col_name}.{c}" for c in df_ress.columns]
    df = df.merge(df_ress)

    return df

def feature_number_of_user_mentions_in_tweet(df : pd.DataFrame, df_timeline : pd.DataFrame, user_id_str_col_name : str = "user_id_str", in_col_name : str = "entities.user_mentions", out_col_name : str = "number_of_user_mentions") -> pd.DataFrame:
    ress = []
    for user_id_str in df[user_id_str_col_name].unique():
        nums = df_timeline.loc[df_timeline[user_id_str_col_name] == user_id_str, in_col_name].apply(lambda x: len(x)).tolist()
        res = distribution_feature(nums)
        res[user_id_str_col_name] = user_id_str
        ress.append(res)

    df_ress = pd.DataFrame(ress)
    df_ress.columns = [c if c == user_id_str_col_name else f"{out_col_name}.{c}" for c in df_ress.columns]
    df = df.merge(df_ress)

    return df

def feature_number_of_media_in_tweet(df : pd.DataFrame, df_timeline : pd.DataFrame, user_id_str_col_name : str = "user_id_str", in_col_name : str = "entities.media", out_col_name : str = "number_of_media") -> pd.DataFrame:
    ress = []
    for user_id_str in df[user_id_str_col_name].unique():
        nums = df_timeline.loc[df_timeline[user_id_str_col_name] == user_id_str, in_col_name].apply(lambda x: len(x)).tolist()
        res = distribution_feature(nums)
        res[user_id_str_col_name] = user_id_str
        ress.append(res)

    df_ress = pd.DataFrame(ress)
    df_ress.columns = [c if c == user_id_str_col_name else f"{out_col_name}.{c}" for c in df_ress.columns]
    df = df.merge(df_ress)

    return df

def feature_number_of_symbols_in_tweet(df : pd.DataFrame, df_timeline : pd.DataFrame, user_id_str_col_name : str = "user_id_str", in_col_name : str = "entities.symbols", out_col_name : str = "number_of_symbols") -> pd.DataFrame:
    ress = []
    for user_id_str in df[user_id_str_col_name].unique():
        nums = df_timeline.loc[df_timeline[user_id_str_col_name] == user_id_str, in_col_name].apply(lambda x: len(x)).tolist()
        res = distribution_feature(nums)
        res[user_id_str_col_name] = user_id_str
        ress.append(res)

    df_ress = pd.DataFrame(ress)
    df_ress.columns = [c if c == user_id_str_col_name else f"{out_col_name}.{c}" for c in df_ress.columns]
    df = df.merge(df_ress)

    return df

def feature_number_of_polls_in_tweet(df : pd.DataFrame, df_timeline : pd.DataFrame, user_id_str_col_name : str = "user_id_str", in_col_name : str = "entities.polls", out_col_name : str = "number_of_polls") -> pd.DataFrame:
    ress = []
    for user_id_str in df[user_id_str_col_name].unique():
        nums = df_timeline.loc[df_timeline[user_id_str_col_name] == user_id_str, in_col_name].apply(lambda x: len(x)).tolist()
        res = distribution_feature(nums)
        res[user_id_str_col_name] = user_id_str
        ress.append(res)

    df_ress = pd.DataFrame(ress)
    df_ress.columns = [c if c == user_id_str_col_name else f"{out_col_name}.{c}" for c in df_ress.columns]
    df = df.merge(df_ress)

    return df

def feature_number_of_entities_in_tweet(df : pd.DataFrame, df_timeline : pd.DataFrame, user_id_str_col_name : str = "user_id_str", in_col_name : str = "entities", out_col_name : str = "number_of_entities") -> pd.DataFrame:
    ress = []
    columns_of_interest = list(filter(lambda x: in_col_name in x, df_timeline.columns))
    number_of_entities_col = df_timeline[columns_of_interest].apply(lambda x: sum([len(c) for c in x]), axis = 1)
    for user_id_str in df[user_id_str_col_name].unique():
        nums = number_of_entities_col.loc[df_timeline[user_id_str_col_name] == user_id_str].tolist()
        res = distribution_feature(nums)
        res[user_id_str_col_name] = user_id_str
        ress.append(res)

    df_ress = pd.DataFrame(ress)
    df_ress.columns = [c if c == user_id_str_col_name else f"{out_col_name}.{c}" for c in df_ress.columns]
    df = df.merge(df_ress)

    return df

def feature_number_of_lower_cases_in_tweet(df : pd.DataFrame, df_timeline : pd.DataFrame, user_id_str_col_name : str = "user_id_str", in_col_name : str = "text", out_col_name : str = "number_of_lower_case") -> pd.DataFrame:
    ress = []
    for user_id_str in df[user_id_str_col_name].unique():
        nums = df_timeline.loc[df_timeline[user_id_str_col_name] == user_id_str, in_col_name].apply(lambda x: len(list(filter(lambda y: y.islower(), x)))).tolist()
        res = distribution_feature(nums)
        res[user_id_str_col_name] = user_id_str
        ress.append(res)

    df_ress = pd.DataFrame(ress)
    df_ress.columns = [c if c == user_id_str_col_name else f"{out_col_name}.{c}" for c in df_ress.columns]
    df = df.merge(df_ress)

    return df

def feature_number_of_upper_cases_in_tweet(df : pd.DataFrame, df_timeline : pd.DataFrame, user_id_str_col_name : str = "user_id_str", in_col_name : str = "text", out_col_name : str = "number_of_upper_case") -> pd.DataFrame:
    ress = []
    for user_id_str in df[user_id_str_col_name].unique():
        nums = df_timeline.loc[df_timeline[user_id_str_col_name] == user_id_str, in_col_name].apply(lambda x: len(list(filter(lambda y: y.isupper(), x)))).tolist()
        res = distribution_feature(nums)
        res[user_id_str_col_name] = user_id_str
        ress.append(res)

    df_ress = pd.DataFrame(ress)
    df_ress.columns = [c if c == user_id_str_col_name else f"{out_col_name}.{c}" for c in df_ress.columns]
    df = df.merge(df_ress)

    return df

def feature_number_of_digits_in_tweet(df : pd.DataFrame, df_timeline : pd.DataFrame, user_id_str_col_name : str = "user_id_str", in_col_name : str = "text", out_col_name : str = "number_of_digit") -> pd.DataFrame:
    ress = []
    for user_id_str in df[user_id_str_col_name].unique():
        nums = df_timeline.loc[df_timeline[user_id_str_col_name] == user_id_str, in_col_name].apply(lambda x: len(list(filter(lambda y: y.isdigit(), x)))).tolist()
        res = distribution_feature(nums)
        res[user_id_str_col_name] = user_id_str
        ress.append(res)

    df_ress = pd.DataFrame(ress)
    df_ress.columns = [c if c == user_id_str_col_name else f"{out_col_name}.{c}" for c in df_ress.columns]
    df = df.merge(df_ress)

    return df

def feature_number_of_emojis_in_tweet(df : pd.DataFrame, df_timeline : pd.DataFrame, user_id_str_col_name : str = "user_id_str", in_col_name : str = "text", out_col_name : str = "number_of_emoji") -> pd.DataFrame:
    ress = []
    for user_id_str in df[user_id_str_col_name].unique():
        nums = df_timeline.loc[df_timeline[user_id_str_col_name] == user_id_str, in_col_name].apply(lambda x: emoji_count(x)).tolist()
        res = distribution_feature(nums)
        res[user_id_str_col_name] = user_id_str
        ress.append(res)

    df_ress = pd.DataFrame(ress)
    df_ress.columns = [c if c == user_id_str_col_name else f"{out_col_name}.{c}" for c in df_ress.columns]
    df = df.merge(df_ress)

    return df

def feature_tweet_length(df : pd.DataFrame, df_timeline : pd.DataFrame, user_id_str_col_name : str = "user_id_str", in_col_name : str = "text", out_col_name : str = "tweet_length") -> pd.DataFrame:
    ress = []
    for user_id_str in df[user_id_str_col_name].unique():
        nums = df_timeline.loc[df_timeline[user_id_str_col_name] == user_id_str, in_col_name].apply(lambda x: len(x)).tolist()
        res = distribution_feature(nums)
        res[user_id_str_col_name] = user_id_str
        ress.append(res)

    df_ress = pd.DataFrame(ress)
    df_ress.columns = [c if c == user_id_str_col_name else f"{out_col_name}.{c}" for c in df_ress.columns]
    df = df.merge(df_ress)

    return df

def feature_ratio_of_tweets(df : pd.DataFrame, df_timeline : pd.DataFrame, user_id_str_col_name : str = "user_id_str", in_col_name : str = "type", out_col_name : str = "ratio_of_tweets") -> pd.DataFrame:
    ress = []
    for user_id_str in df[user_id_str_col_name].unique():
        tmp_col = df_timeline.loc[df_timeline[user_id_str_col_name] == user_id_str, [user_id_str_col_name, in_col_name]]
        num = tmp_col.loc[tmp_col[in_col_name] == "tweet"].shape[0] / tmp_col.shape[0] if tmp_col.shape[0] > 0 else 0.0
        res = {user_id_str_col_name: user_id_str, out_col_name: num}
        ress.append(res)

    df_ress = pd.DataFrame(ress)
    df = df.merge(df_ress)

    return df

def feature_ratio_of_retweets(df : pd.DataFrame, df_timeline : pd.DataFrame, user_id_str_col_name : str = "user_id_str", in_col_name : str = "type", out_col_name : str = "ratio_of_retweets") -> pd.DataFrame:
    ress = []
    for user_id_str in df[user_id_str_col_name].unique():
        tmp_col = df_timeline.loc[df_timeline[user_id_str_col_name] == user_id_str, [user_id_str_col_name, in_col_name]]
        num = tmp_col.loc[tmp_col[in_col_name] == "retweet"].shape[0] / tmp_col.shape[0] if tmp_col.shape[0] > 0 else 0.0
        res = {user_id_str_col_name: user_id_str, out_col_name: num}
        ress.append(res)

    df_ress = pd.DataFrame(ress)
    df = df.merge(df_ress)

    return df

def feature_ratio_of_mentions(df : pd.DataFrame, df_timeline : pd.DataFrame, user_id_str_col_name : str = "user_id_str", in_col_name : str = "type", out_col_name : str = "ratio_of_mentions") -> pd.DataFrame:
    ress = []
    for user_id_str in df[user_id_str_col_name].unique():
        tmp_col = df_timeline.loc[df_timeline[user_id_str_col_name] == user_id_str, [user_id_str_col_name, in_col_name]]
        num = tmp_col.loc[tmp_col[in_col_name] == "mention"].shape[0] / tmp_col.shape[0] if tmp_col.shape[0] > 0 else 0.0
        res = {user_id_str_col_name: user_id_str, out_col_name: num}
        ress.append(res)

    df_ress = pd.DataFrame(ress)
    df = df.merge(df_ress)

    return df

def feature_ratio_of_quotes(df : pd.DataFrame, df_timeline : pd.DataFrame, user_id_str_col_name : str = "user_id_str", in_col_name : str = "type", out_col_name : str = "ratio_of_quotes") -> pd.DataFrame:
    ress = []
    for user_id_str in df[user_id_str_col_name].unique():
        tmp_col = df_timeline.loc[df_timeline[user_id_str_col_name] == user_id_str, [user_id_str_col_name, in_col_name]]
        num = tmp_col.loc[tmp_col[in_col_name] == "quote"].shape[0] / tmp_col.shape[0] if tmp_col.shape[0] > 0 else 0.0
        res = {user_id_str_col_name: user_id_str, out_col_name: num}
        ress.append(res)

    df_ress = pd.DataFrame(ress)
    df = df.merge(df_ress)

    return df

def feature_ratio_of_has_coordinates(df : pd.DataFrame, df_timeline : pd.DataFrame, user_id_str_col_name : str = "user_id_str", in_col_name : str = "coordinates", out_col_name : str = "ratio_of_has_coordinates") -> pd.DataFrame:
    ress = []
    for user_id_str in df[user_id_str_col_name].unique():
        tmp_col = df_timeline.loc[df_timeline[user_id_str_col_name] == user_id_str, [user_id_str_col_name, in_col_name]]
        num = tmp_col.loc[tmp_col[in_col_name].notnull()].shape[0] / tmp_col.shape[0] if tmp_col.shape[0] > 0 else 0.0
        res = {user_id_str_col_name: user_id_str, out_col_name: num}
        ress.append(res)

    df_ress = pd.DataFrame(ress)
    df = df.merge(df_ress)

    return df

def feature_ratio_of_has_place(df : pd.DataFrame, df_timeline : pd.DataFrame, user_id_str_col_name : str = "user_id_str", in_col_name : str = "place", out_col_name : str = "ratio_of_has_place") -> pd.DataFrame:
    ress = []
    for user_id_str in df[user_id_str_col_name].unique():
        tmp_col = df_timeline.loc[df_timeline[user_id_str_col_name] == user_id_str, [user_id_str_col_name, in_col_name]]
        num = tmp_col.loc[tmp_col[in_col_name].notnull()].shape[0] / tmp_col.shape[0] if tmp_col.shape[0] > 0 else 0.0
        res = {user_id_str_col_name: user_id_str, out_col_name: num}
        ress.append(res)

    df_ress = pd.DataFrame(ress)
    df = df.merge(df_ress)

    return df

def feature_ratio_of_possibly_sensitive(df : pd.DataFrame, df_timeline : pd.DataFrame, user_id_str_col_name : str = "user_id_str", in_col_name : str = "possibly_sensitive", out_col_name : str = "ratio_of_possibly_sensitive") -> pd.DataFrame:
    ress = []
    for user_id_str in df[user_id_str_col_name].unique():
        tmp_col = df_timeline.loc[df_timeline[user_id_str_col_name] == user_id_str, [user_id_str_col_name, in_col_name]].dropna()
        num = tmp_col.loc[tmp_col[in_col_name]].shape[0] / tmp_col.shape[0] if tmp_col.shape[0] > 0 else 0.0
        res = {user_id_str_col_name: user_id_str, out_col_name: num}
        ress.append(res)

    df_ress = pd.DataFrame(ress)
    df = df.merge(df_ress)

    return df

def feature_number_of_languages(df : pd.DataFrame, df_timeline : pd.DataFrame, user_id_str_col_name : str = "user_id_str", in_col_name : str = "lang", out_col_name : str = "number_of_languages") -> pd.DataFrame:
    ress = []
    for user_id_str in df[user_id_str_col_name].unique():
        num = len(df_timeline.loc[df_timeline[user_id_str_col_name] == user_id_str, in_col_name].unique())
        res = {user_id_str_col_name: user_id_str, out_col_name: num}
        ress.append(res)

    df_ress = pd.DataFrame(ress)
    df = df.merge(df_ress)

    return df

def feature_entropy_of_languages(df : pd.DataFrame, df_timeline : pd.DataFrame, user_id_str_col_name : str = "user_id_str", in_col_name : str = "lang", out_col_name : str = "entropy_of_languages") -> pd.DataFrame:
    ress = []
    for user_id_str in df[user_id_str_col_name].unique():
        tmp_vals = df_timeline.loc[df_timeline[user_id_str_col_name] == user_id_str, in_col_name].tolist()
        _, counts = np.unique(tmp_vals, return_counts=True)
        num = scipy.stats.entropy(counts)
        res = {user_id_str_col_name: user_id_str, out_col_name: num}
        ress.append(res)

    df_ress = pd.DataFrame(ress)
    df = df.merge(df_ress)

    return df

def feature_number_of_sources(df : pd.DataFrame, df_timeline : pd.DataFrame, user_id_str_col_name : str = "user_id_str", in_col_name : str = "source", out_col_name : str = "number_of_sources") -> pd.DataFrame:
    ress = []
    for user_id_str in df[user_id_str_col_name].unique():
        num = len(df_timeline.loc[df_timeline[user_id_str_col_name] == user_id_str, in_col_name].unique())
        res = {user_id_str_col_name: user_id_str, out_col_name: num}
        ress.append(res)

    df_ress = pd.DataFrame(ress)
    df = df.merge(df_ress)

    return df

def feature_entropy_of_sources(df : pd.DataFrame, df_timeline : pd.DataFrame, user_id_str_col_name : str = "user_id_str", in_col_name : str = "source", out_col_name : str = "entropy_of_sources") -> pd.DataFrame:
    ress = []
    for user_id_str in df[user_id_str_col_name].unique():
        tmp_vals = df_timeline.loc[df_timeline[user_id_str_col_name] == user_id_str, in_col_name].tolist()
        _, counts = np.unique(tmp_vals, return_counts=True)
        num = scipy.stats.entropy(counts)
        res = {user_id_str_col_name: user_id_str, out_col_name: num}
        ress.append(res)

    df_ress = pd.DataFrame(ress)
    df = df.merge(df_ress)

    return df