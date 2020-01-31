import pandas as pd
from emoji.core import emoji_count
import datetime
import pytz
from typing import Optional, Union, Tuple

def user_metadata_features_from_yang_2019(
    df_train_X : pd.DataFrame, 
    df_test_X : Optional[pd.DataFrame] = None,
    is_drop_original_cols : bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    funcs = [
        feature_description_length,
        feature_has_profile_banner,
        feature_is_default_profile,
        feature_is_verified,
        feature_number_of_digits_in_screen_name,
        feature_number_of_digits_in_user_name,
        feature_screen_name_length,
        feature_total_number_of_favourites,
        feature_total_number_of_followers,
        feature_total_number_of_friends,
        feature_total_number_of_listed,
        feature_total_number_of_tweets,
        feature_user_name_length,
        feature_per_hour_number_of_favourites,
        feature_per_hour_number_of_followers,
        feature_per_hour_number_of_friends,
        feature_per_hour_number_of_listed,
        feature_per_hour_number_of_tweets,
        feature_ratio_of_followers_to_friends,
    ]

    if df_train_X.size > 0:

        column_names = df_train_X.columns
        
        for func in funcs:
            df_train_X = func(df_train_X)

        if is_drop_original_cols:
            df_train_X.drop(columns = column_names, inplace = True)

    if df_test_X is not None:

        if df_test_X.size > 0:
            column_names = df_test_X.columns
            
            for func in funcs:
                df_test_X = func(df_test_X)

            if is_drop_original_cols:
                df_test_X.drop(columns = column_names, inplace = True)

        return df_train_X, df_test_X

    else:
        return df_train_X

def user_metadata_features(
    df_train_X : pd.DataFrame, 
    df_test_X : Optional[pd.DataFrame] = None,
    is_drop_original_cols : bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    funcs = [
        feature_account_age_in_days,
        feature_description_length,
        feature_has_description,
        feature_has_location,
        feature_has_profile_banner,
        feature_has_url,
        feature_is_default_profile,
        feature_is_default_profile_image,
        feature_is_protected,
        feature_is_verified,
        feature_number_of_digits_in_description,
        feature_number_of_digits_in_screen_name,
        feature_number_of_digits_in_user_name,
        feature_number_of_lower_case_in_description,
        feature_number_of_lower_case_in_screen_name,
        feature_number_of_lower_case_in_user_name,
        feature_number_of_upper_case_in_description,
        feature_number_of_upper_case_in_screen_name,
        feature_number_of_upper_case_in_user_name,
        feature_number_of_emoji_in_description,
        feature_number_of_emoji_in_screen_name,
        feature_number_of_emoji_in_user_name,
        feature_ratio_of_digits_in_description,
        feature_ratio_of_digits_in_screen_name,
        feature_ratio_of_digits_in_user_name,
        feature_ratio_of_lower_case_in_description,
        feature_ratio_of_lower_case_in_screen_name,
        feature_ratio_of_lower_case_in_user_name,
        feature_ratio_of_upper_case_in_description,
        feature_ratio_of_upper_case_in_screen_name,
        feature_ratio_of_upper_case_in_user_name,
        feature_ratio_of_emoji_in_description,
        feature_ratio_of_emoji_in_screen_name,
        feature_ratio_of_emoji_in_user_name,
        feature_screen_name_length,
        feature_total_number_of_favourites,
        feature_total_number_of_followers,
        feature_total_number_of_friends,
        feature_total_number_of_listed,
        feature_total_number_of_tweets,
        feature_user_name_length,
        feature_per_hour_number_of_favourites,
        feature_per_hour_number_of_followers,
        feature_per_hour_number_of_friends,
        feature_per_hour_number_of_listed,
        feature_per_hour_number_of_tweets,
        feature_ratio_of_followers_to_friends,
    ]

    if df_train_X.size > 0:

        column_names = df_train_X.columns
        
        for func in funcs:
            df_train_X = func(df_train_X)

        if is_drop_original_cols:
            df_train_X.drop(columns = column_names, inplace = True)

    if df_test_X is not None:

        if df_test_X.size > 0:
            column_names = df_test_X.columns
            
            for func in funcs:
                df_test_X = func(df_test_X)

            if is_drop_original_cols:
                df_test_X.drop(columns = column_names, inplace = True)

        return df_train_X, df_test_X

    else:
        return df_train_X


### User metadata features

def feature_ratio_of_followers_to_friends(df : pd.DataFrame, followers_count_col_name : str = "followers_count", friends_count_col_name : str = "friends_count", out_col_name : str = "ratio_of_followers_to_friends"):
    df_friends_counts = df[friends_count_col_name].apply(lambda x: max(1, x))
    df_followers_counts = df[followers_count_col_name]
    df[out_col_name] = df_followers_counts / df_friends_counts
    return df

def feature_screen_name_length(df : pd.DataFrame, in_col_name : str = "screen_name", out_col_name : str = "screen_name_length") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: len(x))
    return df

def feature_number_of_digits_in_screen_name(df : pd.DataFrame, in_col_name : str = "screen_name", out_col_name : str = "number_of_digits_in_screen_name") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: len(list(filter(lambda y: y.isdigit(), x))))
    return df

def feature_number_of_upper_case_in_screen_name(df : pd.DataFrame, in_col_name : str = "screen_name", out_col_name : str = "number_of_upper_case_in_screen_name") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: len(list(filter(lambda y: y.isupper(), x))))
    return df

def feature_number_of_lower_case_in_screen_name(df : pd.DataFrame, in_col_name : str = "screen_name", out_col_name : str = "number_of_lower_case_in_screen_name") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: len(list(filter(lambda y: y.islower(), x))))
    return df

def feature_number_of_emoji_in_screen_name(df : pd.DataFrame, in_col_name : str = "screen_name", out_col_name : str = "number_of_emoji_in_screen_name") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: emoji_count(x))
    return df

def feature_ratio_of_digits_in_screen_name(df : pd.DataFrame, in_col_name : str = "screen_name", out_col_name : str = "ratio_of_digits_in_screen_name") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: len(list(filter(lambda y: y.isdigit(), x))) / len(x) if len(x) > 0 else 0)
    return df

def feature_ratio_of_upper_case_in_screen_name(df : pd.DataFrame, in_col_name : str = "screen_name", out_col_name : str = "ratio_of_upper_case_in_screen_name") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: len(list(filter(lambda y: y.isupper(), x))) / len(x) if len(x) > 0 else 0)
    return df

def feature_ratio_of_lower_case_in_screen_name(df : pd.DataFrame, in_col_name : str = "screen_name", out_col_name : str = "ratio_of_lower_case_in_screen_name") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: len(list(filter(lambda y: y.islower(), x))) / len(x) if len(x) > 0 else 0)
    return df

def feature_ratio_of_emoji_in_screen_name(df : pd.DataFrame, in_col_name : str = "screen_name", out_col_name : str = "ratio_of_emoji_in_screen_name") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: emoji_count(x) / len(x) if len(x) > 0 else 0)
    return df

def feature_user_name_length(df : pd.DataFrame, in_col_name : str = "name", out_col_name : str = "user_name_length") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: len(x))
    return df

def feature_number_of_digits_in_user_name(df : pd.DataFrame, in_col_name : str = "name", out_col_name : str = "number_of_digits_in_user_name") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: len(list(filter(lambda y: y.isdigit(), x))))
    return df

def feature_number_of_upper_case_in_user_name(df : pd.DataFrame, in_col_name : str = "name", out_col_name : str = "number_of_upper_case_in_user_name") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: len(list(filter(lambda y: y.isupper(), x))))
    return df

def feature_number_of_lower_case_in_user_name(df : pd.DataFrame, in_col_name : str = "name", out_col_name : str = "number_of_lower_case_in_user_name") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: len(list(filter(lambda y: y.islower(), x))))
    return df

def feature_number_of_emoji_in_user_name(df : pd.DataFrame, in_col_name : str = "name", out_col_name : str = "number_of_emoji_in_user_name") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: emoji_count(x))
    return df

def feature_ratio_of_digits_in_user_name(df : pd.DataFrame, in_col_name : str = "name", out_col_name : str = "ratio_of_digits_in_user_name") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: len(list(filter(lambda y: y.isdigit(), x))) / len(x) if len(x) > 0 else 0)
    return df

def feature_ratio_of_upper_case_in_user_name(df : pd.DataFrame, in_col_name : str = "name", out_col_name : str = "ratio_of_upper_case_in_user_name") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: len(list(filter(lambda y: y.isupper(), x))) / len(x) if len(x) > 0 else 0)
    return df

def feature_ratio_of_lower_case_in_user_name(df : pd.DataFrame, in_col_name : str = "name", out_col_name : str = "ratio_of_lower_case_in_user_name") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: len(list(filter(lambda y: y.islower(), x))) / len(x) if len(x) > 0 else 0)
    return df

def feature_ratio_of_emoji_in_user_name(df : pd.DataFrame, in_col_name : str = "name", out_col_name : str = "ratio_of_emoji_in_user_name") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: emoji_count(x) / len(x) if len(x) > 0 else 0)
    return df

def feature_description_length(df : pd.DataFrame, in_col_name : str = "description", out_col_name : str = "description_length") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: len(x))
    return df

def feature_number_of_digits_in_description(df : pd.DataFrame, in_col_name : str = "description", out_col_name : str = "number_of_digits_in_description") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: len(list(filter(lambda y: y.isdigit(), x))))
    return df

def feature_number_of_upper_case_in_description(df : pd.DataFrame, in_col_name : str = "description", out_col_name : str = "number_of_upper_case_in_description") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: len(list(filter(lambda y: y.isupper(), x))))
    return df

def feature_number_of_lower_case_in_description(df : pd.DataFrame, in_col_name : str = "description", out_col_name : str = "number_of_lower_case_in_description") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: len(list(filter(lambda y: y.islower(), x))))
    return df

def feature_number_of_emoji_in_description(df : pd.DataFrame, in_col_name : str = "description", out_col_name : str = "number_of_emoji_in_description") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: emoji_count(x))
    return df

def feature_ratio_of_digits_in_description(df : pd.DataFrame, in_col_name : str = "description", out_col_name : str = "ratio_of_digits_in_description") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: len(list(filter(lambda y: y.isdigit(), x))) / len(x) if len(x) > 0 else 0)
    return df

def feature_ratio_of_upper_case_in_description(df : pd.DataFrame, in_col_name : str = "description", out_col_name : str = "ratio_of_upper_case_in_description") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: len(list(filter(lambda y: y.isupper(), x))) / len(x) if len(x) > 0 else 0)
    return df

def feature_ratio_of_lower_case_in_description(df : pd.DataFrame, in_col_name : str = "description", out_col_name : str = "ratio_of_lower_case_in_description") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: len(list(filter(lambda y: y.islower(), x))) / len(x) if len(x) > 0 else 0)
    return df

def feature_ratio_of_emoji_in_description(df : pd.DataFrame, in_col_name : str = "description", out_col_name : str = "ratio_of_emoji_in_description") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: emoji_count(x) / len(x) if len(x) > 0 else 0)
    return df

def feature_is_default_profile(df : pd.DataFrame, in_col_name : str = "default_profile", out_col_name : str = "is_default_profile") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: 1 if x is True else 0)
    return df

def feature_is_default_profile_image(df : pd.DataFrame, in_col_name : str = "default_profile_image", out_col_name : str = "is_default_profile_image") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: 1 if x is True else 0)
    return df

def feature_account_age_in_days(df : pd.DataFrame, in_col_name : str = "created_at", out_col_name : str = "account_age_in_days") -> pd.DataFrame:
    df[in_col_name] = pd.to_datetime(df[in_col_name], utc = True)
    df[out_col_name] = (datetime.datetime.utcnow().replace(tzinfo = pytz.utc) - df[in_col_name]).apply(lambda x: x.days)
    return df

def feature_total_number_of_tweets(df : pd.DataFrame, in_col_name : str = "statuses_count", out_col_name : str = "total_number_of_tweets") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name]
    return df

def feature_total_number_of_favourites(df : pd.DataFrame, in_col_name : str = "favourites_count", out_col_name : str = "total_number_of_favourites") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name]
    return df

def feature_total_number_of_followers(df : pd.DataFrame, in_col_name : str = "followers_count", out_col_name : str = "total_number_of_followers") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name]
    return df

def feature_total_number_of_friends(df : pd.DataFrame, in_col_name : str = "friends_count", out_col_name : str = "total_number_of_friends") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name]
    return df

def feature_total_number_of_listed(df : pd.DataFrame, in_col_name : str = "listed_count", out_col_name : str = "total_number_of_listed") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name]
    return df

def feature_per_hour_number_of_tweets(df : pd.DataFrame, in_col_name : str = "statuses_count", out_col_name : str = "per_hour_number_of_tweets", created_at_col_name : str = "created_at") -> pd.DataFrame:
    df[created_at_col_name] = pd.to_datetime(df[created_at_col_name])
    hours = (datetime.datetime.utcnow().replace(tzinfo = pytz.utc) - df[created_at_col_name]).apply(lambda x: x.total_seconds() / (60 * 60))
    df[out_col_name] = df[in_col_name] / hours
    return df

def feature_per_hour_number_of_favourites(df : pd.DataFrame, in_col_name : str = "favourites_count", out_col_name : str = "per_hour_number_of_favourites", created_at_col_name : str = "created_at") -> pd.DataFrame:
    df[created_at_col_name] = pd.to_datetime(df[created_at_col_name])
    hours = (datetime.datetime.utcnow().replace(tzinfo = pytz.utc) - df[created_at_col_name]).apply(lambda x: x.total_seconds() / (60 * 60))
    df[out_col_name] = df[in_col_name] / hours
    return df

def feature_per_hour_number_of_followers(df : pd.DataFrame, in_col_name : str = "followers_count", out_col_name : str = "per_hour_number_of_followers", created_at_col_name : str = "created_at") -> pd.DataFrame:
    df[created_at_col_name] = pd.to_datetime(df[created_at_col_name])
    hours = (datetime.datetime.utcnow().replace(tzinfo = pytz.utc) - df[created_at_col_name]).apply(lambda x: x.total_seconds() / (60 * 60))
    df[out_col_name] = df[in_col_name] / hours
    return df

def feature_per_hour_number_of_friends(df : pd.DataFrame, in_col_name : str = "friends_count", out_col_name : str = "per_hour_number_of_friends", created_at_col_name : str = "created_at") -> pd.DataFrame:
    df[created_at_col_name] = pd.to_datetime(df[created_at_col_name])
    hours = (datetime.datetime.utcnow().replace(tzinfo = pytz.utc) - df[created_at_col_name]).apply(lambda x: x.total_seconds() / (60 * 60))
    df[out_col_name] = df[in_col_name] / hours
    return df

def feature_per_hour_number_of_listed(df : pd.DataFrame, in_col_name : str = "listed_count", out_col_name : str = "per_hour_number_of_listed", created_at_col_name : str = "created_at") -> pd.DataFrame:
    df[created_at_col_name] = pd.to_datetime(df[created_at_col_name])
    hours = (datetime.datetime.utcnow().replace(tzinfo = pytz.utc) - df[created_at_col_name]).apply(lambda x: x.total_seconds() / (60 * 60))
    df[out_col_name] = df[in_col_name] / hours
    return df

def feature_has_location(df : pd.DataFrame, in_col_name : str = "location", out_col_name : str = "has_location") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: 0 if x is None or len(x) == 0 else 1)
    return df

def feature_has_description(df : pd.DataFrame, in_col_name : str = "description", out_col_name : str = "has_description") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: 0 if x is None or len(x) == 0 else 1)
    return df

def feature_has_url(df : pd.DataFrame, in_col_name : str = "url", out_col_name : str = "has_url") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: 0 if x is None or len(x) == 0 else 1)
    return df

def feature_has_profile_banner(df : pd.DataFrame, in_col_name : str = "profile_banner_url", out_col_name : str = "has_profile_banner") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: 0 if x is pd.np.nan or len(x) == 0 else 1)
    return df

def feature_is_protected(df : pd.DataFrame, in_col_name : str = "protected", out_col_name : str = "is_protected") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: 1 if x is True else 0)
    return df

def feature_is_verified(df : pd.DataFrame, in_col_name : str = "verified", out_col_name : str = "is_verified") -> pd.DataFrame:
    df[out_col_name] = df[in_col_name].apply(lambda x: 1 if x is True else 0)
    return df