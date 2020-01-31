import pandas as pd
from tweepy.models import User, Status
from typing import Dict, List, Optional, Union, Any

def parse_user_metadata(
    user_metadata : Union[User, Dict[str, Any]],
) -> Dict[str, Any]:

    if isinstance(user_metadata, User):
        user_metadata = user_metadata._json

    fields_of_interest = [
        "name",
        "screen_name",
        "location",
        "url",
        "description",
        "protected",
        "verified",
        "followers_count",
        "friends_count",
        "listed_count",
        "favourites_count",
        "statuses_count",
        "created_at",
        "statuses_count",
        "profile_banner_url",
        "profile_image_url_https",
        "default_profile",
        "default_profile_image",
    ]
    
    user_metadata_filtered = {k: v for k,v in user_metadata.items() if k in fields_of_interest}
    user_metadata_filtered["user_id_str"] = user_metadata.get("id_str")

    return user_metadata_filtered

def parse_users_metadata(
    users_metadata : List[Union[User, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    users_metadata_filtered = list(map(parse_user_metadata, users_metadata))
    return users_metadata_filtered

def parse_metadata_into_df(
    users_metadata : List[Union[User, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    users_metadata_filtered = list(map(parse_user_metadata, users_metadata))
    df = pd.DataFrame(users_metadata_filtered)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df

def parse_status(
    status : Union[Status, Dict[str, Any]],
) -> Dict[str, Any]:
    
    if isinstance(status, Status):
        status = status._json

    fields_of_interest = [
        "created_at",
        "source",
        "user",
        "coordinates",
        "place",
        "retweet_count",
        "favorite_count",
        # "entities",
        # "extended_entities",
        "possibly_sensitive",
        "filter_level",
        "lang",
    ]

    status_filtered = {k: v for k, v in status.items() if k in fields_of_interest}
    null_fields = {k: None for k in fields_of_interest if k not in status_filtered}
    status_filtered = {**status_filtered , **null_fields}

    if status.get("full_text") is not None:
        tweet_text = status.get("full_text")
    else:
        tweet_text = status.get("text")
    status_filtered["text"] = tweet_text

    user_id_str = status.get("user").get("id_str")
    status_filtered["user_id_str"] = user_id_str

    status_id_str = status.get("id_str")
    status_filtered["status_id_str"] = status_id_str

    entities_parsed = parse_entities(status.get("entities", {}))
    status_filtered = {**status_filtered , **entities_parsed}

    if status.get("in_reply_to_status_id_str") is not None:
        new_fields = parse_mention(status)

    elif status.get("retweeted_status") is not None:
        new_fields = parse_retweet(status)

    elif status.get("is_quote_status"):
        new_fields = parse_quote(status)

    else:
        new_fields = parse_tweet(status)

    status_filtered = {**status_filtered , **new_fields}

    return status_filtered

def parse_statuses(
    statuses : List[Union[Status, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    statuses_filtered = list(map(parse_status, statuses))
    return statuses_filtered

def parse_statuses_into_df(
    statuses : List[Union[Status, Dict[str, Any]]],
) -> pd.DataFrame:
    fields_of_interest = [
        "created_at",
        "user_id_str",
        "status_id_str",
        "text",
        "source",
        "coordinates",
        "place",
        "retweet_count",
        "favorite_count",
        "possibly_sensitive",
        "lang",
        "type",
        "entities.hashtags",
        "entities.urls",
        "entities.user_mentions",
        "entities.media",
        "entities.symbols",
        "entities.polls",
    ]

    statuses_filtered = list(map(parse_status, statuses))
    df_status_filtered = pd.DataFrame(statuses_filtered)
    df_status_filtered = df_status_filtered[fields_of_interest]
    df_status_filtered["created_at"] = pd.to_datetime(df_status_filtered["created_at"])

    return df_status_filtered

def parse_tweet(status : Dict[str, Any]) -> Dict[str, Any]:
    status_filtered = {"type": "tweet"}

    return status_filtered

def parse_retweet(status : Dict[str, Any]) -> Dict[str, Any]:
    fields_of_interest = [
        "retweeted_status",
    ]

    status_filtered = {k: v for k, v in status.items() if k in fields_of_interest}
    status_filtered["type"] = "retweet"

    return status_filtered

def parse_mention(status : Dict[str, Any]) -> Dict[str, Any]:
    fields_of_interest = [
        "in_reply_to_status_id_str",
        "in_reply_to_user_id_str",
        "in_reply_to_screen_name",
    ]

    status_filtered = {k: v for k, v in status.items() if k in fields_of_interest}
    status_filtered["type"] = "mention"

    return status_filtered

def parse_quote(status : Dict[str, Any]) -> Dict[str, Any]:
    fields_of_interest = [
        "quoted_status_id_str",
        "quoted_status",
    ]

    status_filtered = {k: v for k, v in status.items() if k in fields_of_interest}
    status_filtered["type"] = "quote"

    return status_filtered

def parse_entities(entities_dct : Dict[str, Any]) -> Dict[str, Any]:
    entities_parsed = {
        "entities.hashtags" : entities_dct.get("hashtags", []),
        "entities.urls" : entities_dct.get("urls", []),
        "entities.user_mentions" : entities_dct.get("user_mentions", []),
        "entities.media" : entities_dct.get("media", []),
        "entities.symbols" : entities_dct.get("symbols", []),
        "entities.polls" : entities_dct.get("polls", []),
    }

    return entities_parsed

def parse_connectivity_into_df(
    connections: Dict[str, Dict[str, List[User]]],
    main_user_id_str_col_name: str = "main_user_id_str",
    connection_type_col_name: str = "connection_type",
) -> pd.DataFrame:
    
    list_of_dfs = []

    for user_id, dct in connections.items():
        df_friends = parse_metadata_into_df(dct.get("friends"))
        df_friends[main_user_id_str_col_name] = user_id
        df_friends[connection_type_col_name] = "friend"

        df_followers = parse_metadata_into_df(dct.get("followers"))
        df_followers[main_user_id_str_col_name] = user_id
        df_followers[connection_type_col_name] = "follower"

        list_of_dfs.extend([df_friends, df_followers])

    df = pd.concat(list_of_dfs, sort = True)
    return df