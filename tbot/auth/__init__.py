import tweepy
import json
import os
from typing import Dict, Optional

def load_credentials(
    filepath : str = "",
) -> Optional[Dict[str, str]]:
    
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Exception when reading file from {filepath}, returning None:\n{e}")
        return

    return data

def load_credentials_from_env() -> Optional[Dict[str, str]]:
    creds = {
        "CONSUMER_KEY": os.environ.get("CONSUMER_KEY", None), 
        "CONSUMER_SECRET": os.environ.get("CONSUMER_SECRET", None),
        "ACCESS_TOKEN": os.environ.get("ACCESS_TOKEN", None), 
        "ACCESS_SECRET": os.environ.get("ACCESS_SECRET", None),
    }

    for k, v in creds.items():
        if v is None:
            print(f"Environment variable {k} is not specified, returning None")
            return
    else:
        return creds

def get_api_object(
    credentials_dict : Optional[Dict[str, str]] = None, 
    credentials_filepath : Optional[str] = None,
    wait_on_rate_limit : bool = True,
    wait_on_rate_limit_notify : bool = True,
) -> tweepy.API:

    if credentials_dict is None:
        credentials_dict = load_credentials_from_env()

    if credentials_dict is None and credentials_filepath is not None:
        credentials_dict = load_credentials(credentials_filepath)

    if credentials_dict is not None:
        auth = tweepy.OAuthHandler(credentials_dict["CONSUMER_KEY"], credentials_dict["CONSUMER_SECRET"])
        auth.set_access_token(credentials_dict["ACCESS_TOKEN"], credentials_dict["ACCESS_SECRET"])
        t = tweepy.API(auth, wait_on_rate_limit = wait_on_rate_limit, wait_on_rate_limit_notify = wait_on_rate_limit_notify)
        return t
        
    else:
        raise RuntimeError("No valid credentials, filepath or environment variable provided")