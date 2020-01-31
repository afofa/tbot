import sys
sys.path.append("../tbot")
from dotenv import load_dotenv
load_dotenv(dotenv_path = "main.env")

import pandas as pd
from tbot.auth import get_api_object
from tbot.fetch import fetch_user_metadata, fetch_users_metadata, fetch_user_timeline, fetch_users_timeline, fetch_by_search, fetch_user_all_data, fetch_user_followers, fetch_user_friends, fetch_user_followers_ids, fetch_user_friends_ids, fetch_users_connectivity
from tbot.parse import parse_users_metadata, parse_statuses, parse_statuses_into_df, parse_connectivity_into_df
from tbot.extract.metadata import user_metadata_features
from tbot.extract.timeline import user_timeline_features
from tbot.extract.connectivity import user_connectivity_features

from pprint import pprint

### Fetch

# create tweepy object to fetch data from Twitter API
t = get_api_object(credentials_filepath = "")

# user screenname to fetch metadata
user_to_fetch = 'RealColinPowell'

# user ids to fetch metadata
user_ids_to_fetch = ['1177341700177571840', '25073877', "6253282", "68034431"]

# users = fetch_user_followers_ids(t, user_screen_name = "realDonaldTrump", max_number_to_retrieve = 30000, stringify_ids=False)
# pprint(users)
# print(len(users))

# # fetch metadata for single user
# user_metadata = fetch_user_metadata(t, user_screen_name=user_to_fetch)
# pprint(user_metadata)

# # fetch metadata for multiple users
# users_metadata = fetch_users_metadata(t, user_ids_to_fetch)
# pprint(users_metadata)

# # fetch timeline for single user
# user_timeline = fetch_user_timeline(t, user_id = user_ids_to_fetch[1], count = 200, max_number_to_retrieve = 100)
# pprint(list(map(lambda x: x._json, user_timeline)))
# print(len(user_timeline))

# # fetch by search
# search_result = fetch_by_search(t, q=users_metadata[1].screen_name)
# pprint(search_result)
# print(len(search_result))

# # featch all user data
# all_user_data = fetch_user_all_data(t, user_ids_to_fetch[0])
# pprint(all_user_data)


### Parse

metadatas = fetch_users_metadata(t, user_ids_to_fetch)
timelines = fetch_users_timeline(t, user_ids_to_fetch)
connectivities = fetch_users_connectivity(t, user_ids_to_fetch)

df = pd.DataFrame(parse_users_metadata(metadatas))
df_timeline = parse_statuses_into_df(timelines)
df_connectivity = parse_connectivity_into_df(connectivities)

df.to_csv("metadata.csv")
df_timeline.to_csv("timeline.csv")
df_connectivity.to_csv("connectivities.csv")

df = user_metadata_features(df, is_drop_original_cols = False)
df.to_csv("metadata_features.csv")
df = user_timeline_features(df_timeline, df, is_drop_original_cols = False)
df.to_csv("metadata_and_timeline_features.csv")
df = user_connectivity_features(df_connectivity, df, is_drop_original_cols = False)
df.to_csv("metadata_and_timeline_and_connectivity_features.csv")