import sys
sys.path.append("../tbot")

from tbot.stream import listen_twitter_stream
from tbot.auth import get_api_object
from tbot.utils.mongodb_utils import connection_to_db

api = get_api_object()
db = connection_to_db(is_local = True, client_name = "twitter_streams")
listener_config = {
    "listener_type" : "mongodb",
    "db" : db,
    "collection_name" : "twitter_stream_1",
    "is_save_raw_stream" : True,
    "verbosity" : 2,
}
listen_twitter_stream(api, listener_config, keywords = ["turkey", "türkiye", "tr", "turquie"], is_async = True)