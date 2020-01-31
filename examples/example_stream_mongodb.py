import sys
sys.path.append("../tbot")

from tbot.auth import get_api_object
from tbot.utils.mongodb_utils import connection_to_db
from tbot.stream import StreamListenerMongoDB
from tweepy import Stream

api = get_api_object()
db = connection_to_db(is_local = True, client_name = "twitter_streams")
listener = StreamListenerMongoDB(api, db, collection_name = "twitter_stream_0", verbosity = 2, is_save_raw_stream = True)
stream = Stream(api.auth, listener)
stream.filter(track=["türkiye", "tr", "turkey", "turquie"], is_async = True)