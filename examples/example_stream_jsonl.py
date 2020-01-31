import sys
sys.path.append("../tbot")

from tbot.auth import get_api_object
from tbot.stream import StreamListenerJSONL
from tweepy import Stream

api = get_api_object()
listener = StreamListenerJSONL(api, verbosity = 1, is_save_raw_stream = True)
stream = Stream(api.auth, listener)
stream.filter(track=["türkiye", "tr", "turkey", "turquie"], is_async = True)