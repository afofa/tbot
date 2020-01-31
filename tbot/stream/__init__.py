# http://docs.tweepy.org/en/latest/streaming_how_to.html

from tweepy import StreamListener
from tweepy import API
from tweepy.models import Status
import json
import jsonlines
from ..utils.mongodb_utils import connection_to_db, insert_to_db
from pymongo.database import Database
from tbot.auth import get_api_object
from tweepy import Stream
from typing import Dict, List, Optional

class TwoStreamListener:
    def __init__(self):
        pass


class StreamListenerJSONL(StreamListener):

    def __init__(
        self, 
        api : API, 
        jsonl_filename : str = "stream.jsonl",
        is_save_raw_stream : bool = True,
        verbosity : int = 1, # 0, 1, 2
    ) -> None:
        self.api = api
        self.jsonl_filename = jsonl_filename
        self.is_save_raw_stream = is_save_raw_stream
        self.verbosity = verbosity
        self.fp = open(self.jsonl_filename, mode = "a+")
        self.writer = jsonlines.Writer(self.fp)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        if self.fp is not None:
            self.fp.close()

    def on_connect(self):
        """Called once connected to streaming server.
        This will be invoked once a successful response
        is received from the server. Allows the listener
        to perform some work prior to entering the read loop.
        """
        if self.verbosity >= 1:
            print("Connection established.")

    def on_data(self, raw_data):
        """Called when raw data is received from connection.
        Override this method if you wish to manually handle
        the stream data. Return False to stop stream and close connection.
        """
        data = json.loads(raw_data)
        
        if self.is_save_raw_stream:
            self.on_data_raw(data)
        else:
            self.on_data_filtered(data, raw_data)
            
    def on_data_raw(self, data) -> None:
        if self.verbosity == 1:
            print("New data.")
        elif self.verbosity == 2:
            print(f"New data.\n{data}")
        self.writer.write(data)

    def on_data_filtered(self, data, raw_data) -> None:
        if "in_reply_to_status_id" in data:
            if self.on_status(data) is False:
                return False
        elif "delete" in data:
            if self.on_delete(data) is False:
                return False
        elif "event" in data:
            if self.on_event(data) is False:
                return False
        elif "direct_message" in data:
            if self.on_direct_message(data) is False:
                return False
        elif "friends" in data:
            if self.on_friends(data) is False:
                return False
        elif "limit" in data:
            if self.on_limit(data) is False:
                return False
        elif "disconnect" in data:
            if self.on_disconnect(data) is False:
                return False
        elif "warning" in data:
            if self.on_warning(data) is False:
                return False
        elif "scrub_geo" in data:
            if self.on_scrub_geo(data) is False:
                return False
        elif "status_withheld" in data:
            if self.on_status_withheld(data) is False:
                return False
        elif "user_withheld" in data:
            if self.on_user_withheld(data) is False:
                return False
        else:
            if self.verbosity >= 1:
                print(f"Unknown message type: {raw_data}")

    def on_status(self, data):
        """Called when a new status arrives"""
        if self.verbosity >= 2:
            print(f"New status: {data}.")
        self.on_data_raw(data)

    def on_exception(self, data):
        """Called when an unhandled exception occurs."""
        if self.verbosity >= 1:
            print(f"Exception: {data}.")

    def on_delete(self, data):
        """Called when a delete notice arrives for a status"""
        if self.verbosity >= 2:
            print(f"Deletion: {data}.")
        self.on_data_raw(data)

    def on_event(self, data):
        """Called when a new event arrives"""
        if self.verbosity >= 1:
            print(f"New event: {data}.")

    def on_limit(self, data):
        """Called when a limitation notice arrives"""
        if self.verbosity >= 1:
            print(f"Limitation notice. {data}")

    def on_error(self, data):
        """Called when a non-200 status code is returned"""
        if self.verbosity >= 1:
            print(f"Non-200 status code: {data}.")

    def on_timeout(self):
        """Called when stream connection times out"""
        if self.verbosity >= 1:
            print("Timeout.")
        self.close()

    def on_disconnect(self, notice):
        """Called when twitter sends a disconnect notice
        Disconnect codes are listed here:
        https://developer.twitter.com/en/docs/tweets/filter-realtime/guides/streaming-message-types
        """
        if self.verbosity >= 1:
            print(f"Disconnect notice: {notice}.")
        self.close()

    def keep_alive(self):
        """Called when a keep-alive arrived"""
        if self.verbosity >= 1:
            print("Keep-alive.")

    def on_direct_message(self, data):
        """Called when a new direct message arrives"""
        if self.verbosity >= 1:
            print(f"Direct message: {data}.")

    def on_friends(self, data):
        """Called when a friends list arrives.
        friends is a list that contains user_id
        """
        if self.verbosity >= 1:
            print(f"Friends list: {data}.")

    def on_warning(self, data):
        """Called when a disconnection warning message arrives"""
        if self.verbosity >= 1:
            print(f"Disconnection warning message: {data}.")

    def on_scrub_geo(self, data):
        """Called when a location deletion notice arrives"""
        if self.verbosity >= 1:
            print(f"Scrub geo notice: {data}.")

    def on_status_withheld(self, data):
        """Called when a status withheld content notice arrives"""
        if self.verbosity >= 1:
            print(f"On status withheld notice: {data}.")

    def on_user_withheld(self, data):
        """Called when a user withheld content notice arrives"""
        if self.verbosity >= 1:
            print(f"On user withheld notice: {data}.")

class StreamListenerMongoDB(StreamListener):
    def __init__(
        self, 
        api : API, 
        db : Database,
        collection_name : str,
        is_save_raw_stream : bool = True,
        verbosity : int = 1, # 0, 1, 2
    ) -> None:
        self.api = api
        self.db = db
        self.collection_name = collection_name
        self.is_save_raw_stream = is_save_raw_stream
        self.verbosity = verbosity

    def on_connect(self):
        """Called once connected to streaming server.
        This will be invoked once a successful response
        is received from the server. Allows the listener
        to perform some work prior to entering the read loop.
        """
        if self.verbosity >= 1:
            print("Connection established.")

    def on_data(self, raw_data):
        """Called when raw data is received from connection.
        Override this method if you wish to manually handle
        the stream data. Return False to stop stream and close connection.
        """
        data = json.loads(raw_data)
        
        if self.is_save_raw_stream:
            self.on_data_raw(data)
        else:
            self.on_data_filtered(data, raw_data)

    def on_data_raw(self, data) -> None:
        if self.verbosity == 1:
            print("New data.")
        elif self.verbosity == 2:
            print(f"New data.\n{data}")
        insert_to_db(self.db, self.collection_name, data)

    def on_data_filtered(self, data, raw_data) -> None:
        if "in_reply_to_status_id" in data:
            if self.on_status(data) is False:
                return False
        elif "delete" in data:
            if self.on_delete(data) is False:
                return False
        elif "event" in data:
            if self.on_event(data) is False:
                return False
        elif "direct_message" in data:
            if self.on_direct_message(data) is False:
                return False
        elif "friends" in data:
            if self.on_friends(data) is False:
                return False
        elif "limit" in data:
            if self.on_limit(data) is False:
                return False
        elif "disconnect" in data:
            if self.on_disconnect(data) is False:
                return False
        elif "warning" in data:
            if self.on_warning(data) is False:
                return False
        elif "scrub_geo" in data:
            if self.on_scrub_geo(data) is False:
                return False
        elif "status_withheld" in data:
            if self.on_status_withheld(data) is False:
                return False
        elif "user_withheld" in data:
            if self.on_user_withheld(data) is False:
                return False
        else:
            if self.verbosity >= 1:
                print(f"Unknown message type: {raw_data}")

    def on_status(self, data):
        """Called when a new status arrives"""
        if self.verbosity >= 2:
            print(f"New status: {data}.")
        self.on_data_raw(data)

    def on_exception(self, data):
        """Called when an unhandled exception occurs."""
        if self.verbosity >= 1:
            print(f"Exception: {data}.")

    def on_delete(self, data):
        """Called when a delete notice arrives for a status"""
        if self.verbosity >= 2:
            print(f"Deletion: {data}.")
        self.on_data_raw(data)

    def on_event(self, data):
        """Called when a new event arrives"""
        if self.verbosity >= 1:
            print(f"New event: {data}.")

    def on_limit(self, data):
        """Called when a limitation notice arrives"""
        if self.verbosity >= 1:
            print(f"Limitation notice. {data}")

    def on_error(self, data):
        """Called when a non-200 status code is returned"""
        if self.verbosity >= 1:
            print(f"Non-200 status code: {data}.")

    def on_timeout(self):
        """Called when stream connection times out"""
        if self.verbosity >= 1:
            print("Timeout.")

    def on_disconnect(self, notice):
        """Called when twitter sends a disconnect notice
        Disconnect codes are listed here:
        https://developer.twitter.com/en/docs/tweets/filter-realtime/guides/streaming-message-types
        """
        if self.verbosity >= 1:
            print(f"Disconnect notice: {notice}.")

    def keep_alive(self):
        """Called when a keep-alive arrived"""
        if self.verbosity >= 1:
            print("Keep-alive.")

    def on_direct_message(self, data):
        """Called when a new direct message arrives"""
        if self.verbosity >= 1:
            print(f"Direct message: {data}.")

    def on_friends(self, data):
        """Called when a friends list arrives.
        friends is a list that contains user_id
        """
        if self.verbosity >= 1:
            print(f"Friends list: {data}.")

    def on_warning(self, data):
        """Called when a disconnection warning message arrives"""
        if self.verbosity >= 1:
            print(f"Disconnection warning message: {data}.")

    def on_scrub_geo(self, data):
        """Called when a location deletion notice arrives"""
        if self.verbosity >= 1:
            print(f"Scrub geo notice: {data}.")

    def on_status_withheld(self, data):
        """Called when a status withheld content notice arrives"""
        if self.verbosity >= 1:
            print(f"On status withheld notice: {data}.")

    def on_user_withheld(self, data):
        """Called when a user withheld content notice arrives"""
        if self.verbosity >= 1:
            print(f"On user withheld notice: {data}.")


def start_stream(
    api : API,
    listener : StreamListener,
    user_ids : List[str],
    keywords : List[str],
    is_async : bool = True,
    locations : Optional[List[str]] = None,
    languages : Optional[List[str]] = None,
    stall_warnings : bool = False, 
) -> None:
    stream = Stream(api.auth, listener)

    if isinstance(user_ids, list) and len(user_ids) > 0:
        stream.filter(follow = user_ids, is_async = is_async, locations = locations, languages = languages, stall_warnings = stall_warnings)
    elif isinstance(keywords, list) and len(keywords) > 0:
        stream.filter(track = keywords, is_async = is_async, locations = locations, languages = languages, stall_warnings = stall_warnings)
    else:
        raise RuntimeError("No user_id or keyword provided.")

    return stream

def listen_twitter_stream(
    api : API, 
    listener_config : Dict,
    user_ids : List[str] = [],
    keywords : List[str] = [],
    is_async : bool = True,
    locations : Optional[List[str]] = None,
    languages : Optional[List[str]] = None,
    stall_warnings : bool = False, 
    is_restart : bool = True,
    is_verbose : bool = True,
) -> None:

    listener_type : str = listener_config.get("listener_type")

    if listener_type == "mongodb":
        db = listener_config.get("db")
        if not isinstance(db, Database):
            raise RuntimeError("Provided `db` is not tweepy.database.Database object.")
        collection_name = listener_config.get("collection_name", "twitter_stream_0")
        is_save_raw_stream = listener_config.get("is_save_raw_stream", True)
        verbosity = listener_config.get("verbosity", 1)
        listener = StreamListenerMongoDB(api, db, collection_name, is_save_raw_stream, verbosity)
    elif listener_type == "jsonl":
        jsonl_filename = listener_config.get("jsonl_filename", "stream.jsonl")
        is_save_raw_stream = listener_config.get("is_save_raw_stream", True)
        verbosity = listener_config.get("verbosity", 1)
        listener = StreamListenerJSONL(api, jsonl_filename, is_save_raw_stream, verbosity)
    else:
        raise RuntimeError(f"listener_type is not recognized, got {listener_type}.")
    
    if is_verbose:
        print("Starting stream...")
    stream = start_stream(api, listener, user_ids, keywords, is_async, locations, languages, stall_warnings)
    
    if is_restart:
        while True:
            if not stream.running:
                if is_verbose:
                    print("Restarting stream...")
                stream = start_stream(api, listener, user_ids, keywords, is_async, locations, languages, stall_warnings)
            