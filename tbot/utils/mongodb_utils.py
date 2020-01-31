import pymongo
from sshtunnel import SSHTunnelForwarder
from typing import Dict, Any

def connection_to_db(
        remote_server_ip : str = "52.214.144.166",
        ssh_username : str = "ubuntu",
        ssh_pkey_path : str = "twitter_stream_db.pem",
        client_name : str = "twitter_streams",
        is_local : bool = True,
    ) -> pymongo.database.Database:

    if is_local:
        client = pymongo.MongoClient("127.0.0.1", 27017) # server.local_bind_port is assigned local port
        db = client[client_name]

    else:
        server = SSHTunnelForwarder(
            remote_server_ip,
            ssh_username = ssh_username,
            ssh_pkey = ssh_pkey_path,
            remote_bind_address = ("127.0.0.1", 27017),
            local_bind_address = ("0.0.0.0",)
        )

        server.start()

        client = pymongo.MongoClient("127.0.0.1", server.local_bind_port) # server.local_bind_port is assigned local port
        db = client[client_name]

    return db

def insert_to_db(db : pymongo.database.Database, collection_name : str, document : Dict[str, Any]) -> None:
    db[collection_name].insert(document)