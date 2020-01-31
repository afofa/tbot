import os
import pickle
import json
from typing import Dict, List, Optional, Any

def save_json(
    filepath : str, 
    data : Dict[str, Any], 
) -> None:
    
    with open(filepath, "w+") as f:
        f.write(json.dumps(data))

def save_jsonl(
    filepath : str, 
    datas : List[Dict[str, Any]], 
    is_append : bool = False,
) -> None:
    
    for data in datas:
        if is_append:
            with open(filepath, "a+") as f:
                f.write(json.dumps(data) + "\n")
        else:
            with open(filepath, "w+") as f:
                f.write(json.dumps(data) + "\n")

def save_pickle(obj : Any, filepath : str) -> None:
    with open(filepath, "wb") as f:
        pickle.dump(obj, f, protocol = -1)

def load_pickle(filepath : str) -> Any:
    with open(filepath, "rb") as f:
        obj = pickle.load(f)

    return obj

def check_if_file_exist_in_folder(filename_substr : str, foldername : str) -> Optional[str]:
    for filename in os.listdir(foldername):
        if filename_substr == filename.split(".")[0]:
            return f"{foldername}/{filename}"

    else:
        return