import requests
import zipfile
import gzip
import shutil
import pandas as pd
import os
from ..utils.io_utils import check_if_file_exist_in_folder
from ..utils.datasets_utils import enrich_with_user_metadata
from typing import Optional
def get_fetched_dataset(dataset_name : str) -> Optional[pd.DataFrame]:
    fetched_dataset_path = check_if_file_exist_in_folder(dataset_name, os.environ.get("FETCHED_DATASET_FOLDER"))
    
    if fetched_dataset_path is not None:
        df = read_fetched_dataset(fetched_dataset_path)
        return df
    
    else:
        return None

def read_fetched_dataset(path : str) -> pd.DataFrame:
    dtypes = {
        "user_id": int,
        "label": str, 
        "user_id_str": str, 
        "name": str, 
        "screen_name": str,
        "location": str,
        "description": str,
        "url": str,
        "protected": bool,
        "followers_count": int,
        "friends_count": int,
        "listed_count": int,
        "favourites_count": int,
        "verified": bool,
        "statuses_count": int,
        "profile_image_url_https": str,
        "profile_banner_url": str,
        "default_profile": bool,
        "default_profile_image": bool,
    }

    df = pd.read_csv(path, dtype = dtypes, na_filter = False, lineterminator = "\n")
    df["created_at"] = pd.to_datetime(df["created_at"])

    return df

def make_fetched_dataset(dataset_name : str, creds_filepath : str = "creds/twitter_credentials_tugrulcan.json") -> pd.DataFrame:

    # Get dataset, read from folder if available otherwise download
    df = get_raw_dataset(dataset_name)

    # Get user metadata via Twitter API
    df = enrich_with_user_metadata(df[["user_id", "label"]], creds_filepath = creds_filepath)
    fetched_dataset_foldername = os.environ.get("FETCHED_DATASET_FOLDER")
    df.to_csv(f"{fetched_dataset_foldername}/{dataset_name}.csv", index = False)

    return df

def get_raw_dataset(dataset_name : str) -> pd.DataFrame:
    datasets = {
        "verified-2019": {
            "url": "https://botometer.iuni.iu.edu/bot-repository/datasets/verified-2019/verified-2019.tsv.gz",
            "func": "read_dataset_verified_2019",
        },
        "botwiki-2019": {
            "url": "https://botometer.iuni.iu.edu/bot-repository/datasets/botwiki-2019/botwiki-2019.tsv.gz",
            "func": "read_dataset_botwiki_2019",
        },
        "cresci-rtbust-2019": {
            "url": "https://botometer.iuni.iu.edu/bot-repository/datasets/cresci-rtbust-2019/cresci-rtbust-2019.tsv.gz",
            "func": "read_dataset_cresci_rtbust_2019",
        },
        "political-bots-2019": {
            "url": "https://botometer.iuni.iu.edu/bot-repository/datasets/political-bots-2019/political-bots-2019.tsv.gz",
            "func": "read_dataset_political_bots_2019",
        },
        "botometer-feedback-2019": {
            "url": "https://botometer.iuni.iu.edu/bot-repository/datasets/botometer-feedback-2019/botometer-feedback-2019.tsv.gz",
            "func": "read_dataset_botometer_feedback_2019",
        },
        "vendor-purchased-2019": {
            "url": "https://botometer.iuni.iu.edu/bot-repository/datasets/vendor-purchased-2019/vendor-purchased-2019.tsv.gz",
            "func": "read_dataset_vendor_purchased_2019",
        },
        "celebrity-2019": {
            "url": "https://botometer.iuni.iu.edu/bot-repository/datasets/celebrity-2019/celebrity-2019.tsv.gz",
            "func": "read_dataset_celebrity_2019",
        },
        "pronbots-2019": {
            "url": "https://botometer.iuni.iu.edu/bot-repository/datasets/pronbots-2019/pronbots-2019.tsv.gz",
            "func": "read_dataset_pronbots_2019",
        },
        "midterm-2018": {
            "url": "https://botometer.iuni.iu.edu/bot-repository/datasets/midterm-2018/midterm-2018.tsv.gz",
            "func": "read_dataset_midterm_2018",
        },
        "cresci-stock-2018": {
            "url": "https://botometer.iuni.iu.edu/bot-repository/datasets/cresci-stock-2018/cresci-stock-2018.tsv.gz",
            "func": "read_dataset_cresci_stock_2018",
        },
        "gilani-2017": {
            "url": "https://botometer.iuni.iu.edu/bot-repository/datasets/gilani-2017/gilani-2017.tsv.gz",
            "func": "read_dataset_gilani_2017",
        },
        "varol-2017": {
            "url": "https://botometer.iuni.iu.edu/bot-repository/datasets/varol-2017/varol-2017.dat.gz",
            "func": "read_dataset_varol_2017",
        },
        "caverlee-2011": {
            "url": "https://botometer.iuni.iu.edu/bot-repository/datasets/caverlee-2011/caverlee-2011.zip",
            "func": "read_dataset_caverlee_2011",
        },
        "cresci-2017": {
            "url": "https://botometer.iuni.iu.edu/bot-repository/datasets/cresci-2017/cresci-2017.csv.zip",
            "func": "read_dataset_cresci_2017",
        },
        "cresci-2015": {
            "url": "https://botometer.iuni.iu.edu/bot-repository/datasets/cresci-2015/cresci-2015.csv.tar.gz",
            "func": "read_dataset_cresci_2015",
        },
    }

    if dataset_name in datasets.keys():
        data_filepath = check_if_file_exist_in_folder(dataset_name, os.environ.get("UNZIPPED_DATASET_FOLDER"))

        if data_filepath is None:
            print(f"Downloading {dataset_name}")

            url = datasets[dataset_name]["url"]
            filename = url.split("/")[-1]

            r = requests.get(url, stream=True)
            foldername = os.environ.get("ZIPPED_DATASET_FOLDER")
            zipped_filepath = f"{foldername}/{filename}"

            with open(zipped_filepath, 'wb+') as f:
                f.write(r.content)
            
            unzipped_filepath = unzip_file(zipped_filepath)

        else:
            unzipped_filepath = data_filepath

        read_func_str = datasets[dataset_name]["func"]

        df = eval(f"{read_func_str}(\"{unzipped_filepath}\")")
        return df

    else:
        raise RuntimeError(f"dataset_name = {dataset_name} is not recognized")

# TODO: implement read functions for tweets of caverlee2011, cresci-2017, cresci-2015

def unzip_file(filepath : str) -> None:
    if ".gz" in filepath:
        return unzip_gz_file(filepath)
    elif ".zip" in filepath:
        return unzip_zip_file(filepath)
    else:
        raise RuntimeError("Compression format is not recognised, it can be one of the following: .gz, .zip")

def unzip_gz_file(filename_in : str, filename_out : Optional[str] = None) -> str:
    filepath_out = os.environ.get("UNZIPPED_DATASET_FOLDER")
    if filename_out is None:
        filename_out = filename_in.split("/")[-1].split(".gz")[0]

    unzipped_filepath = f"{filepath_out}/{filename_out}"
    print(f"Unzipping {filename_in}")
    with gzip.open(filename_in, 'rb') as f_in:
        with open(unzipped_filepath, 'wb+') as f_out:
            shutil.copyfileobj(f_in, f_out)

    return unzipped_filepath

def unzip_zip_file(filename_in : str, filename_out : Optional[str] = None) -> None:
    filepath_out = os.environ.get("UNZIPPED_DATASET_FOLDER")
    if filename_out is None:
        filename_out = filename_in.split("/")[-1].split(".zip")[0]

    unzipped_filepath = f"{filepath_out}/{filename_out}"
    print(f"Unzipping {filename_in}")
    with zipfile.ZipFile(filename_in, "r") as zip_ref:
        zip_ref.extractall(unzipped_filepath)

    return unzipped_filepath

def read_labeling_data(filepath : str, delimeter : str = "\t") -> pd.DataFrame:
    df = pd.read_csv(filepath, delimiter = delimeter, header = None, names = ["user_id", "label"], dtype = {"user_id": str, "label": str})
    df.drop_duplicates(subset = ["user_id"], inplace = True)
    return df

def read_dataset_cresci_2015(filepath : str, is_make_binary_labeling : bool = True) -> pd.DataFrame:
    csv_filename = "users.csv"
    dfs_list = []
    files = [os.path.join(filepath, filename) for filename in os.listdir(filepath) if filename.split(".")[-1] == "zip"]
    for zipfile_path in files:
        with zipfile.ZipFile(zipfile_path, "r") as zip_ref:
            zip_ref.extract(csv_filename)
            df = pd.read_csv(csv_filename)
            dfs_list.append(df)
    os.remove(csv_filename)
    df = pd.concat(dfs_list, sort = True)
    df = df[["id", "dataset"]]
    df.rename(columns = {"id" : "user_id", "dataset" : "label"}, inplace = True)
    df.reset_index(drop = True, inplace = True)
    df["user_id"] = df["user_id"].apply(lambda x: str(x))

    if is_make_binary_labeling:
        df.loc[df["label"] == "TFP", "label"] = "human"
        df.loc[df["label"] == "E13", "label"] = "human"
        df.loc[df["label"] == "TWT", "label"] = "bot"
        df.loc[df["label"] == "INT", "label"] = "bot"
        df.loc[df["label"] == "FSF", "label"] = "bot"

    return df

def read_dataset_cresci_2017(filepath : str, is_make_binary_labeling : bool = True) -> pd.DataFrame:
    filepath = os.path.join(filepath, "datasets_full.csv")
    csv_filename = "users.csv"
    dfs_list = []
    files = [filename for filename in os.listdir(filepath) if filename.split(".")[-1] == "zip"]
    for zipfile_path in files:
        try:
            foldername_in_zip = ".".join(zipfile_path.split(".")[:-1])
            label = "_".join(zipfile_path.split(".")[0].split("_")[:2])
            with zipfile.ZipFile(os.path.join(filepath, zipfile_path), "r") as zip_ref:
                zip_ref.extract(os.path.join(foldername_in_zip, csv_filename))
                df = pd.read_csv(os.path.join(foldername_in_zip, csv_filename))
                df["label"] = label
                dfs_list.append(df)
            os.remove(os.path.join(foldername_in_zip, csv_filename))
        except Exception as e:
            continue

    df = pd.concat(dfs_list, sort = True)
    df = df[["id", "label"]]
    df.rename(columns = {"id" : "user_id"}, inplace = True)
    df.reset_index(drop = True, inplace = True)
    df["user_id"] = df["user_id"].apply(lambda x: str(x))

    if is_make_binary_labeling:
        df.loc[df["label"] != "genuine_accounts", "label"] = "bot"
        df.loc[df["label"] == "genuine_accounts", "label"] = "human"

    
    return df

def read_dataset_verified_2019(filepath: str) -> pd.DataFrame:
    df = read_labeling_data(filepath)
    return df

def read_dataset_botwiki_2019(filepath: str) -> pd.DataFrame:
    df = read_labeling_data(filepath)
    return df

def read_dataset_cresci_rtbust_2019(filepath: str) -> pd.DataFrame:
    df = read_labeling_data(filepath)
    return df

def read_dataset_political_bots_2019(filepath: str) -> pd.DataFrame:
    df = read_labeling_data(filepath)
    return df

def read_dataset_botometer_feedback_2019(filepath: str) -> pd.DataFrame:
    df = read_labeling_data(filepath)
    return df

def read_dataset_vendor_purchased_2019(filepath: str) -> pd.DataFrame:
    df = read_labeling_data(filepath)
    return df

def read_dataset_celebrity_2019(filepath: str) -> pd.DataFrame:
    df = read_labeling_data(filepath)
    return df

def read_dataset_pronbots_2019(filepath: str) -> pd.DataFrame:
    df = read_labeling_data(filepath)
    return df

def read_dataset_midterm_2018(filepath: str) -> pd.DataFrame:
    df = read_labeling_data(filepath)
    return df

def read_dataset_cresci_stock_2018(filepath: str) -> pd.DataFrame:
    df = read_labeling_data(filepath)
    return df

def read_dataset_gilani_2017(filepath: str) -> pd.DataFrame:
    df = read_labeling_data(filepath)
    return df

def read_dataset_varol_2017(filepath: str) -> pd.DataFrame:
    # TODO: check if labeling 1 means human or bot
    df = pd.read_csv(filepath, sep = r"\s+", header = None, names = ["user_id", "label"], engine = "python", dtype = {"user_id": str})
    df["label"] = df["label"].apply(lambda x: "human" if x == 0 else "bot")
    df.drop_duplicates(subset = ["user_id"], inplace = True)
    return df

def read_dataset_caverlee_2011(filepath: str) -> pd.DataFrame:
    filepath_to_legitimate_users = filepath + "/social_honeypot_icwsm_2011/legitimate_users.txt"
    filepath_to_content_polluters = filepath + "/social_honeypot_icwsm_2011/content_polluters.txt"
    
    column_names = ["user_id", "created_at", "collected_at", "number_of_followings", "number_of_followers", "number_of_tweets", "length_of_screen_name", "length_of_description_in_user_profile"]
    df_human = pd.read_csv(filepath_to_legitimate_users, delimiter="\t", header = None, names = column_names, dtype = {"user_id": str})
    df_human["created_at"] = pd.to_datetime(df_human["created_at"])
    df_human["collected_at"] = pd.to_datetime(df_human["collected_at"])
    df_human["label"] = "human"
    df_bot = pd.read_csv(filepath_to_content_polluters, delimiter="\t", header = None, names = column_names, dtype = {"user_id": str})
    df_bot["created_at"] = pd.to_datetime(df_bot["created_at"])
    df_bot["collected_at"] = pd.to_datetime(df_bot["collected_at"])
    df_bot["label"] = "bot"
    
    return pd.concat([df_human, df_bot], sort = True).reset_index(drop=True)