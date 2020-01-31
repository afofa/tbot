import sys
sys.path.append("../tbot")

from tbot.datasets import get_raw_dataset, make_fetched_dataset
from dotenv import load_dotenv

load_dotenv(dotenv_path = "main.env")

datasets_names = [
    "verified-2019",
    "botwiki-2019",
    "cresci-rtbust-2019",
    "political-bots-2019",
    "botometer-feedback-2019",
    "vendor-purchased-2019",
    "celebrity-2019",
    "pronbots-2019",
    "midterm-2018",
    "cresci-stock-2018",
    "gilani-2017",
    "varol-2017",
    "caverlee-2011",
    "cresci-2017",
    "cresci-2015",
]

dataset_stats = {}
dataset_stats_fetched = {}
for dataset_name in datasets_names:
    try:
        df = get_raw_dataset(dataset_name)
        dataset_stats[dataset_name] = {
            "human" : len(df[df["label"] == "human"]),
            "bot" : len(df[df["label"] == "bot"]),
        }
        df = make_fetched_dataset(dataset_name)
        dataset_stats_fetched[dataset_name] = {
            "human" : len(df[df["label"] == "human"]),
            "bot" : len(df[df["label"] == "bot"]),
        }
        print(dataset_name)
        print(dataset_stats)
        print(dataset_stats_fetched)
    except Exception as e:
        print(dataset_name)
        print(e)