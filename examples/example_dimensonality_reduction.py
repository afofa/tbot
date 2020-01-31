import sys
sys.path.append("../tbot")

from tbot.pipelines import dimensonality_reduction_pipeline
from dotenv import load_dotenv

load_dotenv(dotenv_path = "main.env")

dimensonality_reduction_pipeline(
    dataset_name = "varol-2017",
    method_config = {"type" : "tsne"},
    preprocessing_config = {"type" : "standard_scaler"},
)