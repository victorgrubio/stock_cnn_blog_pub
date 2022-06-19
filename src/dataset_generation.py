import os
import sys
import time
from pathlib import Path

import pandas as pd

from src.config import ROOT_PATH, OUTPUT_PATH, iter_changes
from src.data_generator import DataGenerator
from loguru import logger


args = sys.argv  # a list of the arguments provided (str)
pd.options.display.width = 0
company_code = args[1]
strategy_type = args[2]
PATH_TO_STOCK_HISTORY_DATA = ROOT_PATH / "stock_history"
Path.mkdir(PATH_TO_STOCK_HISTORY_DATA, exist_ok=True)


if __name__ == "__main__":
    count = 0
    start_time = time.time()
    data_file_name = company_code + ".csv"
    PATH_TO_COMPANY_DATA = os.path.join(PATH_TO_STOCK_HISTORY_DATA, company_code, data_file_name)
    data_gen = DataGenerator(company_code, PATH_TO_COMPANY_DATA, OUTPUT_PATH, strategy_type, False, logger)
