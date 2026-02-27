import os
from utils.config import Config
from utils.train_helper import TrainHelper
from models.cbms import LinearCBM

if __name__ == "__main__":
    config = Config.config()
    trainner = TrainHelper(config=config, Model=LinearCBM)
    trainner.run()
