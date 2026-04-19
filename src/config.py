import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data/interim/master_dataset.csv")
FEATURE_PATH = os.path.join(BASE_DIR, "data/processed/features.csv")

MODEL_PATH = os.path.join(BASE_DIR, "models/model.pkl")
TEST_SIZE = 0.2
RANDOM_STATE = 42