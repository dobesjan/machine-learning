import pandas as pd

class Preprocessing:

    def get_dataset(file_name: str):
        dataset = pd.read_csv(file_name)
        return dataset