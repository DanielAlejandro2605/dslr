import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

def evaluate(df : pd.DataFrame, predictions : np.ndarray):
    house_mapping = {
        'Ravenclaw': 0,
        'Slytherin': 1,
        'Gryffindor': 2,
        'Hufflepuff': 3
    }

    real_houses_values = df['Hogwarts House'].map(house_mapping).to_numpy()
    score = accuracy_score(real_houses_values, predictions)
    return score

def save_feature_list(evaluate_score : int, features_list : list):

    with open('evaluation_logs.txt', 'a+') as file:
        file.write(f"Score {evaluate_score} |  {features_list}\n")


