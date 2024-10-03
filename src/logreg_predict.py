import json
import sys
import numpy as np
import pandas as pd

from logreg_train import prepare_data_X
from logistic_regression.main import predict
from classes.ClassBinarizer import ClassBinarizer

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    high_score = max(data, key=lambda x:x['score'])
    features = high_score['features']
    weight = high_score['weights']
    return features, weight

def main():
    
    file_data_test = sys.argv[1]
    file_weights = sys.argv[2]

    df_test = pd.read_csv(file_data_test)

    features, weights = read_json(file_weights)

    data_x = prepare_data_X(features, df_test)

    predictions_list = []
    house_list = []

    for w in weights:
        house_list.append(w)
        w_array = np.array(weights[w]["weigths"])
        predictions = predict(data_x, w_array)
        predictions_list.append(predictions)

    predictions_matrix = np.column_stack(predictions_list)
    predictions_house_indices = np.argmax(predictions_matrix, axis=1)

    binarizer = ClassBinarizer(house_list)
    
    predictions_house = []

    for prediction_house in predictions_house_indices:
        predictions_house.append(binarizer.get_house_from_binarize(prediction_house))

    df_predictions = pd.DataFrame({'Hogwarts House': predictions_house})
    df_predictions.rename_axis('Index', inplace=True)
    df_predictions.to_csv('./output_csv/houses.csv')
    print("Prediction file created ./output_csv/houses.csv")
    

if __name__ == "__main__":
    main()