from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sys
import json

from main import prepare_data_X
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
    try:
        # Reading data
        df_train = pd.read_csv('../datasets/dataset_train.csv')
        df_test = pd.read_csv('../datasets/dataset_test.csv')

        house_mapping = {
            'Ravenclaw': 0,  
            'Slytherin': 1,
            'Gryffindor' : 2,
            'Hufflepuff' : 3
        }


        features, weights = read_json('evaluation_logs.json')
        X = prepare_data_X(features, df_train)

        y = df_train['Hogwarts House'].map(house_mapping)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


        model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        print(f"Classes détectées : {model.classes_}")


        X_new = prepare_data_X(features, df_test)



        # Prédire les résultats pour les nouvelles données
        y_pred_new = model.predict(X_new)

        # Afficher les prédictions
        # print("Prédictions pour les nouvelles données :", y_pred_new)
        
        binarizer = ClassBinarizer(['Ravenclaw','Slytherin', 'Gryffindor', 'Hufflepuff'])
        
        predictions_house = []

        for prediction_house in y_pred_new:
            predictions_house.append(binarizer.get_house_from_binarize(prediction_house))

        df_predictions = pd.DataFrame({'Hogwarts House': predictions_house})
        df_predictions.rename_axis('Index', inplace=True)
        df_predictions.to_csv('./output_csv/skickit-learn.csv')
        
    except ValueError as e:
        print(e)
    except FileNotFoundError:
        print('Failed to read the dataset')

if __name__ == "__main__":
    main()
