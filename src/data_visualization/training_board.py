import pandas as pd

import pandas as pd
import numpy as np
from typing import List
import pprint

import pandas as pd
import time
import curses

import sys
from classes.ClassBinarizer import ClassBinarizer


# Import exceptions
from exceptions.CourseNotFound import CourseNotFound
# Import prompt
from modules.prompt import prompt
# Import plotters
from modules.plotting import scatter_plot
# Import utils
from modules.utils import create_houses_list_of_course, get_name_for_plot, select_numeric_columns

# Import course
from classes.Course import Course


from logistic_regression.main import logistic_regression, predict


def select_numeric_columns(df : pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=np.number)

def get_tab_courses(df : pd.DataFrame, list_str_courses) -> List[Course]:
    courses : List[Course] = []
    # Sorting data
    for course in list_str_courses:
        # sort_data_course = sort_column(numeric_df, course)
        new_course = Course(course, df[course], True)
        courses.append(new_course)

    return courses

def create_houses_list_of_course(df : pd.DataFrame, hogwarts_houses: dict, list_str_courses : list) -> dict:
    # unique_houses = hogwarts_data['Hogwarts House'].unique()
    courses_by_house = {}
    # Dict {'House' : List[Course]}
    for house in hogwarts_houses:
        house_data = df[df['Hogwarts House'] == house]
        courses_by_house[house] = get_tab_courses(house_data, list_str_courses)

    return courses_by_house

def get_name_for_plot(first_class, second_class, feature, normalized_flag):
    normalized : str = ''
    if normalized_flag:
        normalized = '_normalized'
    return f'./plots/{feature}{normalized}:{get_name_class_for_plot(first_class)}-VS-{get_name_class_for_plot(second_class)}'

def get_name_class_for_plot(class_name : str):
    return '-'.join(class_name.split(' ')).lower()

def get_class_data_from_houses(house_courses : dict, course_searched : str) -> dict:

    grades_of_house : dict = {}
    course_found : bool = False

    for house in house_courses:
        for course in house_courses[house]:
            if course.get_name() == course_searched:
                course_found = True
                grades_of_house[house] = course

    if not course_found:
        raise CourseNotFound(course_searched)

    return grades_of_house

def extract_house_feature_data(house_classes_data,
                               first_class : str, second_class : str,
                               feature : str, normalized_flag : bool) -> dict:


    first_class_data = get_class_data_from_houses(house_classes_data, first_class)
    second_class_data = get_class_data_from_houses(house_classes_data, second_class)

    data_to_be_plotted : dict = {
        first_class : {},
        second_class : {}
    }

    for house in house_classes_data:
        data_to_be_plotted[first_class_data[house].get_name()][house] = first_class_data[house].get_describe_feature(feature, normalized_flag)
        data_to_be_plotted[second_class_data[house].get_name()][house] = second_class_data[house].get_describe_feature(feature, normalized_flag)

    return data_to_be_plotted


def execute_plotter_feature(house_classes_data : dict, classes : tuple, feature : str, normalized_flag : bool):
    # Extracting class from tuple
    first_class = classes[0]
    second_class = classes[1]
    # Getting data to be plotted
    data_to_be_plotted = extract_house_feature_data(house_classes_data, first_class, second_class, feature, normalized_flag)
    # Getting plot name
    plot_name = get_name_for_plot(first_class, second_class, feature, normalized_flag)
    # Plotting
    scatter_plot((first_class, second_class), data_to_be_plotted, plot_name)

def generate_all_the_scatter_plots_of_grades(house_classes_data : dict, list_courses : list):

    for first_class in list_courses:
        for second_class in list_courses:
            if first_class != second_class:
                execute_plotter_feature(house_classes_data, (first_class, second_class), 'grades', False)

# Une fois avoir enregistrer toutes les notes
#creer des listes temporaires a remplir, pour cela on donne la possibilite de choisir des cours
#Soit choisir un cours existant, soit donner la possibilite de creer une combinaison
#Creer un bouton pour mettre fin a la selection (peut etre mettre un maximum le nombre de cours choisis)
# lancer le training
#enregistrer dans un dossier les listes avec avec un taux de reussite elever, avec le nom des listes et le pourcentage de resultat

# Gryffindor = 1
# Slytherin = 2
# Ravenclaw = 3
# Hufflepuff = 4



#mettre a la place de raw_data, un raw data zscore
def prepare_data(chosen_data, data):
    print("start preparation of data")
    courses = []
    result = Course('Hogwarts House', data['Hogwarts House'].tolist(), True)
    # print(result.raw_data)
    # courses.append(Course('Hogwarts House', data['Hogwarts House'].tolist()))
    for course in chosen_data:
        if '/' in course:
            print("combinaison")
            parts = course.split('/')
            combined_data = []
            for part in parts:
                course_data = data[part].tolist()
                combined_data.append(course_data)
            combined_mean_data = np.mean(combined_data, axis=0).tolist()
            courses.append(Course(course, combined_mean_data, True))
        else:
            print("not combinaison")
            course_data = data[course].tolist()
            courses.append(Course(course, course_data, True))
    for course in courses:
        print(course.name)
    df_courses = pd.DataFrame()

    for course in courses:
        df_courses[course.name] = pd.Series(course.data_zscore)
        print("here 1" ,len(course.data_zscore), len(course.raw_data), course.count)
        print("HERE", len(df_courses[course.name]))
    num_rows = len(df_courses)
    num_cols = len(courses) + 1 #car il a Hogwarts House qui n'est pas un cours, donc pas besoin de faire +1
    print(num_rows,num_cols)
    input_matrix = np.ones((num_rows, num_cols))

    for i, course in enumerate(courses):
        input_matrix[:, i + 1] = df_courses[course.name].values
    
    return input_matrix, df_courses

    # matrix = df_courses.to_numpy()
def display_current_selection(stdscr, chosen_courses):
    stdscr.clear()
    stdscr.addstr(0, 0, "Current selection:\n")
    for i, course in enumerate(chosen_courses, start=1):
        stdscr.addstr(i, 0, f"{i}. {course}")
    stdscr.refresh()

def main():
    try:
        # Reading data
        # with open('../datasets/dataset_train_3.csv') as file :
        df = pd.read_csv('../datasets/dataset_train_3.csv')


        # Get name of Hogwarts classes
        unique_houses = df['Hogwarts House'].unique()
        binarizer = ClassBinarizer(unique_houses)

            # Dictionnaire de correspondance
        house_mapping = {
            'Gryffindor': 1,
            'Slytherin': 2,
            'Ravenclaw': 3,
            'Hufflepuff': 4
        }

        # Appliquer la correspondance
        df['Hogwarts House'] = df['Hogwarts House'].map(house_mapping)
        # Getting all the columns with numeric values
        numeric_df = select_numeric_columns(df)
        # Getting names of the numeric columns
        list_courses = [column for column in numeric_df.columns if column != "Index"]
        # print(x for x in list(list_courses))

        # Getting main data structure
        # house_classes_data = create_houses_list_of_course(df, unique_houses, list_courses)

        feature : str = ''
        first_class : str = ''
        second_class : str = ''
        normalized_data_flag : bool = False
        list_features : list =  [x for x  in list_courses if x != 'Hogwarts House']
        list_courses = [x for x  in list_courses if x != 'Hogwarts House']
        list_features.extend(['Combinaison', 'Finish'])
        # print(list_features)
        chosen_courses: list = []
        #verifier qu'on prend pas plusieurs fois le meme cours, mettre une limite de cours, permettre d'enchainer les tests et enregistrer les plus performants

        while True:
            feature = curses.wrapper(lambda stdscr: prompt(stdscr, list_features, chosen_courses, "Choose feature class:\n", False))
            if feature == 'EXIT':
                return
            elif feature == 'Finish':
                break
            elif feature == 'Combinaison':
                first_class = curses.wrapper(lambda stdscr: prompt(stdscr, list_courses, chosen_courses, "Choose first class for combination:\n", False))
                if first_class == 'EXIT':
                    break
                second_class = curses.wrapper(lambda stdscr: prompt(stdscr, list_courses, chosen_courses, "Choose second class for combination:\n", False))
                if second_class == 'EXIT':
                    break
                if first_class != second_class:
                    chosen_courses.append(f"{first_class}/{second_class}")
                else:
                    print("The classes chosen must be different")
                    time.sleep(2)
                    continue
            else:
                chosen_courses.append(feature)
        if not chosen_courses:
            print("no courses were chosen")
        else:
            print("chosen course")
            data_x, df_courses = prepare_data(chosen_courses, numeric_df)
            predictions_list = []
            df2 = pd.read_csv('../datasets/dataset_train_3.csv')
            print("HERE 464 \n")
            for i, house in enumerate(unique_houses):
                print(house)
                data_y_by_house = binarizer.binarize(df2['Hogwarts House'], house)
                w = logistic_regression(data_x, data_y_by_house)
                print(w)
                predictions = predict(data_x, w)
                predictions_list.append(predictions)
            print(predictions_list)
            print("HERE 2")

            expected_results = df2['Hogwarts House'].tolist()
            predicted_results = []

            for i in range(len(predictions_list[0])):
                predictions = [predictions_list[j][i] for j in range(len(predictions_list))]
                max_prediction = max(predictions)
                predictions = [1 if pred == max_prediction else 0 for pred in predictions]
                
                if predictions[0] == 1:
                    predicted_results.append('Ravenclaw')
                elif predictions[1] == 1:
                    predicted_results.append('Slytherin')
                elif predictions[2] == 1:
                    predicted_results.append('Gryffindor')
                elif predictions[3] == 1:
                    predicted_results.append('Hufflepuff')
            correct_predictions = sum([1 for expected, predicted in zip(expected_results, predicted_results) if expected == predicted])
            total_predictions = len(expected_results)
            success_rate = (correct_predictions / total_predictions) * 100

            print(f'Success rate: {success_rate:.2f}%')
                    
            # expected_results = []
            # predicted_results = []
            # w_1 = logistic_regression(data_x, output_gryffindor)
            # w_2 = logistic_regression(data_x, output_hufflepuff)
            # w_3 = logistic_regression(data_x, output_ravenclaw)
            # w_4 = logistic_regression(data_x, output_slytherin)
            # print("avant")
            # for i in range(50):
            #     data_test = get_single_sample_from_dataset_test_2(i, chosen_courses)#jai modifie 
            #     expected_results.append(df_courses['Hogwarts House'][i])
            #     # predictions_1 = predict(data_test, w_1)#jai modifie predict
            #     # predictions_2 = predict(data_test, w_2)
            #     # predictions_3 = predict(data_test, w_3)
            #     # predictions_4 = predict(data_test, w_4)

            #     predictions = [predictions_1, predictions_2, predictions_3, predictions_4]
            #     max_prediction = max(predictions)
            #     predictions = [1 if pred == max_prediction else 0 for pred in predictions]
            #     predictions_1, predictions_2, predictions_3, predictions_4 = predictions

            #     if predictions_1 == 1:
            #         predicted_results.append(1)  # Gryffindor
            #     elif predictions_2 == 1:
            #         predicted_results.append(2)  # Hufflepuff
            #     elif predictions_3 == 1:
            #         predicted_results.append(3)  # Ravenclaw
            #     elif predictions_4 == 1:
            #         predicted_results.append(4)  # Slytherin

            #     print("What we expect",df_courses['Hogwarts House'][i])
            #     if predictions_1 == 1:
            #         print(f'Index {i} is     1 : Gryffindor\n')
            #     if predictions_2 == 1:
            #         print(f'Index {i} is     2 : Hufflepuff\n')
            #     if predictions_3 == 1:
            #         print(f'Index {i} is     3 : Ravenclaw\n')
            #     if predictions_4 == 1:
            #         print(f'Index {i} is     4 : Slytherin\n')
            # correct_predictions = sum([1 for expected, predicted in zip(expected_results, predicted_results) if expected == predicted])
            # total_predictions = len(expected_results)
            # success_rate = (correct_predictions / total_predictions) * 100
            # print(f'Success rate: {success_rate:.2f}%')

    except ValueError as e:
        print(e)
    except CourseNotFound as e:
        print(e)
    except FileNotFoundError:
        print('Failed to read the dataset')

if __name__ == "__main__":
    main()
