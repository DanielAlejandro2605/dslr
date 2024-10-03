import pandas as pd
import time
import curses

# Import exceptions
from exceptions.CourseNotFound import CourseNotFound
# Import prompt
from modules.prompt import prompt
# Import plotters
from modules.plotting import scatter_plot
# Import utils
from modules.utils import create_houses_list_of_course, get_name_for_plot, select_numeric_columns

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

def main():
    try:
        # Reading data
        df = pd.read_csv('../datasets/dataset_train.csv')

        # Getting all the columns with numeric values
        numeric_df = select_numeric_columns(df)
        # Getting names of the numeric columns
        list_courses = [column for column in numeric_df.columns if column != "Index"]
        
        # Get name of Hogwarts classes
        unique_houses = df['Hogwarts House'].unique()

        # Getting main data structure
        house_classes_data = create_houses_list_of_course(df, unique_houses, list_courses)

        feature : str = ''
        first_class : str = ''
        second_class : str = ''
        normalized_data_flag : bool = False
        list_features : list =  ['All grades scatter plots', 'grades', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        
        while True:
            feature = curses.wrapper(lambda stdscr: prompt(stdscr, list_features, [], "Choose feature class:\n", False))
            if feature == 'EXIT':
                return
            
            if feature == 'All grades scatter plots':
                generate_all_the_scatter_plots_of_grades(house_classes_data, list_courses)
                exit()

            first_class = curses.wrapper(lambda stdscr: prompt(stdscr, list_courses, [], "Choose first class:\n", False))
            if first_class == 'EXIT':
                return
            second_class = curses.wrapper(lambda stdscr: prompt(stdscr, list_courses, [], "Choose second class:\n", False))
            if second_class == 'EXIT':
                return
            # Validating classes choices
            if first_class != second_class:
                break
            else:
                print("The classes choised must be different")
                time.sleep(2)

        execute_plotter_feature(house_classes_data, (first_class, second_class), feature, normalized_data_flag)

    except ValueError as e:
        print(e)
    except CourseNotFound as e:
        print(e)
    except FileNotFoundError:
        print('Failed to read the dataset')

if __name__ == "__main__":
    main()