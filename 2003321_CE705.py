import csv
from math import sqrt
import statistics as st
import sys
import subprocess

try:
    ''' We are trying to import numpy if it is not installed then it throws an exception'''
    import numpy as np
except ImportError:
    ''' If an exception is throwed then it will install the numpy by using subprocess package'''
    subprocess.call([sys.executable,"-m",'pip', 'install', 'numpy'])
finally:
    ''' Method will run even if any of the above executed'''
    import numpy as np

def load_from_csv(csvFile):
    '''Loading the CSV file from the same Folder with "," as a delimitter'''
    try:
        array_data = np.genfromtxt(csvFile, delimiter=",")
        '''Converting array to list'''
        array_list = array_data.tolist()
    except OSError:
        ''' If File is not present at .py file location the enter the file path for proceeding furthur'''

        file_name = input(f"please copy or type the path where {csvFile} file is present with(\) in the path")
        '''replacing \ with / as python accepts / as an input.'''
        file_name1 = file_name.replace('\\', '/')
        data = np.genfromtxt(file_name1, delimiter=",")
        array_of_list = data.tolist()   # Converting to List
        return array_of_list
    return array_list



def get_distance(list_a, list_b):
    '''Calculating the Euclidean distance between two list i.e list_a and list_b of same size'''
    return sqrt(sum(pow(a - b, 2) for a, b in zip(list_a, list_b)))


def get_standard_deviation(list_of_lists, column_num):
    '''Calculate the standard deviation of the column number which is passed in the function'''
    # Calculating the average of the column, where axis = 0 which means Columns
    columns_avg = np.average(list_of_lists, axis=0)
    # Selecting the required column to get the average of that column
    column_avg = columns_avg[column_num]
    matrix = np.array(list_of_lists)
    # Subtracting column average from each row in the input matrix
    sub_matrix = matrix[:, column_num] - column_avg
    sum_of_squares = sum(list(map(lambda sub_m: sub_m ** 2, sub_matrix)))
    return sqrt((1 / (len(sub_matrix) - 1)) * sum_of_squares)


def get_standardised_matrix(list_of_lists):
    ''' This function returns standardised version of the matrix which is passed as a parameter'''
    # Calculating the average of the column, where axis = 0 means columns
    columns_avg = np.average(list_of_lists, axis = 0)
    matrix = np.array(list_of_lists)
    # adding rows and columns of the matix
    rows, columns = matrix.shape
    columns_std_dev = list()    # Creating an empty list
    # Creating zero array with the size of input matrix
    std_matrix = np.zeros([rows,columns])
    for column in range(columns):
        columns_std_dev.append(get_standard_deviation(list_of_lists, column))
        for row in range(rows):
            # Using the given formula to calcuate the standardised matrix
            std_matrix[row, column] = ((matrix[row, column] - columns_avg[column])/ columns_std_dev[column])
    std_matrix = std_matrix.tolist()  # Converting array to list
    return std_matrix


def get_k_nearest_labels(a_list, learning_data, learning_data_labels, k):
    ''' This function locates the K-Nearest Neighbor labels'''
    distances = [get_distance(a_list,training_row) for training_row in learning_data]
    # Conveting the distances into dictionary for getting key and values
    distance_dict = dict(zip(distances, learning_data_labels))
    distance_sort = sorted(distance_dict.items(), key=lambda tup: (tup[0], tup[1]))
    # Slicing the first K elements from the sorted list
    sorted_items = distance_sort[0:k]
    # Creating another list for adding k values
    neighbors = list()
    for i in range(k):
        neighbors.append(sorted_items[i][1])
    return neighbors


def get_mode(list_of_lists):
    ''' Returning the most frequently occuring lables from Statistics Package '''
    return st.mode(list_of_lists)


def classify(data, learning_data, learning_data_labels, num_neighbors):
    ''' This fucntion uses K-Nearest Neighbor algorithm and returns a matrix i.e list_of_list'''
    # Calling the get_standardised_matrix function
    standard_matrix = get_standardised_matrix(data)
    standard_learning_data = get_standardised_matrix(learning_data)
    results = []
    count = 0
    for training_data in standard_matrix:
        # Appending the values to the results list and incrementing the count value with 1
        results.append(get_mode(get_k_nearest_labels(training_data, standard_learning_data, learning_data_labels, num_neighbors)))
        count += 1
    return results


def get_accuracy(correct_data_labels, data_labels):
    ''' This function returns accuracy, it calculates the accuracy by comparing both the list. If both list have
        same value then the accuracy will be 100%'''
    count = 0
    for data in range(len(data_labels)):
        if data_labels[data] == correct_data_labels[data]:
            count += 1
        else:
            count += 0
    accuracy = (count / len(data_labels)) * 100
    return accuracy


def run_test():
    ''' This function runs series of tests and creates one matrix for each file'''
    # Calling load_from_csv function to read the data.csv file
    list_of_lists = load_from_csv("data.csv")
    # Calling load_from_csv function to read the learning_data.csv file
    learning_data = load_from_csv("learning_data.csv")
    # Calling load_from_csv function to read the learning_data_labels.csv file
    learning_data_labels = load_from_csv("learning_data_labels.csv")
    # Calling load_from_csv function to read the correct_data_labels.csv file
    correct_data_labels = load_from_csv("correct_data_labels.csv")
    # For each value of K performing classification and finding the accuracy percentage
    for k in range(3, 16):
        data_labels = classify(list_of_lists, learning_data, learning_data_labels, k)
        accuracy = float("{0:.1f}".format(get_accuracy(correct_data_labels, data_labels)))
        # Printing the K value and Accuracy percentage
        print(f"k={k}, Accuracy = {accuracy}%", )
    return list_of_lists, learning_data, learning_data_labels, accuracy


run_test()
