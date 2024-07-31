from model import *
import math
# from random import random

file_path1 = r'/Users/kishorehari/Desktop/SciComm/Conferences/Complexity72h/Project/paris_reduced.csv'


Amsterdam = pd.read_csv(file_path1)


Amsterdam = Amsterdam.drop(Amsterdam.columns[0], axis=1)
# Converting the dataframe to numpy
Amsterdammatrix = Amsterdam.values

# Display the matrix.
print("Amsterdam matrice:")
print(Amsterdammatrix)
# numTrains = []
# for i in range(100):
#     # shuffle_off_diagonal_elements(Amsterdammatrix)
#     numTrains.append(lambdadependingontrain(Amsterdammatrix, 2, 'Amsterdam'+str(i)))
lambdadependingontrain(Amsterdammatrix, 2, 'Paris', csvSave=True, max_length=10)