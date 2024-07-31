from model import *

file_path1 = r'/Users/kishorehari/Desktop/SciComm/Conferences/Complexity72h/Project/Amsterdam_matrix.csv'
# reading the csv file, to dataframe
Amsterdam = pd.read_csv(file_path1)


Amsterdam = Amsterdam.drop(Amsterdam.columns[0], axis=1)
# Converting the dataframe to numpy
Amsterdammatrix = Amsterdam.values

# Display the matrix.
print("Amsterdam matrice:")
print(Amsterdammatrix)
numTrains = []
plt.plot()
for i in range(100):
    # shuffle_off_diagonal_elements(Amsterdammatrix)
    numTrains.append(lambdadependingontrain(Amsterdammatrix, 2, 'Amsterdam'+str(i)))
# lambdadependingontrain(Amsterdammatrix, 2, 'Amsterdam', csvSave=True)

# # # write number of trains added to a file
# # with open('numTrainsAmOriginal.txt', 'w') as f:
# #     for item in numTrains:
# #         f.write("%s\n" % item)

# numTrains = []
# for i in range(100):
#     shuffle_off_diagonal_elements(Amsterdammatrix)
#     numTrains.append(lambdadependingontrain(Amsterdammatrix, 1, 'Amsterdam'+str(i)))

# # write number of trains added to a file
# with open('numTrainsAm.txt', 'w') as f:
#     for item in numTrains:
#         f.write("%s\n" % item)

# numTrains = []
# circuitLengths = []
# for i in range(100):
#     # shuffle_off_diagonal_elements(Milanmatrix)
#     x,A,c = lambdadependingonSplit(Amsterdammatrix, 2, 'Amsterdam')
#     numTrains.append(x)
#     circuitLengths.append(c)

# plt.close()

# circuitMax = []
# for i in range(100):
#     c1 = circuitLengths[i]
#     c2 = []
#     for j in c1:
#         if (len(j)>2):
#             c2.append(np.max(j))
#     circuitMax.append(c2)

# for i in range(100):
#     plt.plot(range(1, len(circuitMax[i])+1), circuitMax[i])

# plt.xlabel('Number of stations split')
# plt.ylabel('Max length of circuits')
# # plt.title('lambda depending on splitting station')
# # plt.show()
# plt.savefig('AmsterdamSplitCircuits.png')

# # [[np.max(j) for j in circuitLengths[i]] for i in range(100)]
# # write circuit lengths to a file

# with open('circuitLengthsMilan.txt', 'w') as f:
#     for item in circuitMax:
#         f.write("%s\n" % item)