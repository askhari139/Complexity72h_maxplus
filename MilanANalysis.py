from model import *
import math
# from random import random

file_path1 = r'/Users/kishorehari/Desktop/SciComm/Conferences/Complexity72h/Project/Datasets/Milan_correct.csv'


# numTrains = []
# for i in range(100):
#     shuffle_off_diagonal_elements(Amsterdammatrix)
#     numTrains.append(lambdadependingontrain(Amsterdammatrix, 2, 'Milan'+str(i)))

# # write number of trains added to a file
# with open('numTrainsMilan.txt', 'w') as f:
#     for item in numTrains:
#         f.write("%s\n" % item)
Milan = pd.read_csv(file_path1)


Milan = Milan.drop(Milan.columns[0], axis=1)
# Converting the dataframe to numpy
Milanmatrix = Milan.values

# Display the matrix.
print("Milan matrice:")
print(Milanmatrix)
Milangraph = create_weighted_graph_from_adjacency_matrix1(Milanmatrix)
max_circuit_milan=find_most_weighted_cycle(Milangraph)
max_circuit_milan[0]
maxnodecomingMilan=find_node_with_max_incoming_arrows([(max_circuit_milan[0][0],max_circuit_milan[0][1])])
split_node_adjacency_matrix(Milanmatrix, maxnodecomingMilan)

numTrains = []
circuitLengths = []
for i in range(100):
    # shuffle_off_diagonal_elements(Milanmatrix)
    x,A,c = lambdadependingonSplit(Milanmatrix, 2, 'Dummy')
    numTrains.append(x)
    circuitLengths.append(c)

plt.close()

circuitMax = []
for i in range(100):
    c1 = circuitLengths[i]
    c2 = []
    for j in c1:
        if (len(j)>2):
            c2.append(np.max(j))
    circuitMax.append(c2)

for i in range(100):
    plt.plot(range(1, len(circuitMax[i])+1), circuitMax[i])

plt.xlabel('Number of stations split')
plt.ylabel('Max length of circuits')
# plt.title('lambda depending on splitting station')
# plt.show()
plt.savefig('MilanSplitCircuits.png')

# [[np.max(j) for j in circuitLengths[i]] for i in range(100)]
# write circuit lengths to a file

with open('circuitLengthsMilan.txt', 'w') as f:
    for item in circuitMax:
        f.write("%s\n" % item)