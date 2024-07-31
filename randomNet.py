from model import *

# generate a random matrix with 100 nodes, 20% density

n = 20

RFCN=np.random_matrix = np.random.rand(5, 5)
for i in range(5):
    RFCN[i][i]=-np.inf

# for i in range(20):
#     for j in range(20):
#         if RFCN[i][j] > 0.9:
#             RFCN[i][j] = np.random.rand(1)
#         else:
#             RFCN[i][j] = -np.inf

RFCN=RFCN*10+5
RFCN=(RFCN+RFCN.T)/2
RFCN
print(RFCN)
lambdadependingontrain(RFCN, 2, 'RFCN_5')
RFCNgraph=create_weighted_graph_from_adjacency_matrix1(RFCN)

B = shuffle_off_diagonal_elements(RFCN)


# numTrains = []
# for i in range(100):
#     shuffle_off_diagonal_elements(B)
#     numTrains.append(lambdadependingontrain(B, 2, 'RFCN_20'+str(i)))

# # write number of trains added to a file
# with open('numTrains_rand_25.txt', 'w') as f:
#     for item in numTrains:
#         f.write("%s\n" % item)