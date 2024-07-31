#In this code, we define the max-plus product between 2 matrix

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random


def max_plus_product(A, B):
    """
    Computes the max-plus product of two matrices.
    
    Parameters:
    - A: 2D numpy array
    - B: 2D numpy array
    
    Returns:
    - C: 2D numpy array, max plus product of A and B
    """
    C = np.zeros((A.shape[0], B.shape[1]))
    
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            C[i, j] = np.max(A[i, :] + B[:, j])
    

    return C

#calculates the eigenvalue of a matrix in max-plus algebra


def eigenvalue(A, maxdiff=np.inf, maxpow=1000):
    """
    Computes the eigenvalue of a matrix in max-plus algebra.
    
    Parameters:
    - A: 2D numpy array
    - maxdiff: int, maximum number of iterations for finding the cyclicity
    - maxpow: int, maximum number of times A matrix is powered
    
    Returns:
    - lambda_val: float, eigenvalue of matrix A in max-plus algebra
    - cyclicity: int, length of the critical circuit
    - max_pow: int, number of maximum A power needed
    - A_pow: 2D numpy array, maximum A power needed
    """
    # Initialize Pows to store powers of A
    Pows = np.zeros([A.shape[0], A.shape[1], maxpow])
    Pows[:, :, 0] = A

    for i in range(1, maxpow):
        # Compute a new power
        Pows[:, :, i] = max_plus_product(Pows[:, :, i-1], A)

        for diff in range(1, min(maxdiff, i+1)):
            # Look for a constant matrix
            M = Pows[:, :, i] - Pows[:, :, i-diff]

            # Tolerance check
            diag_diff = np.abs(np.diag(Pows[:, :, diff-1]) - np.max(M))
            SUM = np.sum(diag_diff < 1e-5)
            
            if SUM == 0 or SUM % diff != 0:
                continue

            if np.max(M) == np.min(M):  # If this happens, the matrix is constant
                lambda_ = np.max(M) / diff
                max_pow = i
                cyclicity = diff
                A_pow = Pows[:, :, diff-1]
                return lambda_, cyclicity, max_pow, A_pow
        

    print('Not found lambda with the bounds given')
    return np.nan, np.nan, np.nan, np.nan

#calculates the eigenvectors of a matrix in max-plus algebra

def eigenvector(A):
    """
    Computes the eigenvectors of a matrix in max-plus algebra.
    
    Parameters:
    - A: 2D numpy array
    
    Returns:
    - realeigenvector: 2D numpy array, matrix whose columns are eigenvectors of A
    - A_star: 2D numpy array, the star matrix
    """
    
    lambda_val, _, maxpow, _ = eigenvalue(A, 100, 10000)
    
    if np.isnan(lambda_val):
        print("Eigenvalue not found within the given bounds.")
        return np.nan, np.nan, np.nan, np.nan

    A_lambda = A - lambda_val
    
    P = -np.inf * np.ones(A.shape)
    np.fill_diagonal(P, 0)
    
    A_star = P.copy()

    
    for _ in range(int(maxpow) + 1):
        P = max_plus_product(P, A_lambda)
        A_star = np.maximum(A_star, P)
    
    eigenvectors = A_star.copy()
    
    for col in range(A_star.shape[1]):
        minimum = np.min(A_star[:, col])
        if minimum < 0:
            eigenvectors[:, col] = A_star[:, col] - minimum
    
    realeigenvector = []

    pivotnode=[]
    
    for col in range(eigenvectors.shape[1]):
        vec = eigenvectors[:, col]
        res = max_plus_product(A, vec[:, np.newaxis]).flatten()
        if np.all(res == vec + lambda_val):
            realeigenvector.append(vec)
            pivotnode.append(col)
    
    return np.array(realeigenvector).T, lambda_val, pivotnode, A_star

#add a new (virtual) station to a matrix of distances between train stations. 
#This new station is inserted between two existing stations, a and b, and its position is determined by parameter c 
#(which must be either a or b to indicate which station the new station is closer to).
#The aim is to add an intermediate station and update distances accordingly



def add_train(ini_matrix, a, b, c):

    # in this code, we insert one train from station a to station b, 
    # c means this virtual station is more close to which station of a or b (c=a/b)

    infty = -np.inf
    if c != a and c != b:
        print("Incorrect input")
        return

    if ini_matrix[b][a] == infty:
        print("Incorrect input")
        return   
    

    # Get the current size of the matrix
    n = len(ini_matrix)
    
    # Create a new matrix with increased size
    new_matrix = [[infty] * (n + 1) for _ in range(n + 1)]
    
    # Copy the original matrix elements into the new matrix
    for i in range(n):
        for j in range(n):
            new_matrix[i][j] = ini_matrix[i][j]
        
 
    new_matrix[b][a] = infty  

    if c==a:
        new_matrix[n][a] = 0
        new_matrix[b][n] = ini_matrix[b][a]

    if c==b:
        new_matrix[n][a] = ini_matrix[b][a]
        new_matrix[b][n] = 0
    
    new_matrix_np = np.array(new_matrix)
    return new_matrix_np


#create a weighted graph from an adjacency matrix in max-plus algebra. Int :an adjacency matrix, output an oriented 
#weighted graph




def create_weighted_graph_from_adjacency_matrix1(adjacency_matrix):
    # Vérifie si la matrice est carrée
    assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1], "La matrice d'adjacence doit être carrée"

    # Créating weighted graph
    G = nx.DiGraph()

    # Parcourir la matrice d'adjacence pour ajouter les arêtes avec poids
    num_nodes = adjacency_matrix.shape[0]
    for i in range(num_nodes):
        for j in range(num_nodes):
            weight = adjacency_matrix[i, j]
            if weight != -np.inf:  # Seuls les poids différents de -inf sont ajoutés comme arêtes
                G.add_edge(i, j, weight=weight)

    return G

# Function to display the weighted graph with curved arcs to distinguish directions
def visualize_weighted_graph(adjacency_matrix):
    # Créating weighted graph
    weighted_graph = create_weighted_graph_from_adjacency_matrix1(adjacency_matrix)

    # Positioning nodes for visualization
    pos = nx.spring_layout(weighted_graph)

    # List of edges in the graph
    edges = weighted_graph.edges()

    # List of bidirectional edges (arrows on both sides)
    bidirectional_edges = [(u, v) for (u, v) in edges if weighted_graph.has_edge(v, u)]

    # Viewing the graph
    plt.figure(figsize=(10, 8))

    # Drawing knots
    nx.draw_networkx_nodes(weighted_graph, pos, node_color='lightblue', node_size=500, alpha=0.8)

    # Draw node labels
    nx.draw_networkx_labels(weighted_graph, pos, font_size=12, font_color='black')

    # Drawing unidirectional edges with curved arcs
    nx.draw_networkx_edges(weighted_graph, pos, edgelist=[(u, v) for (u, v) in edges if (u, v) not in bidirectional_edges], 
                           arrows=True, connectionstyle='arc3,rad=0.3')

    # Draw bidirectional edges with curved arcs in both directions
    nx.draw_networkx_edges(weighted_graph, pos, edgelist=bidirectional_edges, 
                           arrows=True, connectionstyle='arc3,rad=0.3', style='dashed')

    # Add weight labels to edges
    labels = nx.get_edge_attributes(weighted_graph, 'weight')
    nx.draw_networkx_edge_labels(weighted_graph, pos, edge_labels=labels)

    # Display graph
    plt.title('Weighted graph from adjacency matrix in max-plus algebra')
    plt.axis('off')
    plt.show()

#find the longest cycle in a directed graph, by Finding all strongly related components,
#then For each strongly connected component, find all cycles,
#then Browse all cycles found


def find_largest_cycle(G):
    largest_cycle = []
    max_length = 0
    
    # Find all strongly related components
    scc = list(nx.strongly_connected_components(G))
    
    # For each strongly connected component, find all cycles
    for component in scc:
        subgraph = G.subgraph(component)
        cycles = nx.simple_cycles(subgraph)
        
        # Browse all cycles found
        for cycle in cycles:
            if len(cycle) > max_length:
                max_length = len(cycle)
                largest_cycle = cycle
    
    return largest_cycle

#Another direct function (see documentation), by listing all cycles, and picking the largest one.
#Find simple cycles (elementary circuits) of a graph.


def find_largest_cycle1(G):
    largest_cycle = None
    max_length = 0
    
    # Trouver tous les cycles élémentaires
    all_cycles = nx.simple_cycles(G)
    
    # Parcourir tous les cycles
    for cycle in all_cycles:
        if len(cycle) > max_length:
            max_length = len(cycle)
            largest_cycle = cycle
    
    return largest_cycle

#Calculation of the largest average cycle weight

def get_cycle_weights(G, max_length = 5):
    all_cycles = list(nx.simple_cycles(G, length_bound=max_length))
    #This variable will be used to store the maximum value of the average weights of the cycles found.
    max_average_weight = -np.inf
    #this variable will be used to store the cycle with the highest average weight among all the cycles found
    most_weighted_cycle = None
    average_weight = []
    for cycle in all_cycles:
        # Calculate the sum of the cyclic weights
        total_weight = sum(G[u][v]['weight'] for u, v in zip(cycle, cycle[1:] + cycle[:1]))
        av = total_weight / len(cycle)
        # Calculate average cycle weight
        average_weight.append(av)     
    
    return all_cycles, average_weight


def find_most_weighted_cycle(G, all_cycles = "", weights = [], max_length = 5):
    if all_cycles == "" and weights == []:
        all_cycles, weights = get_cycle_weights(G, max_length=max_length)
    max_average_weight = max(weights)
    most_weighted_cycle = all_cycles[weights.index(max_average_weight)]
    return most_weighted_cycle, max_average_weight

#Example with Amsterdam, importing the adjecency Matrix

# def update_cycle_weight(G, cycle)



#modifies a matrix of distances between train stations, progressively adding intermediate stations. 
#The function also plots the weighted average length of the most weighted cycles as a function of the number 
#of train additions made. 
#combines the creation and modification of a matrix of distances between stations, 
#the tracking of cycles in a weighted graph, and the visualization of results at each stage, 
#with the graph of lmabda as a function of the number of trains added.

def lambdadependingontrain(A, w=2, key = '', csvSave = False, max_length = 5, eigenValues = False):
    Agraph=create_weighted_graph_from_adjacency_matrix1(A)
    x=0
    X=[0]
    p,z = find_most_weighted_cycle(Agraph, max_length=max_length)
    if eigenValues:
        y = eigenvalue(A)[0]
    else:
        y = z
    Y=[y]
    # runs = 100
    # Z = [find_most_weighted_cycle(Agraph)[2]]
    # add a progress bar
    # pbar = tqdm(total = runs+1)
    while y>w:
        x=x+1
        # pbar.update(1)
        # p = find_most_weighted_cycle(Agraph)[0]
        A=add_train(A,p[0],p[len(p)-1],p[0])
        #visualize_weighted_graph(A)
        Agraph= create_weighted_graph_from_adjacency_matrix1(A)
        p,z = find_most_weighted_cycle(Agraph, max_length=max_length)
        if eigenValues:
            y = eigenvalue(A)[0]
        else:
            y = z
        # print(find_most_weighted_cycle(Agraph)[1])
        print(y)
        X.append(x)
        Y.append(y)
        # Z.append(z)
    # pbar.close()
    plt.plot(X,Y)
    plt.xlabel('nb of train we add')
    plt.ylabel('lambda')
    plt.title('lambda depending on adding train')
    # plt.show()
    plt.savefig(key + 'optim.png')
    print("the dimension of the new final matrix is",A.shape, "we added", len(X), "train")
    # save A to csv if needed
    if (csvSave):
        np.savetxt(key + 'matrix.csv', A, delimiter=',')
    # occurrences = {x: Y.count(x) for x in set(Y)}
    # Affichage des résultats
    # for element, count in occurrences.items():
    #     print(f"L'élément {element} apparaît {count} fois dans la liste.")
    return len(X)

#Example with Amsterdam

# 
def shuffle_off_diagonal_elements(matrix):
    size = matrix.shape[0]
    
    # Extraction des éléments hors diagonale supérieure
    off_diag_indices = np.triu_indices(size, k=1)
    off_diag_elements = matrix[off_diag_indices]
    
    # Mélange aléatoire des éléments hors diagonale
    np.random.shuffle(off_diag_elements)
    
    # Réaffectation des éléments mélangeés aux positions symétriques
    matrix[off_diag_indices] = off_diag_elements
    matrix.T[off_diag_indices] = off_diag_elements
    
    return matrix

# import concurrent.futures
# import multiprocessing
# import math

# # Global list to store results
# manager = multiprocessing.Manager()
# global_results = manager.list()

# # Sample function to be executed in parallel



def addEdges(matrix, numEdges, distanceMatrix, scale=1):
    # sample pairs of nodes to add edges between, make sure that they are -Inf in matrix
    for i in range(numEdges):
        while True:
            a = np.random.randint(0, matrix.shape[0])
            b = np.random.randint(0, matrix.shape[1])
            if matrix[a][b] == -np.inf:
                break
        # add edge with random weight
        matrix[a][b] = distanceMatrix[a][b]*scale
    return matrix

def split_node_adjacency_matrix(A, node_to_split):
    n = A.shape[0]
    new_node = n
    
    # Create a new adjacency matrix with one additional row and column
    new_A = np.full((n+1, n+1), -np.inf)
    
    # Copy existing values to the new matrix
    new_A[:n, :n] = A
    
    # Redistribute the edges of the node_to_split
    for i in range(n):
        if i == node_to_split:
            continue
        if A[node_to_split, i] > -np.inf:
            if random.random() < 0.5:
                new_A[new_node, i] = A[node_to_split, i]
                new_A[node_to_split, i] = -np.inf
            else:
                new_A[node_to_split, i] = A[node_to_split, i]
        if A[i, node_to_split] > -np.inf:
            if random.random() < 0.5:
                new_A[i, new_node] = A[i, node_to_split]
                new_A[i, node_to_split] = -np.inf
            else:
                new_A[i, node_to_split] = A[i, node_to_split]
    
    # Set the diagonal elements to -np.inf
    np.fill_diagonal(new_A, -np.inf)
    
    return new_A

def find_node_with_max_incoming_arrows(circuit):
    incoming_arrows_count = {}  # compter les flèches entrantes pour chaque nœud

    #  le comptage pour chaque nœud du circuit
    for edge in circuit:
        u, v = edge  # u -> v (u est la source, v est la destination)
        
        # Incrémenter le compteur d'arêtes entrantes pour v
        if v in incoming_arrows_count:
            incoming_arrows_count[v] += 1
        else:
            incoming_arrows_count[v] = 1
    
    # Trouver le nœud avec le maximum d'arêtes entrantes
    max_node = None
    max_arrows = -1
    
    for node, arrows_count in incoming_arrows_count.items():
        if arrows_count > max_arrows:
            max_arrows = arrows_count
            max_node = node
    # print("Dans le circuit {}, le nœud qui reçoit le maximum d'arêtes entrantes est : {}".format(circuit, max_node))
    return max_node


def get_circuits_lengths(graph):
    """
    Given a NetworkX graph, find all circuits (simple cycles) and their lengths.
    
    Args:
    graph (nx.Graph): The input graph.
    
    Returns:
    list: A list of tuples where each tuple contains a circuit (as a list of nodes)
          and the length of that circuit.
    """
    circuits = list(nx.simple_cycles(graph))
    circuits_lengths = [len(circuit) for circuit in circuits]
    return circuits_lengths


def lambdadependingonSplit(A, w=2, key = '', csvSave = False, circuitLen = True):
    Agraph=create_weighted_graph_from_adjacency_matrix1(A)
    x=0
    X=[0]
    Y=[find_most_weighted_cycle(Agraph)[1]]
    circuit_lengths = [get_circuits_lengths(Agraph)]
    while find_most_weighted_cycle(Agraph)[1]>w:
        x=x+1
        p=find_node_with_max_incoming_arrows([(find_most_weighted_cycle(create_weighted_graph_from_adjacency_matrix1(A))[0][0],find_most_weighted_cycle(create_weighted_graph_from_adjacency_matrix1(A))[0][1])])
        A=split_node_adjacency_matrix(A, p)
        
        # visualize_weighted_graph(A)
        # print(A)
        Agraph= create_weighted_graph_from_adjacency_matrix1(A)
        Atemp = Agraph.copy()
        iter = 0
        # print(iter)
        # print(nx.is_strongly_connected(Atemp))
        # while (not nx.is_strongly_connected(Atemp)) & iter < 20:
        #     # print('Scc going on')
        #     p=find_node_with_max_incoming_arrows([(find_most_weighted_cycle(create_weighted_graph_from_adjacency_matrix1(A))[0][0],find_most_weighted_cycle(create_weighted_graph_from_adjacency_matrix1(A))[0][1])])
        #     At=split_node_adjacency_matrix(A, p)
        #     Atemp= create_weighted_graph_from_adjacency_matrix1(A)
        #     iter += 1
        # # print(iter)
        # if (iter == 20):
        #     print("Graph is not strongly connected")
        #     break
        # else:
        #     A = At.copy()
        #     Agraph = Atemp.copy()

        find_most_weighted_cycle(Agraph)
        # print(find_most_weighted_cycle(Agraph)[1])
        X.append(x)
        Y.append(find_most_weighted_cycle(Agraph)[1])
        if circuitLen:
            circuit_lengths.append(get_circuits_lengths(Agraph))
    plt.plot(X,Y)
    plt.xlabel('Number of Stations Split')
    plt.ylabel('Lambda')
    plt.title('')
    # plt.show()
    plt.savefig(key + 'Split.png')
    # print("the dimension of the new final matrix is",A.shape, "we added", len(X), "train")
    # occurrences = {x: Y.count(x) for x in set(Y)}
    # Affichage des résultats
    # for element, count in occurrences.items():
    #     print(f"L'élément {element} apparaît {count} fois dans la liste.")
    return len(X),A, circuit_lengths


