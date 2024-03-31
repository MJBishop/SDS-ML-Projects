import numpy as np
import kmeans
import common
import naive_em
import em

# X = np.loadtxt("toy_data.txt")

# TODO: Your code here
#   K = 1->4
#   Try: seed = 0->4

#  K-Means
# print('k-means:')
# for k in range(1, 5):
#     print('For K =', k)
#     for i in range(5):
#         seed = i
#         mixture, post = common.init(X, k, seed)
#         mixture, post, cost = kmeans.run(X, mixture, post)
#         title = f'k-means: k = {k}, cost for seed: {i} = {cost}'
#         print(title)
#         common.plot(X, mixture, post, title=f'k-means({k})_seed({i}).png')

# EM
# print('EM:')
# for k in range(1, 5):
#     print('For K =', k)
#     for i in range(5):
#         seed = i
#         mixture, post = common.init(X, k, seed)
#         mixture, post, cost = naive_em.run(X, mixture, post)
#         title = f'EM: k = {k}, cost for seed: {i} = {cost}'
#         print(title)
#         common.plot(X, mixture, post, title=f'EM({k})_seed({i}).png')

#  BIC
# print('EM:')
# for k in range(1, 5):
#     mixture, post = common.init(X, k)
#     mixture, post, cost = naive_em.run(X, mixture, post)
#     title = f'EM: k = {k}, cost = {cost}'
#     print(title)
#     bic = common.bic(X, mixture, cost)
#     print('BIC: ', bic)


# EM - Matrix Completion
# np.seterr(all='raise') 
print('EM - Matrix Completion:')
X = np.loadtxt("netflix_incomplete.txt")

# def em_matrix_completion(k, X):
#     print('For K =', k)
#     for i in range(5):
#         seed = i
#         mixture, post = common.init(X, k, seed)
#         mixture, post, cost = em.run(X, mixture, post)
#         title = f'EM: k = {k}, cost for seed: {i} = {cost}'
#         print(title)
#         # common.plot(X, mixture, post, title=f'EM({k})_seed({i}).png')

# em_matrix_completion(1, X)
# em_matrix_completion(12, X)


# EM - Matrix Completion
k = 12
seed = 1
mixture, post = common.init(X, k, seed)
mixture, post, cost = em.run(X, mixture, post)
title = f'EM: k = {k}, cost for seed: {1} = {cost}'
print(title)
X_pred = em.fill_matrix(X, mixture)

X_gold = np.loadtxt('netflix_complete.txt')
rmse = common.rmse(X_gold, X_pred)
print("RMSE: ", rmse)

