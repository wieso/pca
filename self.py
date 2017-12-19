import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

dataset = datasets.load_digits()
X = dataset.data

X_std = StandardScaler().fit_transform(X)

# mean_vec = np.mean(X_std, axis=0)
# cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0] - 1)
# print('Covariance matrix \n%s' % cov_mat)


cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' % eig_vecs)
print('\nEigenvalues \n%s' % eig_vals)

u, s, v = np.linalg.svd(X_std.T)

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0])
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

tot = sum(eig_vals)
var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# print(var_exp)
# print(cum_var_exp)

matrix_w = np.hstack((eig_pairs[0][1].reshape(X.shape[1], 1),
                      eig_pairs[1][1].reshape(X.shape[1], 1),
                      eig_pairs[2][1].reshape(X.shape[1], 1),
                      eig_pairs[3][1].reshape(X.shape[1], 1),
                      ))

print('Matrix W:\n', matrix_w)

Y = X_std.dot(matrix_w)

print(Y)
