import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

np.random.seed(5)

centers = [[1, 1], [-1, -1], [1, -1]]
dataset = datasets.load_digits()
X = dataset.data
y = dataset.target

X = StandardScaler().fit_transform(X)

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
# ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
pca = decomposition.PCA()
pca.fit(X)

print(pca.explained_variance_)


def kaiser(vars):
    mean_var = np.mean(vars)
    return len(list(filter(lambda x: x > mean_var, vars)))


def broken_stick(vars):
    b = [sum([1 / (j + 1) for j in range(i, len(vars))]) for i in range(len(vars))]
    k = 0
    while b[k] < vars[k]:
        k += 1

    return k


new_size = kaiser(pca.explained_variance_ratio_)
broken_size = broken_stick(pca.explained_variance_)
print('Kaiser: ', new_size)
print('Broken: ', broken_size)

plt.xlabel('Number component')
plt.ylabel('% Variance')
plt.plot(pca.explained_variance_ratio_)

plt.scatter(new_size, pca.explained_variance_ratio_[new_size])
plt.scatter(broken_size, pca.explained_variance_ratio_[broken_size], marker='^')
plt.show()

X = pca.transform(X)

colors = ['navy', 'turquoise', 'darkorange', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
plt.figure(figsize=(8, 8))
for color, i, target_name in zip(colors, list(set(dataset.target)), dataset.target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1],
                color=color, lw=2, label=target_name)

plt.legend()
# plt.show()

#
# for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
#     ax.text3D(X[y == label, 0].mean(),
#               X[y == label, 1].mean() + 1.5,
#               X[y == label, 2].mean(), name,
#               horizontalalignment='center',
#               bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
# y = np.choose(y, [1, 2, 0]).astype(np.float)
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.spectral,
#             edgecolor='k')
#
