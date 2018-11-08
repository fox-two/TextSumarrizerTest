import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
def plotaPontos(pontos):
    pca = PCA(n_components=2)
    pontos_2d = pca.fit_transform(pontos)
    plt.scatter(pontos_2d[:, 0], pontos_2d[:, 1])
    plt.show()


