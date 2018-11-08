from gensim.models import KeyedVectors
model = KeyedVectors.load('modelo_word2vec')


vetor_pesquisa = model['ferrugem']

from sklearn.metrics.pairwise import cosine_similarity

distancias = [(model.index2word[i], resultado[0]) for i, resultado in enumerate(cosine_similarity(model.vectors, [vetor_pesquisa]))]

distancias = sorted(distancias, key=lambda x: x[1], reverse=True)

from pprint import pprint


pprint(distancias[0:20])


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AffinityPropagation, DBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


cores = AffinityPropagation(convergence_iter=50, damping=0.5)

c  = cores.fit_predict(model.vectors)

clusters_calculados = set(c)

palavras_no_cluster = {}
for item in clusters_calculados:
    palavras_no_cluster[item] =  [model.index2word[i] for i, cluster_atual in enumerate(c) if cluster_atual == item]
    

pca = PCA(n_components=2)
pontos_2d = pca.fit_transform(model.vectors)
plt.scatter(pontos_2d[:, 0], pontos_2d[:, 1], c=c)
plt.show()
