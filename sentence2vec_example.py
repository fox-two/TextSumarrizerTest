from ler_dataset import getPhrases, processa_frase, getTrainingData
from gensim.models import KeyedVectors
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AffinityPropagation, DBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

model = KeyedVectors.load('modelo_word2vec')


emails, texto_original = getTrainingData()
frases = getPhrases()

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from pprint import pprint

from sentence2vec import sentence_vectorizer

sent_vectorizer = sentence_vectorizer(frases, model)


def sumarize_email(texto):
    linhas = [x for x in texto.split('\n') if x != ""]

    vetores = [sent_vectorizer.infer_vector(x) for x in linhas]

    clusterizador = AffinityPropagation()
    c = clusterizador.fit_predict(vetores)

    clusters_calculados = set(c)
    ######################
    #exclusivo kmeans

    '''
    dists = clusterizador.transform(vetores)

    centros = []

    for cluster_atual in clusters_calculados:
        itens_cluster_atual = [i for i, x in enumerate(c) if x == cluster_atual]

        distancias_cluster_atual = dists[:, cluster_atual]

        centros.append(linhas[sorted([(x, distancias_cluster_atual[x]) for x in itens_cluster_atual], key=lambda k: k[1])[0][0]])


    
    #############
'''
    
    clusters_itens = {}
    for classe in clusters_calculados:
        clusters_itens[classe] = [x for i, x in enumerate(linhas) if c[i] == classe]

    pprint(clusters_itens)

    pca = PCA(n_components=2)
    pontos_2d = pca.fit_transform(np.array(vetores))

    #np.save('pca_frases', pontos_2d)
    plt.scatter(pontos_2d[:, 0], pontos_2d[:, 1], c=c)
    plt.show()
    return centros
  

texto_id = 91
sumarize_email(texto_original[texto_id])


def mostraSimilares(vetor_pesquisa):
    vetores = [sent_vectorizer.infer_vector(x) for x in frases]
    vetores = np.array(vetores)
    

    distancias = [(frases[i], resultado[0]) for i, resultado in enumerate(euclidean_distances(vetores, [vetor_pesquisa]))]

    distancias = sorted(distancias, key=lambda x: x[1])


    pprint(distancias[0:50])
#mostraSimilares(vetores[12])



def clusters_teste():
    vetores = [sent_vectorizer.infer_vector(x) for x in frases]
    vetores = np.array(vetores)
    
    teste_cluster = KMeans(n_clusters=200)

    c = teste_cluster.fit_predict(vetores)


    clusters_calculados = set(c)

    frases_no_cluster = {}
    for item in clusters_calculados:
        frases_no_cluster[item] =  [frases[i] for i, cluster_atual in enumerate(c) if cluster_atual == item]

    pca = PCA(n_components=2)
    pontos_2d = pca.fit_transform(np.array(vetores))

    #np.save('pca_frases', pontos_2d)
    plt.scatter(pontos_2d[:, 0], pontos_2d[:, 1], c=c)
    plt.show()

    print('fim')
