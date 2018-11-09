from ler_dataset import getPhrases, processa_frase, getTrainingData
from gensim.models import KeyedVectors
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AffinityPropagation, DBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt
import pickle
from nltk.tokenize import sent_tokenize




from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from pprint import pprint

from sentence2vec import sentence_vectorizer



def train():
    model = KeyedVectors.load('modelo_word2vec')
    phrases, original_phrases = getPhrases()
    sent_vectorizer = sentence_vectorizer(phrases, model)
    pickle.dump(sent_vectorizer, open("sent_vectorizer.bin", "wb"))




def sumarize_email(texto):
    sent_vectorizer = pickle.load( open( "sent_vectorizer.bin", "rb" ) )
    linhas = sent_tokenize(texto, language='portuguese')

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

        centros.append(sorted([(x, distancias_cluster_atual[x]) for x in itens_cluster_atual], key=lambda k: k[1])[0][0])
    centros = sorted(centros)
    centros = list(map(lambda x: linhas[x], centros))

    
    '''
    clusters_itens = {}
    for classe in clusters_calculados:
        clusters_itens[classe] = [x for i, x in enumerate(linhas) if c[i] == classe]


    resumo = ""
    
    pprint(clusters_itens)

    return resumo



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
