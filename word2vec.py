import csv
import json
import os.path
import gensim
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.corpora import Dictionary
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.snowball import SnowballStemmer
import gensim
from gensim.utils import simple_preprocess
import spacy
from nltk.tokenize import sent_tokenize


stemmer = SnowballStemmer("portuguese")
nlp = None#spacy.load('pt', disable=['parser', 'tagger', 'ner', 'textcat', 'tokenizer'])

def processa_frase(frase):
	palavras = gensim.utils.simple_preprocess(str(frase), deacc=True)
	return [x for x in palavras]

def getTrainingData():
		dataset = []
		texto_original = []

		if os.path.isfile('emails_processados.json'):
				with open('emails_processados.json', 'r') as f:
						dados = json.load(f)
						return dados[0], dados[1]

		
		print('Lendo emails...')
		with open('dados_email.csv', encoding='utf-8') as csvfile:
				table = csv.reader(csvfile, delimiter=',', quotechar='"')
				for i, row in enumerate(table):
						if i == 0:
								continue
						print("Email %d" % i)
						
						linhas = row[4].lower()
						frase_atual = processa_frase(linhas)

						dataset.append(frase_atual)
						texto_original.append(linhas)


		print("%i emails lidos" % len(dataset))

		with open('emails_processados.json', 'w') as f:
				salvar = [dataset, texto_original]
				json.dump(salvar, f)
		return dataset, texto_original

dados, texto_original = getTrainingData()

model = Word2Vec(dados, size=35, window=3, min_count=10, workers=4, sg=1, iter=300)
model = model.wv
model.save("modelo_word2vec")



vetor_pesquisa = model['motorista']

from sklearn.metrics.pairwise import cosine_similarity

distancias = [(model.index2word[i], resultado[0]) for i, resultado in enumerate(cosine_similarity(model.vectors, [vetor_pesquisa]))]

distancias = sorted(distancias, key=lambda x: x[1], reverse=True)

from pprint import pprint


pprint(distancias[0:20])

print("fim")
#from gensim.models import KeyedVectors
#model = KeyedVectors.load_word2vec_format('glove_s50.txt')


