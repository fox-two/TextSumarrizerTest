import os.path
import json
import csv
from nltk.stem.snowball import SnowballStemmer
import gensim
from gensim.utils import simple_preprocess
from nltk.tokenize import sent_tokenize

stemmer = SnowballStemmer("portuguese")

def processa_frase(frase, stem=True):
	palavras = gensim.utils.simple_preprocess(str(frase), deacc=True)
	
	if stem:
		return [stemmer.stem(x) for x in palavras]
	else:
		return [x for x in palavras]

def getTrainingData():
	dataset = []
	texto_original = []

	
	print('Lendo emails...')
	with open('articles.csv', encoding='utf-8') as csvfile:
		table = csv.reader(csvfile, delimiter=',', quotechar='"')
		for i, row in enumerate(table):
			if i == 0:
				continue		
			print("Email %d" % i)
			frase_original = row[1].lower()
			frase_processada = processa_frase(frase_original, False)

			#texto_original.append(frase_original)
			dataset.append(frase_processada)

	print("%i emails lidos" % len(dataset))

	return dataset, texto_original
