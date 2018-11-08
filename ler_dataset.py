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
			frase_original = row[4].lower()
			frase_processada = processa_frase(frase_original)
			texto_original.append(frase_original)
			dataset.append(frase_processada)

	print("%i emails lidos" % len(dataset))

	with open('emails_processados.json', 'w') as f:
		salvar = [dataset, texto_original]
		json.dump(salvar, f)
	return dataset, texto_original

def getPhrases():
	if os.path.isfile('frases.json'):
		with open('frases.json', 'r') as f:
			dados = json.load(f)
			return dados

	frases = []
	dados, texto_original = getTrainingData()
	
	for item in texto_original:
		linhas = item.split('\n')
		for linha in linhas:
			linhas_separadas = [processa_frase(x, False) for x in sent_tokenize(linha)]
			frases.extend([x for x in linhas_separadas if len(x) > 0])
	with open('frases.json', 'w') as f:
		json.dump(frases, f)
	return frases			