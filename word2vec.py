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
from ler_dataset import getTrainingData

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

dados, texto_original = getTrainingData()

model = Word2Vec(dados, size=150, window=10, min_count=10, workers=8, sg=1, iter=30)
model = model.wv
model.save("modelo_word2vec")



