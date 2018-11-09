import gensim
from ler_dataset import getTrainingData

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

dados, texto_original = getTrainingData()

model = Word2Vec(dados, size=150, window=10, min_count=10, workers=8, sg=1, iter=30)
model = model.wv
model.save("modelo_word2vec")



