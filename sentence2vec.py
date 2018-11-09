#!/usr/bin/python3

#
#  Copyright 2016-2018 Peter de Vocht
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import normalize
import gensim

def processa_frase(frase):
	palavras = gensim.utils.simple_preprocess(str(frase), deacc=True)
	return [x for x in palavras]

# A SIMPLE BUT TOUGH TO BEAT BASELINE FOR SENTENCE EMBEDDINGS
# Sanjeev Arora, Yingyu Liang, Tengyu Ma
# Princeton University
# convert a list of sentence with word2vec items into a set of sentence vectors
# cada frase é formada por uma lista de inteiros que identifica cada palavra
class sentence_vectorizer:
    def __init__(self, sentence_list, model, a: float=1e-3):
        word_vectors = model.vectors
        word_freq    = [model.vocab[x].count for x in model.vocab]
        self.model = model
        self.embedding_size = len(word_vectors[0])
        self.a = a
        self.word_vectors = word_vectors
        self.word_frequency = word_freq

        sentence_set = [self._preprocess_sentence(x) for x in sentence_list]

        # calculate PCA of this sentence set
        pca = IncrementalPCA(n_components=self.embedding_size)
        pca.fit(np.array(sentence_set))
        self.u = pca.components_[0]  # the PCA vector
        self.u = np.multiply(self.u, np.transpose(self.u))  # u x uT
    
    def converter_pra_int(self, frase):
        palavras = processa_frase(frase)

        resultado = []
        for palavra in palavras:       
            if palavra in self.model:
                resultado.append(self.model.vocab[palavra].index)
        return resultado

    def infer_vector(self, sentence):
        vs = self._preprocess_sentence(sentence)
        sub = np.multiply(self.u,vs)
        return (np.subtract(vs, sub))


    def _preprocess_sentence(self, sentence):
        sentence = self.converter_pra_int(sentence)
        vs = np.zeros(self.embedding_size)  # add all word2vec values into one vector for the sentence
        
        for word_id in sentence:
            a_value = self.a / (self.a + self.word_frequency[word_id])  # smooth inverse frequency, SIF
            vs += (self.word_vectors[word_id] * self.a) #np.add(vs, np.multiply(a_value, word.vector))  # vs += sif * word_vector

        if len(sentence) > 0:
            vs *= 1/len(sentence)  # weighted average
        return vs  # add to our existing re-calculated set of sentences
