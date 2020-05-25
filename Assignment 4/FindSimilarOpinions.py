import gensim.models.keyedvectors as word2vec
from ExtractOpinions import sentiments_dict
import numpy as np
from numpy.linalg import norm


class FindSimilarOpinions:
	extracted_opinions = {}
	word2VecObject = []
	cosine_sim = 0
	noun_sim_threshold = 0.27
	adj_sim_threshold = 0.288
	
	def __init__(self, input_cosine_sim, input_extracted_ops):
		self.cosine_sim = input_cosine_sim
		self.extracted_opinions = input_extracted_ops
		word2vec_add = "data//assign4_word2vec_for_python.bin"
		self.word2VecObject = word2vec.KeyedVectors.load_word2vec_format(word2vec_add, binary=True)
		# print(sentiments_dict)
		return
	
	def get_word_sim(self, word_1, word_2):
		try:
			return self.word2VecObject.similarity(word_1, word_2)
		except KeyError:
			return 0
	
	def phrase2vec(self, phrase):
		words = phrase.split(' ')
		m = []
		for w in words:
			try:
				m.append(self.word2VecObject.get_vector(w))
			except KeyError:
				continue
		
		m = np.array(m)
		vector = m.sum(axis=0)
		if type(vector) != np.ndarray:
			return np.zeros(300)
		return vector / np.sqrt((vector ** 2).sum())
	
	def get_noun_sim(self, noun1, noun2):
		noun1 = self.phrase2vec(noun1)
		noun2 = self.phrase2vec(noun2)
		return np.inner(noun1, noun2) / (norm(noun1) * norm(noun2))
	
	def get_adj_sim(self, adj1, adj2):
		if adj1 not in sentiments_dict['Positive'] and adj1 not in sentiments_dict['Negative']:
			return self.get_word_sim(adj1, adj2)
		elif adj2 not in sentiments_dict['Positive'] and adj2 not in sentiments_dict['Negative']:
			return self.get_word_sim(adj1, adj2)
		elif (adj1 in sentiments_dict['Positive']) == (adj2 in sentiments_dict['Positive']):
			return self.get_word_sim(adj1, adj2)
		else:
			return 0
	
	def findSimilarOpinions(self, query_opinion):
		# example data, which you will need to remove in your real code. Only for demo.
		similar_opinions = {}
		query = query_opinion.split(', ')
		noun1 = query[0]
		adj1 = query[1]
		for opinion in self.extracted_opinions.keys():
			op = opinion.split(', ')
			noun2 = op[0]
			adj2 = op[1]
			if self.get_noun_sim(noun1, noun2) > self.noun_sim_threshold and \
					self.get_adj_sim(adj1, adj2) > self.adj_sim_threshold:
				similar_opinions[opinion] = self.extracted_opinions[opinion]
		
		return similar_opinions
