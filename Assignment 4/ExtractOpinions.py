import stanfordnlp
import warnings
from gensim.models import KeyedVectors

warnings.filterwarnings('ignore')

word2vec_add = "data/assign4_word2vec_for_python.bin"
model = KeyedVectors.load_word2vec_format(word2vec_add, binary=True)

words_list = {'Positive': ['good', 'best', 'love', 'attentive', 'huge', 'excellent', 'nice', 'large', 'warm', 'fast'],
              'Negative': ['bad', 'sick', 'rude', 'small', 'busy', 'slow', 'disappoint', 'hard']}

uncheck_sentiments_dict = {}
uncheck_sentiments_dict['Positive'] = [el[0] for el in
                                       model.most_similar(positive=words_list['Positive'],
                                                          negative=words_list['Negative'],
                                                          topn=100)]
uncheck_sentiments_dict['Positive'].extend(words_list['Positive'])

uncheck_sentiments_dict['Negative'] = [el[0] for el in
                                       model.most_similar(positive=words_list['Negative'],
                                                          negative=words_list['Positive'],
                                                          topn=100)]
uncheck_sentiments_dict['Negative'].extend(words_list['Negative'])

# print(uncheck_sentiments_dict)

sentiments_dict = {'Positive': set(), 'Negative': set()}  # create a dictionary for sentiments analysis


class ExtractOpinions:
	# Extracted opinions and corresponding review id is saved in extracted_pairs, where KEY is the opinion and VALUE
	# is the set of review_ids where the opinion is extracted from.
	# Opinion should in form of "attribute, assessment", such as "service, good".
	extracted_opinions = {}
	nlp = stanfordnlp.Pipeline()
	
	def __init__(self):
		return
	
	def add_to_dict(self, dictionary, target, position):
		if target not in dictionary:
			dictionary[target] = [position] if type(position) != list else position
		else:
			if type(position) == list:
				dictionary[target].extend(position)
			else:
				dictionary[target].append(position)
			dictionary[target] = list(set(dictionary[target]))  # remove duplication
	
	def check(self, table, start, end, pos):
		for i in range(start, end):  # [start, end)
			el = table[i].split('\t')
			if el[3] == pos:
				return el[2]
		return ''
	
	def check_pos_between(self, table, end, target_pos, until_pos):
		for i in range(end, -1, -1):
			el = table[i].split('\t')
			if el[3] == until_pos:
				return False
			elif el[3] == target_pos:
				return True
		return False
	
	def handle_negative(self, table, start, expect_pos='VERB', until_pos='ADJ'):  # change 'not huge' to 'small'
		for i in range(start, len(table)):
			el = table[i].split('\t')
			if el[3] == until_pos:
				return True
			elif el[3] == expect_pos:
				return False
		return True
	
	def extract_pairs(self, review_id, review_content):
		# example data, which you will need to remove in your real code. Only for demo.
		doc = self.nlp(review_content)
		table = doc.conll_file.conll_as_string()
		print(table)
		sentences = table.split('\n\n')
		for sentence in sentences:
			word_table = sentence.split('\n')
			token_dict = {}
			coordinate_dict = {}
			adj_dict = {}
			noun_dict = {}
			negative_need_handle = False
			for el in word_table:
				el = el.split('\t')
				if len(el) == 1:
					continue
				# ['16', 'wonderful', 'wonderful', 'ADJ', 'JJ', 'Degree=Pos', '17', 'amod', '_', '_']
				#  el[0]     el[1]       el[2]      el[3]                     el[6]
				# print(el)
				el[0] = int(el[0])
				el[6] = int(el[6])
				token_dict[el[0]] = el[2]
				if el[3] == 'ADJ':
					if negative_need_handle:
						options = [x[0] for x in model.most_similar(el[2], topn=20)]
						positive = el[2] in uncheck_sentiments_dict['Positive']
						for opt in options:
							if positive:
								if opt in uncheck_sentiments_dict['Negative']:
									self.add_to_dict(adj_dict, opt, el[6])
									token_dict[el[0]] = opt
									break
							else:
								if opt in uncheck_sentiments_dict['Positive']:
									self.add_to_dict(adj_dict, opt, el[6])
									token_dict[el[0]] = opt
									break
					else:
						self.add_to_dict(adj_dict, el[2], el[6])
					if el[2] in uncheck_sentiments_dict['Positive']:
						sentiments_dict['Positive'].add(el[2])
					elif el[2] in uncheck_sentiments_dict['Negative']:
						sentiments_dict['Negative'].add(el[2])
				elif el[3] == 'PROPN' or el[3] == 'NOUN':
					self.add_to_dict(noun_dict, el[2], el[6])
				elif el[2] == 'not':
					negative_need_handle = self.handle_negative(word_table, el[0])
					
			for el in word_table:
				el = el.split('\t')
				if len(el) == 1:
					continue
				el[0] = int(el[0])
				el[6] = int(el[6])
				if el[3] == 'ADJ':
					"""
					For example:
					11	huge	huge	ADJ	JJ	Degree=Pos	6	acl:relcl	_	_
					12	and	and	CCONJ	CC	_	13	cc	_	_
					13	delicious	delicious	ADJ	JJ	Degree=Pos	11	conj	_	_
					"""
					if el[6] in token_dict and token_dict[el[6]] in adj_dict and \
							self.check(word_table, el[6] - 1, el[0], 'NOUN') == '':
						self.add_to_dict(coordinate_dict, el[6], el[2])
						self.add_to_dict(adj_dict, el[2], adj_dict[token_dict[el[6]]])
						self.add_to_dict(adj_dict, token_dict[el[6]], adj_dict[el[2]])
				elif el[2] == 'be':
					if el[6] in token_dict and token_dict[el[6]] in adj_dict and token_dict[el[0] - 1] in noun_dict and \
							token_dict[noun_dict[token_dict[el[0] - 1]][0]] not in noun_dict:
						self.add_to_dict(adj_dict, token_dict[el[6]], el[0] - 1)
						
			current_noun = ''
			for el in word_table:
				el = el.split('\t')
				if len(el) == 1:
					continue
				el[0] = int(el[0])
				el[6] = int(el[6])
				if el[3] == 'PROPN' or el[3] == 'NOUN':
					if current_noun == '':
						current_noun = el[2]
					else:
						current_noun += ' ' + el[2]
					
					# add to opinions
					modified = False
					for adj in adj_dict:
						if el[0] in adj_dict[adj]:
							self.add_to_dict(self.extracted_opinions, current_noun + ', ' + adj, review_id)
							modified = True
						if not self.check_pos_between(word_table, el[0] - 2, 'PUNCT', 'NOUN') and el[6] in token_dict \
								and token_dict[el[6]] in noun_dict:
							specific_el = word_table[el[6] - 1].split('\t')
							specific_el[0] = int(specific_el[0])
							specific_el[6] = int(specific_el[6])
							# if self.check(word_table, specific_el[0] - 1, el[0], 'ADP') != 'of':
							# 	break
							if specific_el[0] in adj_dict[adj]:
								if specific_el[2] not in current_noun:
									adp_word = self.check(word_table, specific_el[0] - 1, el[0], 'ADP')
									if adp_word == 'of':
										current_noun += ' ' + specific_el[2]
										self.add_to_dict(self.extracted_opinions, current_noun + ', ' + adj, review_id)
										modified = True
							# self.add_to_dict(self.extracted_opinions, current_noun + ', ' + adj, review_id)
							# modified = True
							if not modified and specific_el[6] in token_dict and token_dict[specific_el[6]] in adj_dict:
								if specific_el[2] not in current_noun:
									adp_word = self.check(word_table, specific_el[0] - 1, el[0], 'ADP')
									if adp_word == 'of':
										current_noun += ' ' + specific_el[2]
								self.add_to_dict(self.extracted_opinions,
								                 current_noun + ', ' + token_dict[specific_el[6]], review_id)
								if specific_el[6] in coordinate_dict:
									for word in coordinate_dict[specific_el[6]]:
										self.add_to_dict(self.extracted_opinions, current_noun + ', ' + word, review_id)
								modified = True
					if not modified and el[6] in token_dict and token_dict[el[6]] in adj_dict:
						self.add_to_dict(self.extracted_opinions, current_noun + ', ' + token_dict[el[6]], review_id)
						if el[6] in coordinate_dict:
							for word in coordinate_dict[el[6]]:
								self.add_to_dict(self.extracted_opinions, current_noun + ', ' + word, review_id)
				elif el[2] == '/':
					if current_noun != '':
						for adj in adj_dict:
							if el[0] - 1 in adj_dict[adj]:
								self.add_to_dict(adj_dict, adj, el[0] + 1)
							elif el[0] + 1 in adj_dict[adj]:
								self.add_to_dict(self.extracted_opinions, current_noun + ', ' + adj, review_id)
						current_noun = current_noun[:current_noun.rindex(' ')]
				else:
					current_noun = ''
