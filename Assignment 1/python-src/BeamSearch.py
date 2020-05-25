import StringDouble
import ExtractGraph
import heapq


class BeamSearch:
	graph = []
	
	def __init__(self, input_graph):
		self.graph = input_graph
		return
	
	def beamSearchV1(self, pre_words, beamK, maxToken):
		# Basic beam search.
		
		prev_beam = Beam(beamK)
		prefix = pre_words.split(' ')
		# math.log(1.0) = 0.0
		prefix_prob = 0.0  # <s>
		for i in range(len(prefix) - 1):
			prefix_prob += self.graph.graph[prefix[i]][prefix[i + 1]]
		
		# if the last word is the end symbol
		if prefix[-1] == '</s>':
			prev_beam.add(prefix_prob, True, prefix)
		else:
			prev_beam.add(prefix_prob, False, prefix)
		
		# Beam Search
		while True:
			curr_beam = Beam(beamK)
			
			for (prefix_prob, complete, prefix) in prev_beam:
				if complete is True:
					curr_beam.add(prefix_prob, True, prefix)
				else:
					for next_word, next_prob in self.graph.graph[prefix[-1]].items():
						if next_word == '</s>':
							curr_beam.add(prefix_prob + next_prob, True, prefix)
						else:
							curr_beam.add(prefix_prob + next_prob, False, prefix + [next_word])
			
			(best_prob, best_complete, best_prefix) = max(curr_beam)
			if best_complete is True or len(best_prefix) - 1 == maxToken:
				sentence = " ".join(best_prefix[1:])
				probability = best_prob
				return StringDouble.StringDouble(sentence, probability)
			
			prev_beam = curr_beam
	
	def beamSearchV2(self, pre_words, beamK, param_lambda, maxToken):
		# Beam search with sentence length normalization.
		
		prev_beam = Beam(beamK)
		prefix = pre_words.split(' ')
		# math.log(1.0) = 0.0
		prefix_prob = 0.0  # <s>
		for i in range(len(prefix) - 1):
			prefix_prob += self.graph.graph[prefix[i]][prefix[i + 1]]
		
		# if the last word is the end symbol
		if prefix[-1] == '</s>':
			prev_beam.add2(prefix_prob / len(prefix) ** param_lambda, prefix_prob, True, prefix)
		else:
			prev_beam.add2(prefix_prob / len(prefix) ** param_lambda, prefix_prob, False, prefix)
		
		# Beam Search, store both score and prob each time
		while True:
			curr_beam = Beam(beamK)
			
			for (prefix_score, prefix_prob, complete, prefix) in prev_beam:
				if complete is True:
					curr_beam.add2(prefix_score, prefix_prob, True, prefix)
				else:
					for next_word, next_prob in self.graph.graph[prefix[-1]].items():
						if next_word == '</s>':
							curr_beam.add2((prefix_prob + next_prob) / (len(prefix) + 1) ** param_lambda,
							               prefix_prob + next_prob, True, prefix)
						else:
							curr_beam.add2((prefix_prob + next_prob) / (len(prefix) + 1) ** param_lambda,
							               prefix_prob + next_prob, False, prefix + [next_word])
			
			(best_score, best_prob, best_complete, best_prefix) = max(curr_beam)
			if best_complete is True or len(best_prefix) - 1 == maxToken:
				sentence = " ".join(best_prefix[1:])
				probability = best_score
				return StringDouble.StringDouble(sentence, probability)
			
			prev_beam = curr_beam


class Beam:
	
	def __init__(self, beamK):
		"""
		:param beamK: width of beam
		"""
		self.heap = list()
		self.beamK = beamK
	
	def add(self, prob, complete, prefix):
		heapq.heappush(self.heap, (prob, complete, prefix))
		if len(self.heap) > self.beamK:
			heapq.heappop(self.heap)
	
	def add2(self, score, prob, complete, prefix):
		heapq.heappush(self.heap, (score, prob, complete, prefix))
		if len(self.heap) > self.beamK:
			heapq.heappop(self.heap)
	
	def __iter__(self):
		return iter(self.heap)
