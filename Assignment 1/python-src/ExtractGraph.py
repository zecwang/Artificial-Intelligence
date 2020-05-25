import math


class ExtractGraph:
    # key is head word; value stores next word and corresponding probability.
    graph = {}

    sentences_add = "data\\assign1_sentences.txt"

    head_word_count = {}

    def __init__(self):
        # Extract the directed weighted graph, and save to {head_word, {tail_word, probability}}
        with open(self.sentences_add, 'r') as f:
            for line in f.readlines():
                words = line.strip().split(' ')
                # print(words[:-1])  # ignore </s>
                for i in range(len(words) - 1):
                    # count each head word
                    if words[i] in self.head_word_count:
                        self.head_word_count[words[i]] += 1
                    else:
                        self.head_word_count[words[i]] = 1

                    # add to graph
                    if words[i] in self.graph:
                        if words[i + 1] in self.graph[words[i]]:
                            self.graph[words[i]][words[i + 1]] += 1
                        else:
                            self.graph[words[i]][words[i + 1]] = 1
                    else:
                        self.graph[words[i]] = {words[i + 1]: 1}
            # calculate log(prob)
            for head_word in self.graph.keys():
                for tail_word in self.graph[head_word].keys():
                    self.graph[head_word][tail_word] = math.log(
                        self.graph[head_word][tail_word] / self.head_word_count[head_word])

        return

    def getProb(self, head_word, tail_word):
        try:
            return math.e ** self.graph[head_word][tail_word]
        except KeyError:  # head_word not exists in graph
            return 0.0
