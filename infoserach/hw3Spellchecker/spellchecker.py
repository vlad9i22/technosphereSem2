#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import sys
import pickle
import Levenshtein as lev
from tqdm import tqdm
from collections import defaultdict
import heapq
from collections import deque
from catboost import CatBoostClassifier


# In[7]:


def extract_features(cur_quer, right_quer):
    prob_cur = lang_model.proba(cur_quer)
    right_quer_prob = lang_model.proba(right_quer)
    return [prob_cur, len(cur_quer.split()), right_quer_prob, len(right_quer.split()),
            err_model.proba(cur_quer, right_quer), prob_cur / right_quer_prob]


# In[8]:


def give_me_zero():
    return 0


# In[4]:


class language_model:
    def __init__(self):
        self.default_prob = 0
        self.all_words = 0
        self.count_pairs = 0
        self.count_words = {}
        self.count_pred_word = defaultdict(give_me_zero)
    def fit(self, path):
        with open(path, mode='r', encoding='utf-8') as file:
            for query in file:
                line = query.split('\t')
                line = line[-1]
                line = line.lower()
                words = line.split()
                first_word = words[0]
                self.all_words += len(words)
                for word in words[1:]:
                    self.count_pairs += 1
                    self.count_pred_word[word] += 1
                    if first_word not in self.count_words:
                        self.count_words[first_word] = [{}, 0]
                    self.count_words[first_word][1] += 1
                    if word not in self.count_words[first_word][0]:
                        self.count_words[first_word][0][word] = 0
                    self.count_words[first_word][0][word] += 1
                    first_word = word
            self.default_prob = 1 / self.all_words
    
    
    def proba(self, query):
        query = query.lower()
        words = query.split()
        prob = 1
        first_word = words[0]
        if first_word in self.count_words and len(words) == 1:
            prob *= self.count_words[first_word][1] / self.all_words
        elif len(words) == 1:
            prob = self.default_prob
        flag = 0
        for word in words[1:]:
            flag = 1
            if first_word in self.count_words and word in self.count_words[first_word][0]:
                count_left = self.count_words[first_word][0][word]
                count_right = self.count_pred_word[word]
            else:
                count_left = 1
                count_right = self.count_pairs
            prob *= count_left / count_right
            first_word = word
        if flag:
            if self.count_pred_word[word] == 0:
                self.count_pred_word[word] += 1
            prob *= self.count_pred_word[word] / self.all_words
        if prob == 0:
            prob = 1 / self.all_words
            prob *= 1e-30
        return prob


# In[5]:


from catboost import CatBoostClassifier


# In[6]:


def gen_zero():
    return 0
def gen_for_dict():
    return [0, defaultdict(gen_zero)]
class error_model:
    def __init__(self, alpha = 1.2):
        self.all_bigrams = 0
        self.model = defaultdict(gen_for_dict)
        self.alpha = alpha
    def fit(self, path):
        with open(path, mode='r', encoding='utf-8') as file:
            num_lines = sum(1 for line in open(path, mode='r', encoding='utf-8'))
            for query in tqdm(file, total = num_lines):
                line = query.split('\t')
                if len(line) != 2:
                    continue
                correct_words = line[1].strip().split()
                wrong_words = line[0].strip().split()
                if len(correct_words) != len(wrong_words):
                    continue
                flag = 0
                for i in range(len(correct_words)):
                    orig = wrong_words[i].lower()
                    fix = correct_words[i].lower()
                    fix_len = len(fix)
                    cur_pos = 0
                    for operation, spos, dpos in lev.editops(orig, fix):
                        if operation == "insert":
                            if spos == 0:
                                cur_orig_bigram = "^" + orig[spos]
                                cur_fix_bigram = fix[dpos] + orig[spos]
                            else:
                                cur_orig_bigram = orig[spos - 1] + '_'
                                cur_fix_bigram = orig[spos - 1] + fix[dpos]
                        elif operation == "delete":
                            if spos == 0:
                                cur_orig_bigram = '^' + orig[spos]
                                cur_fix_bigram = '^_'
                            else:
                                cur_orig_bigram = orig[spos - 1] + orig[spos]
                                cur_fix_bigram = orig[spos - 1] + '_'
                        elif operation == "replace":
                            if spos == 0:
                                cur_orig_bigram = '^' + orig[spos]
                                cur_fix_bigram = '^' + fix[dpos]
                            else:
                                cur_orig_bigram = orig[spos - 1] + orig[spos]
                                cur_fix_bigram = orig[spos - 1] + fix[dpos]
                        self.model[cur_orig_bigram][0] += 1
                        self.model[cur_orig_bigram][1][cur_fix_bigram] += 1
                        self.all_bigrams += 1
                
            self.default_p = 1 / self.all_bigrams
            for orig_bigrams in self.model:
                cur_sum = 0
                for fixed_bigrams in self.model[orig_bigrams][1]:
                    cur_sum += self.model[orig_bigrams][1][fixed_bigrams]
                for fixed_bigrams in self.model[orig_bigrams][1]:
                    self.model[orig_bigrams][1][fixed_bigrams] /= cur_sum           
                    
    def proba(self, orig, fix):
        orig = orig.lower()
        fix = fix.lower()
        prob = 1
        for operation, spos, dpos in lev.editops(orig, fix):
            if operation == "insert":
                if spos == 0:
                    cur_orig_bigram = "^" + orig[spos]
                    cur_fix_bigram = fix[dpos] + orig[spos]
                else:
                    cur_orig_bigram = orig[spos - 1] + '_'
                    cur_fix_bigram = orig[spos - 1] + fix[dpos]
            elif operation == "delete":
                if spos == 0:
                    cur_orig_bigram = '^' + orig[spos]
                    cur_fix_bigram = '^_'
                else:
                    cur_orig_bigram = orig[spos - 1] + orig[spos]
                    cur_fix_bigram = orig[spos - 1] + '_'
            elif operation == "replace":
                if spos == 0:
                    cur_orig_bigram = '^' + orig[spos]
                    cur_fix_bigram = '^' + fix[dpos]
                else:
                    cur_orig_bigram = orig[spos - 1] + orig[spos]
                    cur_fix_bigram = orig[spos - 1] + fix[dpos]
            tmp = self.model[cur_orig_bigram][1][cur_fix_bigram]
            if tmp == 0:
                self.model[cur_orig_bigram][1][cur_fix_bigram] = self.default_p
                tmp = self.default_p
            prob *= tmp
        return prob


# In[7]:


with open('lang_model', 'rb') as f:
    lang_model = pickle.load(f)
with open('error_model', 'rb') as f:
    err_model = pickle.load(f)


# In[8]:


import heapq
from collections import deque
class Node():
    def __init__(self):
        self.nodes = {}
        self.is_term = False
        self.count = 0
        self.skipped = False
    def add(self, letter):
        if letter not in self.nodes:
            self.nodes[letter] = Node()
            
class Trie:
    def __init__(self, alpha = 0.015):
        self.head = Node()
        self.alpha = alpha
                
    def add_word(self, word):
        cur = self.head
        for i in range(len(word)):
            letter = word[i]
            if letter not in cur.nodes:
                cur.nodes[letter] = Node()
            if cur == self.head:
                cur.count += 1
            cur = cur.nodes[letter]
            cur.count += 1
            if i == len(word) - 1:
                cur.is_term = True
    
    
    def fit(self, path):
        with open(path, mode='r', encoding='utf-8') as file:
            num_lines = sum(1 for line in open(path, mode='r', encoding='utf-8'))
            for query in tqdm(file, total = num_lines):
                line = query.split('\t')
                line = line[-1]
                line = line.lower()
                words = line.split()
                for word in words:
                    self.add_word(word)
                
    def find(self, word):
        cur = self.head
        for i in range(len(word)):
            letter = word[i]
            if letter not in cur.nodes:
                return False
            else:
                cur = cur.nodes[letter]
            if i == len(word) - 1:
                return cur.is_term
            
    
    def loss(self, freq, probab):
        return self.alpha * np.log2(freq) + np.log2(probab)
    
    def suggest_for_one_word(self, word):
        queue = deque()
        prior_queue = []
        queue.append((self.head, ""))
        while len(queue) != 0:
            cur = queue.popleft()
            prefix_freq = cur[0].count
            cur_prior_queue = []
            for letter in cur[0].nodes:
                fix = cur[1] + letter
                loss = self.loss(prefix_freq, err_model.proba(word, fix))
                heapq.heappush(cur_prior_queue, (loss, fix, cur[0].nodes[letter]))
                if cur[0].nodes[letter].is_term and fix != word and abs(len(fix) - len(word)) < 5:
                    heapq.heappush(prior_queue, (loss, fix))
            #skip_letter part
#             if not cur[0].skipped:
#                 fix = cur[1]
#                 loss = self.loss(prefix_freq, err_model.proba(word, cur[1] + "_"))
#                 cur[0].skipped = True
#                 heapq.heappush(cur_prior_queue, (loss, fix, cur[0]))

            for i in heapq.nlargest(15, cur_prior_queue):
                if len(i[1]) < 24 and err_model.proba(word, i[1]) > 0.00000001:
                    queue.append((i[2], i[1]))
            while len(prior_queue) > 5:
                heapq.heappop(prior_queue)
        res = []
        for i in heapq.nlargest(5, prior_queue):
            res.append(i[1])
        return res
    
    
    def search(self, quary):
        words = quary.split()
        for i, word in enumerate(words):
            best_bulling = []
            for sug_word in self.suggest_for_one_word(word):
                bulling_words = words.copy()
                bulling_words[i] = sug_word
                fixed_quary = " ".join(bulling_words)
                features = extract_features(quary, fixed_quary)
                predicted = model.predict_proba(features)[1]
                if model.predict(features) and predicted > 0.8:
                    best_bulling.append((predicted, fixed_quary))
            if len(best_bulling) > 0:
                return sorted(best_bulling)[-1][1]
        return None


# In[ ]:


with open('model', 'rb') as f:
    model = pickle.load(f)


# In[9]:


a = Trie()
a.fit("./queries_all.txt")


# In[11]:


import sys


# In[12]:


iterations = 5
for query in sys.stdin :
    for j in range(iterations):
        fixed_query = a.search(query)
        if fixed_query == None:
            break
        else:
            cur_quer = fixed_query.strip()
            query = fixed_query.strip()
    print(query.strip())


# In[ ]:




