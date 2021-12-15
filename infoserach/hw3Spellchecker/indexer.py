#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys
import Levenshtein as lev
from collections import defaultdict
from tqdm import tqdm


# In[2]:


def give_me_zero():
    return 0


# In[3]:


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


# In[4]:


lang_model = language_model()


# In[5]:


lang_model.fit("./queries_all.txt")


# In[6]:


import pickle


# In[7]:


with open('lang_model', 'wb') as f:
    pickle.dump(lang_model, f)


# In[39]:


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


# In[40]:


err_model = error_model()
err_model.fit("./queries_all.txt")


# In[41]:


with open('error_model', 'wb') as f:
    pickle.dump(err_model, f)


# In[11]:


import heapq
from collections import deque


# In[84]:


class Node():
    def __init__(self):
        self.nodes = {}
        self.is_term = False
        self.count = 0
        self.skipped = False
    def add(self, letter):
        if letter not in self.nodes:
            self.nodes[letter] = Node()


# In[117]:


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


def add_best_mistake_to_word(word):
    lang_change = 0
    res = word
    for i, ch in enumerate(word):
        change = word
        model_max = 0
        best_bi_change = ""
        if i == 0:
            left_bi = "^"
        else:
            left_bi = word[i - 1]
        for key, item in err_model.model[left_bi + word[i]][1].items():
            if 1 - item > model_max:
                model_max = item
                best_bi_change = key
        change = list(change)
        if len(best_bi_change) == 2:
            if i != 0:
                change[i - 1] = best_bi_change[0]
                change[i] = best_bi_change[1]
            else:
                change[i] = best_bi_change[1]
            change = "".join(change).strip()
            if len(change) >= 1:
                cur_lang_mod = lang_model.proba(change)
                if cur_lang_mod > lang_change and change != word:
                    lang_change = cur_lang_mod
                    res = change
    if res == word:
        return None
    return res


# In[23]:


def create_mistake(query):
    words = [i.strip() for i in query.split()]
    lang_proba = 0
    best_bully = None
    for i in range(len(words)):
        bully_words = words.copy()
        bully_words[i] = add_best_mistake_to_word(words[i])
        if bully_words[i] != None:
            lang_cur = lang_model.proba(" ".join(bully_words))
            if lang_cur > lang_proba:
                lang_proba = lang_cur
                best_bully = bully_words.copy()
    if best_bully == None:
        return query
    return " ".join(best_bully)


# In[16]:


from catboost import CatBoostClassifier


# In[42]:


model = CatBoostClassifier(iterations=200)


# In[191]:


y_labels = []
X = []


# In[43]:


def extract_features(cur_quer, right_quer):
    prob_cur = lang_model.proba(cur_quer)
    right_quer_prob = lang_model.proba(right_quer)
    return [prob_cur, len(cur_quer.split()), right_quer_prob, len(right_quer.split()),
            err_model.proba(cur_quer, right_quer), prob_cur / right_quer_prob]


# In[193]:


with open("./queries_all.txt", mode='r', encoding='utf-8') as file:
    num_lines = sum(1 for line in open("./queries_all.txt", mode='r', encoding='utf-8'))
    for query in tqdm(file, total = num_lines):
        line = query.split('\t')
        cur_quer = line[0]
        if len(line) == 2:
            y_labels.append(1)
            y_labels.append(0)
            change_for = line[-1]
        else:
            y_labels.append(0)
            y_labels.append(1)
            change_for = create_mistake(cur_quer)
        X.append(extract_features(cur_quer, change_for))
        X.append(extract_features(change_for, cur_quer))


# In[195]:


# with open('X_features', 'wb') as f:
#     pickle.dump(X, f)
# with open('y_labels', 'wb') as f:
#     pickle.dump(y_labels, f)


# In[18]:


# with open('X_features', 'rb') as f:
#     X = pickle.load(f)
# with open('y_labels', 'rb') as f:
#     y_labels = pickle.load(f)


# In[44]:


model.fit(X, y_labels)


# In[20]:


with open('model', 'wb') as f:
    pickle.dump(model, f)


# In[21]:


# with open('model', 'rb') as f:
#     model = pickle.load(f)


# In[118]:


# a = Trie()
# a.fit("./queries_all.txt")


# In[119]:


# iterations = 5
# for query in sys.stdin :
#     for j in range(iterations):
#         fixed_query = a.search(query)
#         if model.predict(extract_features(query, fixed_query)):
#             query = fixed_query
#     print(query)


# In[135]:


# correct = 0
# incorrect = 0


# In[136]:


# iterations = 5
# with open("./queries_all.txt", mode='r', encoding='utf-8') as file:
#     num_lines = sum(1 for line in open("./queries_all.txt", mode='r', encoding='utf-8'))
#     for query in tqdm(file, total = num_lines):
        
#         line = query.split('\t')
#         cur_quer = line[0].strip()
#         if len(line) == 1:
#             continue
#         for j in range(iterations):
#             fixed_query = a.search(cur_quer)
#             if fixed_query == None:
#                 break
#             else:
#                 cur_quer = fixed_query.strip()
#         if line[-1].strip() == cur_quer.strip():
#             correct += 1
#         else:
#             incorrect += 1
#             print("was: ", line[0])
#             print("should  have been: \n", line[-1])
#             print("but is: \n", cur_quer)
#         print(correct, incorrect, (correct / (correct + incorrect)) * 100)
#         if correct + incorrect == 100:
#             break


# In[ ]:




