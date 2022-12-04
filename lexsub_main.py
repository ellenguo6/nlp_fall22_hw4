#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import gensim
import transformers 

from typing import List
import string

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    lemma = lemma.lower()
    candidates: set[str] = {lemma}
    for synset in wn.synsets(lemma, pos=pos):
        for sys_lemma in synset.lemmas():
            candidate = sys_lemma.name().lower().replace("_", " ")
            candidates.add(candidate)
    candidates.remove(lemma)
    return list(candidates)

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    # part 2
    lemma = context.lemma.lower()
    max_count = 0
    max_candidate = None
    for synset in wn.synsets(lemma, pos=context.pos):
        for sys_lemma in synset.lemmas():
            candidate = sys_lemma.name().lower().replace("_", " ")
            if candidate == lemma:
                continue
            new_count = sys_lemma.count()
            if new_count >= max_count:
                max_count = new_count
                max_candidate = candidate
    return max_candidate

def wn_simple_lesk_predictor(context : Context) -> str:
    # part 3
    def get_tokens_of_synset(s) -> List[str]:
        tokens: list[str] = tokenize(s.definition())
        stop_words = stopwords.words('english')
        for example in s.examples():
            tokens += tokenize(example)
        filt_tokens = [word.lower() for word in tokens if word not in stop_words]
        return filt_tokens

    lemma = context.lemma.lower()
    max_score = 0
    best_candidate = None
    for synset in wn.synsets(lemma, pos=context.pos):
        full_context = context.left_context + context.right_context
        filt_context = set([word.lower() for word in full_context])

        definition_tokens = get_tokens_of_synset(synset)
        for hypernym in synset.hypernyms():
            definition_tokens += get_tokens_of_synset(hypernym)
        definition_tokens = set(definition_tokens)

        # do weighted tiebreak as described in Ed #536
        a = len(filt_context.intersection(definition_tokens))   # overlap 

        b = 0           # The frequency of <target,synset>  
        for sys_lemma in synset.lemmas():
            candidate = sys_lemma.name().lower().replace("_", " ")
            if candidate == lemma:
                b = sys_lemma.count()
                break
        
        for sys_lemma in synset.lemmas():
            candidate = sys_lemma.name().lower().replace("_", " ")
            if candidate == lemma:
                continue
            c = sys_lemma.count()       # The frequency of <lemma,synset>  
            score = 1000 * a + 100 * b + c
            if score > max_score:
                max_score = score
                best_candidate = candidate
    
    return best_candidate
   

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        lemma = context.lemma.lower()
        candidates = get_candidates(lemma, context.pos)

        max_similarity = 0
        best_candidate = None
        for candidate in candidates:
            try:
                sim = self.model.similarity(candidate, lemma)
            except KeyError:
                continue
            if sim > max_similarity:
                max_similarity = sim
                best_candidate = candidate

        return best_candidate if best_candidate else "smurf"


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        # part 5
        candidates = set(get_candidates(context.lemma, context.pos))
        input = " ".join(context.left_context) + " [MASK] " + " ".join(context.right_context)
        
        input_toks = self.tokenizer.encode(input)
        idx = self.tokenizer.convert_ids_to_tokens(input_toks).index("[MASK]")
        input_mat = np.array(input_toks).reshape((1,-1))
        outputs = self.model.predict(input_mat, verbose=False)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][idx])[::-1]
        tokens = self.tokenizer.convert_ids_to_tokens(best_words)
        for token in tokens:
            if token in candidates:
                return token

        return "smurf"

class BetterBertPredictor(object):

    def __init__(self, filename): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert_model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
        self.word_2_vec_model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)  
        print("finished initializing BetterBertPredictor")

    def predict(self, context : Context, alpha: float) -> str:
        # part 6 attempt 1
        # since BERT is good at finding words that fit in the sentence but might not mean the same,
        # also incorporate word2vec scoring
        def normalize(arr, min_val, max_val):
            output = []
            diff = max_val - min_val
            arr_min = min(arr)
            arr_diff = max(arr) - arr_min
            for e in arr:
                val = (((e - arr_min) * diff) / arr_diff) + min_val
                output.append(val)
            return output
            
        input = " ".join(context.left_context) + " [MASK] " + " ".join(context.right_context)
        
        input_toks = self.tokenizer.encode(input)
        idx = self.tokenizer.convert_ids_to_tokens(input_toks).index("[MASK]")
        input_mat = np.array(input_toks).reshape((1,-1))
        outputs = self.bert_model.predict(input_mat, verbose=False)
        predictions = outputs[0][0][idx]
        tokens = self.tokenizer.convert_ids_to_tokens([i for i in range(0, len(predictions))])

        max_score = 0
        best_prediction = None
        l = len(predictions)
        predictions = normalize(predictions, 0, 1)
        for i, bert_prediction in enumerate(predictions):
            tok = tokens[i]
            if tok == context.lemma:
                continue
            try:
                sim = self.word_2_vec_model.similarity(tok, context.lemma)
            except KeyError:
                continue

            score = alpha * bert_prediction + (1-alpha) * sim
            if score > max_score:
                max_score = score
                best_prediction = tok

        return best_prediction if best_prediction else "smurf"

class BestBertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        # part 6 attempt 2
        candidates = get_candidates(context.lemma, context.pos)
        new_candidates = []
        for candidate in candidates:
            new_candidates += get_candidates(candidate, context.pos)
        candidates = set(new_candidates + candidates)

        input = " ".join(context.left_context) + " [MASK] " + " ".join(context.right_context)
        input_toks = self.tokenizer.encode(input)
        idx = self.tokenizer.convert_ids_to_tokens(input_toks).index("[MASK]")
        input_mat = np.array(input_toks).reshape((1,-1))
        outputs = self.model.predict(input_mat, verbose=False)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][idx])[::-1]
        tokens = self.tokenizer.convert_ids_to_tokens(best_words)
        for token in tokens:
            if token in candidates:
                return token

        return "smurf"
    

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    # predictor = Word2VecSubst(W2VMODEL_FILENAME)

    # bert_predictor = BertPredictor()
    better_bert_predictor = BetterBertPredictor(W2VMODEL_FILENAME)
    # best_bert_predictor = BestBertPredictor()

    # PART 1
    # print(get_candidates('slow', 'a'))

    for context in read_lexsub_xml(sys.argv[1]):

        # print(context)  # useful for debugging
        # prediction = smurf_predictor(context) 

        # PART 2
        # Total = 298, attempted = 298
        # precision = 0.097, recall = 0.097
        # Total with mode 206 attempted 206
        # precision = 0.131, recall = 0.131
        # prediction = wn_frequency_predictor(context) 

        # PART 3
        # Total = 298, attempted = 298
        # precision = 0.114, recall = 0.114
        # Total with mode 206 attempted 206
        # precision = 0.160, recall = 0.160
        # prediction = wn_simple_lesk_predictor(context) 

        # PART 4
        # Total = 298, attempted = 298
        # precision = 0.115, recall = 0.115
        # Total with mode 206 attempted 206
        # precision = 0.170, recall = 0.170
        # prediction = predictor.predict_nearest(context)

        # PART 5
        # Total = 298, attempted = 298
        # precision = 0.115, recall = 0.115
        # Total with mode 206 attempted 206
        # precision = 0.170, recall = 0.170
        # prediction = bert_predictor.predict(context)

        # PART 6 attempt 1
        # tried a couple alphas, found that 0.6 was the best
        # (.predict files reflect which alpha was used, 
        # so 0.2.predict is for alpha = 0.2)
        # though still didn't outperform Part 5
        # ---SCORING FOR 0.0---
        # Total = 298, attempted = 298
        # precision = 0.075, recall = 0.075
        # Total with mode 206 attempted 206
        # precision = 0.121, recall = 0.121
        # ---SCORING FOR 0.2---
        # Total = 298, attempted = 298
        # precision = 0.085, recall = 0.085
        # Total with mode 206 attempted 206
        # precision = 0.136, recall = 0.136
        # ---SCORING FOR 0.4---
        # Total = 298, attempted = 298
        # precision = 0.089, recall = 0.089
        # Total with mode 206 attempted 206
        # precision = 0.136, recall = 0.136
        # ---SCORING FOR 0.6---
        # Total = 298, attempted = 298
        # precision = 0.091, recall = 0.091
        # Total with mode 206 attempted 206
        # precision = 0.160, recall = 0.160
        # ---SCORING FOR 0.8---
        # Total = 298, attempted = 298
        # precision = 0.060, recall = 0.060
        # Total with mode 206 attempted 206
        # precision = 0.102, recall = 0.102
        # ---SCORING FOR 1.0---
        # Total = 298, attempted = 298
        # precision = 0.035, recall = 0.035
        # Total with mode 206 attempted 206
        # precision = 0.058, recall = 0.058
        
        prediction = better_bert_predictor.predict(context, alpha=0.6)

        # PART 6 attempt 2 -- lol even worse
        # Total = 298, attempted = 298
        # precision = 0.042, recall = 0.042
        # Total with mode 206 attempted 206
        # precision = 0.063, recall = 0.063
        # prediction = best_bert_predictor.predict(context)

        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
