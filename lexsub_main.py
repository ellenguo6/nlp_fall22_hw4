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
        return None # replace for part 5

    

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)

    # PART 1
    # print(get_candidates('slow', 'a'))

    for context in read_lexsub_xml(sys.argv[1]):
        # print(context)  # useful for debugging
        # prediction = smurf_predictor(context) 

        # PART 2
        # prediction = wn_frequency_predictor(context) 

        # PART 3
        # prediction = wn_simple_lesk_predictor(context) 

        # PART 4
        prediction = predictor.predict_nearest(context)


        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
