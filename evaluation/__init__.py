from .bleu import Bleu
from .meteor import Meteor
from .rouge import Rouge
from .cider import Cider 
from .tokenizer import PTBTokenizer 
from .distinct import distinct_n_corpus_level
import numpy as np 

def flatten(x): 
    seq_list = []
    for k, v in x.items(): 
        for vv in v: 
            seq_list.append(vv.split()) 
    return seq_list

def set_combine(x): 
    s = set()
    for xx in x: 
        s = set.union(s, set(xx)) 
    # print(s)
    return len(s)

def compute_scores(gts, gen):
    # metrics = (Bleu(), Meteor(), Rouge(), Cider())
    metrics = (Bleu(), Cider())
    all_score = {}
    all_scores = {}
    for metric in metrics:
        score, scores = metric.compute_score(gts, gen)
        all_score[str(metric)] = score
        all_scores[str(metric)] = scores
    #gen_list = [v[0] for k, v in gen.items()] 
    #gts_list = [v for k, v in gts.items()] 
    gen_list = flatten(gen) 
    gts_list = flatten(gts) 
    all_score['distinct'] = [distinct_n_corpus_level(gen_list, n=1), distinct_n_corpus_level(gen_list, n=2)]
    all_scores['distinct'] = [distinct_n_corpus_level(gts_list, n=1), distinct_n_corpus_level(gts_list, n=2)]
    
    all_score['vocab'] = set_combine(gen_list)
    all_scores['vocab'] = set_combine(gts_list)

    return all_score, all_scores

