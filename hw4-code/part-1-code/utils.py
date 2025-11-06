import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    text = example.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return example

    # typos and synonym
    p_typo = 0.2
    p_syn  = 0.2

    qwerty_neighbors_vowel = {
        'a': 'qwsz',   
        'e': 'wsdr',
        'i': 'ujko',
        'o': 'iklp',
        'u': 'yhji'
    }

    tokens = word_tokenize(text)
    detok = TreebankWordDetokenizer()

    def add_simple_typo(w: str):
        idxs = [i for i, ch in enumerate(w) if ch.lower() in qwerty_neighbors_vowel]
        if not idxs:
            return None
        i = random.choice(idxs)
        ch = w[i]
        pool = qwerty_neighbors_vowel[ch.lower()]
        rep = random.choice(pool)
        if ch.isupper(): rep = rep.upper()
        return w[:i] + rep + w[i+1:]

    def synonym_replace(w: str):
        if not w.isalpha():
            return None
        base = w.lower()
        syns = wordnet.synsets(base)
        if not syns:
            return None
        cands = []
        for s in syns:
            for l in s.lemmas():
                cand = l.name().replace('_', ' ')
                if cand.isalpha() and cand.lower() != base:
                    cands.append(cand)
        if not cands:
            return None
        rep = random.choice(cands)
        if w[0].isupper():
            rep = rep.capitalize()
        return rep

    out = []
    for w in tokens:
        changed = False
        # random typo
        if not changed and random.random() < p_typo:
            tw = add_simple_typo(w)
            if tw:
                out.append(tw)
                changed = True
        # random syns
        if not changed and random.random() < p_syn: #we only want to apply change if we did not apply typo
            sw = synonym_replace(w)
            if sw:
                out.append(sw)
                changed = True
        if not changed:
            out.append(w)

    example["text"] = detok.detokenize(out)
    return example

