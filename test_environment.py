import os
import sys
import unittest
from unittest import TestCase
import pandas as pd
import pytest
import re
import self
from nltk.corpus import stopwords
from tqdm import tqdm
import numpy as np
import sklearn.manifold
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from gensim.test.utils import datapath
from smart_open import open
from gensim.models import Word2Vec, KeyedVectors
import pandas as pd
from scipy.stats import spearmanr

stopWords = stopwords.words('english')

def clean_data(sentence):
    # Clean the data from all punctuation and remove all the stopwords.
    sentence = re.sub(r'[^A-Za-z0-9\s.]', r'', str(sentence).lower())
    sentence = re.sub(r'\n', r' ', sentence)
    sentence = " ".join([word for word in sentence.split() if word not in stopWords])
    return sentence

class TryTesting(unittest.TestCase):
    def test_always_passes(self):
        self.assertTrue(True)

    #     # Load pre-trained Word2Vec model (or train your own)
    #     model_path = "path/to/your/word2vec/model.bin"  # Use Google's or custom Word2Vec
    #     model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    #
    #     # Retrieve synonyms for a given word
    #     def get_synonyms(word, top_n=10):
    #         try:
    #             return model.most_similar(word, topn=top_n)
    #         except KeyError:
    #             return []
    #
    #     # Load SimLex-999 dataset
    #     simlex_path = "path/to/SimLex-999.txt"  # Ensure it's the correct path
    #     simlex = pd.read_csv(simlex_path, sep="\t")
    #
    # def evaluate_using_simplex
    #     # Load SimLex-999 dataset
    #     simlex = pd.read_csv('C:\\Users\\USER-PC\\Projects\\msc-2024-kwanele\\Simlex-999.txt', sep='\t')
    #
    #     # Load pre-trained word embeddings (e.g., Word2Vec)
    #     model = KeyedVectors.load_word2vec_format('path_to_embeddings.bin', binary=True)
    #
    #     # Function to compute cosine similarity
    #     def cosine_similarity(vec1, vec2):
    #         return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    #
    #     # Calculate cosine similarities for word pairs
    #     similarities = []
    #     valid_pairs = 0
    #
    #     for _, row in simlex.iterrows():
    #         word1, word2, simlex_score = row['word1'], row['word2'], row['SimLex999']
    #
    #         if word1 in model and word2 in model:
    #             vec1, vec2 = model[word1], model[word2]
    #             cosine_sim = cosine_similarity(vec1, vec2)
    #             similarities.append((simlex_score, cosine_sim))
    #             valid_pairs += 1
    #
    #     # Extract scores
    #     human_scores, model_scores = zip(*similarities)
    #
    #     # Compute Spearman's correlation
    #     spearman_corr, _ = spearmanr(human_scores, model_scores)
    #
    #     print(f"Evaluated on {valid_pairs}/{len(simlex)} word pairs.")
    #     print(f"Spearman's correlation: {spearman_corr:.4f}")


    # # Evaluate Word2Vec on SimLex-999
    # def evaluate_simlex(model, simlex_df):
    #     actual_scores = []
    #     predicted_scores = []
    #
    #     for _, row in simlex_df.iterrows():
    #         word1, word2, human_score = row['word1'], row['word2'], row['SimLex999']
    #         if word1 in model and word2 in model:
    #             predicted_scores.append(model.similarity(word1, word2))
    #             actual_scores.append(human_score)
    #
    #     correlation, _ = spearmanr(actual_scores, predicted_scores)
    #     return correlation
    #
    # # Example usage
    # word = "car"
    # print(f"Synonyms for {word}: {get_synonyms(word)}")
    # correlation = evaluate_simlex(model, simlex)
    # print(f"Spearman correlation with SimLex-999: {correlation:.4f}")

    def test_evaluate_word_pairs(self):
        # Read all the Game of Thrones books and combine them into single corpus.
        file = open("data/got1.txt", encoding='UTF-8')
        corpus1 = file.read()
        file = open("data/got2.txt", encoding='UTF-8')
        corpus2 = file.read()
        file = open("data/got3.txt", encoding='UTF-8')
        corpus3 = file.read()
        file = open("data/got4.txt", encoding='UTF-8')
        corpus4 = file.read()
        file = open("data/got5.txt", encoding='UTF-8')
        corpus5 = file.read()
        data = corpus1 + corpus2 + corpus3 + corpus4 + corpus5

        # Cleansing the big corpus and converting into DataFrame
        data = data.splitlines()
        data = list(filter(None, data))
        data = pd.DataFrame(data)
        data[0] = data[0].map(lambda x: clean_data(x))
        tmp_corpus = data[0].map(lambda x: x.split('.'))

        corpus = []
        for i in tqdm(range(len(tmp_corpus))):
            for line in tmp_corpus[i]:
                words = [x for x in line.split()]
                corpus.append(words)
        # Define the required parameters
        size = 100
        window_size = 10  # sentences weren't too long
        epochs = 100
        min_count = 5
        workers = 4
        model = Word2Vec(corpus, sg=1, window=window_size, vector_size=size, min_count=min_count, workers=workers,
                         epochs=epochs, sample=0.01)
        model.save('w2v_model')
        Word2Vec.load('w2v_model')
        print(os.path.exists('file:///C:/Users/USER-PCProjects/msc-2024-kwanele/Simlex-999.txt'))
        x=model.wv.evaluate_word_pairs(datapath('C:\\Users\\USER-PC\\Projects\\msc-2024-kwanele\\Simlex-999.txt'))
        print(x)
        # model.wv.evaluate_word_pairs(self,'file:///C:/Users/USER-PCProjects/msc-2024-kwanele/Simlex-999.txt',restrict_vocab=300000,case_insensitive=True,dummy4unknown=False)
