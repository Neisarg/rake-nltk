# -*- coding: utf-8 -*-
"""Implementation of Rapid Automatic Keyword Extraction algorithm.

As described in the paper `Automatic keyword extraction from individual
documents` by Stuart Rose, Dave Engel, Nick Cramer and Wendy Cowley.
"""

import string
from collections import Counter, defaultdict
from itertools import chain, groupby, product

import nltk
from enum import Enum
from nltk.tokenize import wordpunct_tokenize
import os
from multiprocessing import Lock, Value, Process
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import reduce
import pickle
import re
from sqlitedict import SqliteDict
import time

class DocCounter(object):
    def __init__(self, initval=0):
        self.val = Value('L', initval)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1
            if self.val.value % 1000 == 0:
                print("{} documents scanned".format(self.val.value))

    def value(self):
        with self.lock:
            v = self.val.value
        return v

class Metric(Enum):
    """Different metrics that can be used for ranking."""
    DEGREE_TO_FREQUENCY_RATIO = 0  # Uses d(w)/f(w) as the metric
    WORD_DEGREE = 1  # Uses d(w) alone as the metric
    WORD_FREQUENCY = 2  # Uses f(w) alone as the metric


cntr = DocCounter()
doc_preprocess_func = None
pargs = None


def file_writer_process(output_folder, prefix, q, lock):
    print("writer process started")
    pl_dict = SqliteDict(os.path.join(output_folder, "{0}_{1}.sqlite".format(prefix, "phrase_list")), tablename="phrase_list", autocommit=True)
    coc_dict = SqliteDict(os.path.join(output_folder, "{0}_{1}.sqlite".format(prefix, "co_occ")), tablename="co_occ", autocommit=True)
    wd_dict = SqliteDict(os.path.join(output_folder, "{0}_{1}.sqlite".format(prefix, "word_dist")), tablename="word_dist", autocommit=True)
    local_cntr = 0
    while(True):
        data = None
        lock.acquire()
        if(not q.empty()):
            data = q.get()
        lock.release()
        if data is not None:
            word_dist, co_occ, phrase_list = data
            pl_dict[local_cntr] = phrase_list
            coc_dict[local_cntr] = co_occ
            wd_dict[local_cntr] = word_dist
            local_cntr += 1
            if(local_cntr%1000 == 0):
                print("Written {0} documents".format(local_cntr))
        else:
            time.sleep(0.1)


def rake_process(raw_doc):
    cntr.increment()
    if doc_preprocess_func is None:
        doc = raw_doc
    else:
        doc = doc_preprocess_func(raw_doc)
    rake = RakeProcess(**pargs)
    rake.extract_keywords_from_text(doc)
    word_dist = rake.get_word_frequency_distribution()
    co_occ = rake.get_word_co_occurance_graph()
    phrase_list = rake.get_phrase_list()
    return [word_dist, co_occ, phrase_list]

class DistributedRake(object):
    def __init__(
        self,
        output_folder,
        prefix = "",
        stopwords=None,
        punctuations=None,
        language="english",
        ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO,
        max_length=100000,
        min_length=1,
    ):
        global pargs
        pargs = {"stopwords": stopwords,
                     "punctuations": punctuations,
                     "language": language,
                     "ranking_metric": ranking_metric,
                     "max_length": max_length,
                     "min_length": min_length}

        if isinstance(ranking_metric, Metric):
            self.metric = ranking_metric
        else:
            self.metric = Metric.DEGREE_TO_FREQUENCY_RATIO

        self.output_folder = output_folder
        self.prefix = prefix

    def digest_docs(self, doc_itr, doc_func = None, num_workers=mp.cpu_count(), chunksize=1000):
        global doc_preprocess_func
        mp_manager = mp.Manager()
        q = mp_manager.Queue()
        lock = Lock()
        doc_preprocess_func = doc_func
        print("Spawning writer process")
        writer_process = Process(target = file_writer_process, args = (self.output_folder, self.prefix, q, lock))
        writer_process.start()
        print("Starting Process Pool")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            result = executor.map(rake_process, doc_itr, chunksize=chunksize)
            while(True):
                try:
                    data = next(result)
                    lock.acquire()
                    q.put(data)
                    lock.release()
                except Exception as e:
                    print(e)
                    break

        while(not q.empty()):
            time.sleep(1)

        writer_process.terminate()



    # def store_results(self, folder, prefix=""):
    #     with open(os.path.join(folder, prefix + "ranked_phrases.pkl"), "wb") as wf:
    #         pickle.dump(self.ranked_phrases, wf)
    #
    #     with open(os.path.join(folder, prefix + "rank_list.pkl"), "wb") as wf:
    #         pickle.dump(self.rank_list, wf)
    #
    #     with open(os.path.join(folder, prefix + "frequency_dist.pkl"), "wb") as wf:
    #         pickle.dump(self.frequency_dist, wf)
    #
    #     with open(os.path.join(folder, prefix + "word_degree.pkl"), "wb") as wf:
    #         pickle.dump(self.degree, wf)

    # def get_phrase_list(self):
    #     return self.phrase_list
    #
    # def get_word_frequency_distribution(self):
    #     """Method to fetch the word frequency distribution in the given text.
    #
    #     :return: Dictionary (defaultdict) of the format `word -> frequency`.
    #     """
    #     return self.frequency_dist
    #
    # def get_word_co_occurance_graph(self):
    #     """Method to fetch the degree of words in the given text. Degree can be
    #     defined as sum of co-occurances of the word with other words in the
    #     given text.
    #     """
    #     return self.co_occ
    #
    # def get_ranked_phrases(self):
    #     """Method to fetch ranked keyword strings.
    #
    #     :return: List of strings where each string represents an extracted
    #              keyword string.
    #     """
    #     return self.ranked_phrases
    #
    # def get_ranked_phrases_with_scores(self):
    #     """Method to fetch ranked keyword strings along with their scores.
    #
    #     :return: List of tuples where each tuple is formed of an extracted
    #              keyword string and its score. Ex: (5.68, 'Four Scoures')
    #     """
    #     return self.rank_list
    #
    # def get_word_degree(self):
    #     return self.degree

    # def build_word_degree(self):
    #     self.degree = defaultdict(int)
    #     for key in self.co_occ:
    #         self.degree[key] = sum(self.co_occ[key].values())
    #
    # def build_ranklist(self):
    #     """Method to rank each contender phrase using the formula
    #
    #           phrase_score = sum of scores of words in the phrase.
    #           word_score = d(w)/f(w) where d is degree and f is frequency.
    #
    #     :param phrase_list: List of List of strings where each sublist is a
    #                         collection of words which form a contender phrase.
    #     """
    #     self.rank_list = []
    #     for phrase in self.phrase_list:
    #         rank = 0.0
    #         for word in phrase:
    #             if self.metric == Metric.DEGREE_TO_FREQUENCY_RATIO:
    #                 rank += 1.0 * self.degree[word] / self.frequency_dist[word]
    #             elif self.metric == Metric.WORD_DEGREE:
    #                 rank += 1.0 * self.degree[word]
    #             else:
    #                 rank += 1.0 * self.frequency_dist[word]
    #         self.rank_list.append((rank, " ".join(phrase)))
    #     self.rank_list.sort(reverse=True)
    #     self.ranked_phrases = [ph[1] for ph in self.rank_list]


class RakeProcess(object):
    """Rapid Automatic Keyword Extraction Algorithm."""

    def __init__(
        self,
        stopwords=None,
        punctuations=None,
        language="english",
        ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO,
        max_length=100000,
        min_length=1,
    ):
        """Constructor.

        :param stopwords: List of Words to be ignored for keyword extraction.
        :param punctuations: Punctuations to be ignored for keyword extraction.
        :param language: Language to be used for stopwords
        :param max_length: Maximum limit on the number of words in a phrase
                           (Inclusive. Defaults to 100000)
        :param min_length: Minimum limit on the number of words in a phrase
                           (Inclusive. Defaults to 1)
        """
        # By default use degree to frequency ratio as the metric.
        if isinstance(ranking_metric, Metric):
            self.metric = ranking_metric
        else:
            self.metric = Metric.DEGREE_TO_FREQUENCY_RATIO

        # If stopwords not provided we use language stopwords by default.
        self.stopwords = stopwords
        if self.stopwords is None:
            self.stopwords = nltk.corpus.stopwords.words(language)
        else:
            self.stopwords.extend(nltk.corpus.stopwords.words(language))
            self.stopwords = list(set(self.stopwords))

        # If punctuations are not provided we ignore all punctuation symbols.
        self.punctuations = punctuations
        if self.punctuations is None:
            self.punctuations = string.punctuation

        # All things which act as sentence breaks during keyword extraction.
        self.to_ignore = set(chain(self.stopwords, self.punctuations)).union(set(['',' ']))

        # Assign min or max length to the attributes
        self.min_length = min_length
        self.max_length = max_length

        # Stuff to be extracted from the provided text.
        self.frequency_dist = None
        self.word_co_occurance_graph = None
        self.phrase_list = None

    def extract_keywords_from_text(self, text):
        """Method to extract keywords from the text provided.

        :param text: Text to extract keywords from, provided as a string.
        """
        sentences = nltk.tokenize.sent_tokenize(text)
        self.extract_keywords_from_sentences(sentences)

    def extract_keywords_from_sentences(self, sentences):
        """Method to extract keywords from the list of sentences provided.

        :param sentences: Text to extraxt keywords from, provided as a list
                          of strings, where each string is a sentence.
        """
        self.phrase_list = self._generate_phrases(sentences)
        self._build_frequency_dist(self.phrase_list)
        self._build_word_co_occurance_graph(self.phrase_list)

    def get_phrase_list(self):
        return self.phrase_list

    def get_word_frequency_distribution(self):
        """Method to fetch the word frequency distribution in the given text.

        :return: Dictionary (defaultdict) of the format `word -> frequency`.
        """
        return self.frequency_dist

    def get_word_co_occurance_graph(self):
        """Method to fetch the degree of words in the given text. Degree can be
        defined as sum of co-occurances of the word with other words in the
        given text.
        """
        return self.word_co_occurance_graph

    def _build_frequency_dist(self, phrase_list):
        """Builds frequency distribution of the words in the given body of text.

        :param phrase_list: List of List of strings where each sublist is a
                            collection of words which form a contender phrase.
        """
        self.frequency_dist = Counter(chain.from_iterable(phrase_list))

    def _build_word_co_occurance_graph(self, phrase_list):
        """Builds the co-occurance graph of words in the given body of text to
        compute degree of each word.

        :param phrase_list: List of List of strings where each sublist is a
                            collection of words which form a contender phrase.
        """


        co_occurance_graph = defaultdict(self.dd_lambda)
        for phrase in phrase_list:
            # For each phrase in the phrase list, count co-occurances of the
            # word with other words in the phrase.
            #
            # Note: Keep the co-occurances graph as is, to help facilitate its
            # use in other creative ways if required later.
            for (word, coword) in product(phrase, phrase):
                co_occurance_graph[word][coword] += 1

        self.word_co_occurance_graph = co_occurance_graph


    def _generate_phrases(self, sentences):
        """Method to generate contender phrases given the sentences of the text
        document.

        :param sentences: List of strings where each string represents a
                          sentence which forms the text.
        :return: Set of string tuples where each tuple is a collection
                 of words forming a contender phrase.
        """
        phrase_list = set()
        # Create contender phrases from sentences.
        regex = re.compile('[^a-zA-Z]')
        for sentence in sentences:
            word_list = [word.lower() for word in wordpunct_tokenize(sentence)]
            word_list = list(chain(*[regex.sub(' ', w).split(' ') for w in word_list]))
            phrase_list.update(self._get_phrase_list_from_words(word_list))
        return phrase_list

    def _get_phrase_list_from_words(self, word_list):
        """Method to create contender phrases from the list of words that form
        a sentence by dropping stopwords and punctuations and grouping the left
        words into phrases. Only phrases in the given length range (both limits
        inclusive) would be considered to build co-occurrence matrix. Ex:

        Sentence: Red apples, are good in flavour.
        List of words: ['red', 'apples', ",", 'are', 'good', 'in', 'flavour']
        List after dropping punctuations and stopwords.
        List of words: ['red', 'apples', *, *, good, *, 'flavour']
        List of phrases: [('red', 'apples'), ('good',), ('flavour',)]

        List of phrases with a correct length:
        For the range [1, 2]: [('red', 'apples'), ('good',), ('flavour',)]
        For the range [1, 1]: [('good',), ('flavour',)]
        For the range [2, 2]: [('red', 'apples')]

        :param word_list: List of words which form a sentence when joined in
                          the same order.
        :return: List of contender phrases that are formed after dropping
                 stopwords and punctuations.
        """
        groups = groupby(word_list, self.filter_word)
        phrases = [tuple(group[1]) for group in groups if group[0]]
        return list(
            filter(
                self.filter_phrase, phrases
            )
        )

    def filter_word(self, x):
        return x not in self.to_ignore

    def filter_phrase(self, x):
        return self.min_length <= len(x) <= self.max_length

    def dd_lambda(self):
        return defaultdict(int)