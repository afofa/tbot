from __future__ import annotations
import copy
from functools import reduce
from math import log
from nltk.util import ngrams
import numpy as np
import pickle
from typing import Dict, Tuple, Union, List

NGram = Union[Dict[Tuple[str, ...], int], Dict[Tuple[str, ...], float]]
NGramCountType = Dict[str, Dict[int, int]]

class NGramCount:

    def __init__(self, words : List[str], n : int, is_stringify_ngrams : bool = True) -> None:
        self.num_of_words : int = len(words)
        self.words : List[str] = words
        self.n : int = n
        self.is_stringify_ngrams : bool = is_stringify_ngrams
        self.ngram_counts : NGramCountType = self.char_ngrams_of_words_with_count(words, self.n, self.is_stringify_ngrams)

    def char_ngrams_of_words_with_count(
        self,
        words: List[str], 
        n: int, 
        is_stringify_ngrams : bool = True,
    ) -> NGramCount:
        """
        val_type:
            0 -> counts of ngrams
            1 -> probabilities of ngrams
            2 -> log-probabilities of ngrams
        """

        dct = {}
        for i, word in enumerate(words):
            if is_stringify_ngrams:
                char_ngrams_of_word = map(lambda x: "".join(x), ngrams(word, n))
            else:
                char_ngrams_of_word = ngrams(word, n)

            for char_ngram in char_ngrams_of_word:
                if char_ngram not in dct.keys():
                    dct[char_ngram] = {i: 0}
                if char_ngram in dct.keys() and i not in dct[char_ngram].keys():
                    dct[char_ngram][i] = 0
                dct[char_ngram][i] += 1
        
        return dct

    def get_ngram_counts(self) -> NGramCountType:
        return self.ngram_counts

    def get_ngram_counts_with_removed_words(
        self,
        removed_words : List[str] = [],
        is_sum : bool = True,
    ) -> NGramCountType:
        removed_word_indexes = [i for i, word in enumerate(self.words) if word in removed_words]

        dct = copy.deepcopy(self.ngram_counts)

        keys_to_be_deleted = []
        for char_ngram in dct.keys():
            new_value = {k: v for k, v in dct[char_ngram].items() if k not in removed_word_indexes}
            if len(new_value) > 0:
                if is_sum:
                    dct[char_ngram] = sum(new_value.values())
                else:
                    dct[char_ngram] = new_value
            else:
                keys_to_be_deleted.append(char_ngram)
        [dct.pop(key) for key in keys_to_be_deleted] 

        return dct

    @staticmethod
    def from_count_to_prob(ngram_count_dict : NGramCountType, is_log : bool = True) -> NGramCountType:
        total_num_of_ngrams = sum(ngram_count_dict.values())
        ngram_count_dict = {k: v / total_num_of_ngrams for k, v in ngram_count_dict.items()}
        if is_log:
            ngram_count_dict = {k: log(v, 10) for k, v in ngram_count_dict.items()}
        return ngram_count_dict


class NGramStreamCount:

    def __init__(self, n : int, is_stringify_ngrams : bool = True) -> None:
        self.filepaths : List[str] = []
        self.num_of_words : int = 0
        self.words : Dict[str, None] = {}
        self.n : int = n
        self.is_stringify_ngrams : bool = is_stringify_ngrams
        self.ngram_counts : NGramCountType = {}

    def add_filepath(self, filepath : str) -> None:
        self.filepaths.append(filepath)
            
    def add_word(self, word : str) -> None:

        if word not in self.words:

            word_key = self.num_of_words
            self.words.update({word: None})
            self.num_of_words += 1

            if self.is_stringify_ngrams:
                char_ngrams_of_word = map(lambda x: "".join(x), ngrams(word, self.n))
            else:
                char_ngrams_of_word = ngrams(word, self.n)

            for char_ngram in char_ngrams_of_word:
                if char_ngram not in self.ngram_counts.keys():
                    self.ngram_counts[char_ngram] = {word_key: 0}
                if char_ngram in self.ngram_counts.keys() and word_key not in self.ngram_counts[char_ngram].keys():
                    self.ngram_counts[char_ngram][word_key] = 0
                self.ngram_counts[char_ngram][word_key] += 1

    def save_pickle(self, filepath : str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump([self.filepaths, self.num_of_words, list(self.words.keys()), self.n, self.is_stringify_ngrams, self.ngram_counts], f, protocol = -1)

    @staticmethod
    def load_pickle(filepath : str) -> NGramStreamCount:
        with open(filepath, "rb") as f:
            filepaths, num_of_words, words, n, is_stringify_ngrams, ngram_counts = pickle.load(f)

        obj = NGramStreamCount(n, is_stringify_ngrams)
        obj.filepaths = filepaths
        obj.num_of_words = num_of_words
        obj.words = dict.fromkeys(words) if isinstance(words, list) else words
        obj.set_ngram_counts(ngram_counts)

        return obj
        
    def set_ngram_counts(self, ngram_counts : NGramCountType) -> None:
        self.ngram_counts = ngram_counts

    def get_ngram_counts(self) -> NGramCountType:
        return self.ngram_counts

    def get_ngram_counts_with_removed_words(
        self,
        removed_words : List[str] = [],
        is_sum : bool = True,
    ) -> NGramCountType:
        removed_word_indexes = [i for i, word in enumerate(self.words) if word in removed_words]

        dct = copy.deepcopy(self.ngram_counts)

        keys_to_be_deleted = []
        for char_ngram in dct.keys():
            new_value = {k: v for k, v in dct[char_ngram].items() if k not in removed_word_indexes}
            if len(new_value) > 0:
                if is_sum:
                    dct[char_ngram] = sum(new_value.values())
                else:
                    dct[char_ngram] = new_value
            else:
                keys_to_be_deleted.append(char_ngram)
        [dct.pop(key) for key in keys_to_be_deleted] 

        return dct

    @staticmethod
    def from_count_to_prob(ngram_count_dict : NGramCountType, is_log : bool = True) -> NGramCountType:
        total_num_of_ngrams = sum(ngram_count_dict.values())
        ngram_count_dict = {k: v / total_num_of_ngrams for k, v in ngram_count_dict.items()}
        if is_log:
            ngram_count_dict = {k: log(v, 10) for k, v in ngram_count_dict.items()}
        return ngram_count_dict

def make_char_ngrams_from_words(
    words: List[str], 
    n: int, 
    val_type: int = 0,
) -> NGram:
    """
    val_type:
        0 -> counts of ngrams
        1 -> probabilities of ngrams
        2 -> log-probabilities of ngrams
    """
    dct = dict()
    for word in words:
        word_ngrams = map(lambda x: "".join(x), ngrams(word, n))
        
        for i in word_ngrams:
            if i in dct.keys():
                dct[i] += 1
            else:
                dct[i] = 1
    
    if val_type == 0:
        pass
    elif val_type == 1:
        sums = reduce(lambda x, y: x + y, dct.values(), 0)
        dct = dict((k, v/sums) for k, v in dct.items())
    elif val_type == 2:
        sums = reduce(lambda x, y: x + y, dct.values(), 0)
        dct = dict((k, log(v/sums, 10)) for k, v in dct.items())
    else:
        raise RuntimeError(f"unrecognized `val_type` = {val_type}")

    return dct

def count_char_ngrams_of_words(
        words: List[str], 
        n: int, 
        is_stringify_ngrams : bool = True,
    ) -> NGramCountType:
        """
        val_type:
            0 -> counts of ngrams
            1 -> probabilities of ngrams
            2 -> log-probabilities of ngrams
        """

        dct = {}
        for i, word in enumerate(words):
            if is_stringify_ngrams:
                char_ngrams_of_word = map(lambda x: "".join(x), ngrams(word, n))
            else:
                char_ngrams_of_word = ngrams(word, n)

            for char_ngram in char_ngrams_of_word:
                if char_ngram not in dct.keys():
                    dct[char_ngram] = {i: 0}
                if char_ngram in dct.keys() and i not in dct[char_ngram].keys():
                    dct[char_ngram][i] = 0
                dct[char_ngram][i] += 1
        
        return dct


def calculate_mean_log_prob_of_word(word: str, ngram_dct: Dict[Tuple, float], n: int) -> float:
    word_ngrams = map(lambda x: "".join(x), ngrams(word, n)) if isinstance(word, str) else []

    count_of_ngrams = 0
    sum_of_log_probs = 0

    for word_ngram in word_ngrams:
        if word_ngram in ngram_dct.keys():
            count_of_ngrams += 1
            sum_of_log_probs += ngram_dct[word_ngram]
        else: # if ngram does not exist, do not account for that
            count_of_ngrams += 1
            sum_of_log_probs += 0
        
    return sum_of_log_probs / count_of_ngrams if count_of_ngrams > 0 else -10

def calculate_probability_of_word(word : str, ngram_dictionary : Dict[str, float], n : int, mean : str = "geometric", dummy_value : float = -5.0) -> float:

    word_ngrams = map(lambda x: "".join(x), ngrams(word, n))
    probabilities_ngrams = map(lambda x: ngram_dictionary.get(x, dummy_value * n), word_ngrams)

    if mean == "geometric":
        probability = np.exp(np.mean(list(map(lambda x: np.log(x), probabilities_ngrams))))
    elif mean == "arithmetic":
        probability = np.mean(probabilities_ngrams)
    else:
        probability = np.exp(np.mean(map(lambda x: np.log(x), probabilities_ngrams)))

    return probability


if __name__ == "__main__":
    words = ["ahmet", "faruk", "ömer", "dasdsadsa", "melih mahmutoğlu"]
    n = 2
    is_stringify_ngrams = True
    obj = NGramStreamCount(n = n, is_stringify_ngrams = is_stringify_ngrams)

    print(obj.get_ngram_counts())

    for word in words:
        obj.add_word(word)
        print(obj.get_ngram_counts())

    obj.save_pickle("filepath.pkl")
    new_obj = NGramStreamCount.load_pickle("filepath.pkl")

    print(new_obj.get_ngram_counts())
    print(new_obj.words)
    # obj = NGramCount(words, n, is_stringify_ngrams)
    # print(obj.get_ngram_counts())
    # print(obj.get_ngram_counts_with_removed_words(["ömer", "mdadsad", "melih mahmutoğlu"]))