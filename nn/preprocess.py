# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
from collections import Counter

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    assert len(set(labels)) == 2


    a = max(labels,key=labels.count)
    b = min(labels,key=labels.count)
    a_counts, b_counts = labels.count(a), labels.count(b)

    seqs = np.array(seqs)
    labels = np.array(labels)
    a_seqs = seqs[labels == a]
    b_seqs = seqs[labels == b]

    #case where the values are already equal
    if a_counts == b_counts:
        return seqs, labels

    new_b_seqs_idx = np.random.choice(len(b_seqs), a_counts, replace=True)
    new_b_seqs = [b_seqs[i] for i in new_b_seqs_idx]
    new_b_labels = [b] * len(new_b_seqs_idx)
    sampled_seqs = list(a_seqs) + new_b_seqs
    sampled_labels = [a] * a_counts + new_b_labels
    return sampled_seqs, sampled_labels


def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    
    alphabet = 'ATCG'
    one_hot_seqs = []
   
    for seq in seq_arr:
        
        out = np.zeros(len(seq) * len(alphabet))
        
        for i, letter in enumerate(seq):
            out[(i*4) + alphabet.index(letter)] = 1
            
        one_hot_seqs += [out]

    return one_hot_seqs