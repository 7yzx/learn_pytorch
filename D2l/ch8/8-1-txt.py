import collections

import torch
from torch import nn
from d2l import torch as d2l

lines = d2l.read_time_machine()
print(f'# 文本总行数: {len(lines)}')
# print(lines[0])
# print(lines[10])

tokens = d2l.tokenize(lines)
for i in range(11):
    print(tokens[i])


# class Vocab:
#     def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
#         if tokens is None:
#             tokens = []
#         if reserved_tokens is None:
#             reserved_tokens = []
#         counter = cou
#
# def count_corpus(tokens):
#     if len(tokens) == 0 or isinstance(tokens[0], list):
#         tokens = [token for line in tokens for token in line]
#         return collections.Counter(tokens)

vocab = d2l.Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])
