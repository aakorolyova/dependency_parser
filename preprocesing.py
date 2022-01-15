import os
import numpy as np
from collections import Counter
from tqdm import tqdm
import pickle
import pandas as pd

def parse_connlu(file_path):
  sentences = []
  with open(file_path, 'r', encoding='utf-8') as input_file:
    sentence = []
    for l in input_file.readlines():
      if l.find('\t') == -1:
        if len(sentence) > 0:
          sentences.append(sentence[::-1])
          sentence = []
      else:
        tok_id, token, lemma, pos_simple, pos_detailed, morphology, head, dep, head_dep, spacing = l.strip().split('\t')
        if tok_id.find('-') == -1 and tok_id.find('.') == -1:
          sentence.append((int(tok_id), token, pos_detailed, int(head)))
  return sentences


def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 2  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix


def get_word_embeddings(filepath, embedding_dim):
  word_embeddings = {}
  with open(filepath, 'r', encoding='utf-8') as f:
    for line in f:
      word, *vector = line.split()
      word_embeddings[word] = np.array(
          vector, dtype=np.float32)[:embedding_dim]

    return word_embeddings


def onehot(unique_items):
    unique_items = sorted(unique_items)
    onehot_mapping = {}
    for i, item in enumerate(unique_items):
        onehot_mapping[item] = np.zeros(len(unique_items))
        onehot_mapping[item][i] = 1
    return onehot_mapping


def sentence2sequences_3l(sentence, ops2labels, debug = False):
  # 3 labels: shift, left, right
  sequences = []
  labels = []
  stack = ['ROOT']
  tok_id2head = {tok_id: head for tok_id, token, pos_detailed, head in sentence}
  buffer = [(tok_id, token, pos_detailed) for tok_id, token, pos_detailed, head in sentence]
  #sequences.append((stack, buffer, ops2labels['shift']))
  n = 0
  while len(buffer) > 0 or len(stack) > 1 and n < 200:
    if debug:
      print('buffer:', buffer)
      print('stack:', stack)

    if len(stack) == 1:
      if len(buffer) > 0:
        action = 'shift'
      # else:
      #   stack.pop()
    elif len(stack) == 2:
      if len(buffer) > 0:
        action = 'shift'
      else:
        action = 'right'
    else:
      tok_1, tok_2 = stack[-2], stack[-1]
      tok_1_id = tok_1[0]
      tok_2_id = tok_2[0]
      head_1 = tok_id2head[tok_1_id]
      head_2 = tok_id2head[tok_2_id]
      if debug:
        print(tok_1_id, tok_2_id)
        print(head_1, head_2)
      remaining_heads = [head for tok_id, token, pos_detailed, head in sentence if tok_id > last_popped]
      if debug:
        print(remaining_heads)
      if tok_1_id == head_2:
        if tok_2_id in remaining_heads:
          action = 'shift'
        else:
          action = 'right'
      elif tok_2_id == head_1:
        if tok_1_id in remaining_heads:
          action = 'shift'
        else:
          action = 'left'
      else:
          action = 'shift'

    if debug:
      print(action)
      print()
    sequences.append((stack[-2:], buffer.copy()))
    labels.append(ops2labels[action])

    if debug:
      print('added', stack[-2:], buffer, ops2labels[action])
    n += 1
    if action == 'shift' and len(buffer) > 0:
      next_tok = buffer.pop()
      last_popped = next_tok[0]
      stack.append(next_tok)
    elif action == 'right':
      stack.pop()
    elif action == 'left':
      stack.pop(-2)

  return sequences, labels, stack


def sentence2sequences_5l(sentence, ops2labels, debug=False):
    # 5 labels: shift, left, right, left_1, right_1
    sequences = []
    labels = []
    stack = ['ROOT']
    tok_id2head = {tok_id: head for tok_id, token, pos_detailed, head in sentence}
    buffer = [(tok_id, token, pos_detailed) for tok_id, token, pos_detailed, head in sentence]
    # sequences.append((stack, buffer, ops2labels['shift']))
    n = 0
    while len(buffer) > 0 or len(stack) > 1 and n < 200:
        if debug:
            print('buffer:', buffer)
            print('stack:', stack)

        if len(stack) == 1:
            if len(buffer) > 0:
                action = 'shift'
            # else:
            #   stack.pop()
        elif len(stack) == 2:
            if len(buffer) > 0:
                action = 'shift'
            else:
                action = 'right'
        else:
            tok_1, tok_2 = stack[-2], stack[-1]

            # 3rd token in stack, in case of non-projective trees
            tok_3 = None
            if len(stack) > 3:
                tok_3 = stack[-3]
                tok_3_id = tok_3[0]
                head_3 = tok_id2head[tok_3_id]

            tok_1_id = tok_1[0]
            tok_2_id = tok_2[0]
            head_1 = tok_id2head[tok_1_id]
            head_2 = tok_id2head[tok_2_id]
            if debug:
                print(tok_1_id, tok_2_id)
                print(head_1, head_2)
            remaining_heads = [head for tok_id, token, pos_detailed, head in sentence if tok_id > last_popped]
            if debug:
                print(remaining_heads)
            if tok_1_id == head_2:
                if tok_2_id in remaining_heads:
                    action = 'shift'
                else:
                    if tok_3 is not None:
                        if tok_2_id == head_3:
                            action = 'left_1'
                        else:
                            action = 'right'
                    else:
                        action = 'right'
            elif tok_2_id == head_1:
                if tok_1_id in remaining_heads:
                    action = 'shift'
                else:
                    action = 'left'
            else:
                # here something for non-proj
                # if len(buffer) == 0:
                if tok_3 is not None and tok_3_id == head_2 and tok_2_id not in remaining_heads:
                    action = 'right_1'
                else:
                    action = 'shift'

        if debug:
            print(action)
            print()
        sequences.append((stack[-3:], buffer.copy()))
        labels.append(ops2labels[action])

        if debug:
            print('added', stack[-2:], buffer, ops2labels[action])
        n += 1
        if action == 'shift' and len(buffer) > 0:
            next_tok = buffer.pop()
            last_popped = next_tok[0]
            stack.append(next_tok)
        elif action == 'right' or action == 'right_1':
            stack.pop()
        elif action == 'left':
            stack.pop(-2)
        elif action == 'left_1':
            stack.pop(-3)

    return sequences, labels, stack


def dataset2sequences(dataset, ops2labels, sentence2sequences_func, debug = False):
  dataset_sequences = []
  dataset_labels = []
  success = 0
  fail = 0
  for s in dataset:
    sequences, labels, stack = sentence2sequences_func(s, ops2labels, debug = debug)
    if stack == ['ROOT']:
      dataset_sequences.extend(sequences)
      dataset_labels.extend(labels)
      success += 1
  print('Successfully parsed sentences:', success, success / len(dataset))
  return dataset_sequences, dataset_labels


def get_features_element_3l(element, word_embeddings, postags_onehot):
  """
  3 labels scenario, using 2 last tokens in stack
  Features:
  - length of stack
  - length of buffer
  - is ROOT in stack
  - embeddings of tokens in stack
  - distance between tokens in stack
  - positions of tokens in stack
  - postag onehot encoding
  """
  seq_features = []
  stack, buffer = element
  # lengths of buffer & stack
  seq_features.append(len(stack))
  seq_features.append(len(buffer))

  # is ROOT in stack
  seq_features.append(1 if 'ROOT' in stack else 0)

  postag_len = len(list(postags_onehot.keys()))
  word_embeddings_len = len(word_embeddings[list(word_embeddings.keys())[0]])

  if stack == ['ROOT']:
    dist = 0
    position_1 = 0
    position_2 = 0
    tok_1_emb = np.zeros(word_embeddings_len)
    tok_2_emb = np.zeros(word_embeddings_len)
    postag_1 = np.zeros(postag_len)
    postag_2 = np.zeros(postag_len)

    seq_features.append(dist)
    seq_features.append(position_1)
    seq_features.append(position_2)
    seq_features.extend(list(tok_1_emb))
    seq_features.extend(list(tok_2_emb))
    seq_features.extend(list(postag_1))
    seq_features.extend(list(postag_2))

  else:
    tok_1, tok_2 = stack
    if tok_1 != 'ROOT':
      dist = tok_2[0] - tok_1[0]
      position_1 = tok_1[0]
      postag_1 = postags_onehot[tok_1[2]]
      if tok_1[1] in word_embeddings:
        tok_1_emb = word_embeddings[tok_1[1]]
      else:
        tok_1_emb = np.zeros(word_embeddings_len)
    else:
      dist = tok_2[0]
      position_1 = 0
      postag_1 = np.zeros(postag_len)
      tok_1_emb = np.zeros(word_embeddings_len)

    position_2 = tok_2[0]
    postag_2 = postags_onehot[tok_2[2]]
    if tok_2[1] in word_embeddings:
      tok_2_emb = word_embeddings[tok_2[1]]
    else:
      tok_2_emb = np.zeros(word_embeddings_len)

    seq_features.append(dist)
    seq_features.append(position_1)
    seq_features.append(position_2)
    seq_features.extend(list(tok_1_emb))
    seq_features.extend(list(tok_2_emb))
    seq_features.extend(list(postag_1))
    seq_features.extend(list(postag_2))
  return seq_features


def get_features_element_5l(element, word_embeddings, postags_onehot):
    """
    5 labels scenario, using 2 last tokens in stack
    Features:
    - length of stack
    - length of buffer
    - is ROOT in stack
    - embeddings of tokens in stack
    - distance between tokens in stack
    - positions of tokens in stack
    - postag onehot encoding
    """
    seq_features = []
    stack, buffer = element
    # lengths of buffer & stack
    seq_features.append(len(stack))
    seq_features.append(len(buffer))

    # is ROOT in stack
    seq_features.append(1 if 'ROOT' in stack else 0)

    postag_len = len(list(postags_onehot.keys()))
    word_embeddings_len = len(word_embeddings[list(word_embeddings.keys())[0]])

    if stack == ['ROOT']:
        dist = 0
        position_1 = 0
        position_2 = 0
        position_3 = 0
        tok_1_emb = np.zeros(word_embeddings_len)
        tok_2_emb = np.zeros(word_embeddings_len)
        tok_3_emb = np.zeros(word_embeddings_len)
        postag_1 = np.zeros(postag_len)
        postag_2 = np.zeros(postag_len)
        postag_3 = np.zeros(postag_len)

    elif len(stack) == 2:
        tok_2, tok_1 = stack
        if tok_2 != 'ROOT':
            dist = tok_1[0] - tok_2[0]
            position_2 = tok_2[0]
            postag_2 = postags_onehot[tok_2[2]]
            if tok_2[1] in word_embeddings:
                tok_2_emb = word_embeddings[tok_2[1]]
            else:
                tok_2_emb = np.zeros(word_embeddings_len)
        else:
            dist = tok_1[0]
            position_2 = 0
            postag_2 = np.zeros(postag_len)
            tok_2_emb = np.zeros(word_embeddings_len)

        position_1 = tok_1[0]
        postag_1 = postags_onehot[tok_1[2]]
        if tok_1[1] in word_embeddings:
            tok_1_emb = word_embeddings[tok_1[1]]
        else:
            tok_1_emb = np.zeros(word_embeddings_len)

        position_3 = 0
        postag_3 = np.zeros(postag_len)
        tok_3_emb = np.zeros(word_embeddings_len)

    elif len(stack) == 3:
        tok_3, tok_2, tok_1 = stack

        dist = tok_1[0] - tok_2[0]
        position_1 = tok_1[0]
        postag_1 = postags_onehot[tok_1[2]]
        if tok_1[1] in word_embeddings:
            tok_1_emb = word_embeddings[tok_1[1]]
        else:
            tok_1_emb = np.zeros(word_embeddings_len)

        position_2 = tok_2[0]
        postag_2 = postags_onehot[tok_2[2]]
        if tok_2[1] in word_embeddings:
            tok_2_emb = word_embeddings[tok_2[1]]
        else:
            tok_2_emb = np.zeros(word_embeddings_len)

        if tok_3 != 'ROOT':
            position_3 = tok_3[0]
            postag_3 = postags_onehot[tok_3[2]]
            if tok_3[1] in word_embeddings:
                tok_3_emb = word_embeddings[tok_3[1]]
            else:
                tok_3_emb = np.zeros(word_embeddings_len)
        else:
            position_3 = 0
            postag_3 = np.zeros(postag_len)
            tok_3_emb = np.zeros(word_embeddings_len)
            seq_features.append(dist)

    seq_features.append(position_3)
    seq_features.append(position_2)
    seq_features.append(position_1)
    seq_features.extend(list(tok_3_emb))
    seq_features.extend(list(tok_2_emb))
    seq_features.extend(list(tok_1_emb))
    seq_features.extend(list(postag_3))
    seq_features.extend(list(postag_2))
    seq_features.extend(list(postag_1))

    return seq_features


def get_features(dataset_sequences, word_embeddings, postags_onehot, get_features_element_fuct):
  """
  Features:
  - length of stack
  - length of buffer
  - is ROOT in stack
  - embeddings of tokens in stack
  - distance between tokens in stack
  - positions of tokens in stack
  - postag onehot encoding
  """
  features = [] #[get_features_element(seq, word_embeddings, postags_onehot) for seq in dataset_sequences]
  for seq in tqdm(dataset_sequences):
    seq_features = get_features_element_fuct(seq, word_embeddings, postags_onehot)
    features.append(seq_features)
  return features