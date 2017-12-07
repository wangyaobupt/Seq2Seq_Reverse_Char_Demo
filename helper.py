import os
import numpy as np


def load_data(path):
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data


def extract_vocab(data):
    special_words = ['<pad>', '<unk>', '<s>',  '<\s>']

    set_words = set([word for line in data.split('\n') for word in line.split()])
    int_to_vocab = {word_i: word for word_i, word in enumerate(special_words + list(set_words))}
    vocab_to_int = {word: word_i for word_i, word in int_to_vocab.items()}

    return int_to_vocab, vocab_to_int
    
def extract_alphabet_vocab():
    special_words = ['<pad>', '<unk>', '<s>',  '<\s>']

    alphabet_list =  [chr(ord('a')+i) for i in range(0, 26)]
    int_to_vocab = {word_i: word for word_i, word in enumerate(special_words + alphabet_list)}
    vocab_to_int = {word: word_i for word_i, word in int_to_vocab.items()}

    return int_to_vocab, vocab_to_int


def pad_id_sequences(source_ids, source_vocab_to_int, target_ids, target_vocab_to_int, sequence_length):
    new_source_ids = [list(reversed(sentence + [source_vocab_to_int['<pad>']] * (sequence_length - len(sentence)))) \
                      for sentence in source_ids]
    new_target_ids = [sentence + [target_vocab_to_int['<pad>']] * (sequence_length - len(sentence)) \
                      for sentence in target_ids]

    return new_source_ids, new_target_ids


def batch_data(source, target, batch_size):
    """
    Batch source and target together
    """
    for batch_i in range(0, len(source)//batch_size):
        start_i = batch_i * batch_size
        source_batch = source[start_i:start_i + batch_size]
        target_batch = target[start_i:start_i + batch_size]
        yield source_batch, target_batch


def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
    """Batch targets, sources, and the lengths of their sentences together"""
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))
        
        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))
        
        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))
        
        yield pad_targets_batch, pad_sources_batch, pad_targets_lengths, pad_source_lengths

def removeFileInDir(targetDir):
  if os.path.isdir(targetDir):
    for file in os.listdir(targetDir):
      targetFile = os.path.join(targetDir, file)
      if os.path.isfile(targetFile):
        print ('Delete Old Log FIle:', targetFile)
        os.remove(targetFile)
      elif os.path.isdir(targetFile):
        print ('Delete olds in log dir: ', targetFile)
        removeFileInDir(targetFile)
  else:
    os.mkdir(targetDir)
