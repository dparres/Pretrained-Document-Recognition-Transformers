import torch
from torch.nn.utils.rnn import pad_sequence

# CTC collate
def custom_collate(data):

    target_lengths = [len(d['label']) for d in data]
    labels = [d['label'] for d in data]
    inputs = [d['img'].tolist() for d in data]
    idx = [d['idx'] for d in data]
    raw_label = [d['raw_label'] for d in data]

    target_lengths = torch.tensor(target_lengths)
    labels = pad_sequence(labels, batch_first=True)
    inputs = torch.tensor(inputs)
    idx = torch.tensor(idx)

    return { #(6)
        'idx': idx,
        'img': inputs,
        'label': labels,
        'target_lengths': target_lengths,
        'raw_label': raw_label,
    }

def create_char_dicts(list_strings):
    text_to_seq = {}
    seq_to_text = {}
    value = 1 # 0 is blank token

    for text in list_strings:
        for character in text:
            if character not in text_to_seq:
                text_to_seq[character] = value
                seq_to_text[value] = character
                value += 1
    return text_to_seq, seq_to_text

def sample_text_to_seq(list_strings, mydict):
    return [mydict.get(character, "") for character in list_strings]

def sample_seq_to_text(list_strings, mydict):
    return ''.join([mydict.get(character, "") for character in list_strings])