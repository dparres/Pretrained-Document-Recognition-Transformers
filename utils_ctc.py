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