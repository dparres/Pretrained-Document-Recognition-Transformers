def create_char_dicts(lista_cadenas):
    text_to_seq = {}
    seq_to_text = {}
    valor = 1 # 0 is blank token

    for cadena in lista_cadenas:
        for caracter in cadena:
            if caracter not in text_to_seq:
                text_to_seq[caracter] = valor
                seq_to_text[valor] = caracter
                valor += 1
    return text_to_seq, seq_to_text

def sample_text_to_seq(lista_cadenas, diccionario):
    return [diccionario.get(caracter, "") for caracter in lista_cadenas]

def sample_seq_to_text(lista_cadenas, diccionario):
    return ''.join([diccionario.get(caracter, "") for caracter in lista_cadenas])