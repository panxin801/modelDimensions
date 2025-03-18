from text import cleaners
from text.symbols import (symbols, symbols_zh)


# Mappings from symbol to numeric ID and vice versa:
# _symbol_to_id = {s: i for i, s in enumerate(symbols)}
# _id_to_symbol = {i: s for i, s in enumerate(symbols)}

chinese_mode = True
if chinese_mode:
    # symbols_zh=[中文符号和标点]
    _symbol_to_id = {v: idx for idx, v in enumerate(symbols_zh)}
    _id_to_symbol = {idx: v for idx, v in enumerate(symbols_zh)}
else:
    _symbol_to_id = {v: idx for idx, v in enumerate(symbols)}
    _id_to_symbol = {idx: v for idx, v in enumerate(symbols)}


def text_to_sequence(text, cleaner_names):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
      Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through
      Returns:
        List of integers corresponding to the symbols in the text
    '''

    sequence = []
    # text 是输入文本，cleaner_names 是config json里选择的cleaner
    clean_text = _clean_text(text, cleaner_names)
    # clean_text 是经过cleaner处理后的音素，中文就是带音调的拼音。
    for symbol in clean_text:
        symbol_id = _symbol_to_id[symbol]  # 拼音转换为数字
        sequence += [symbol_id]
    return sequence  # 返回文本的id序列


def cleaned_text_to_sequence(cleaned_text, chinese_mode=True):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
      Args:
        text: string to convert to a sequence
      Returns:
        List of integers corresponding to the symbols in the text
    '''
    # if chinese_mode:
    #   sequence = [_symbol_to_id_zh[symbol] for symbol in cleaned_text]
    # else:
    sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
    return sequence


def sequence_to_text(sequence):
    '''Converts a sequence of IDs back to a string'''
    result = ''
    for symbol_id in sequence:
        s = _id_to_symbol[symbol_id]
        result += s
    return result


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception(f"Unknown cleaner: {name}")
        text = cleaner(text)
    return text
