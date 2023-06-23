import math
import pandas as pd

def get_list_word(readydoc):
    list_word = []

    for sentence in readydoc:
        for word in sentence.split(" "):
            if word not in list_word:
                list_word.append(word)

    sorted_word = sorted(list_word)

    return sorted_word

def get_tf(clean_list_querycontent, list_word_content):
    length_list_querycontent = len(clean_list_querycontent)

    tf = []
    for x in range(length_list_querycontent):
        tf.append(dict(zip(list_word_content, [0 for x in range(len(list_word_content))])))

    for index, sentence in enumerate(clean_list_querycontent):
        for word in sentence.split(" "):
            if word in tf[index]:
                tf[index][word] += 1

    return tf

def get_idf(tf, list_word_content):
    length_list_content = len(tf)

    df = dict(zip(list_word_content, [0 for x in range(len(list_word_content))]))
    for index, document in enumerate(tf):
        if index >= 0:
            for key, value in document.items():
                if value:
                    df[key] += 1

    d_df = {}
    for key, value in df.items():
        d_df[key] = length_list_content / value

    idf = {}
    for key, value in d_df.items():
        idf[key] = round(math.log10(value), 3)

    return idf

def get_wqt(tf, idf):
    wqt = []

    for index, document in enumerate(tf):
        wqt.append({})
        for key, value in document.items():
            wqt[index][key] = round(value * idf[key], 3)

    return wqt

def get_df_tf_wqt(data):
    length_list_querycontent = len(data)
    df = pd.DataFrame(data, index=[f'D{i}' for i in range(length_list_querycontent)])
    t_df = df.T

    return t_df

def get_df_idf(data):
    df = pd.DataFrame.from_dict(data, orient='index', columns=['IDF'])

    return df