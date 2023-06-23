from nltk.tokenize import RegexpTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

def tokenizing(doc):
    tokenizer = RegexpTokenizer(r'\w+')
    tokenizing_doc = []

    for x in doc:
        lowerdoc = x.lower()
        proses_token = tokenizer.tokenize(lowerdoc)
        tokenizing_doc.append(proses_token)

    return tokenizing_doc

def stopword_removal(doc):
    stopwords_factory = StopWordRemoverFactory()
    stopwords = stopwords_factory.get_stop_words()
    sw_doc = []
    tuple_stopword = tuple(stopwords)

    i = 0
    for a in doc:
        sw_doc.append([])
        for b in a:
            if b not in tuple_stopword:
                sw_doc[i].append(b)
        i += 1

    return sw_doc

def case_folding(doc):
    case_folding_doc = []

    i = 0
    for a in doc:
        case_folding_doc.append([])
        for b in a:
            if b.isalpha():
                case_folding_doc[i].append(b)
        i += 1

    return case_folding_doc

def stemming(doc):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemmed_doc = []

    i = 0
    for a in doc:
        stemmed_doc.append([])
        for b in a:
            stemmed_doc[i].append(stemmer.stem(b))
        i += 1

    return stemmed_doc

def clean_doc(doc):
    readydoc = []

    i = 0
    for x in doc:
        readydoc.append((' '.join(map(str, doc[i]))))
        i += 1

    return readydoc