
import glob
#import logging
import os
import re

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

#logging.basicConfig(format='%(asctime)s : %(levelame)s : %(message)s', level=logging.INFO)
path = "/Users/Lydia/Downloads/DesktopFiles/GMTCorpus/GMTRaw"
lmtzr = WordNetLemmatizer()
stemmer = SnowballStemmer("english")

def textNaming():
    os.chdir(path)
    textID = {}
    i = 0
    for dir in glob.glob("*.txt"):
        textID[i] = dir
        i += 1
    return textID

def readCorpus():
    documents = []
    os.chdir(path)
    for dir in glob.glob("*.txt"):
        docPath = os.path.join(path,dir)
        doc = open(docPath, 'r')
        documents.append(doc.read())
    return documents

def preprocessCorpus():
    i = 0
    documents = readCorpus()
    textID = textNaming()
    for document in documents:
        cleanDoc = re.sub(r'\d+', '', document)
        cleanDoc = re.sub('[^A-Za-z0-9]+', ' ', cleanDoc)
        text = [unicode(word) for word in cleanDoc.lower().split() if word not in stopwords.words('english')
            and len(word) > 3 and not any(c.isdigit() for c in word)]
        text = [lmtzr.lemmatize(t) for t in text]
        text = [stemmer.stem(t) for t in text]
        lemmaTexts = ' '.join(text)
        filename = "/Users/Lydia/Downloads/DesktopFiles/GMTCorpus/clean2/" + textID[i] + ".txt"
        file = open(filename, 'w')
        file.write(lemmaTexts)
        file.close()
        i += 1

freqwrods = [u'georg', u'massey', u'tunnel', u'replac', u'project', u'bridg']

def preprocessQuery(query):
    cleanDoc = re.sub(r'\d+', '', query)
    cleanDoc = re.sub('[^A-Za-z0-9]+', ' ', cleanDoc)
    querylist = [word for word in cleanDoc.lower().split() if word not in stopwords.words('english')
            and len(word) > 3 and not any(c.isdigit() for c in word)]
    querylist = [lmtzr.lemmatize(q) for q in querylist]
    querylist = [stemmer.stem(q) for q in querylist]

    return querylist

def preprocessQuery2(query):
    cleanDoc = re.sub(r'\d+', '', query)
    cleanDoc = re.sub('[^A-Za-z0-9]+', ' ', cleanDoc)
    querylist = [word for word in cleanDoc.lower().split() if word not in stopwords.words('english')
            and len(word) > 3 and not any(c.isdigit() for c in word)]
    querylist = [lmtzr.lemmatize(q) for q in querylist]
    querylist = [stemmer.stem(q) for q in querylist]

    #remove frequently occuring words from the query. Does this lead to any improvements?
    i = 0
    l = len(querylist)
    while i < l:
        q = querylist[i]
        if q in freqwrods:
            del querylist[i]
            l -= 1
            i -= 1
        i += 1
    return querylist