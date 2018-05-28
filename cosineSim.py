import logging
import re
from nltk.corpus import stopwords
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from porter2stemmer import Porter2Stemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(format='%(asctime)s : %(levelame)s : %(message)s', level=logging.INFO)
path = "/Users/Lydia/Desktop/corpus/Summary/Extracted_Sentences/newCandidateSentencesT1.txt"
filename = "/Users/Lydia/Desktop/corpus/Summary/Extracted_Sentences/newCandidateSentencescleanT1.txt"
fileMatrix = "/Users/Lydia/Desktop/corpus/Summary/Extracted_Sentences/newMatrixT1.txt"
thresh = "/Users/Lydia/Desktop/corpus/Summary/Extracted_Sentences/newMatrixThresholdT1.txt"
stemmer = Porter2Stemmer()

file = open(filename, 'w')
with open(path) as fp:
    for line in fp:
        cleanDoc = re.sub('[^A-Za-z0-9]+', ' ', line)
        text = [unicode(word) for word in cleanDoc.lower().split() if word not in stopwords.words('english')]
        #text = [stemmer.stem(t) for t in text]
        t = ' '.join(text)
        file.write(t)
        file.write("\n")
file.close()
print "C'est Finni!!"

document = []

with open(filename) as f:
    for l in f:
        document.append(l.strip("\n"))
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(document)
w = open(fileMatrix, 'w')
t = open(thresh, 'w')
cosine = (tfidf_matrix * tfidf_matrix.T).A
for list in cosine:
    for v in list:
        w.write(str(v))
        w.write("\t")
        #if v > 0.99:
            #t.write(str(v))
            #t.write("\t")
        if v < 0.35:
            t.write(str(v))
            t.write("\t")
        else: t.write("-" + "\t")
    t.write("\n")
    w.write("\n")
w.close()
t.close()