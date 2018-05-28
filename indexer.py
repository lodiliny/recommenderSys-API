#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import print_function
from __future__ import division
from rake_nltk import Rake
from nltk.corpus import stopwords
import glob
import os
import math
from math import sqrt
import operator
import probability
import preprocess
import re
import random
import sklearn

#logging.basicConfig(format='%(asctime)s : %(levelame)s : %(message)s', level=logging.INFO)
path = "/Users/Lydia/Downloads/DesktopFiles/GMTCorpus/clean2"
#output = open("/Users/Lydia/Downloads/DesktopFiles/GMTCorpus/vocab.txt", 'w')
os.chdir(path)

topic_word_weights = probability.computeTopicWordProbability()

docID = {}
index = {}
tokenize = lambda doc: doc.lower().split()
#sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)

def getLinks():
    output = {}
    with open("/Users/Lydia/Downloads/DesktopFiles/GMTCorpus/Article_title_weblinks.txt", 'r') as f:
        for line in f:
            odas = []
            line = line.rstrip('\n')
            details = line.split('\t')
            saved_title = details[0]
            odas.append(details[1])
            odas.append(details[2])
            output[saved_title] = odas
    return output

def getTopicWords():
    alltopicwords = []
    topicwords = {}
    topicID = 0
    with open("/Users/Lydia/Downloads/DesktopFiles/GMTCorpus/topic-words.txt", 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            words = line.split()
            topicwords[topicID] = words
            for w in words:
                alltopicwords.append(w)
            topicID += 1
    return alltopicwords


def readCleanCorpus():
    i = 0
    documentID = {}
    documents = []
    os.chdir(path)
    read_files = glob.glob("*.txt")
    for f in read_files:
        documentID[i] = f
        i += 1
        with open (f, 'rb') as input:
            content = input.read()
            documents.append(content)
    return documents, documentID

def term_frequency(term, tokenized_document):
    return tokenized_document.count(term)

def sublinear_term_frequency(term, tokenized_document):
    count = tokenized_document.count(term)
    if count == 0:
        return 0
    return 1 + math.log(count)

def inverse_document_frequencies(tokenized_documents):
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    for tkn in all_tokens_set:
        if len(tkn) > 1:
            contains_token = map(lambda doc: tkn in doc, tokenized_documents)
            idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
    #print idf_values
    return idf_values


def tfidf_sklearn(documents):
    tokenized_documents = [tokenize(d) for d in documents]
    idf = inverse_document_frequencies(tokenized_documents)
    idf_list = idf.keys()
    tfidf_documents = {}
    docID = 0
    for document in tokenized_documents:
        doc_tfidf = []
        for term in idf.keys():
            tf = sublinear_term_frequency(term, document)
            doc_tfidf.append(tf * idf[term])
        #tfidf_documents[docID] = doc_tfidf
        norm_vec = sklearn.preprocessing.normalize([doc_tfidf], norm='l2')
        norm_vec = [item for sublist in norm_vec for item in sublist]
        tfidf_documents[docID] = norm_vec
        docID +=1
    i = 0
    for token in idf_list: #i in range(0,len(idf_list)):
        #token = idf_list[i]
        posting = []
        for documentID in tfidf_documents:
            tfidf_val = tfidf_documents[documentID]
            if tfidf_val[i] > 0.0:
                posting_pair = [documentID, tfidf_val[i]]
                posting.append(posting_pair)
        index[token] = posting
        i += 1
    return tfidf_documents

def tfidf(documents):
    tokenized_documents = [tokenize(d) for d in documents]
    idf = inverse_document_frequencies(tokenized_documents)
    idf_list = idf.keys()
    tfidf_documents = {}
    docID = 0

    for document in tokenized_documents:
        doc_tfidf = []
        for term in idf.keys():
            tf = sublinear_term_frequency(term, document)
            doc_tfidf.append(tf * idf[term])
        tfidf_documents[docID] = doc_tfidf
        docID += 1

    for i in range(0,len(idf_list)):
        token = idf_list[i]
        posting = []
        for documentID in tfidf_documents:
            tfidf_val = tfidf_documents[documentID]
            if tfidf_val[i] > 0.0:
                posting_pair = [documentID, tfidf_val[i]]
                posting.append(posting_pair)
        index[token] = posting
    return tfidf_documents

def doclenghts(documents):
    lengths = {}
    doc_tfidf = tfidf(documents)
    for k in doc_tfidf.keys():
        lengths[k] = 0.0
        total = 0.0
        for items in doc_tfidf[k]:
            square_x = items ** 2
            total += square_x
        item_length = sqrt(total)
        lengths[k] += item_length
    return lengths

def relevance(highlightlist):
    documents, docIDs = readCleanCorpus()
    for k in docIDs.keys():
        print(k, docIDs[k])
    tfidf(documents) #to get the global variable 'index' -- inverted index PL populated
    tokenized_documents = [tokenize(d) for d in documents]
    idf = inverse_document_frequencies(tokenized_documents)
    score = {}
    alltop5docs = []
    article_dict = {}
    article_title = preprocess.textNaming()
    doclen = doclenghts(documents)
    for highlight in highlightlist:
        for i in range(0, 50):
            score[i] = 0.0
        highlight = highlight[0]
        querylist = preprocess.preprocessQuery(highlight)
        #print(querylist)
        for term in querylist:
            if term in idf.keys():
                q_tf = sublinear_term_frequency(term, querylist)
                q_tfidf = q_tf * idf[term]
                t_pl = index[term]
                #print(term, t_pl)
            for list in t_pl:
                docID = list[0]
                dt_tfidf = list[1]
                score[docID] += q_tfidf * dt_tfidf
        for id in score.keys():
            score[id] = round((score[id] / doclen[id]), 4)
        sorted_similarity = sorted(score.iteritems(), key = operator.itemgetter(1), reverse=True)
        article_list = {}
        article_list2 = {}
        for item in sorted_similarity:
            title2 = docIDs[item[0]]
            title = article_title[item[0]]
            article_list2[title2] = item[1]
            article_list[title] = item[1]
        sorted_relevance = sorted(article_list.iteritems(), key=operator.itemgetter(1), reverse=True)
        sorted_r = sorted(article_list2.iteritems(), key=operator.itemgetter(1), reverse=True)
        top5docs = []
        top5c = []
        top5ID = []
        for r in range(0, 5):
            top5docs.append(sorted_relevance[r])
            top5c.append(sorted_r[r])
            top5ID.append(sorted_similarity[r])
        print('new mapping', top5c)
        #print('old mapping', top5docs)
        #print('similarity', top5ID)
        alltop5docs += top5docs
    '''for d in alltop5docs:
        title = d[0]
        score = d[1]
        if title in article_dict.keys():
            article_dict[title] += score
        else: article_dict[title] = score
    final_output = sorted(article_dict.iteritems(), key=operator.itemgetter(1), reverse=True)
    print "Final Output"
    for a in final_output:
        print a
    return sorted_relevance'''

def uniqueWordsDoc(list):
    dict = {}
    for w in list:
        if w in dict.keys():
            dict[w] += 1
        else: dict[w] = 1
    unique = dict.keys()
    return unique

def ensemble(highlight):
    article_title = preprocess.textNaming()
    documents = readCleanCorpus()
    tfidf_sklearn(documents)  # to get the global variable 'index' -- inverted index PL populated
    tokenized_documents = [tokenize(d) for d in documents]
    idf = inverse_document_frequencies(tokenized_documents)
    val = {}
    score = {}
    doclen = doclenghts(documents)
    for i in range(0, 50):
        score[i] = 0.0
    querylist = preprocess.preprocessQuery(highlight)
    #to get the query tfidf normalized
    for term in querylist:
        if term in idf.keys():
            q_tf = sublinear_term_frequency(term, querylist)
            q_tfidf = q_tf * idf[term]
            val[term] = q_tfidf
    norm_query = sklearn.preprocessing.normalize([val.values()], norm='l2')
    norm_query = [item for sublist in norm_query for item in sublist]
    query_dict = dict(zip(val.keys(), norm_query))
    for term in query_dict.keys():
        t_pl = index[term]
        for list in t_pl:
            docID = list[0]
            dt_tfidf = list[1]
            score[docID] += q_tfidf * dt_tfidf
    for id in score.keys():
        score[id] = round((score[id] / doclen[id]), 4)
    sorted_similarity = sorted(score.iteritems(), key=operator.itemgetter(1), reverse=True)
    article_list = {}
    for item in sorted_similarity:
        title = article_title[item[0]]
        article_list[title] = item[1]
    sorted_relevance = sorted(article_list.iteritems(), key=operator.itemgetter(1), reverse=True)
    top5docs = []
    for r in range(0, 5):
        top5docs.append(sorted_relevance[r])
    print('ensemble function', top5docs)
    return top5docs

def queryDocSimilarity(highlights):
    article_title = preprocess.textNaming()
    documents, docIDs = readCleanCorpus()
    docweights = docTopicWeights(documents)
    docvectors = {}
    alltop5docs = []
    top5docs = []
    for docID in range(0, 50):
        weights = []
        for topicID in range(0, 10):
            weights.append(docweights[topicID][docID])
        docvectors[docID] = weights
     #compute the similarity usiing Q.Di for each document i in the collection
    number = len(highlights)
    if number == 1:
        for highlight in highlights:
            highlight = highlight[0]
            querylist = preprocess.preprocessQuery2(highlight)
            queryvector = computeQueryWeights(querylist)
            scores = {}
            for doc in range(0, 50):
                score = 0.0
                for topic in range(0, 10):
                    score += queryvector[topic] * docvectors[doc][topic]
                scores[doc] = score
            sorted_similarity = sorted(scores.iteritems(), key=operator.itemgetter(1), reverse=True)
            article_list = {}
            for item in sorted_similarity:
                title = article_title[item[0]]
                article_list[title] = item[1]
            sorted_relevance = sorted(article_list.iteritems(), key=operator.itemgetter(1), reverse=True)
            for i in range(0, 5):
                tuplex = sorted_relevance[i]
                tuplex = tuplex + (sorted_similarity[i][0],)
                alltop5docs.append(tuplex)
    else:
        threshold = 0.0
        for highlight in highlights:
            highlight = highlight[0]
            querylist = preprocess.preprocessQuery2(highlight)
            queryvector = computeQueryWeights(querylist)
            scores = {}
            for doc in range(0, 50):
                score = 0.0
                for topic in range(0, 10):
                    score += queryvector[topic] * docvectors[doc][topic]
                scores[doc] = score
            sorted_similarity = sorted(scores.iteritems(), key=operator.itemgetter(1), reverse=True)
            article_list = {}
            for item in sorted_similarity:
                title = article_title[item[0]]
                article_list[title] = item[1]
            sorted_relevance = sorted(article_list.iteritems(), key=operator.itemgetter(1), reverse=True)
            ttle = []
            tuplex = sorted_relevance[0]
            tuplex = tuplex + (sorted_similarity[0][0],)
            if len(alltop5docs) == 0:
                alltop5docs.append(tuplex)
            else:
                for e in alltop5docs:
                    ttle.append(e[0])
                if tuplex[0] not in ttle:
                    alltop5docs.append(tuplex)
            for i in range(1, 5):
                tuplex = sorted_relevance[i]
                if tuplex[1] >= threshold:
                    tuplex = tuplex + (sorted_similarity[i][0],)
                    top5docs.append(tuplex)
        sorted_top5docs = sorted(top5docs, key=operator.itemgetter(1), reverse=True)
        num = len(alltop5docs)
        if num < 5:
            for j in range(0, 5 - num):
                alltop5docs.append(sorted_top5docs[j])
        else:
            if num > 7:
                alltop5docs2 = alltop5docs
                alltop5docs =[]
                for j in range(0, 7):
                    alltop5docs.append(alltop5docs2[j])
    return alltop5docs


def computeQueryWeights(querylist):
    #queryvector = uniqueWordsDoc(querylist)
    queryvector = querylist
    queryweights = {}
    L = len(querylist)
    for topic in topic_word_weights.keys():
        sum = 0
        for w in queryvector:
            if w in topic_word_weights[topic].keys():
                w_jk = topic_word_weights[topic][w]
                sum += w_jk
        queryweights[topic] = float(sum / L)
    return queryweights

def docTopicWeights(documents):
    output = {}
    tokenized_documents = [tokenize(d) for d in documents]
    for topicID in topic_word_weights.keys():
        wordlist =  topic_word_weights[topicID]
        doc_topic_weights = {}
        for d in range(0, len(tokenized_documents)):
            doc = tokenized_documents[d]
            unique_doclist = uniqueWordsDoc(doc)
            sum = 0.0
            di_N = len(doc)
            for word in unique_doclist:
                if word in wordlist.keys():
                    n_ij = term_frequency(word, doc)
                    w_jk = topic_word_weights[topicID][word]
                    prod = w_jk * n_ij
                    sum += prod
            d_ik = float(sum / di_N)
            if d_ik > 1.0:
                print(word, n_ij, w_jk, sum, di_N)
            doc_topic_weights[d] = d_ik
        output[topicID] = doc_topic_weights
    return output

def recsDetails(sfu_id, highlightlist):
    top5docs = queryDocSimilarity(highlightlist)
    keywords = rakeKeywords(highlightlist)#lydskeywords(highlightlist)
    output = {}
    data = {}
    article_titles = []
    article_URLs = []
    article_snippets = []
    articles = getLinks()
    num = len(highlightlist)
    if num == 1:
        for i in range(0, len(top5docs)):
            snippet = snippetText(highlightlist[0][0], [top5docs[i][2]])
            snippet = snippet.lstrip('\n')
            saved_title = top5docs[i][0]
            article_details = articles[saved_title]
            atitle = article_details[0]
            aurl = article_details[1]
            atitle = removeNonAscii(atitle)
            if atitle not in article_titles:
                article_titles.append(atitle)
                article_URLs.append(aurl)
                article_snippets.append(snippet)
            output[saved_title] = article_details
    elif num == len(top5docs) or num > 7:
        for i in range(0, len(top5docs)):
            snippet = snippetText(highlightlist[i][0], [top5docs[i][2]])
            snippet = snippet.lstrip('\n')
            saved_title = top5docs[i][0]
            article_details = articles[saved_title]
            atitle = article_details[0]
            aurl = article_details[1]
            atitle = removeNonAscii(atitle)
            if atitle not in article_titles:
                article_titles.append(atitle)
                article_URLs.append(aurl)
                article_snippets.append(snippet)
            output[saved_title] = article_details
    else:
        for i in range(0, num):
            snippet = snippetText(highlightlist[i][0], [top5docs[i][2]])
            snippet = snippet.lstrip('\n')
            saved_title = top5docs[i][0]
            article_details = articles[saved_title]
            atitle = article_details[0]
            aurl = article_details[1]
            atitle = removeNonAscii(atitle)
            if atitle not in article_titles:
                article_titles.append(atitle)
                article_URLs.append(aurl)
                article_snippets.append(snippet)
            output[saved_title] = article_details
        for j in range(num, len(top5docs)):
            rand = random.randint(0, num-1)
            snippet = snippetText(highlightlist[rand][0], [top5docs[j][2]])
            snippet = snippet.lstrip('\n')
            saved_title = top5docs[j][0]
            article_details = articles[saved_title]
            atitle = article_details[0]
            aurl = article_details[1]
            atitle = removeNonAscii(atitle)
            if atitle not in article_titles:
                article_titles.append(atitle)
                article_URLs.append(aurl)
                article_snippets.append(snippet)
            output[saved_title] = article_details
    data['sfu_id'] = sfu_id
    data['titles'] = ';, '.join(article_titles)
    data['URLs'] = ';, '.join(article_URLs)
    data['snippets'] = ';, '.join(article_snippets)
    data['keywords'] = ';, '.join(keywords)
    print(data['titles'])
    return data

def extendedPostingList():
    documents = preprocess.readCorpus()
    corpusdocsents = {}
    finaldocwords = {}
    for i in range(0, len(documents)):
        finaldocsents = []
        docwords = {}
        doc = documents[i]
        doc = removeNonAscii(doc)
        docsent = doc.split('.')
        docsentlen = len(docsent)
        s = 0 #counter for number of sentences in the document
        while s < docsentlen:
            sent = docsent[s]
            sentlist = sent.lower().split()
            for w in range(0, len(sentlist)):
                words = sentlist[w]
                words = words.lstrip('\n')
                if ('$' in words or (words.isdigit() and len(words) < 2) or words == 'mr' or words == 'ms') and words == sentlist[len(sentlist)-1]:
                    s += 1
                    nextsent = docsent[s]
                    nextsent_list = nextsent.split()
                    if len(nextsent_list) > 0:
                        w1 = nextsent_list[0]
                    else:
                        s -= 1
                        break
                    mergedlist = []
                    if w1.isdigit() or w1 == '5-billion' or words == 'mr' or words == 'ms':
                        del docsent[s]
                        docsentlen -= 1
                        s -= 1
                        w1 = str(sentlist[len(sentlist)-1]) + '.'
                        w2 = str(nextsent_list[0])
                        mergedstr = w1 + w2
                        for j in range(0, len(sentlist)-1):
                            mergedlist.append(sentlist[j])
                        mergedlist.append(mergedstr)
                        for k in range(1, len(nextsent_list)):
                            mergedlist.append(nextsent_list[k])
                        sentlist = mergedlist
            for d in range(0, len(sentlist)):
                w = sentlist[d]
                if w in docwords.keys():
                    value = docwords[w]
                    item = (s, d) #sentence number and word position in sentence
                    value.append(item)
                    docwords[w] = value
                else:
                    value = []
                    item = (s, d)
                    value.append(item)
                    docwords[w] = value
            finaldocsents.append(sentlist)
            s += 1
        finaldocwords[i] = docwords
        corpusdocsents[i] = finaldocsents
    return corpusdocsents, finaldocwords

def snippetPositions(highlight, top5_docs):
    '''We would try it with one highlight/query for now...
    TO-DO: all associated functions should be able to work with the different highlights as in a list format...
    and not concatenated as a string'''
    #here, the highlights that would be passed would be the keywords from the highlights obtained, and not the actual highlight list themselves
    finaloutput = []
    corpus, e_PL = extendedPostingList()
    '''top5_docs = []
    retrieved_docs = queryDocSimilarity(highlight)
    for item in retrieved_docs:
        dID = item[0]
        top5_docs.append(dID)'''
    highlightvector = [word for word in highlight.lower().split() if word not in stopwords.words('english')
            and len(word) > 3 and not any(c.isdigit() for c in word)]
    highlightvector = uniqueWordsDoc(highlightvector)
    for docID in top5_docs:
        doc_ePL = e_PL[docID]
        keys = doc_ePL.keys()
        for word in highlightvector:
            if word in doc_ePL.keys():
                wordPos_details = [word, docID, doc_ePL[word]] #list: [word, docID, [(sentNum, WordPosInSent)]]
                finaloutput.append(wordPos_details)
                #get all variations of the query word
                windex = keys.index(word)
                del keys[windex]
                for w in keys:
                    if w.startswith(word):
                        wordPos_details = [w, docID, doc_ePL[w]]
                        finaloutput.append(wordPos_details)
    return finaloutput, corpus

def snippetScores(highlight, top5docs):
    finaloutput = {}
    sorted_finaloutput = {}
    docquerywords = {}
    snippetPOS, corpus = snippetPositions(highlight, top5docs)
    for dID in top5docs:
        docSents = {}
        docqwords = []
        for item in snippetPOS:
            tup = item[2]
            if dID == item[1]:
                docqwords.append(item[0])
                #compute scores for each sentence in snippetPos
                for t in tup:
                    s = t[0]
                    if s == 0:
                        if s in docSents.keys():
                            docSents[s] += 1
                        else: docSents[s] = 1
                    elif s == 1:
                        if s in docSents.keys():
                            docSents[s] += 2
                        else: docSents[s] = 2
                    elif s > 1:
                        if s in docSents.keys():
                            docSents[s] += 1
                        else: docSents[s] = 1
        finaloutput[dID] = docSents
        docquerywords[dID] = docqwords
    for k in finaloutput.keys():
        sent_details = finaloutput[k]
        sorted_similarity = sorted(sent_details.iteritems(), key=operator.itemgetter(1), reverse=True)
        sorted_finaloutput[k] = sorted_similarity
        #print k, sorted_similarity
    return corpus, sorted_finaloutput

def snippetText(highlight, top5docs):
    corpus, snippetsc = snippetScores(highlight, top5docs)
    snippet_seg = {}
    for dID in snippetsc.keys():
        sent = []
        score = []
        segment = {}
        doc_list = snippetsc[dID]
        for d in doc_list :
            tup = d
            sent.append(tup[0])
            score.append(tup[1])

        for i in range(0, len(sent)):
            sentID = sent[i]
            seg1 = ''
            seg2 = ''
            seg_score1 = 0
            seg_score2 = 0
            if sentID > 1:
                for j in range(sentID-2, sentID+1):
                    if j in sent:
                        seg1 += str(j) + ' '
                        indx = sent.index(j)
                        seg_score1 += score[indx]
                seg1 = seg1.rstrip(' ')
                if seg1 in segment.keys():
                    a = 0
                else: segment[seg1] = seg_score1

                for l in range(sentID, sentID+3):
                    if l in sent:
                        seg2 += str(l) + ' '
                        indx = sent.index(l)
                        seg_score2 += score[indx]
                seg2 = seg2.rstrip(' ')
                if seg2 in segment.keys():
                    a = 0
                else: segment[seg2] = seg_score2
            else:
                for l in range(sentID, sentID+3):
                    if l in sent:
                        seg2 += str(l) + ' '
                        indx = sent.index(l)
                        seg_score2 += score[indx]
                seg2 = seg2.rstrip(' ')
                if seg2 in segment.keys():
                    a = 0
                else: segment[seg2] = seg_score2
        sorted_segment = sorted(segment.iteritems(), key=operator.itemgetter(1), reverse=True)
        if len(sorted_segment) > 0:
            snippet_seg[dID] = sorted_segment[0]
        else: snippet_seg[dID] = ('0 1 2', 10)
    snippet_segment = {}
    snippet_text = {}
    for k in snippet_seg.keys():
        snippet = ''
        stup = snippet_seg[k]
        sID = stup[0]
        keylist = sID.split()
        keylist = [int(i) for i in keylist]
        start_seg = min(keylist)
        end_seg = max(keylist)
        snippet_segment[k] = [start_seg, end_seg]
        if start_seg == end_seg:
            for s in range(0, 3):
                doc = corpus[k]
                sentences = doc[s]
                summary = ' '.join(sentences)
                snippet += str(summary) + '. '
            snippet = snippet + '...'
            snippet_text[k] = snippet
        else:
            for s in range(int(start_seg), int(end_seg)+1):
                doc = corpus[k]
                sentences = doc[s]
                summary = ' '.join(sentences)
                snippet += str(summary) + '. '
            snippet = snippet + '...'
            snippet_text[k] = snippet
    return snippet

def removeNonAscii(text):
    output = []
    if isinstance(text, str):
        return re.sub(r'[^\x00-\x7F]+', ' ', text)
    else:
        for t in text:
            tt = t[0]
            tt = [re.sub(r'[^\x00-\x7F]+', ' ', tt)]
            output.append(tt)
        return output

def rakeKeywords(querylist):
    r = Rake()
    output = []
    finalOutput = []
    for query in querylist:
        query = query[0]
        r.extract_keywords_from_text(query)
        no_integers = [x for x in r.get_ranked_phrases() if not (x.isdigit()
                                                 or x[0] == '-' and x[1:].isdigit())]
        no_integers = [x for x in no_integers if len(x) > 1]
        output.append(no_integers)
    for list in output:
        str = ', '.join(list)
        finalOutput.append(str)
    print (finalOutput)
    return finalOutput

def querykeywordsFromTopics(query):
    allquerytopics = {}
    alltopicwords = getTopicWords() #change the function to return the dictionary not the list of all topic words
    cleanDoc = re.sub(r'\d+', '', query)
    cleanDoc = re.sub('[^A-Za-z0-9]+', ' ', cleanDoc)
    querylist = [word for word in cleanDoc.lower().split() if word not in stopwords.words('english')
                 and len(word) > 3 and not any(c.isdigit() for c in word)]
    querylist = uniqueWordsDoc(querylist)
    print("query list: ", querylist)
    for t in range(0, len(alltopicwords)):
        querytopic = []
        topicwords = alltopicwords[t]
        for tw in topicwords:
            for q in querylist:
                if tw in q:
                    querytopic.append(q)
        allquerytopics[t] = querytopic
    return allquerytopics

def lydskeywords(highlightlist):
    output = []
    alltopicwords = getTopicWords()
    for list in highlightlist:
        highlight = list[0]
        if ',' in highlight:
            highlight = highlight.split(',')
            for h in highlight:
                phrase = h
                phrase = phrase.split(' ')
                wordPOS = []
                for w in range(0, len(phrase)):
                    words = phrase[w]
                    words = preprocess.lmtzr.lemmatize(words)
                    words = preprocess.stemmer.stem(words)
                    if words in alltopicwords:
                        wordPOS.append(w)
                keyword = ''
                if len(wordPOS) > 1:
                    minval = min(wordPOS)
                    maxval = max(wordPOS)
                    for g in range(minval, maxval + 1):
                        keyword += phrase[g] + ' '
                    output.append(keyword)
                elif len(wordPOS) == 1:
                    minval = wordPOS[0]
                    keyword = phrase[minval]
                    output.append(keyword)
        else:
            highlight = highlight.split()
            wordPOS = []
            for w in range(0, len(highlight)):
                words = highlight[w]
                words = preprocess.lmtzr.lemmatize(words)
                words = preprocess.stemmer.stem(words)
                if words in alltopicwords:
                    wordPOS.append(w)
            finalKey = ''
            if len(wordPOS) > 1:
                minval = min(wordPOS)
                maxval = max(wordPOS)
                for g in range(minval, maxval + 1):
                    finalKey += highlight[g] + ' '
                output.append(finalKey)

    outputs ={}
    for item in output:
        itemlist = item.split()
        if len(itemlist) > 1 and len(itemlist) < 11:
            #answers = {}
            if item not in outputs.keys():
                outputs[item] = len(itemlist)
                '''for k in outputs.keys():
                    if k.lower() in item.lower() or item.lower() in k.lower():
                        answers['yes'] = k
                    else:
                        answers['no'] = k
                if 'yes' in answers.keys():
                    continue
                else: outputs[item] = len(itemlist)'''
        elif len(itemlist) > 10 and len(itemlist) < 30:
            itemlist2 = []
            for c in itemlist:
                c = preprocess.lmtzr.lemmatize(c)
                c = preprocess.stemmer.stem(c)
                itemlist2.append(c)
            for h in range(1, len(itemlist2)-1):
                if h + 2 < len(itemlist2) and itemlist2[h] in alltopicwords and itemlist2[h + 1] in alltopicwords and itemlist2[h + 2] in alltopicwords:
                    answers = {}
                    itm = ''
                    reststr = ''
                    wrdpos = []
                    for w in range(0, h + 3):
                        itm += itemlist[w] + ' '
                    itm = itm.rstrip(' ')
                    itmlist = itm.split()
                    if len(itmlist) < 11:
                        if itm not in outputs.keys():
                            for k in outputs.keys():
                                if k.lower() not in itm.lower():
                                    answers['no'] = k
                                else:
                                    answers['yes'] = k
                            if 'yes' in answers.keys():
                                continue
                            else: outputs[itm] = len(itmlist)
                    for q in range(h + 3, len(itemlist)):
                        if itemlist2[q] in alltopicwords:
                            wrdpos.append(q)
                        if len(wrdpos) > 1:
                            minv = min(wrdpos)
                            maxv = max(wrdpos)
                            for z in range(minv, maxv + 1):
                                reststr += itemlist[z] + ' '
                elif h + 1 < len(itemlist2) and itemlist2[h] in alltopicwords and itemlist2[h + 1] in alltopicwords:
                    answers = {}
                    itm = ''
                    reststr = ''
                    wrdpos = []
                    for w in range(0, h + 2):
                        itm += itemlist[w] + ' '
                    itm = itm.rstrip(' ')
                    itmlist = itm.split()
                    if len(itmlist) < 11:
                        if itm not in outputs.keys():
                            for k in outputs.keys():
                                if k.lower() not in itm.lower():
                                    answers['no'] = k
                                else:
                                    answers['yes'] = k
                            if 'yes' in answers.keys():
                                continue
                            else: outputs[itm] = len(itmlist)
                    for q in range(h + 2, len(itemlist)):
                        if itemlist2[q] in alltopicwords:
                            wrdpos.append(q)
                        if len(wrdpos) > 1:
                            minv = min(wrdpos)
                            maxv = max(wrdpos)
                            for z in range(minv, maxv + 1):
                                reststr += itemlist[z] + ' '
                elif h + 2 < len(itemlist2) and itemlist2[h] in alltopicwords:
                    if itemlist2[h + 1] not in alltopicwords or itemlist2[h + 2] not in alltopicwords:
                        answers = {}
                        itm = ''
                        reststr = ''
                        wrdpos = []
                        for w in range(0, h + 1):
                            itm += itemlist[w] + ' '
                        itm = itm.rstrip(' ')
                        itmlist = itm.split()
                        if len(itmlist) > 5 and len(itmlist) < 11:
                            if itm not in outputs.keys():
                                for k in outputs.keys():
                                    if k.lower() not in itm.lower():
                                        answers['no'] = k
                                    else:
                                        answers['yes'] = k
                                if 'yes' in answers.keys():
                                    continue
                                else: outputs[itm] = len(itmlist)
                        for q in range(h + 1, len(itemlist)):
                            if itemlist2[q] in alltopicwords:
                                wrdpos.append(q)
                            if len(wrdpos) > 1:
                                minv = min(wrdpos)
                                maxv = max(wrdpos)
                                for z in range(minv, maxv + 1):
                                    reststr += itemlist[z] + ' '
    #for key in outputs.keys():
        #print(key)
    return outputs.keys()

def randomArticleTitles():
    articles = preprocess.textNaming()
    entry = {}
    for i in range(0, 5):
        rand = random.randint(0, 49)
        if rand in entry:
            rand = random.randint(0, 49)
            entry[rand] = articles[rand]
        else:
            entry[rand] = articles[rand]
    print(entry)

#query = '''The cost of the project is thought to be about $3.5 billion. it’s not technically feasible to upgrade the tunnel to meet current seismic standards. highlighting how its the most economically and environmentally feasible option bridge would cause less harm to the environment. the project needs to pass reviews related to environmental and agricultural impact. The economic impacts would be long‐term, far reaching and impact the entire region. the multi billion-dollar project is too costly and will only serve to increase congestion in our region.'''
#query = [['the cost to replace the Massey Tunnel with a new bridge could triple'], ['The cost of the project is thought to be about $3.5 billion'], ['it’s not technically feasible to upgrade the tunnel to meet current seismic standards'], ['highlighting how its the most economically and environmentally feasible option'], ['bridge would cause less harm to the environment'], ['the project needs to pass reviews related to environmental and agricultural impact'], ['The economic impacts would be long‐term, far reaching and impact the entire region'], ['the multi billion-dollar project is too costly and will only serve to increase congestion in our region']]
#query1 = 'Mr. Stone said the province has considered the views of stakeholders over the five years since the premier committed to the project'
#query1 = 'The cost of the project is thought to be about $3.5 billion'
#query = [['debt adds billions to the cost of massey bridge']]
#query = [['twinning the tunnel might be a better option']]
#query = [["infrastructure and the associated land use and agricultural impacts"], ["traffic impacts on local roads"]]
#query = [["public transit and other alternative forms of transit"], ["regional but a provincial perspective"], ["$3.5-billion project"]]
'''query = []
with open("/Users/Lydia/Downloads/DesktopFiles/Sample-highlights/Zahia1.txt", 'r') as f:
    for line in f:
        line = [line.rstrip('\n')]
        query.append(line)
query = removeNonAscii(query)'''
#query1 = removeNonAscii(query1)
#lydskeywords(query)
#rakeKeywords(query)
#snippetText(query)
#relevance(query)
#queryDocSimilarity(query)
#recsDetails('lodiliny', query)
#ensemble(query1)
#randomArticleTitles()
print("COMPLETED!!")