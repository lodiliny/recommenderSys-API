import matplotlib.pyplot as plt
import preprocess
import numpy as np

def readDocTopicWeights():
    topicProportions = {}
    with open("/Users/Lydia/mallet/output/dec-doc-topic-weights.txt", 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            line = line.rstrip('\r')
            doc_weights = line.split('\t')
            docID = int(doc_weights[0])
            composition = []
            for i in range(2, len(doc_weights)):
                composition.append(float(doc_weights[i]))
            topicProportions[docID] = composition
            #print docID, composition
    return topicProportions

def docTopicThreshold():
    finalweights = {}
    docTopics = readDocTopicWeights()
    for doc in range(0, len(docTopics)):
        proportion = docTopics[doc]
        docweightthreshold = {}
        for topic in range(0, len(proportion)):
            weight = proportion[topic]
            if weight > 0.15:
                docweightthreshold[topic] = weight
        finalweights[doc] = docweightthreshold
    N = 2
    menMeans = (0.20, 0.35)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence
    p1 = plt.bar(ind, menMeans, width)
    plt.show()
    return finalweights