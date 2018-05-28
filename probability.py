def sumTopicWordWeights():
    #gets the topic word weights from the topic-word-weights file. For each of the topics, the function sums the weights of the words
    sumprobability = {}
    with open("/Users/Lydia/mallet/output/dec2-topic-word-weights.txt", 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            line = line.rstrip('\r')
            topic = line.split('\t')
            topicID = int(topic[0])
            topicWord = topic[1]
            wordWeight = float(topic[2])
            if topicID in sumprobability.keys():
                sumprobability[topicID] += wordWeight
            else: sumprobability[topicID] = wordWeight

    return sumprobability

def getTopicWordProbability():
    #gets the topic word probabilty from the diagnosis file
    topicWordDetail = {}
    with open("/Users/Lydia/mallet/output/dec2-diagnosis.txt", 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            line = line.rstrip('\r')
            content = line.split(' ')
            identifier = content[0]
            if identifier == "topicID":
                num = int(content[1])
                details = {}
            if identifier == "count":
                topword = content[20]
                prob = float(content[3])
                details[topword] = prob
            topicWordDetail[num] = details
    for k in topicWordDetail.keys():
        print(k, topicWordDetail[k])
    return topicWordDetail

def computeTopicWordProbability():
    #file = open("/Users/Lydia/Downloads/DesktopFiles/GMTCorpus/vocab2.txt", 'w')
    output = {}
    topic0 = {}
    topic1 = {}
    topic2 = {}
    topic3 = {}
    topic4 = {}
    topic5 = {}
    topic6 = {}
    topic7 = {}
    topic8 = {}
    topic9 = {}
    wordlist = {}
    sumTopicWeights = sumTopicWordWeights()
    with open("/Users/Lydia/mallet/output/dec2-topic-word-weights.txt", 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            line = line.rstrip('\r')
            topic = line.split('\t')
            topicID = int(topic[0])
            topicWord = topic[1]
            wordWeight = float(topic[2])
            if topicID == 0:
                topic0[topicWord] = wordWeight
            elif topicID == 1:
                topic1[topicWord] = wordWeight
                output[0] = topic0
            elif topicID == 2:
                topic2[topicWord] = wordWeight
                output[1] = topic1
            elif topicID == 3:
                topic3[topicWord] = wordWeight
                output[2] = topic2
            elif topicID == 4:
                topic4[topicWord] = wordWeight
                output[3] = topic3
            elif topicID == 5:
                topic5[topicWord] = wordWeight
                output[4] = topic4
            elif topicID == 6:
                topic6[topicWord] = wordWeight
                output[5] = topic5
            elif topicID == 7:
                topic7[topicWord] = wordWeight
                output[6] = topic6
            elif topicID == 8:
                topic8[topicWord] = wordWeight
                output[7] = topic7
            elif topicID == 9:
                topic9[topicWord] = wordWeight
                output[8] = topic8
            else: print(topicID, topicWord, wordWeight)
        output[9] = topic9
    #computes topic word weights by: weight(topici-wordj) divided by sum(weights(topici-wordj))
    for k in sumTopicWeights.keys():
        weight = sumTopicWeights[k]
        wordlist = output[k]
        #print('the wordlist is:', k, wordlist)
        #print "Previous Values:", "Topic:", k, wordlist
        for w in wordlist:
            value = wordlist[w]
            wordlist[w] = (value / weight)
        output[k] = wordlist
    #for key in wordlist.keys():
    #    file.write(str(key + "\n"))
    #file.close()

    #compute Wjk for indexing, starting by computing the sum of the topic-word weights for each word in the vocabulary
    topic_word_weights = {}
    for keys in wordlist:
        sum_beta_j = 0.0
        for j in range(0, 10):
            beta_j = output[j][keys]
            sum_beta_j += beta_j
        topic_word_weights[keys] = sum_beta_j

    for ks in topic_word_weights.keys():
        w_weight = topic_word_weights[ks]
        for l in range(0, 10):
            wjk = output[l][ks] / w_weight
            output[l][ks] = wjk

    return output
#computeTopicWordProbability()
#sumTopicWordWeights()