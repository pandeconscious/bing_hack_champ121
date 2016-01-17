#arg1 : all possible features
#arg2 : train data 
#arg3 : file to store good features

import math
import nltk
import sys
import csv
from fileinput import filename
from _collections import defaultdict

#returns -0.0 for all class labels of same type
#return -1.0 for empty set
def entropy(setLabels):
    if len(setLabels) == 0:
        return -1.0
    freqdist = nltk.FreqDist(setLabels)
    probs = [freqdist.freq(l) for l in freqdist]
    return -sum(p * math.log(p,2) for p in probs)

def informationGain(setBeforeSplit, setsAfterSplit):
    entropyBefore = entropy(setBeforeSplit)
    entropyAfter = 0.0
    for set in setsAfterSplit:
        currEntropy = entropy(set)
        currEntropy *= float(len(set))/float(len(setBeforeSplit))
        entropyAfter += currEntropy
    
    return entropyBefore - entropyAfter
 
#features is all the features
#data is a list of <string, labels> 

def getEntropyDocFreqAllFeatures(features, data, featureDocFreq):
    #featureWiseTopicLabels = {k: [] for k in features} #a dictionary of feature to the topic labels the feature is in
    featureWiseTopicLabels = defaultdict(list) #a dictionary of feature to the topic labels the feature is in
    for f in features:
        for row in data:
            if f in row[0]:
                featureWiseTopicLabels[f].append(row[1])
    
    average_labels_per_feature = 0.0
    
    featureWithEntropies = {}
    for k in featureWiseTopicLabels:
        currEntropy = entropy(featureWiseTopicLabels[k])
        featureWithEntropies[k] = currEntropy
        featureDocFreq[k] = len(featureWiseTopicLabels[k])
        average_labels_per_feature += len(featureWiseTopicLabels[k])
    
    average_labels_per_feature /= len(features)
    print "average labels per feature: " + str(average_labels_per_feature)
    
    return featureWithEntropies

    
        
def getAllLinesAsListFrom_file(fileName):
    l = []
    f = open(fileName, 'r')
    reader = csv.reader(f)
    for row in reader:
        l.append(row[0])
    f.close()
    return l

#data is a list of <string, labels> 
def getListStringLabels(fileName):
    data = []
    f = open(fileName, 'r')
    reader = csv.reader(f)
    for row in reader:
        currRow = [row[0], row[1]]
        data.append(currRow)
    f.close()
    return data

def printGoodFeaturesFile(allFeaturesEntropies, featureDocFreq, fileName, entropyThreshold, docFreqThreshold):
    f = open(fileName, 'w')
    for k, v in allFeaturesEntropies.iteritems():
        if v <= entropyThreshold and featureDocFreq[k] >= docFreqThreshold:
            f.write(k + '\n')
    f.close()
            
allFeatures =  getAllLinesAsListFrom_file("D:/Bing/all_uniq_words_in_titles_train.txt")#sys.argv[1]


data = getListStringLabels("D:/Bing/train_data_title_pub_year.csv")#sys.argv[2]

featureDocFreq = {}

allFeaturesEntropies =  getEntropyDocFreqAllFeatures(allFeatures, data, featureDocFreq)

#print featureDocFreq
printGoodFeaturesFile(allFeaturesEntropies, featureDocFreq, "Good_title_feature.txt", 2, 5)#sys.argv[3]

       

'''test code        
setBefore = ['m', 'm','m','m','m','m','m','m','m','f','f','f','f','f']
setRight = ['m', 'm', 'm', 'f', 'f', 'f', 'f']
setLeft = ['m', 'm', 'm', 'm', 'm', 'm', 'f']    
setsAfter = [setLeft, setRight]

print informationGain(setBefore, setsAfter)
'''