__author__ = 'FAHAD'
import pandas as pd
from os.path import isfile, join
import numpy as np
import re
import csv
train = pd.read_csv('D:/Bing/BingHackathonTrainingData.txt', header = -1, delimiter = '\t')
test = pd.read_csv('D:/Bing/BingHackathonTestData.txt', header = -1, delimiter = '\t')

# Train data
train_summry_data = open( 'D:/Bing/summary_data_train.txt', 'r+')
train_summary = []
for line in train_summry_data:
    temp = line.split()
    train_summary.append(temp)

train_title_data = open('D:/Bing/title_data_train.txt', 'r+')
train_title = []
for line in train_title_data:
    temp = line.split()
    train_title.append(temp)

train_author_data = open('D:/Bing/author_data.txt')
train_author = []
for line in train_author_data:
    line = re.sub('[^0-9]', ' ', line)
    temp = line.split()
    train_author.append(temp)

# Loading Good feature for summary, title and author
summary = open( 'D:/Bing/Good_summary_feature.txt', 'r+')
good_summary_feature = []
for line in summary:
    good_summary_feature.append(','.join(line.split()))

title = open('D:/Bing/Good_title_feature.txt', 'r+')
good_title_feature = []
for line in title:
    good_title_feature.append(','.join(line.split()))

author = open('D:/Bing/authors_written_atleast_5_docs.txt', 'r+')
good_author_feature = []
for line in author:
    good_author_feature.append(','.join(line.split()))

# Testing data
# summary_test = open( 'D:/Bing/summary_data_test.txt', 'r+')
# valData = []
# for line in summary_test:
#     temp = line.split()
#     valData.append(temp)
#
# title = open('D:/Bing/title_data_test.txt', 'r+')
# valData_title = []
# for line in title:
#     temp = line.split()
#     valData_title.append(temp)
#
# author = open('D:/Bing/author_data_test.txt', 'r+')
# valData_author = []
# for line in author:
#     line = re.sub('[^0-9]', ' ', line)
#     temp = line.split()
#     valData_author.append(temp)


num_train = 4*len(train[1])/5
num_test = len(train[1])
trainData1 = train[0:num_train]
valData1 = train[num_train:num_test+1]

trainData = train_summary[0:num_train]
trainData_title = train_title[0:num_train]
trainData_author = train_author[0:num_train]

# Validation set (20 % traning data)
valData = train_summary[num_train:num_test+1]
valData_title = train_title[num_train:num_test+1]
valData_author = train_author[num_train:num_test+1]


def bow(words, good_features):
    feature_vec = np.zeros(len(good_features), dtype="int32")

    for word in words:
        if word in good_features:
            index = good_features.index(word)
            feature_vec[index] += 1
    return feature_vec

# Creating Bag of Word representation for summary, title and authors datasets
good_summary_matrix = np.zeros((len(trainData), len(good_summary_feature)), dtype="int32")
count = 0
for token in range(0, len(trainData)):
    good_summary_matrix[count] = bow(trainData[token], good_summary_feature)
    count =count + 1

good_title_matrix = np.zeros((len(trainData_title), len(good_title_feature)), dtype="int32")
count =0
for token in range(0, len(trainData_title)):
    good_title_matrix[count] = bow(trainData_title[token], good_title_feature)
    count  = count + 1

good_author_matrix = np.zeros((len(trainData_author), len(good_author_feature)), dtype="int32")
count = 0
for token in range(0, len(trainData_author)):
    good_author_matrix[count] = bow(trainData_author[token], good_author_feature)
    count = count + 1

summary_title_matrix = np.append(good_summary_matrix, good_title_matrix,1)
summary_title_author_matrix = np.append(summary_title_matrix, good_author_matrix, 1)
print 'train shape', summary_title_author_matrix.shape

# Bag of Word representation for test data(validation data)
good_summary_matrix_val = np.zeros((len(valData), len(good_summary_feature)), dtype="int32")
count=0
for token in valData:
    good_summary_matrix_val[count] = bow(token, good_summary_feature)
    count = count + 1

good_title_matrix_val = np.zeros((len(valData_title), len(good_title_feature)), dtype="int32")
count =0
for token in range(0, len(valData_title)):
    good_title_matrix_val[count] = bow(valData_title[token], good_title_feature)
    count  = count + 1

good_author_matrix_val = np.zeros((len(valData_author), len(good_author_feature)), dtype="int32")
count = 0
for token in range(0, len(valData_author)):
    good_author_matrix_val[count] = bow(valData_author[token], good_author_feature)
    count = count + 1

summary_title_val_matrix = np.append(good_summary_matrix_val, good_title_matrix_val, 1)
summary_title_author_val_matrix = np.append(summary_title_val_matrix, good_author_matrix_val, 1)
print 'test shape', summary_title_author_val_matrix.shape

# Training SVR
print "svr Training ..."

from sklearn.svm import SVR
reg = SVR(kernel='linear',C=.4, epsilon=0.5)
reg = reg.fit(summary_title_author_matrix, trainData1[2])

result = reg.predict(summary_title_author_val_matrix)
result = [round(elem) for elem in result]
result = ['%d' % elem for elem in result]
result = [int(x) for x in result]
output = pd.DataFrame(data={"id":valData1[0], "Year":valData1[2], "Year_predicted":result})
print output
# output = pd.DataFrame(data={ "Year":result, "record id":test[0]})
# print output
# output.to_csv('D:/Bing/Predicted_year.txt', index=False, header=False)
# print 'result', result

import math
sum = 0
for i in range(0, len(result)):
    sum = sum + math.pow((valData1[2][i+num_train]-result[i]),2)

mse  = sum/len(result)
print 'MSE=', mse,
