# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 23:12:35 2014

@author: maelrazavet

"""

#Database
import MySQLdb

#Sklearn
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

#other
import numpy as np
from random import gauss
import random
import pandas
#import pylab as pl
import nltk
from nltk.corpus import stopwords
import sys
from time import time
from datetime import datetime
import string
from scipy.spatial import distance
from collections import deque

class DBConnector():
    def __init__(self):
        # Open database connection
        self.db = MySQLdb.connect("localhost","root","root","maelrazabdd");

    def insertRecord(self, title, description, date, url, content):
        # prepare a cursor object using cursor() method
        cursor = self.db.cursor()
        cursor.execute("select count(*) From News ")
        for reading in cursor.fetchall():
            print str(reading[0])

        # disconnect from server
        #self.db.close()

    def selectNews(self):
        t0 = time()
        data = []
        # prepare a cursor object using cursor() method
        cursor = self.db.cursor()
        cursor.execute("select * From News where length(Summary) > 0")

        for row in cursor.fetchall():
            d = Document(int(row[0]), int(row[1]), row[2], row[3], row[4], row[5], row[6], int(row[7]));
            data.append(d) 

        # disconnect from server
        #self.db.close()
        print("Computational time: %0.3fs" % float(time() - t0))
        return data
    
    def updateRecord(self, data):
        # prepare a cursor object using cursor() method
        cursor = self.db.cursor()
        for item in data:
            cursor.execute("update News set Processed = %s where ID = %s", ("1", str(item.ID)))
            
        self.db.close()

class Document():        
    def __init__(self, ID, company_ID, title, date, summary, link, content, processed):
            self.ID = ID
            self.Company_ID = company_ID
            self.Title = title
            self.Date = date
            self.Summary = summary
            self.Link = link
            self.Content = content  
            self.Processed = processed
            self.Vector = []    
        
class Corpus():
    def __init__(self, data):
        self.data = data
        self.train = []
        self.test = []
        self.splitTrainTest()
    
    def splitTrainTest(self):
        for item in self.data:
            if item.Processed == 1:
                self.train.append(item)
            else:
                self.test.append(item)
               
    def spliTrainTestRandomly(self):
        self.train, self.test = train_test_split(self.data, train_size=0.7, test_size=0.3)
        
    def searchDocumentById(self, id):
        res = None
        for item in self.data:
            if item.ID == id:
                res = item
                break
        return res

class Preprocessing():
    def __init__(self, data, train ,test, nbFeatures):
        self.data = data
        self.train = train
        self.test = test
        #as part of pre-processing, we remove the stopwords, punctuation as we analize the words only
        #max_features is in the parameters
        self.tfidf = TfidfVectorizer(tokenizer=self.tokenize, stop_words='english', analyzer='word', norm='l2', lowercase=True, decode_error='ignore', max_features=nbFeatures)        
        
    def tokenize(self, data):
        tokens = nltk.word_tokenize(data)        
        return tokens        

    #function that trains the tf idf function
    def trainTfIDF(self):     
        #extract only the summary
        l = [item.Summary for item in self.train if len(item.Summary) > 0]
        tfs_train = self.tfidf.fit_transform(l)
        return tfs_train.toarray(), list(item.ID for item in self.train if len(item.Summary) > 0)
    
    #funtion that applies the tf idf method previously trained
    def applyTfIDF(self):
        l = [item.Summary for item in self.test if len(item.Summary) > 0]        
        tfs_test = self.tfidf.transform(l)     
        return tfs_test.toarray(), list(item.ID for item in self.test if len(item.Summary) > 0)                 

class Bucket():
    def __init__(self, parameters):
        self.param = parameters #tuning parameters
        self.h = deque(maxlen=self.param.n)
        self.r = {} #dictionnary of the random vectors
        self.hashFunction() #initialisation of the hash function

    def randomVector(self):
        #a random vector is a value obtained with the Gaussian Distribution of mean 0 and variance 1
        lr = [] #vector of a fixed number of bits, which is the length of the vector spaces
        for i in range(0, self.param.length):
            lr.append(gauss(0, 1))
        return lr
    
    #definition of the hash function based on the random projection method
    def hashFunction(self):
        for i in range(0, self.param.k):
            self.r[str(i)] = self.randomVector()
      
    #get the hash value of a vector space in this bucket     
    def getHashValue(self, u):
        bits = []
        for i in range(0, self.param.k):
            if np.dot(self.r[str(i)], u) >= 0:
                bits.append(1)
            else:
                bits.append(0)
        return bits
    
    def printHashFunction(self):
        for item in self.r.iteritems():
            print ("Hash value: " + str(item))        

class Neighbors():
    def __init__(self):
        self.candidates = []
        
    def getShortestDistance(self, new):
        distance = 0
        shortest = None
        for item in self.candidates:
            #if the cosine similarity is closer to 1, then the candidate is closer
            if cosine_similarity(item.Vector, new.Vector) >= distance:
                distance = cosine_similarity(item.Vector, new.Vector)
                shortest = item 
        #return the similarity score and the item itself
        return distance, shortest
        
class Parameters():
    def __init__(self):    
        self.listb = [i for i in range(20, 100)]
        self.listT = [i for i in np.arange(0,0.5,0.01)]
        self.listk = [i for i in range(3, 40)]
        self.listn = [i for i in range(20, 100)]
        self.listLength = [i for i in range(200, 1000)]
        self.length = 200 #fixed vector size of documents
        self.b = 20 #number of buckets
        self.T = 0.3 #threshold to compare the nearest neighbor        
        self.k = 3 #k random vectors
        self.n = 20 #n number of stored documents with same hash in a bucket        

    def writeResults(self, results):
        t = datetime.now()
        filename = "./Logs/log_" + str(t.day) + "_" + str(t.month) + ".txt"
        file = open(filename, "w")   
        file.write("Parameters:\n")
        file.write("Length of documents (length):" + str(self.length) + "\n")
        file.write("Number of buckets (b):" + str(self.b) + "\n")
        file.write("Number of random vectors (k):" + str(self.k) + "\n")
        file.write("Number of stored documents (n):" + str(self.n) + "\n")
        file.write("Threshold (T):" + str(self.T) + "\n")
        file.write("\n\n")
        
        file.write("Results of the nearest neighbor if found:\n")
        if type(results) == tuple:
            file.write("Distance: " + str(results[0][0]) + "\n")
            file.write("Doc Id: " + str(results[1].ID) + "\n")
        else:
            file.write(results)
        
        file.write("\n\n")
        file.close()         

if __name__ == '__main__':
    print ("Start of program")
    #Parameters of the program
    p = Parameters()    
    db = DBConnector()
    data = db.selectNews() 
    corpus = Corpus(data)
    processing = Preprocessing(corpus.data, corpus.train, corpus.test, p.length)
    candidates = Neighbors()
    
    matrixTrain, Ids = processing.trainTfIDF()
    i = 0
    for item in Ids:
        d = corpus.searchDocumentById(item)
        d.Vector = matrixTrain[i]
        i += 1        
    #new document arriving
    matrixTest, Ids = processing.applyTfIDF()    
    i = 0
    for item in Ids:
        d = corpus.searchDocumentById(item)
        d.Vector = matrixTest[i]
        i += 1     
    
    #initialisation of b buckets and store them in a list named buckets
    buckets = []
    for i in range(0, p.b):
        buck = Bucket(parameters=p)
        buckets.append(buck)   
    #simulation -- initialisation of the buckets with n 
    for item in buckets:    
        for doc in corpus.train:
            tup = doc, item.getHashValue(doc.Vector)
            item.h.append(tup) 
    
    #Get the hash value for the new document and comparison
    for item in buckets:
        for hash in item.h:
            #compare hashes with the hash value of the bucket for v
            if hash[1] == item.getHashValue(corpus.test[0].Vector):
                #store the doc in the candidates list
                candidates.candidates.append(hash[0])
        #don't forget to add the new document's hash value to the queue's bucket
        tup = corpus.test[0], item.getHashValue(corpus.test[0].Vector)
        item.h.append(tup)    
        
    distance, shortest = candidates.getShortestDistance(corpus.test[0])
    print str(distance) + "\n" 
    
    #comparison with the treshold
    if distance[0][0] <= p.T:
        print "New story"
        tupRes = distance, shortest
    else:
        tupRes = "No new story found!"
    
    p.writeResults(tupRes)
    
    #we treated each new document
    #db.updateRecord(corpus.test)
        
    print ("End of program")
    
    
    ##n docs simulated and generated
    #documents = []
    #id = 1
    #for k in range(0, p.n):
        ##initialise a doc
        #u = [] #vector of a fixed number of bits, which is the length of the vector spaces
        #for i in range(0, p.length):
            #u.append(gauss(0, 1))   
        #d = Document(id, 0, "", "", "", "", "", u)
        #documents.append(d)
        #id+=1    
    #corpus = Corpus(documents)
        
        
    #new document arriving
    #v = []
    #for i in range(0, p.length):
        #v.append(gauss(0, 1))  
    #d = Document(id, 0, "", "", "", "", "", v)
    #corpus.corpus.append(d)        
            
    ##Get the hash value for the new document and comparison
    #for item in buckets:
        #for hash in item.h:
            ##compare hashes with the hash value of the bucket for v
            #if hash[1] == item.getHashValue(v):
                ##store the doc in the candidates list
                #candidates.candidates.append(hash[0])
        ##don't forget to add the new document's hash value to the queue's bucket
        #tup = v, item.getHashValue(v)
        #item.h.append(tup)    