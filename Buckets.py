from script import *

import pickle
import os.path

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



def generateBuckets(p):
    #initialisation of b buckets and store them in a list named buckets
    buckets = []
    for i in range(0, p.b):
        buck = Bucket(parameters=p)
        buckets.append(buck)   
    return buckets

def saveBucketsIntoFile(buckets, filename):
    with open(name=filename, mode='wb') as f:
        pickle.dump(buckets, f)    

#load buckets from file, if parameters are new, then we save a new file, before loading
def loadBucketsFromFile(filename, p):
    if os.path.isfile(filename):
        with open(name=filename, mode='rb') as f:
            buckets = pickle.load(f)
    else:
        buckets = generateBuckets(p)
        saveBucketsIntoFile(buckets, filename) 
    return buckets



if __name__ == '__main__':
    print ("Start of program")
    #Parameters of the program
    p = Parameters()      
    buckets = generateBuckets()
    filename = 'Buckets/buckets_' + str(p.b) + '_' + str(p.k) + '.pickle'
    saveBucketsIntoFile(buckets, filename)      
        
    