from script import *

if __name__ == '__main__':
    print ("Start of program")
    #variables for evaluating tuning parameters
    docRes = set()
    #Parameters of the program
    p = Parameters()    
    db = DBConnector()
    data = db.selectNews() 
    corpus = Corpus(data)
    candidates = Neighbors()
    
    for l in p.listLength:
        p.length = l;
        
        processing = Preprocessing(corpus.data, corpus.train, corpus.test, p.length)
        
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
        
        for n in p.listn:
            p.n = n;
            for b in p.listb:
                p.b = b;
                for k in p.listk:
                    p.k = k;
                    
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
                    
                    for T in p.listT:
                        p.T = T;
                        
                        #comparison with the treshold
                        if distance[0][0] <= p.T:
                            print "New story"
                            tupRes = distance, shortest
                            docRes.add(shortest.ID)
                            db.insertResults(p.T, p.n, p.b, p.k, p.length, shortest.ID)
                        else:
                            tupRes = "No new story found!"
                            db.insertResults(p.T, p.n, p.b, p.k, p.length, None)
                        
                        p.writeResults(tupRes)
                        
    
    #we treated each new document
    #db.updateRecord(corpus.test)
    db.closeConnection()
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