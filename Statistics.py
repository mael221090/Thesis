from script import *

class Statistics():
    def __init__(self, db): 
        self.db = db;
        
    def getDistinctDocID(self):
        cursor = self.db.executeQuery("SELECT COUNT(DISTINCT News_ID) FROM Results")
        for i in range(cursor.rowcount):
                
            row = cursor.fetchone()
            print row[0]  


if __name__ == '__main__':
    print ("Start of program")
    db = DBConnector()
    stats = Statistics(db)
    stats.getDistinctDocID()    