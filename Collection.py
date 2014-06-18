# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 20:54:23 2014

@author: maelrazavet
"""

import json
from pandas import *
from numpy import arange,array,ones,linalg
from pylab import plot,show
import matplotlib.pyplot as plt
import numpy as np
import csv
import scipy
import sklearn
from scipy.stats.stats import pearsonr
import feedparser
import urllib2
import datetime
import time
import MySQLdb

def writeInLogFile(str):
    i = datetime.datetime.now()
    filename = "log_%s_%s_%s" % (i.day, i.month, i.year)
    f = open(filename, 'a')
    f.write(datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S') + " << " +str+"\n")
    f.close()
    
def sendEmail(subject, msg):
    # Import smtplib for the actual sending function
    import smtplib
    
    try:
        writeInLogFile("msg")
        s=smtplib.SMTP_SSL()
        s.connect("smtp.gmail.com",465)
        s.login("mael.razavet@gmail.com", "dimk221090")
        
        s.sendmail("mael.razavett@gmail.com", "mael.razavett@gmail.com", msg)
        s.quit()
    except Exception,R:
            print R
    
def insertInDB(db, title, description, date, url, content):
    if db.find_one({"title": title}):
        writeInLogFile("Already in the database : %s" %(title))
    else:
        db.insert({
           "title": title, 
           "description": description,
           "date": date,
           "url": url,
           "content": content,
        })
    
                   
def launchDataCollection():
    try:
        db = MySQLdb.connect("localhost","root","root","maelrazabdd")
        # prepare a cursor object using cursor() method
        cursor = db.cursor()
            
        # Execute the SQL command
        cursor.execute("SELECT * FROM Companies")
        for company in cursor.fetchall():
            data = feedparser.parse('http://finance.yahoo.com/rss/headline?s=' + company[2])
            for item in data.entries:
                if "RSS feed not found" not in item.summary:
                    insertRecord(company[0], item.title.encode('utf-8'), item.summary.encode('utf-8'), item.published, item.link, "")
                    #insertInDB(db.news, item.title.encode('utf-8'), item.summary.encode('utf-8'), item.published, item.link, "")
    except Exception, e:
        print(e.message)

def insertRecord(company_ID, title, description, date, url, content):
    # Open database connection
    db = MySQLdb.connect("localhost","root","root","maelrazabdd")
    print(title)
    print(description)
    print(date)
    print(url)
    print(content)
    # prepare a cursor object using cursor() method
    cursor = db.cursor()
    
    # Execute the SQL command
    cursor.execute("SELECT * FROM News WHERE Title=%s", [title])
    
    # Fetch all the rows in a list of lists.
    if cursor.rowcount > 0:
        print('yep')
    else:
        print('test')
        # Prepare SQL query to INSERT a record into the database
        try:
               # Execute the SQL command
            cursor.execute("INSERT INTO News (Company_ID, Title, Date, Summary, Link, Content, Processed) values (%i,%s,%s,%s,%s,%s,%i)", (company_ID, title, date, description, url, content, 0))
             
            # Commit your changes in the database
            db.commit()
               
        except:
           # Rollback in case there is any error
            db.rollback()
           

        # disconnect from server
        db.close()

if __name__ == "__main__":
    print("Program launched")
    while True:
        print("Start of day")
        launchDataCollection()
        #writeInLogFile("Day 1 terminated")
        print("End of Day")
        time.sleep(86400) # 3600 seconds = 1 hour