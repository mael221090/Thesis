import datetime
import matplotlib.pyplot as plt

import pandas as pd
import pandas.io.data
from pandas import Series, DataFrame
import time
import numpy as np

if __name__ == '__main__':
    print ("Start of program")
    today = datetime.datetime.now()
    sp = pd.io.data.get_data_yahoo('^GSPC', 
                                     start=datetime.datetime(2014, 6, 10), 
                                     end=datetime.datetime.today())
    
    data = pd.io.data.get_quote_yahoo('^GSPC')[['last', 'time']]
    print sp.tail()    
    
    t = [data['time'][0]]
    x = 0
    y = data['last'][0]
    fig=plt.figure(1)
    ax=fig.add_subplot(111)
    ax.set_xlim(0,10)
    ax.set_ylim(1941.0,1943.0)    
    line,=ax.plot(x,y,'ko-', label='Current')
    plt.legend()
    plt.title("S&P 500")  
    
    for i in range(10):
        print i
        newData = pd.io.data.get_quote_yahoo('^GSPC')[['last', 'time']]
        data.append(newData)
        t = np.concatenate((line.get_xdata(), [newData['time'][0]]))
        x = np.concatenate((line.get_xdata(),[i]))
        y = np.concatenate((line.get_ydata(),[newData['last'][0]]))  
        line.set_data(x, y)
        print y
        print t
        plt.pause(0.0001)
        
    plt.pause(100)  
         
        