import datetime
import matplotlib.pyplot as plt

import pandas as pd
import pandas.io.data
from pandas import Series, DataFrame

if __name__ == '__main__':
    print ("Start of program")
    today = datetime.datetime.now()
    sp = pd.io.data.get_data_yahoo('^GSPC', 
                                     start=datetime.datetime(2014, 1, 1), 
                                     end=datetime.datetime(2014, 6, 15))
    print sp.tail()    
    sp['Close'].plot(label="Close")
    plt.legend()
    plt.title("S&P 500")
    plt.show()