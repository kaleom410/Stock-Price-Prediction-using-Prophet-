import pandas_datareader.data as web
import pandas as df
import numpy as np
# from .Stock_prediction import get_historical_stock_price as st
import datetime
import config as config
import numpy as np




startDate = datetime.datetime(config.start_year, config.start_month, config.start_date)
endDate = datetime.datetime(config.end_year, config.end_month, config.end_date+7)

pd = web.DataReader('googl', 'yahoo', startDate, endDate)

pd.to_csv('static/acc.csv',index= False)

pd2 = df.read_csv('static/acc.csv')

pd3 = df.read_csv('static/numbers.csv')
pred = pd3['Forecasted'][-8:-3]
act = pd2['Close'][-5:]
pred1 = pred.reset_index(drop=True)
act1 = act.reset_index(drop=True)

acc = []
acc = (pred1-act1)/act1
acc = 100 - abs(acc*100)

print(np.mean(acc))
