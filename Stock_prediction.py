
from flask import Flask, render_template, request,session,logging,flash,url_for,redirect,jsonify,Response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import secrets
import json
import pandas as pd
import numpy as np
import pandas_datareader.data as web
from fbprophet import Prophet
import datetime
import numpy as np
from flask import Flask, render_template
from flask import request, redirect
from pathlib import Path
import os
import os.path
import csv
import config as config
import logging as log

from itertools import zip_longest

with open('config.json', 'r') as c:
    params = json.load(c)["params"]
# Define a flask app
local_server = True
app = Flask(__name__,template_folder='templates')

app.secret_key = 'super-secret-key'

if(local_server):
    app.config['SQLALCHEMY_DATABASE_URI'] = params['local_uri']
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = params['prod_uri']

db = SQLAlchemy(app)


class Register(db.Model):
    '''
    sno, name phone_num, msg, date, email
    '''
    rno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(20), nullable=False)
    password = db.Column(db.String(12), nullable=False)
    password2 = db.Column(db.String(120), nullable=False)


@app.route("/")
def home():
    return render_template('index1.html',params=params)


@app.route("/register", methods=['GET','POST'])
def register():
    if(request.method=='POST'):
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        password2 = request.form.get('password2')
        error=""
        avilable_email= Register.query.filter_by(email=email).first()
        print(avilable_email)
        if avilable_email:
            error="email is already exists"
            print(error)
        else:
            if (password==password2):
                entry = Register(name=name,email=email,password=password, password2=password2)
                db.session.add(entry)
                db.session.commit()
                return redirect(url_for('login'))
            else:
                flash("plz enter right password")
        return render_template('register.html', params=params, error=error)
    return render_template('register.html',params=params)


@app.route("/login",methods=['GET','POST'])
def login():
    if('email' in session and session['email']):
        return render_template('index.html',params=params)

    if (request.method== "POST"):
        email = request.form["email"]
        password = request.form["password"]
        
        login = Register.query.filter_by(email=email, password=password).first()
        print(login)
        if login is not None:
            session['email']=email
            return render_template('index.html',params=params)
        else:
            flash("plz enter right password")
    return render_template('login.html',params=params)




@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response
    
@app.route("/")
def first_page():
    """
    original_end = 175
    forecast_start = 200
    stock = "IBM"
    return render_template("plot.html", original = original_end, forecast = forecast_start, stock_tinker = stock)
    """
    tmp = Path("static/prophet.png")
    tmp_csv = Path("static/numbers.csv")
    if tmp.is_file():
        os.remove(tmp)
    if tmp_csv.is_file():
        os.remove(tmp_csv)
    return render_template("index.html", params=params)

#function to get stock data
def yahoo_stocks(symbol, start, end):
    print(start)
    return web.DataReader(symbol, 'yahoo', start, end)

def get_historical_stock_price(stock):
    print ("Getting historical stock prices for stock ", stock)
    
    #get 7 year stock data for Apple
    startDate = datetime.datetime(config.start_year, config.start_month, config.start_date)
    #date = datetime.datetime.now().date()
    #endDate = pd.to_datetime(date)
    endDate = datetime.datetime(config.end_year, config.end_month, config.end_date)
    stockData = yahoo_stocks(stock, startDate, endDate)
    return stockData


def get_historical_stock_price2(stock):
    print("Getting historical stock prices for stock ", stock)

    # get 7 year stock data for Apple
    startDate = datetime.datetime(2021, 5, 3)
    # date = datetime.datetime.now().date()
    # endDate = pd.to_datetime(date)
    endDate = datetime.datetime(2021, 5, 12)
    stockData = yahoo_stocks(stock, startDate, endDate)
    return stockData

@app.route("/plot" , methods = ['POST', 'GET'] )
def main():

    if request.method == 'POST':
        # link = "https://in.tradingview.com/symbols/NASDAQ-"+stock_tinker+"/technicals/"
        stock = request.form['companyname']

        df_whole = get_historical_stock_price(stock)
        df_whole.to_csv('static/yahoo_data.csv')


        #getting whole values

        df_whole2 = get_historical_stock_price2(stock)
        df3 = df_whole2.filter(['Close'])

        df3.to_csv("static/whole_values.csv")

        df = df_whole.filter(['Close'])
        # df.to_csv("static/predicted_values.csv")

        link = "https://in.tradingview.com/symbols/NASDAQ-"+stock+"/technicals/"
        finance = "https://in.tradingview.com/symbols/NASDAQ-"+stock+"/financials-overview/"
        df['ds'] = df.index
        #log transform the ‘Close’ variable to convert non-stationary data to stationary.
        df['y'] = np.log(df['Close'])
        print(df)
        original_end = df['Close'][-1]
        ns = "NASDAQ:"+stock
        model = Prophet()
        #
        # model.add(Dense(8, input_dim=1, activation='relu'))
        # model.add(Dense(1))
        # model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(df)

        #num_days = int(input("Enter no of days to predict stock price for: "))
        
        num_days = 11
        future = model.make_future_dataframe(periods=num_days)
        forecast = model.predict(future)
        
        print (forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        df.set_index('ds', inplace=True)
        forecast.set_index('ds', inplace=True)

        viz_df = df.join(forecast[['yhat', 'yhat_lower','yhat_upper']], how = 'outer')
        viz_df['yhat_scaled'] = np.exp(viz_df['yhat'])


        close_data = viz_df.Close
        forecasted_data = viz_df.yhat_scaled
        date = future['ds']
        forecast_start = forecasted_data[-num_days]


        d = [date, close_data, forecasted_data]
        export_data = zip_longest(*d, fillvalue = '')
        with open('static/numbers.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(("Date", "Actual", "Forecasted"))
            wr.writerows(export_data)
        myfile.close()


        """accuracy"""

        startDate = datetime.datetime(config.start_year, config.start_month, config.start_date)
        endDate = datetime.datetime(config.end_year, config.end_month, config.end_date + 7)

        tr = web.DataReader(stock, 'yahoo', startDate, endDate)

        tr.to_csv('static/acc.csv', index=False)

        tr2 = pd.read_csv('static/acc.csv')

        tr3 = pd.read_csv('static/numbers.csv')
        pred = tr3['Forecasted'][-8:-3]
        act = tr2['Close'][-5:]
        pred1 = pred.reset_index(drop=True)
        act1 = act.reset_index(drop=True)

        acc = []
        acc = (pred1 - act1) / act1
        acc = 100 - abs(acc * 100)
        accuracy = np.mean(acc)
        print(accuracy)


        return render_template("plot.html", original = round(original_end,2), forecast = round(forecast_start,2), stock_tinker = stock.upper(),link = link,finance = finance,stock = stock,ns =ns,accuracy=accuracy)
'''
if __name__ == "__main__":
    main()
'''
@app.route("/logout", methods = ['GET','POST'])
def logout():
    session.pop('email')
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
