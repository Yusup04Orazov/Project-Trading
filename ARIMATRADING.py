import json
import time
import numpy as np
import robin_stocks.robinhood as rh
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from datetime import datetime  # Add this import at the top of your file

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import os

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Login to Robinhood
login_response = rh.authentication.login(
    username='y.orazov2018@gmail.com',
    password='Buzmeyin2004@', 
    expiresIn=86400,
    scope='internal',
    by_sms=True,
    store_session=True
)

# symbols = ["AAPL", "TSLA", "AMZN", "NVDA", "META"]
symbols = ["BTC", "ETH", "BNB", "LINK", "AVAX"]

data = {
    symbol: {
        'price_change_count': 1,
        'prices': [],
        'moving_average_200': [],
        'macd': [],
        'signal_line': [],
        'last_logged_price': 0,
        'last_trade_action': None,
        'last_saved_change_number': 0,
        'future_price': [],
        'price_direction': 0,
        'coins_bought': 0,
        'coins_sold': 0,
        'last_logged_ask_price': 0,
        'last_logged_bid_price': 0,
        'ask_price': 0,
        'bid_price': 0,
        'price_direction_counter':[]
    }
    for symbol in symbols
}

# Define the file paths to save the graphs
graph_file_paths = {symbol: f'{symbol}_currentPrice.png' for symbol in symbols}
macd_file_paths = {symbol: f'{symbol}_macd.png' for symbol in symbols}
currentPrice_file_paths = {symbol: f'{symbol}_prices.jsonl' for symbol in symbols}
predictedPrice_paths = {symbol: f'{symbol}_predictedPrices.png' for symbol in symbols}

# Define file paths for saving data
predictions_file_paths = {symbol: f'{symbol}_predictions.jsonl' for symbol in symbols}
analysis_file_paths = {symbol: f'{symbol}_analyze.jsonl' for symbol in symbols}

# Define the file path for trade signals
buy_sell_signals_file = 'trade.jsonl'

bought_price = 0
sold_price = 0
prediction_n = 200

investment = 25000
safety_investment = 25000 / 2
safety_investment_trade = 25000 * 0.1

def sell_fractional_shares(symbol, account_number=None, timeInForce='gfd', extendedHours=False, jsonify=True):
    global amountInDollars
    amountInDollars = 1
    """
    Submits a market order to sell fractional shares by specifying the dollar amount and saves the response to a file.

    Parameters:
        symbol (str): The stock ticker of the stock to sell.
        account_number (Optional[str]): Robinhood account number.
        timeInForce (str): How long the order will be in effect. 'gfd' = good for the day.
        extendedHours (bool): Trading during extended hours (premium users only).
        jsonify (bool): If True, returns a dictionary with order details; if False, returns the request object.

    Returns:
        dict: Information about the transaction.
    """
    response = rh.orders.order_sell_fractional_by_price(
        symbol,
        amountInDollars,
        account_number=account_number,
        timeInForce=timeInForce,
        extendedHours=extendedHours,
        jsonify=jsonify
    )
    
    with open('trade.jsonl', 'a') as file:
        json.dump(response, file)
        file.write('\n')
    
    return response


def buy_fractional_shares(symbol, account_number=None, timeInForce='gfd', extendedHours=False, jsonify=True):
    global amountInDollars
    amountInDollars = 1
    """
    Submits a market order to buy fractional shares by specifying the dollar amount and saves the response to a file.

    Parameters:
        symbol (str): The stock ticker of the stock to purchase.
        account_number (Optional[str]): Robinhood account number.
        timeInForce (str): How long the order will be in effect. 'gfd' = good for the day.
        extendedHours (bool): Trading during extended hours (premium users only).
        jsonify (bool): If True, returns a dictionary with order details; if False, returns the request object.

    Returns:
        dict: Information about the transaction.
    """
    response = rh.orders.order_buy_fractional_by_price(
        symbol,
        amountInDollars,
        account_number=account_number,
        timeInForce=timeInForce,
        extendedHours=extendedHours,
        jsonify=jsonify
    )
    
    with open('trade.jsonl', 'a') as file:
        json.dump(response, file)
        file.write('\n')
    
    return response

# def sell_fractional_shares(symbol, amountInDollars, account_number=None, timeInForce='gfd', extendedHours=False, jsonify=True):
    
#     """
#     Submits a market order to sell fractional shares by specifying the dollar amount.

#     Parameters:
#         symbol (str): The stock ticker of the stock to sell.
#         amountInDollars (float): The dollar amount of the fractional shares to sell.
#         account_number (Optional[str]): Robinhood account number.
#         timeInForce (str): How long the order will be in effect. 'gfd' = good for the day.
#         extendedHours (bool): Trading during extended hours (premium users only).
#         jsonify (bool): If True, returns a dictionary with order details; if False, returns the request object.

#     Returns:
#         dict: Information about the transaction.
#     """
#     return rh.orders.order_sell_fractional_by_price(
#         symbol,
#         amountInDollars,
#         account_number=account_number,
#         timeInForce=timeInForce,
#         extendedHours=extendedHours,
#         jsonify=jsonify
#     )


# def buy_fractional_shares(symbol, amountInDollars, account_number=None, timeInForce='gfd', extendedHours=False, jsonify=True):
#     """
#     Submits a market order to buy fractional shares by specifying the dollar amount.

#     Parameters:
#         symbol (str): The stock ticker of the stock to purchase.
#         amountInDollars (float): The dollar amount of the fractional shares to buy.
#         account_number (Optional[str]): Robinhood account number.
#         timeInForce (str): How long the order will be in effect. 'gfd' = good for the day.
#         extendedHours (bool): Trading during extended hours (premium users only).
#         jsonify (bool): If True, returns a dictionary with order details; if False, returns the request object.

#     Returns:
#         dict: Information about the transaction.
#     """
#     return rh.orders.order_buy_fractional_by_price(
#         symbol,
#         amountInDollars,
#         account_number=account_number,
#         timeInForce=timeInForce,
#         extendedHours=extendedHours,
#         jsonify=jsonify
#     )


def is_price_trend_linear(prices):
    if len(prices) >= 200:
        recent_prices = prices[-200:]
        
        # Prepare data for linear regression
        X = np.arange(0, 200).reshape(-1, 1)  # Independent variable: time or indices
        y = np.array(recent_prices)  # Dependent variable: prices
        
        # Fit linear regression model
        model = LinearRegression().fit(X, y)
        predictions = model.predict(X)
        
        # Calculate R^2 to evaluate the fit
        r_squared = r2_score(y, predictions)
        
        # Convert NumPy bool to Python bool before returning
        return bool(r_squared > 0.8)
    else:
        return False  # Not enough data to determine if the trend is linear

# Function to fit ARIMA model and make predictions
def fit_arima_and_predict(prices):

    # Take the natural logarithm of the prices to stabilize the variance
    log_prices = np.log(prices)
    # Use auto_arima from pmdarima to find the optimal ARIMA parameters
    auto_arima_results = pm.auto_arima(log_prices, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True, stationary=True, d=None)
    # Extract the p, d, q values
    p, d, q = auto_arima_results.order
    # Fit the ARIMA model using statsmodels with the parameters obtained from auto_arima
    arima_model = ARIMA(log_prices, order=(p, d, q))
    arima_result = arima_model.fit()
    # Make predictions on the log-transformed data
    log_predictions = arima_result.forecast(steps=prediction_n)
    # Reverse the log transformation to get predictions in the original scale
    predictions = np.exp(log_predictions)
    
    return predictions 

# Function to calculate profit/loss based on buy/sell signals
def check_total_profit_threshold():
    total_profit_loss_percentage = 0.0
    try:
        with open(buy_sell_signals_file, 'r') as file:
            for line in file:
                trade = json.loads(line.strip())
                if 'profit or loss %' in trade:
                    total_profit_loss_percentage += trade['profit or loss %']
                    # print(total_profit_loss_percentage)
    except FileNotFoundError:
        print("The trade.jsonl file does not exist. Creating an empty file.")
        with open(buy_sell_signals_file, 'w') as file:
            # Creates an empty file, or clears it if it somehow exists
            pass
        return False  # Since no file existed, we assume no history and thus can proceed with buying.

    return total_profit_loss_percentage >= 1.0


def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum()/period
    down = -seed[seed < 0].sum()/period

    if down == 0:
        rs = np.nan  # Avoid division by zero
    else:
        rs = up/down

    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100./(1.+rs)

    for i in range(period, len(prices)):
        delta = deltas[i-1]  # because the diff is 1 shorter
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(period-1) + upval)/period
        down = (down*(period-1) + downval)/period

        if down == 0:
            rs = np.nan  # Handle division by zero
        else:
            rs = up/down

        rsi[i] = 100. - 100./(1.+rs)

    return rsi

# Function to calculate moving averages
def calculate_moving_averages(prices, moving_average_list):
    if len(prices) >= 200:
        moving_average_list.append(round(np.mean(prices[-200:]), 2))

# Function to calculate MACD
def calculate_macd_and_adjust_for_sideways(prices, macd_list, signal_line_list, period_for_rsi=14, threshold_multiplier=0.5):
    prices_df = pd.DataFrame(prices, columns=['current_price'])
    prices_df['EMA_12'] = prices_df['current_price'].ewm(span=12, adjust=False).mean()
    prices_df['EMA_26'] = prices_df['current_price'].ewm(span=26, adjust=False).mean()
    
    prices_df['MACD'] = prices_df['EMA_12'] - prices_df['EMA_26']
    prices_df['Signal_Line'] = prices_df['MACD'].ewm(span=9, adjust=False).mean()

    # Calculate RSI
    rsi_values = calculate_rsi(prices_df['current_price'].values, period=period_for_rsi)
    prices_df['RSI'] = rsi_values

    # Round MACD and Signal_Line to nearest 3 decimals
    prices_df['MACD'] = prices_df['MACD'].round(3)
    prices_df['Signal_Line'] = prices_df['Signal_Line'].round(3)

    # Calculate the dynamic threshold based on the standard deviation of MACD values
    macd_std_dev = prices_df['MACD'].std()
    dynamic_threshold = macd_std_dev * threshold_multiplier

    # Check for sideways market conditions with dynamic threshold
    latest_macd = prices_df['MACD'].iloc[-1]
    latest_rsi = prices_df['RSI'].iloc[-1]
    if abs(latest_macd) < dynamic_threshold and 40 <= latest_rsi <= 60:
        adjusted_macd_value = 0.0
        adjusted_signal_line_value = 0.0
    else:
        adjusted_macd_value = latest_macd
        adjusted_signal_line_value = prices_df['Signal_Line'].iloc[-1]

    macd_list.append(adjusted_macd_value)
    signal_line_list.append(adjusted_signal_line_value)
    
    return prices_df

# make another if that checks for RSI, MACD and Moving average.
def generate_buy_signal(symbol, current_price, macd, signal_line):
    global investment
    global bought_price
    global sold_price
    global buy_signal

    # Calculate RSI
    prices = np.array(data[symbol]['prices'])
    rsi = calculate_rsi(prices)[-1]  # Get the most recent RSI value
    rsi_mean = np.mean(calculate_rsi(prices))

    # Check if the collective profit threshold has been met
    if check_total_profit_threshold():
        print("Collective profit threshold reached. Holding off on new buy signals.")
        return  # Exit the function if threshold is met
    
    elif(len(data[symbol]['moving_average_200']) > 0 and 
        current_price > data[symbol]['moving_average_200'][-1] and 
        data[symbol]['last_trade_action'] == None and 
        macd < 0 and signal_line < 0 and 
        macd > signal_line and 
        data[symbol]['price_direction'] == 1 and investment > (safety_investment)):

        # # Execute the buy on Robinhood
        # buy_response = buy_fractional_shares(symbol)
        # print(f"Sold {symbol}! Response: {buy_response}")
        
        investment_amount = safety_investment_trade / (len(symbols)) 

        bought_price = current_price
        data[symbol]['coins_bought'] = investment_amount / bought_price

        investment = investment - (data[symbol]['coins_bought'] * bought_price)
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Formats time as Year-Month-Day Hour:Minute:Second


        buy_signal = {
            'action': 'buy',
            'symbol': symbol,
            'bought_price': bought_price,
            'coins_bought': data[symbol]['coins_bought'],
            'portfolio': investment,
            'rsi': rsi,
            'current_time':current_time
        }

        with open(buy_sell_signals_file, 'a') as f:
            json.dump(buy_signal, f)
            f.write('\n')

        # Update the last trade action
        data[symbol]['last_trade_action'] = "buy"

    elif(len(data[symbol]['moving_average_200']) > 0 and 
        current_price> data[symbol]['moving_average_200'][-1] and 
        data[symbol]['last_trade_action'] == "sell" and 
        macd < 0 and signal_line < 0 and 
        macd > signal_line and 
        data[symbol]['price_direction'] == 1 and investment > (safety_investment)):

        # # Execute the buy on Robinhood
        # buy_response = buy_fractional_shares(symbol)
        # print(f"Sold {symbol}! Response: {buy_response}")
        
        bought_price = current_price

        investment_amount = safety_investment_trade / (len(symbols)) 

        data[symbol]['coins_bought'] = investment_amount / bought_price
        investment = investment - (data[symbol]['coins_bought'] * bought_price)
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Formats time as Year-Month-Day Hour:Minute:Second

        buy_signal = {
            'action': 'buy',
            'symbol': symbol,
            'bought_price': bought_price,
            'coins_bought': data[symbol]['coins_bought'],
            'portfolio': investment,
            'rsi':rsi,
            'current_time':current_time
        }
        with open(buy_sell_signals_file, 'a') as f:
            json.dump(buy_signal, f)
            f.write('\n')

        # Update the last trade action
        data[symbol]['last_trade_action'] = "buy"

def generate_sell_signal(symbol, current_price, macd, signal_line):
    global investment
    global sold_price
    global trade_result
    global profit_percent
    global sell_decision

    # Calculate RSI
    prices = np.array(data[symbol]['prices'])
    rsi = calculate_rsi(prices)[-1]  # Get the most recent RSI value

    if data[symbol]['last_trade_action'] == "buy":
        with open(buy_sell_signals_file, 'r') as f:
            buy_signals = [json.loads(line.strip()) for line in f]

        # Find the most recent buy signal for the symbol
        recent_buy_signal = next((buy_signal for buy_signal in reversed(buy_signals) if buy_signal['symbol'] == symbol), None)

        if recent_buy_signal:
            # Check conditions to generate a sell signal
            if (rsi > 60 and macd > 0 and signal_line > 0 and macd < signal_line) or (current_price >= (1.003 * recent_buy_signal['bought_price'])) or (current_price <= 0.98 * recent_buy_signal['bought_price']):
                sold_price = current_price
                data[symbol]['coins_sold'] = data[symbol]['coins_bought']      
                investment = investment + (data[symbol]['coins_sold'] * sold_price)
                trade_result = sold_price - recent_buy_signal['bought_price']
                
                # # Execute the sale on Robinhood
                # sell_response = sell_fractional_shares(symbol)
                # print(f"Sold {symbol}! Response: {sell_response}")

                if (rsi > 60 and macd > 0 and signal_line > 0 and macd < signal_line):
                    sell_decision = 1
                elif (current_price >= (1.003 * recent_buy_signal['bought_price'])):
                    sell_decision = 2
                else:
                    sell_decision = 3

                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Formats time as Year-Month-Day Hour:Minute:Second
                sell_signal = {
                    'action': 'sell',
                    'symbol': symbol,
                    'sold_price': sold_price,
                    'coins_sold': data[symbol]['coins_sold'],
                    'portfolio': investment,
                    'profit or loss %': (trade_result * 100 / recent_buy_signal['bought_price']),
                    'reason for selling' : (sell_decision),
                    'rsi':rsi,
                    'current_time':current_time
                }

                with open(buy_sell_signals_file, 'a') as f:
                    json.dump(sell_signal, f)
                    f.write('\n')

                print("sold!", symbol)

                data[symbol]['last_trade_action'] = "sell"

# Initialize the plot for derivatives
plt.figure(figsize=(12, 6))
plt.xlabel('Change Number')
plt.ylabel('Derivative Value')
plt.title('1st and 2nd Derivatives')

# Initialize a counter for the collected prices
price_count = 0

# # Call the main loop
# main_loop()

while True:
    for symbol in symbols:
        # For Crypto
        instrument = rh.crypto.get_crypto_quote(symbol)
        if instrument is not None and 'mark_price' in instrument:
            last_trade_price = float(instrument['mark_price'])
        else:
            print(f"Error getting price for {symbol}")
            time.sleep(0.1)
            continue

        # # For Stocks   .get_latest_price(inputSymbols, priceType=None, includeExtendedHours=True)
        # instrument = rh.stocks.get_stock_quote_by_symbol(symbol)
        # if instrument is not None and 'last_trade_price' and 'bid_size' and 'ask_size' and 'ask_price' and 'bid_price'in instrument:
        #     ask_price = float(instrument['ask_price'])
        #     bid_price = float(instrument['bid_price'])
        #     ask_size = float(instrument['ask_size'])
        #     bid_size = float(instrument['bid_size'])
        #     last_trade_price = float(instrument['bid_price'])
        # else:
        #     print(f"Error getting price for {symbol}")
        #     time.sleep(0.0000000000001)
        #     continue

        data[symbol]['rounded_price'] = round((last_trade_price), 2)

        last_logged_price = data[symbol]['last_logged_price']

        # Check to not save the same saved value before
        if data[symbol]['rounded_price'] != last_logged_price:
            #This line is to add +1 if current price is higher than previous and -1 else
            if data[symbol]['rounded_price'] > last_logged_price:
                data[symbol]['price_direction_counter'].append(1)
            elif data[symbol]['rounded_price'] < last_logged_price:
                data[symbol]['price_direction_counter'].append(-1)

            data[symbol]['prices'].append(data[symbol]['rounded_price'])
            # calculate_macd(data[symbol]['prices'], data[symbol]['macd'], data[symbol]['signal_line'])
            calculate_macd_and_adjust_for_sideways(data[symbol]['prices'], data[symbol]['macd'], data[symbol]['signal_line'])
            calculate_moving_averages(data[symbol]['prices'], data[symbol]['moving_average_200'])

            # Proceed if there's enough data to calculate RSI
            if len(data[symbol]['prices']) > 14:
                current_rsi = calculate_rsi(np.array(data[symbol]['prices']))[-1]  # Latest RSI value
            else:
                current_rsi = None  # Insufficient data for RSI calculation

            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Formats time as Year-Month-Day Hour:Minute:Second
            trend_is_linear = is_price_trend_linear(data[symbol]['prices'])

            # Log data
            log_data = {
                'change_number': data[symbol]['price_change_count'],
                'current_price': data[symbol]['rounded_price'],
                'moving_average_200': data[symbol]['moving_average_200'][-1] if data[symbol]['moving_average_200'] else None,
                'macd': data[symbol]['macd'][-1],
                'signal_line': data[symbol]['signal_line'][-1],
                'rsi': current_rsi,
                'trend_is_linear':trend_is_linear,
                'price_direction_counter':data[symbol]['price_direction_counter'],
                'logged_at': current_time
            }

            with open(currentPrice_file_paths[symbol], 'a') as f:
                json.dump(log_data, f)
                f.write('\n')

            data[symbol]['price_change_count'] += 1
            price_count += 1

            # Update the last_logged_price
            data[symbol]['last_logged_price'] = data[symbol]['rounded_price']

            # Perform ARIMA modeling and prediction
            if len(data[symbol]['prices']) >= 200:  # Adjust the window size as needed
                # prices_for_arima = data[symbol]['prices'][-200:]  # Adjust the window size as needed
                prices_for_arima = data[symbol]['prices']  # Adjust the window size as needed

                # Call fit_arima_and_predict with the computed values of p, d, and q
                predictions = fit_arima_and_predict(prices_for_arima)

                # Save new predictions to file
                with open(predictions_file_paths[symbol], 'a') as predictions_file:
                    for i, pred in enumerate(predictions):
                        if len(data[symbol]['prices']) + i + 1 > data[symbol]['last_saved_change_number']:
                            data[symbol]['future_price'].append(round(pred, 2))

                            if len(data[symbol]['future_price']) >= prediction_n:
                                mean_futurePrices = np.mean(data[symbol]['future_price'][-prediction_n])

                                if((mean_futurePrices) > data[symbol]['rounded_price']):
                                    data[symbol]['price_direction'] = 1
                                    print("(1 is positive and 0 is negative)")
                                    print("Price Direction is 1 (1 is positive and 0 is negative)")
                                else:
                                    data[symbol]['price_direction'] = 0
                                    print("(1 is positive and 0 is negative)")
                                    print("Price Direction is 0")

                            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Formats time as Year-Month-Day Hour:Minute:Second
                            prediction_data = {
                                'predicted_change_number': len(data[symbol]['prices']) + i + 1,
                                'future_price': round(pred, 2),
                                'direction':data[symbol]['price_direction'],
                                'current_time':current_time
                            }
                            json.dump(prediction_data, predictions_file)
                            predictions_file.write('\n')
                            data[symbol]['last_saved_change_number'] = len(data[symbol]['prices']) + i + 1

            # Generate buy and sell signals
            generate_buy_signal(symbol, log_data['current_price'], log_data['macd'], log_data['signal_line'])
            generate_sell_signal(symbol, log_data['current_price'], log_data['macd'], log_data['signal_line'])

            # Update cryptocurrency data
            if len(data[symbol]['prices']) > 200:
                data[symbol]['moving_average_200'].append(np.mean(data[symbol]['prices'][-200:]))


