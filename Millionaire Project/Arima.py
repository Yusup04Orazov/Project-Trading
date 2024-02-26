import json
import time
import numpy as np
import robin_stocks.robinhood as rh
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
import pandas as pd

# Login to Robinhood
login_response = rh.authentication.login(
    username="username",
    password="password", 
    expiresIn=86400,
    scope='internal',
    by_sms=True,
    store_session=True)


symbols = ["BTC", "ETH", "DOGE", "BNB"]

data = {
    symbol: {
        'price_change_count': 1,
        'prices': [],
        'moving_average_200': [],
        'macd': [],
        'signal_line': [],
        'last_logged_price': 0,
        'last_trade_action': None,
        'last_saved_change_number':0,
        'future_price': [],
        'price_direction': 0,
        'coins_bought': 0,
        'coins_sold':0
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
prediction_n = 35

investment = 25000

# Function to fit ARIMA model and make predictions
def fit_arima_and_predict(prices):
    # Remove NaN values from the input data
    prices = np.nan_to_num(prices)

    # Fit ARIMA model using auto_arima
    arima_model = pm.auto_arima(prices, seasonal=False, trace=True)
    
    # Make predictions
    predictions = arima_model.predict(n_periods=prediction_n)
    return predictions


# Function to calculate moving averages
def calculate_moving_averages(prices, moving_average_list):
    if len(prices) >= 200:
        moving_average_list.append(round(np.mean(prices[-200:]), 2))

# Function to calculate MACD
def calculate_macd(prices, macd_list, signal_line_list, threshold_multiplier=1.15):
    prices_df = pd.DataFrame(prices, columns=['current_price'])
    prices_df['EMA_12'] = prices_df['current_price'].ewm(span=12, adjust=False).mean()
    prices_df['EMA_26'] = prices_df['current_price'].ewm(span=26, adjust=False).mean()
    
    prices_df['MACD'] = prices_df['EMA_12'] - prices_df['EMA_26']
    prices_df['Signal_Line'] = prices_df['MACD'].ewm(span=9, adjust=False).mean()

    # Round MACD and Signal_Line to nearest 4 decimals
    prices_df['MACD'] = prices_df['MACD'].round(3)
    prices_df['Signal_Line'] = prices_df['Signal_Line'].round(3)

    macd_values = prices_df['MACD'].values
    # Calculate the range of MACD values
    macd_range = np.mean(macd_values)
    # Set threshold based on MACD range and multiplier
    threshold = threshold_multiplier * macd_range

    macd_value = prices_df['MACD'].iloc[-1]
    signal_line_value = prices_df['Signal_Line'].iloc[-1]

    # Set MACD and Signal_Line to 0 if abs(MACD) is less than the threshold
    if abs(macd_value) < threshold:
        macd_value = 0.0
        signal_line_value = 0.0
    macd_list.append(macd_value)
    signal_line_list.append(signal_line_value)
    
    return prices_df

# TODO FIX HOW PORTOFLIO IS UPDATED. IT'S WRONG
def generate_buy_signal(symbol, current_price, macd, signal_line):
    global investment
    global bought_price
    global sold_price
    global buy_signal

    if (len(data[symbol]['moving_average_200']) > 0 and 
        current_price > data[symbol]['moving_average_200'][-1] and 
        data[symbol]['last_trade_action'] == None and 
        macd < 0 and signal_line < 0 and 
        macd > signal_line and 
        data[symbol]['price_direction'] == 1 and investment >= (investment*0.5)):
        
        bought_price = current_price
        data[symbol]['coins_bought'] = (investment*0.1) / bought_price
        investment = investment - (data[symbol]['coins_bought'] * bought_price)

        buy_signal = {
            'action': 'buy',
            'symbol': symbol,
            'bought_price': bought_price,
            'coins_bought': data[symbol]['coins_bought'],
            'portfolio': investment
        }
        with open(buy_sell_signals_file, 'a') as f:
            json.dump(buy_signal, f)
            f.write('\n')

        # Update the last trade action
        data[symbol]['last_trade_action'] = "buy"

    elif(len(data[symbol]['moving_average_200']) > 0 and 
        current_price > data[symbol]['moving_average_200'][-1] and 
        data[symbol]['last_trade_action'] == "sell" and 
        macd < 0 and signal_line < 0 and 
        macd > signal_line and 
        data[symbol]['price_direction'] == 1 and investment >= (investment*0.5)):
        
        bought_price = current_price
        data[symbol]['coins_bought'] = (investment*0.1) / bought_price
        investment = investment - (data[symbol]['coins_bought'] * bought_price)
        
        buy_signal = {
            'action': 'buy',
            'symbol': symbol,
            'bought_price': bought_price,
            'coins_bought': data[symbol]['coins_bought'],
            'portfolio': investment
        }
        with open(buy_sell_signals_file, 'a') as f:
            json.dump(buy_signal, f)
            f.write('\n')

        # Update the last trade action
        data[symbol]['last_trade_action'] = "buy"

# TODO FIX HOW PORTOFLIO IS UPDATED. IT'S WRONG
# TODO Implement RSI for selling the stock
def generate_sell_signal(symbol, current_price, macd, signal_line):
    global investment
    global bought_price
    global sold_price

    if data[symbol]['last_trade_action'] == "buy":
        with open(buy_sell_signals_file, 'r') as f:
            buy_signals = [json.loads(line.strip()) for line in f]

        # Find the most recent buy signal for the symbol
        recent_buy_signal = next((buy_signal for buy_signal in reversed(buy_signals) if buy_signal['symbol'] == symbol), None)

        if recent_buy_signal:
            # Check conditions to generate a sell signal
            if (macd > 0 and signal_line > 0 and macd < signal_line) or (current_price <= 0.87 * recent_buy_signal['bought_price']) or (current_price >= (1.00054 * recent_buy_signal['bought_price'])):
                
                sold_price = current_price
                data[symbol]['coins_sold'] = data[symbol]['coins_bought']      
                investment = investment + (data[symbol]['coins_sold'] * sold_price)
                sell_signal = {
                    'action': 'sell',
                    'symbol': symbol,
                    'sold_price': sold_price,
                    'coins_sold': data[symbol]['coins_sold'],
                    'portfolio': investment
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

while True:
    for symbol in symbols:

        ## For Crypto
        instrument = rh.crypto.get_crypto_quote(symbol)
        if instrument is not None and 'mark_price' in instrument:
            latest_price = float(instrument['mark_price'])
        else:
            print(f"Error getting price for {symbol}")
            time.sleep(0.1)
            continue

        # # For Stocks
        # instrument = rh.stocks.get_stock_quote_by_symbol(symbol)
        # if instrument is not None and 'ask_price' in instrument:
        #     latest_price = float(instrument['ask_price'])
        # else:
        #     print(f"Error getting price for {symbol}")
        #     time.sleep(0.1)
        #     continue

        data[symbol]['rounded_price'] = round(latest_price, 2)
        last_logged_price = data[symbol]['last_logged_price']

        # Check to not save the same saved value before
        if data[symbol]['rounded_price'] != last_logged_price:

            data[symbol]['prices'].append(data[symbol]['rounded_price'])
            calculate_macd(data[symbol]['prices'], data[symbol]['macd'], data[symbol]['signal_line'])
            calculate_moving_averages(data[symbol]['prices'], data[symbol]['moving_average_200'])

            # Log data
            log_data = {
                'change_number': data[symbol]['price_change_count'],
                'current_price': data[symbol]['rounded_price'],
                'moving_average_200': data[symbol]['moving_average_200'][-1] if data[symbol]['moving_average_200'] else None,
                'macd': data[symbol]['macd'][-1],
                'signal_line': data[symbol]['signal_line'][-1],
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
                prices_for_arima = data[symbol]['prices'][-200:]  # Adjust the window size as needed

                # Call fit_arima_and_predict with the computed values of p, d, and q
                predictions = fit_arima_and_predict(prices_for_arima)

                # Save new predictions to file
                with open(predictions_file_paths[symbol], 'a') as predictions_file:
                    for i, pred in enumerate(predictions):
                        if len(data[symbol]['prices']) + i + 1 > data[symbol]['last_saved_change_number']:
                            data[symbol]['future_price'].append(round(pred, 2))

                            if len(data[symbol]['future_price']) >= prediction_n:
                                mean_futurePrices = np.mean(data[symbol]['future_price'][-prediction_n])

                                if(mean_futurePrices > data[symbol]['rounded_price']):
                                    data[symbol]['price_direction'] = 1
                                    print("(1 is positive and 0 is negative)")
                                    print("Price Direction is 1 (1 is positive and 0 is negative)")
                                else:
                                    data[symbol]['price_direction'] = 0
                                    print("(1 is positive and 0 is negative)")
                                    print("Price Direction is 0")

                            prediction_data = {
                                'predicted_change_number': len(data[symbol]['prices']) + i + 1,
                                'future_price': round(pred, 2),
                                'directoin':data[symbol]['price_direction']
                                # 'future_price': data[symbol]['future_price']
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

        # Graphing for current price
        plt.clf()
        plt.plot(range(1, len(data[symbol]['prices']) + 1), data[symbol]['prices'], label='Price', marker='o')

        if data[symbol]['moving_average_200']:
            plt.plot(range(len(data[symbol]['prices']) - len(data[symbol]['moving_average_200']) + 1, len(data[symbol]['prices']) + 1),
                     data[symbol]['moving_average_200'], label='Moving Average (200)', marker='o')

        plt.legend()
        plt.draw()
        plt.pause(0.0000001)
        plt.savefig(graph_file_paths[symbol])

        # Graphing for MACD
        plt.clf()
        plt.plot(range(len(data[symbol]['prices']) - len(data[symbol]['macd']) + 1, len(data[symbol]['prices']) + 1),data[symbol]['macd'], label='MACD', marker='o')
        plt.axhline(y=0, color='k', linestyle='-')
        plt.plot(range(len(data[symbol]['prices']) - len(data[symbol]['signal_line']) + 1, len(data[symbol]['prices']) + 1),
                 data[symbol]['signal_line'], label='Signal Line', marker='o')

        plt.legend()
        plt.draw()
        plt.pause(0.0000001)
        plt.savefig(macd_file_paths[symbol])

        # Graph for Predicted Values
        if(len(data[symbol]['future_price']) > 1):
            # Read predictions from the corresponding prediction file
            with open(predictions_file_paths[symbol], 'r') as predictions_file:
                prediction_data = [json.loads(line.strip()) for line in predictions_file]

            # Extract data for plotting
            predicted_change_numbers = [entry['predicted_change_number'] for entry in prediction_data]
            future_prices = [entry['future_price'] for entry in prediction_data]

            # Plot the data
            plt.clf()
            plt.plot(predicted_change_numbers, future_prices, label='Predicted Prices', marker = 'o')
            plt.xlabel('Predicted Change Number')
            plt.ylabel('Future Price')
            plt.title(f'Predicted Prices for {symbol}')
            plt.legend()
            plt.savefig(predictedPrice_paths[symbol])
