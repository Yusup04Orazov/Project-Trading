

import json
import time
import numpy as np
import robin_stocks.robinhood as rh
import matplotlib.pyplot as plt
import pandas as pd

# Login to Robinhood
login_response = rh.authentication.login(
    username="username",
    password="password", 
    expiresIn=86400,
    scope='internal',
    by_sms=True,
    store_session=True
)

# last_logged_prices = {symbol: 0 for symbol in ["AAPL", "GOOGL", "TSLA", "MSFT", "AMZN", "NVDA", "META"]}
last_logged_prices = {symbol: 0 for symbol in ["BTC", "ETH", "AVAX", "LTC"]}

# Initialize variables for each cryptocurrency
# crypto_symbols = ["AAPL", "GOOGL", "TSLA", "MSFT", "AMZN", "NVDA", "META"]
crypto_symbols = ["BTC", "ETH", "AVAX", "LTC"]


crypto_data = {
    symbol: {
        'price_change_count': 1,
        'prices': [],
        'moving_average_200': [],
        'macd': [],
        'signal_line': [],
        'first_derivative': [0],
        'second_derivative': [0],
        'last_logged_price': 0,
        'last_trade_action': None
    }
    for symbol in crypto_symbols
}

# Define the file paths to save the graphs
graph_file_paths = {symbol: f'{symbol}_currentPrice.png' for symbol in crypto_symbols}
macd_file_paths = {symbol: f'{symbol}_macd.png' for symbol in crypto_symbols}
first_derivative_file_paths = {symbol: f'{symbol}_first_derivative.png' for symbol in crypto_symbols}
derivative_file_paths = {symbol: f'{symbol}_derivatives.png' for symbol in crypto_symbols}

# Define the file path for trade signals
buy_sell_signals_file = 'trade.jsonl'

bought_price = 0
sold_price = 0

# Function to calculate moving averages
def calculate_moving_averages(prices, moving_average_list):
    if len(prices) >= 200:
        moving_average_list.append(round(np.mean(prices[-200:]), 2))

# Function to detect sideways movement
def detect_sideways_movement(prices):
    global sideways_window_size
    global sideways_epsilon
    sideways_window_size = 200
    sideways_epsilon = int(crypto_data[symbol]['rounded_price'])*0.002

    if len(prices) > sideways_window_size:
        moving_average = np.mean(prices[-sideways_window_size:])
        current_price = prices[-1]
        delta = current_price - moving_average
        # return abs(delta) < sideways_epsilon
        if abs(delta) < sideways_epsilon:
            return True
        
    return False

def calculate_macd(prices, macd_list, signal_line_list, threshold_multiplier=1.18):
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
    # macd_range = np.max(macd_values) - np.min(macd_values)
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


    

# Function to calculate the first and second derivatives
def calculate_derivatives(prices, first_derivative_list, second_derivative_list):
    if len(prices) > 1:
        # Calculate the first derivative (rate of change)
        first_derivative = np.gradient(prices)
        first_derivative_list.extend(first_derivative)

        # Calculate the second derivative
        second_derivative = np.gradient(first_derivative)
        second_derivative_list.extend(second_derivative)


# Function to generate buy signal
def generate_buy_signal(symbol, current_price, macd, signal_line):
    global bought_price
    global sold_price
    global buy_signal
    sideways_movement = detect_sideways_movement(crypto_data[symbol]['prices'])
    if (len(crypto_data[symbol]['moving_average_200']) > 1 and 
        current_price > crypto_data[symbol]['moving_average_200'][-1] and 
        crypto_data[symbol]['last_trade_action'] == None and 
        macd < 0 and signal_line < 0 and 
        macd > signal_line):

        # print(symbol, volatility)
        bought_price = current_price
        buy_signal = {
            'action': 'buy',
            'symbol': symbol,
            'bought_price': bought_price
        }
        with open(buy_sell_signals_file, 'a') as f:
            json.dump(buy_signal, f)
            f.write('\n')

        # Update the last trade action
        crypto_data[symbol]['last_trade_action'] = "buy"

    elif(len(crypto_data[symbol]['moving_average_200']) > 1 and 
        current_price > crypto_data[symbol]['moving_average_200'][-1] and 
        crypto_data[symbol]['last_trade_action'] == "sell" and 
        macd < 0 and signal_line < 0 and macd > signal_line):
        print(sideways_movement)
        
        bought_price = current_price
        buy_signal = {
            'action': 'buy',
            'symbol': symbol,
            'bought_price': bought_price
        }
        with open(buy_sell_signals_file, 'a') as f:
            json.dump(buy_signal, f)
            f.write('\n')

        # Update the last trade action
        crypto_data[symbol]['last_trade_action'] = "buy"

# Function to generate sell signal
def generate_sell_signal(symbol, current_price, macd, signal_line):
    global bought_price
    global sold_price

    if crypto_data[symbol]['last_trade_action'] == "buy":
        with open(buy_sell_signals_file, 'r') as f:
            buy_signals = [json.loads(line.strip()) for line in f]

        # Find the most recent buy signal for the symbol
        recent_buy_signal = next((buy_signal for buy_signal in reversed(buy_signals) if buy_signal['symbol'] == symbol), None)

        if recent_buy_signal:
            # Check conditions to generate a sell signal
            if (macd > 0 and signal_line > 0 and macd < signal_line) or (current_price <= 0.85 * recent_buy_signal['bought_price']) or (current_price >= 1.05 * recent_buy_signal['bought_price']):
    
                sold_price = current_price
                sell_signal = {
                    'action': 'sell',
                    'symbol': symbol,
                    'sold_price': sold_price
                }

                with open(buy_sell_signals_file, 'a') as f:
                    json.dump(sell_signal, f)
                    f.write('\n')

                print("sold!", symbol)

                crypto_data[symbol]['last_trade_action'] = "sell"

# Initialize the plot for derivatives
plt.figure(figsize=(12, 6))
plt.xlabel('Change Number')
plt.ylabel('Derivative Value')
plt.title('1st and 2nd Derivatives')

# Initialize a counter for the collected prices
price_count = 0

while True:
    for symbol in crypto_symbols:
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

        crypto_data[symbol]['rounded_price'] = round(latest_price, 2)
        last_logged_price = crypto_data[symbol]['last_logged_price']

        # Check to not save the same saved value before
        if crypto_data[symbol]['rounded_price'] != last_logged_price:

            crypto_data[symbol]['prices'].append(crypto_data[symbol]['rounded_price'])
            calculate_macd(crypto_data[symbol]['prices'], crypto_data[symbol]['macd'], crypto_data[symbol]['signal_line'])
            calculate_moving_averages(crypto_data[symbol]['prices'], crypto_data[symbol]['moving_average_200'])
            calculate_derivatives(crypto_data[symbol]['prices'], crypto_data[symbol]['first_derivative'], crypto_data[symbol]['second_derivative'])

            # Log data
            log_data = {
                'change_number': crypto_data[symbol]['price_change_count'],
                'current_price': crypto_data[symbol]['rounded_price'],
                'moving_average_200': crypto_data[symbol]['moving_average_200'][-1] if crypto_data[symbol]['moving_average_200'] else None,
                'macd': crypto_data[symbol]['macd'][-1],
                'signal_line': crypto_data[symbol]['signal_line'][-1],
                'first_derivative': crypto_data[symbol]['first_derivative'][-1] if crypto_data[symbol]['first_derivative'] else None,
                'second_derivative': crypto_data[symbol]['second_derivative'][-1] if crypto_data[symbol]['second_derivative'] else None,

            }

            with open(f'{symbol}_prices.jsonl', 'a') as f:
                json.dump(log_data, f)
                f.write('\n')

            crypto_data[symbol]['price_change_count'] += 1
            price_count += 1

            # Update the last_logged_price
            crypto_data[symbol]['last_logged_price'] = crypto_data[symbol]['rounded_price']

            # Generate buy and sell signals
            generate_buy_signal(symbol, log_data['current_price'], log_data['macd'], log_data['signal_line'])
            generate_sell_signal(symbol, log_data['current_price'], log_data['macd'], log_data['signal_line'])

        # Graphing for current price
        plt.clf()
        plt.plot(range(1, len(crypto_data[symbol]['prices']) + 1), crypto_data[symbol]['prices'], label='Price', marker='o')

        if crypto_data[symbol]['moving_average_200']:
            plt.plot(range(len(crypto_data[symbol]['prices']) - len(crypto_data[symbol]['moving_average_200']) + 1, len(crypto_data[symbol]['prices']) + 1),
                     crypto_data[symbol]['moving_average_200'], label='Moving Average (200)', marker='o')

        plt.legend()
        plt.draw()
        plt.pause(0.0000001)
        plt.savefig(graph_file_paths[symbol])

        # Graphing for MACD
        plt.clf()
        plt.plot(range(len(crypto_data[symbol]['prices']) - len(crypto_data[symbol]['macd']) + 1, len(crypto_data[symbol]['prices']) + 1),crypto_data[symbol]['macd'], label='MACD', marker='o')
        plt.axhline(y=0, color='k', linestyle='-')
        plt.plot(range(len(crypto_data[symbol]['prices']) - len(crypto_data[symbol]['signal_line']) + 1, len(crypto_data[symbol]['prices']) + 1),
                 crypto_data[symbol]['signal_line'], label='Signal Line', marker='o')

        plt.legend()
        plt.draw()
        plt.pause(0.0000001)
        plt.savefig(macd_file_paths[symbol])
        
        # Graphing for 2nd derivative
        plt.clf()
        plt.plot(range(len(crypto_data[symbol]['prices']) - len(crypto_data[symbol]['first_derivative']) + 1, len(crypto_data[symbol]['prices']) + 1),
                 crypto_data[symbol]['first_derivative'], label='1st Derivative', marker='o')
        
        plt.plot(range(len(crypto_data[symbol]['prices']) - len(crypto_data[symbol]['second_derivative']) + 1, len(crypto_data[symbol]['prices']) + 1),
                 crypto_data[symbol]['second_derivative'], label='2nd Derivative', marker='x')

        plt.legend()
        plt.draw()
        plt.pause(0.0000001)
        plt.savefig(derivative_file_paths[symbol])


        

        # Graphing for 1st derivative
        plt.clf()
        plt.plot(range(len(crypto_data[symbol]['prices']) - len(crypto_data[symbol]['first_derivative']) + 1, len(crypto_data[symbol]['prices']) + 1),
                 crypto_data[symbol]['first_derivative'], label='1st Derivative', marker='o')

        plt.legend()
        plt.draw()
        plt.pause(0.0000001)
        plt.savefig(first_derivative_file_paths[symbol])
