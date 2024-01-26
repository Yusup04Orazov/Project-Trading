

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
# last_logged_prices = {symbol: 0 for symbol in ["BTC", "ETH", "AVAX", "LTC", "LINK", "BNB", "BCH"]}
last_logged_prices = {symbol: 0 for symbol in ["BTC", "ETH", "AVAX", "LTC"]}

# Initialize variables for each cryptocurrency
# crypto_symbols = ["AAPL", "GOOGL", "TSLA", "MSFT", "AMZN", "NVDA", "META"]
# crypto_symbols = ["BTC", "ETH", "AVAX", "LTC", "LINK", "BNB", "BCH"]
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
# # first_derivative_file_paths = {symbol: f'{symbol}_first_derivative.png' for symbol in crypto_symbols}
# derivative_file_paths = {symbol: f'{symbol}_derivatives.png' for symbol in crypto_symbols}

# Define the file path for trade signals
buy_sell_signals_file = 'trade.jsonl'

bought_price = 0
sold_price = 0

# Function to calculate moving averages
def calculate_moving_averages(prices, moving_average_list):
    if len(prices) >= 200:
        moving_average_list.append(round(np.mean(prices[-200:]), 2))

# Initialize variables for sideways movement detection
# sideways_window_size = 20  # Adjust the window size as needed
# sideways_epsilon = 0.03  # Adjust the epsilon value as needed

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


# Function to calculate volatility for a given symbol
# def calculate_volatility(prices):
#     if len(prices) > 200:
#         prices_array = np.array(prices)
#         returns = np.log(prices_array[1:] / prices_array[:-1])
#         volatility = np.std(returns)

#         volatility_threshold = np.mean(volatility)
#         print(volatility, volatility_threshold)
        
#         if volatility > volatility_threshold:
#             return True
#     else:
#         return False


# Function to calculate MACD
# def calculate_macd(prices, macd_list, signal_line_list):
#     prices_df = pd.DataFrame(prices, columns=['current_price'])
#     prices_df['EMA_12'] = prices_df['current_price'].ewm(span=12, adjust=False).mean()
#     prices_df['EMA_26'] = prices_df['current_price'].ewm(span=26, adjust=False).mean()
    
#     prices_df['MACD'] = prices_df['EMA_12'] - prices_df['EMA_26']
#     prices_df['Signal_Line'] = prices_df['MACD'].ewm(span=9, adjust=False).mean()
    
#     macd_list.append(prices_df['MACD'].iloc[-1])
#     signal_line_list.append(prices_df['Signal_Line'].iloc[-1])
    
#     return prices_df

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

# def calculate_macd(prices):
#     prices_df = pd.DataFrame(prices, columns=['current_price'])
#     prices_df['EMA_12'] = prices_df['current_price'].ewm(span=12, adjust=False).mean()
#     prices_df['EMA_26'] = prices_df['current_price'].ewm(span=26, adjust=False).mean()
    
#     prices_df['MACD'] = prices_df['EMA_12'] - prices_df['EMA_26']
#     prices_df['Signal_Line'] = prices_df['MACD'].ewm(span=9, adjust=False).mean()
    
#     macd_value = prices_df['MACD'].iloc[-1]
#     signal_line_value = prices_df['Signal_Line'].iloc[-1]

#     # Set MACD and Signal_Line to 0 if abs(MACD) is less than 0.1
#     if abs(macd_value) < 0.1:
#         macd_value = 0.0
#         signal_line_value = 0.0
#     crypto_data[symbol]['macd'].append(macd_value)
#     # macd.append(macd_value)
#     crypto_data[symbol]['signal_line'].append(signal_line_value)
#     # signal_line.append(signal_line_value)
    
#     return prices_df
# Function to calculate MACD
# def calculate_macd(prices):
#     prices_df = pd.DataFrame(prices, columns=['current_price'])
#     prices_df['EMA_12'] = prices_df['current_price'].ewm(span=12, adjust=False).mean()
#     prices_df['EMA_26'] = prices_df['current_price'].ewm(span=26, adjust=False).mean()
    
#     prices_df['MACD'] = prices_df['EMA_12'] - prices_df['EMA_26']
#     prices_df['Signal_Line'] = prices_df['MACD'].ewm(span=9, adjust=False).mean()
    
#     macd_values = prices_df['MACD'].values

#     # Calculate the range of MACD values
#     macd_range = np.max(macd_values) - np.min(macd_values)

#     # Set threshold based on MACD range
#     threshold = macd_range * 0.8

#     macd_value = prices_df['MACD'].iloc[-1]
#     signal_line_value = prices_df['Signal_Line'].iloc[-1]

#     # Set MACD and Signal_Line to 0 if abs(MACD) is less than the threshold
#     if abs(macd_value) <= threshold:
#         macd_value = 0.0
#         signal_line_value = 0.0

#     crypto_data[symbol]['macd'].append(macd_value)
#     crypto_data[symbol]['signal_line'].append(signal_line_value)
    
#     return prices_df

    

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
    # global first_summation_int
    # global second_summation_int
    # global first_derivative_int
    # global second_derivative_int
    # volatility = calculate_volatility(crypto_data[symbol]['prices'])
    sideways_movement = detect_sideways_movement(crypto_data[symbol]['prices'])

    # first_derivative_int = crypto_data[symbol]['first_derivative'][-1]
    # second_derivative_int = crypto_data[symbol]['second_derivative'][-1]
    # first_summation_int = sum(crypto_data[symbol]['first_derivative'])/len(crypto_data[symbol]['first_derivative'])
    # second_summation_int = sum(crypto_data[symbol]['second_derivative'])/len(crypto_data[symbol]['second_derivative'])

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
            
        # print("bought!", symbol, volatility)

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
        
        # # Graphing for 2nd derivative
        # plt.clf()
        # plt.plot(range(len(crypto_data[symbol]['prices']) - len(crypto_data[symbol]['first_derivative']) + 1, len(crypto_data[symbol]['prices']) + 1),
        #          crypto_data[symbol]['first_derivative'], label='1st Derivative', marker='o')
        
        # plt.plot(range(len(crypto_data[symbol]['prices']) - len(crypto_data[symbol]['second_derivative']) + 1, len(crypto_data[symbol]['prices']) + 1),
        #          crypto_data[symbol]['second_derivative'], label='2nd Derivative', marker='x')

        # plt.legend()
        # plt.draw()
        # plt.pause(0.0000001)
        # plt.savefig(derivative_file_paths[symbol])


        

        # Graphing for 1st derivative
        # plt.clf()
        # plt.plot(range(len(crypto_data[symbol]['prices']) - len(crypto_data[symbol]['first_derivative']) + 1, len(crypto_data[symbol]['prices']) + 1),
        #          crypto_data[symbol]['first_derivative'], label='1st Derivative', marker='o')

        # plt.legend()
        # plt.draw()
        # plt.pause(0.0000001)
        # plt.savefig(first_derivative_file_paths[symbol])






























# import json
# import time
# import numpy as np
# import robin_stocks.robinhood as rh
# import matplotlib.pyplot as plt
# import pandas as pd

# # Login to Robinhood
# login_response = rh.authentication.login(
#     username="y.orazov2018@gmail.com",
#     password="Turkmenistan123@", 
#     expiresIn=86400,
#     scope='internal',
#     by_sms=True,
#     store_session=True
# )

# # last_logged_prices = {symbol: 0 for symbol in ["AAPL", "GOOGL", "TSLA", "MSFT", "AMZN", "NVDA", "META", "JNJ", "UNH"]}
# last_logged_prices = {symbol: 0 for symbol in ["BTC", "ETH", "AVAX", "LTC", "LINK", "BNB", "BCH"]}

# # Initialize variables for each cryptocurrency
# # crypto_symbols = ["AAPL", "GOOGL", "TSLA", "MSFT", "AMZN", "NVDA", "META", "JNJ", "UNH"]
# crypto_symbols = ["BTC", "ETH", "AVAX", "LTC", "LINK", "BNB", "BCH"]

# crypto_data = {
#     symbol: {
#         'price_change_count': 1,
#         'prices': [],
#         'moving_average_200': [],
#         'macd': [],
#         'signal_line': [],
#         'first_derivative': [],
#         'second_derivative': [],
#         'last_logged_price': 0,
#         'last_trade_action': None
#     }
#     for symbol in crypto_symbols
# }

# # Define the file paths to save the graphs
# graph_file_paths = {symbol: f'{symbol}_currentPrice.png' for symbol in crypto_symbols}
# macd_file_paths = {symbol: f'{symbol}_macd.png' for symbol in crypto_symbols}

# # Define the file path for trade signals
# buy_sell_signals_file = 'trade.jsonl'

# bought_price = 0
# sold_price = 0
# buy_que = True

# # Function to calculate moving averages
# def calculate_moving_averages(prices, moving_average_list):
#     if len(prices) >= 200:
#         moving_average_list.append(round(np.mean(prices[-200:]), 2))

# # Function to calculate MACD
# def calculate_macd(prices, macd_list, signal_line_list):
#     prices_df = pd.DataFrame(prices, columns=['current_price'])
#     prices_df['EMA_12'] = prices_df['current_price'].ewm(span=12, adjust=False).mean()
#     prices_df['EMA_26'] = prices_df['current_price'].ewm(span=26, adjust=False).mean()
    
#     prices_df['MACD'] = prices_df['EMA_12'] - prices_df['EMA_26']
#     prices_df['Signal_Line'] = prices_df['MACD'].ewm(span=9, adjust=False).mean()
    
#     macd_list.append(prices_df['MACD'].iloc[-1])
#     signal_line_list.append(prices_df['Signal_Line'].iloc[-1])
    
#     return prices_df

# # Function to calculate the first and second derivatives
# def calculate_derivatives(prices, first_derivative_list, second_derivative_list):
#     if len(prices) >= 2:

#         # Calculate the first derivative (rate of change)
#         first_derivative = np.gradient(prices)
#         first_derivative_list.extend(first_derivative)

#         # Calculate the second derivative
#         second_derivative = np.gradient(first_derivative)
#         second_derivative_list.extend(second_derivative)


# # Function to generate buy signal
# def generate_buy_signal(symbol, current_price, macd, signal_line):
#     global bought_price
#     global sold_price

#     if (len(crypto_data[symbol]['moving_average_200']) > 1 and 
#         current_price > crypto_data[symbol]['moving_average_200'][-1] and 
#         crypto_data[symbol]['last_trade_action'] == None and 
#         macd < 0 and signal_line < 0 and macd > signal_line):
        
#         bought_price = current_price
#         buy_signal = {
#             'action': 'buy',
#             'symbol': symbol,
#             'bought_price': bought_price
#         }
#         with open(buy_sell_signals_file, 'a') as f:
#             json.dump(buy_signal, f)
#             f.write('\n')
            
#         print("bought!", symbol)

#         # Update the last trade action
#         crypto_data[symbol]['last_trade_action'] = "buy"

#     elif (len(crypto_data[symbol]['moving_average_200']) > 1 and 
#           current_price > crypto_data[symbol]['moving_average_200'][-1] and 
#           crypto_data[symbol]['last_trade_action'] == "sell" and 
#           macd < 0 and signal_line < 0 and macd > signal_line):
        
#         bought_price = current_price
#         buy_signal = {
#             'action': 'buy',
#             'symbol': symbol,
#             'bought_price': bought_price
#         }
#         with open(buy_sell_signals_file, 'a') as f:
#             json.dump(buy_signal, f)
#             f.write('\n')
            
#         print("bought!", symbol)

#         # Update the last trade action
#         crypto_data[symbol]['last_trade_action'] = "buy"

# # Function to generate sell signal
# def generate_sell_signal(symbol, current_price, macd, signal_line):
#     global bought_price
#     global sold_price

#     if crypto_data[symbol]['last_trade_action'] == "buy":
#         with open(buy_sell_signals_file, 'r') as f:
#             buy_signals = [json.loads(line.strip()) for line in f]

#         # Find the most recent buy signal for the symbol
#         recent_buy_signal = next((buy_signal for buy_signal in reversed(buy_signals) if buy_signal['symbol'] == symbol), None)

#         if recent_buy_signal:
#             # Check conditions to generate a sell signal
#             if (macd > 0 and signal_line > 0 and macd < signal_line) or (current_price <= 0.9 * recent_buy_signal['bought_price'] or current_price >= 1.05 * recent_buy_signal['bought_price']):
    
#                 sold_price = current_price
#                 sell_signal = {
#                     'action': 'sell',
#                     'symbol': symbol,
#                     'sold_price': sold_price
#                 }

#                 with open(buy_sell_signals_file, 'a') as f:
#                     json.dump(sell_signal, f)
#                     f.write('\n')

#                 print("sold!", symbol)

#                 crypto_data[symbol]['last_trade_action'] = "sell"

# # Initialize the plot
# plt.figure(figsize=(12, 6))
# plt.xlabel('Change Number')
# plt.ylabel('Price')
# plt.title('Price vs. Predicted Price')

# # Initialize a counter for the collected prices
# price_count = 0

# while True:
#     for symbol in crypto_symbols:
        
#         # Get data for each cryptocurrency
        # instrument = rh.crypto.get_crypto_quote(symbol)
        # if instrument is not None and 'mark_price' in instrument:
        #     latest_price = float(instrument['mark_price'])
        # else:
        #     print(f"Error getting price for {symbol}")
        #     time.sleep(0.1)
        #     continue

#         crypto_data[symbol]['rounded_price'] = round(latest_price, 2)
#         last_logged_price = crypto_data[symbol]['last_logged_price']

#         # Check to not save the same saved value before
#         if crypto_data[symbol]['rounded_price'] != last_logged_price:

#             crypto_data[symbol]['prices'].append(crypto_data[symbol]['rounded_price'])
#             calculate_macd(crypto_data[symbol]['prices'], crypto_data[symbol]['macd'], crypto_data[symbol]['signal_line'])
#             calculate_moving_averages(crypto_data[symbol]['prices'], crypto_data[symbol]['moving_average_200'])
#             calculate_derivatives(crypto_data[symbol]['prices'], crypto_data[symbol]['first_derivative'], crypto_data[symbol]['second_derivative'])

#             # Log data
#             log_data = {
#                 'change_number': crypto_data[symbol]['price_change_count'],
#                 'current_price': crypto_data[symbol]['rounded_price'],
#                 'moving_average_200': crypto_data[symbol]['moving_average_200'][-1] if crypto_data[symbol]['moving_average_200'] else None,
#                 'macd': crypto_data[symbol]['macd'][-1],
#                 'signal_line': crypto_data[symbol]['signal_line'][-1],
#                 'first_derivative': crypto_data[symbol]['first_derivative'][-1] if crypto_data[symbol]['first_derivative'] else None,
#                 'second_derivative': crypto_data[symbol]['second_derivative'][-1] if crypto_data[symbol]['second_derivative'] else None,

#             }

#             with open(f'{symbol}_prices.jsonl', 'a') as f:
#                 json.dump(log_data, f)
#                 f.write('\n')

#             crypto_data[symbol]['price_change_count'] += 1
#             price_count += 1

#             # Update the last_logged_price
#             crypto_data[symbol]['last_logged_price'] = crypto_data[symbol]['rounded_price']

#             # Generate buy and sell signals
#             generate_buy_signal(symbol, log_data['current_price'], log_data['macd'], log_data['signal_line'])
#             generate_sell_signal(symbol, log_data['current_price'], log_data['macd'], log_data['signal_line'])

#         # Graphing start
#         plt.clf()
#         plt.plot(range(1, len(crypto_data[symbol]['prices']) + 1), crypto_data[symbol]['prices'], label='Price', marker='o')

#         if crypto_data[symbol]['moving_average_200']:
#             plt.plot(range(len(crypto_data[symbol]['prices']) - len(crypto_data[symbol]['moving_average_200']) + 1, len(crypto_data[symbol]['prices']) + 1),
#                      crypto_data[symbol]['moving_average_200'], label='Moving Average (200)', marker='o')

#         plt.legend()
#         plt.draw()
#         plt.pause(0.0000001)
#         plt.savefig(graph_file_paths[symbol])

#         # Graphing start
#         plt.clf()
#         plt.plot(range(len(crypto_data[symbol]['prices']) - len(crypto_data[symbol]['macd']) + 1, len(crypto_data[symbol]['prices']) + 1),
#                  crypto_data[symbol]['macd'], label='MACD', marker='o')
#         plt.axhline(y=0, color='k', linestyle='-')
#         plt.plot(range(len(crypto_data[symbol]['prices']) - len(crypto_data[symbol]['signal_line']) + 1, len(crypto_data[symbol]['prices']) + 1),
#                  crypto_data[symbol]['signal_line'], label='Signal Line', marker='o')

#         plt.legend()
#         plt.draw()
#         plt.pause(0.0000001)
#         plt.savefig(macd_file_paths[symbol])





























# import json
# import time
# import numpy as np
# import robin_stocks.robinhood as rh
# import matplotlib.pyplot as plt
# import pandas as pd

# # Login to Robinhood
# login_response = rh.authentication.login(
#     username="y.orazov2018@gmail.com",
#     password="Turkmenistan123@", 
#     expiresIn=86400,
#     scope='internal',
#     by_sms=True,
#     store_session=True
# )

# # last_logged_prices = {symbol: 0 for symbol in ["AAPL", "GOOGL", "TSLA", "MSFT", "AMZN", "NVDA", "META", "JNJ", "UNH"]}
# last_logged_prices = {symbol: 0 for symbol in ["BTC", "ETH", "AVAX", "LTC", "LINK", "BNB", "BCH"]}

# # Initialize variables for each cryptocurrency
# # crypto_symbols = ["AAPL", "GOOGL", "TSLA", "MSFT", "AMZN", "NVDA", "META", "JNJ", "UNH"]
# crypto_symbols = ["BTC", "ETH", "AVAX", "LTC", "LINK", "BNB", "BCH"]

# crypto_data = {symbol: {'price_change_count': 1, 'prices': [], 'moving_average_200': [], 'macd': [], 'signal_line': [], 'last_logged_price': 0, 'last_trade_action': None} for symbol in crypto_symbols}

# # # Define the file paths to save the graphs
# graph_file_paths = {symbol: f'{symbol}_currentPrice.png' for symbol in crypto_symbols}
# macd_file_paths = {symbol: f'{symbol}_macd.png' for symbol in crypto_symbols}

# # Define the file path for trade signals
# buy_sell_signals_file = 'trade.jsonl'

# bought_price = 0
# sold_price = 0
# buy_que = True
# # Function to calculate moving averages
# def calculate_moving_averages(prices, moving_average_list):
#     if len(prices) >= 200:
#         moving_average_list.append(round(np.mean(prices[-200:]), 2))


# # Function to calculate MACD
# def calculate_macd(prices, macd_list, signal_line_list):
#     prices_df = pd.DataFrame(prices, columns=['current_price'])
#     prices_df['EMA_12'] = prices_df['current_price'].ewm(span=12, adjust=False).mean()
#     prices_df['EMA_26'] = prices_df['current_price'].ewm(span=26, adjust=False).mean()
    
#     prices_df['MACD'] = prices_df['EMA_12'] - prices_df['EMA_26']
#     prices_df['Signal_Line'] = prices_df['MACD'].ewm(span=9, adjust=False).mean()
    
#     macd_list.append(prices_df['MACD'].iloc[-1])
#     signal_line_list.append(prices_df['Signal_Line'].iloc[-1])
    
#     return prices_df


# # Function to generate buy signal
# def generate_buy_signal(symbol, current_price, macd, signal_line):
#     global bought_price
#     global sold_price

#     # if len(crypto_data[symbol]['moving_average_200']) > 1:

#         # if (crypto_data[symbol]['last_trade_action'] == 'buy' and current_price > crypto_data[symbol]['moving_average_200'][-1] and macd < 0 and signal_line < 0 and macd > signal_line):
#         # if (crypto_data[symbol]['last_trade_action'] == 'buy' and macd < 0 and signal_line < 0 and macd > signal_line):
#     if (len(crypto_data[symbol]['moving_average_200']) > 1 and current_price > crypto_data[symbol]['moving_average_200'][-1] and crypto_data[symbol]['last_trade_action'] == None and macd < 0 and signal_line < 0 and macd > signal_line ):
#     # if (crypto_data[symbol]['last_trade_action'] == None and macd < 0 and signal_line < 0 and macd > signal_line):

#         bought_price = current_price
#         buy_signal = {
#             'action': 'buy',
#             'symbol': symbol,
#             'bought_price': bought_price
#         }
#         with open(buy_sell_signals_file, 'a') as f:
#             json.dump(buy_signal, f)
#             f.write('\n')
            
#         print("bought!", symbol)

                
#         # Update the last trade action
#         crypto_data[symbol]['last_trade_action'] = "buy"
        

#     elif (len(crypto_data[symbol]['moving_average_200']) > 1 and current_price > crypto_data[symbol]['moving_average_200'][-1] and crypto_data[symbol]['last_trade_action'] == "sell" and macd < 0 and signal_line < 0 and macd > signal_line):
#     # elif (crypto_data[symbol]['last_trade_action'] == "sell" and macd < 0 and signal_line < 0 and macd > signal_line):
    
#         bought_price = current_price
#         buy_signal = {
#             'action': 'buy',
#             'symbol': symbol,
#             'bought_price': bought_price
#         }
#         with open(buy_sell_signals_file, 'a') as f:
#             json.dump(buy_signal, f)
#             f.write('\n')
            
#         print("bought!", symbol)

                
#         # Update the last trade action
#         crypto_data[symbol]['last_trade_action'] = "buy"



# def generate_sell_signal(symbol, current_price, macd, signal_line):
#     global bought_price
#     global sold_price

#     if crypto_data[symbol]['last_trade_action'] == "buy":
#         with open(buy_sell_signals_file, 'r') as f:
#             buy_signals = [json.loads(line.strip()) for line in f]

#         # Find the most recent buy signal for the symbol
#         recent_buy_signal = next((buy_signal for buy_signal in reversed(buy_signals) if buy_signal['symbol'] == symbol), None)

#         if recent_buy_signal:
#             # Check conditions to generate a sell signal
#             # if (current_price < crypto_data[symbol]['moving_average_200'][-1] and macd > 0 and signal_line > 0 and macd < signal_line) or (current_price <= 0.80 * recent_buy_signal['bought_price'] or current_price >= 1.01 * recent_buy_signal['bought_price']):
#             if (macd > 0 and signal_line > 0 and macd < signal_line) or (current_price <= 0.9 * recent_buy_signal['bought_price'] or current_price >= 1.006 * recent_buy_signal['bought_price']):
    
#                 sold_price = current_price
#                 sell_signal = {
#                     'action': 'sell',
#                     'symbol': symbol,
#                     'sold_price': sold_price
#                 }

#                 with open(buy_sell_signals_file, 'a') as f:
#                     json.dump(sell_signal, f)
#                     f.write('\n')

#                 print("sold!", symbol)

#                 crypto_data[symbol]['last_trade_action'] = "sell"


# # Initialize the plot
# plt.figure(figsize=(12, 6))
# plt.xlabel('Change Number')
# plt.ylabel('Price')
# plt.title('Price vs. Predicted Price')

# # Initialize a counter for the collected prices
# price_count = 0


# while True:
#     for symbol in crypto_symbols:
        
#         # Get data for each cryptocurrency
#         instrument = rh.crypto.get_crypto_quote(symbol)
#         if instrument is not None and 'mark_price' in instrument:
#             latest_price = float(instrument['mark_price'])
#         else:
#             print(f"Error getting price for {symbol}")
#             time.sleep(0.1)
#             continue
        
        # instrument = rh.stocks.get_stock_quote_by_symbol(symbol)
        # if instrument is not None and 'ask_price' in instrument:
        #     latest_price = float(instrument['ask_price'])
        # else:
        #     print(f"Error getting price for {symbol}")
        #     time.sleep(0.1)
        #     continue

#         crypto_data[symbol]['rounded_price'] = round(latest_price, 2)
#         last_logged_price = crypto_data[symbol]['last_logged_price']

#         # Check to not save the same saved value before
#         if crypto_data[symbol]['rounded_price'] != last_logged_price:

#             crypto_data[symbol]['prices'].append(crypto_data[symbol]['rounded_price'])
#             calculate_macd(crypto_data[symbol]['prices'], crypto_data[symbol]['macd'], crypto_data[symbol]['signal_line'])
#             calculate_moving_averages(crypto_data[symbol]['prices'], crypto_data[symbol]['moving_average_200'])

#             # Log data
#             log_data = {
#                 'change_number': crypto_data[symbol]['price_change_count'],
#                 'current_price': crypto_data[symbol]['rounded_price'],
#                 'moving_average_200': crypto_data[symbol]['moving_average_200'][-1] if crypto_data[symbol]['moving_average_200'] else None,
#                 'macd': crypto_data[symbol]['macd'][-1],
#                 'signal_line': crypto_data[symbol]['signal_line'][-1],
#             }

#             with open(f'{symbol}_prices.jsonl', 'a') as f:
#                 json.dump(log_data, f)
#                 f.write('\n')

#             crypto_data[symbol]['price_change_count'] += 1
#             price_count += 1

#             # Update the last_logged_price
#             crypto_data[symbol]['last_logged_price'] = crypto_data[symbol]['rounded_price']

#             # Generate buy and sell signals
#             generate_buy_signal(symbol, log_data['current_price'], log_data['macd'], log_data['signal_line'])
#             generate_sell_signal(symbol, log_data['current_price'], log_data['macd'], log_data['signal_line'])

#         # Graphing start
#         plt.clf()
#         plt.plot(range(1, len(crypto_data[symbol]['prices']) + 1), crypto_data[symbol]['prices'], label='Price', marker='o')

#         if crypto_data[symbol]['moving_average_200']:
#             plt.plot(range(len(crypto_data[symbol]['prices']) - len(crypto_data[symbol]['moving_average_200']) + 1, len(crypto_data[symbol]['prices']) + 1),
#                      crypto_data[symbol]['moving_average_200'], label='Moving Average (200)', marker='o')

#         plt.legend()
#         plt.draw()
#         plt.pause(0.0000001)
#         plt.savefig(graph_file_paths[symbol])

#         # Graphing start
#         plt.clf()
#         plt.plot(range(len(crypto_data[symbol]['prices']) - len(crypto_data[symbol]['macd']) + 1, len(crypto_data[symbol]['prices']) + 1),
#                  crypto_data[symbol]['macd'], label='MACD', marker='o')
#         plt.axhline(y=0, color='k', linestyle='-')
#         plt.plot(range(len(crypto_data[symbol]['prices']) - len(crypto_data[symbol]['signal_line']) + 1, len(crypto_data[symbol]['prices']) + 1),
#                  crypto_data[symbol]['signal_line'], label='Signal Line', marker='o')

#         plt.legend()
#         plt.draw()
#         plt.pause(0.0000001)
#         plt.savefig(macd_file_paths[symbol])





























# # This code version works fine but just produces buy signals back to back
# import json
# import time
# import numpy as np
# import robin_stocks.robinhood as rh
# import matplotlib.pyplot as plt
# import pandas as pd

# # Login to Robinhood
# login_response = rh.authentication.login(
#     username="y.orazov2018@gmail.com",
#     password="Turkmenistan123@", 
#     expiresIn=86400,
#     scope='internal',
#     by_sms=True,
#     store_session=True
# )

# last_logged_prices = {symbol: 0 for symbol in ["BTC", "ETH", "BCH"]}

# # Initialize variables for each cryptocurrency
# crypto_symbols = ["BTC", "ETH", "BCH"]
# crypto_data = {symbol: {'price_change_count': 1, 'prices': [], 'moving_average_200': [], 'macd': [], 'signal_line': [], 'last_logged_price': 0} for symbol in crypto_symbols}

# # Define the file paths to save the graphs
# graph_file_paths = {symbol: f'{symbol}_currentPrice.png' for symbol in crypto_symbols}
# macd_file_paths = {symbol: f'{symbol}_macd.png' for symbol in crypto_symbols}

# # Define the file path for trade signals
# buy_sell_signals_file = 'trade.jsonl'

# # Function to calculate moving averages
# def calculate_moving_averages(prices, moving_average_list):
#     if len(prices) > 50:
#         moving_average_list.append(round(np.mean(prices[-50:]), 2))

# # Function to calculate MACD
# def calculate_macd(prices, macd_list, signal_line_list):
#     prices_df = pd.DataFrame(prices, columns=['current_price'])
#     prices_df['EMA_12'] = prices_df['current_price'].ewm(span=12, adjust=False).mean()
#     prices_df['EMA_26'] = prices_df['current_price'].ewm(span=26, adjust=False).mean()
    
#     prices_df['MACD'] = prices_df['EMA_12'] - prices_df['EMA_26']
#     prices_df['Signal_Line'] = prices_df['MACD'].ewm(span=9, adjust=False).mean()
    
#     macd_list.append(prices_df['MACD'].iloc[-1])
#     signal_line_list.append(prices_df['Signal_Line'].iloc[-1])
    
#     return prices_df

# # Function to generate buy signal
# def generate_buy_signal(symbol, current_price, macd, signal_line):
#     if crypto_data[symbol]['moving_average_200']:
#         if current_price > crypto_data[symbol]['moving_average_200'][-1] and macd < 0 and signal_line < 0 and macd > signal_line:
#             buy_signal = {
#                 'action': 'buy',
#                 'symbol': symbol,
#                 'bought_price': current_price
#             }
#             with open(buy_sell_signals_file, 'a') as f:
#                 json.dump(buy_signal, f)
#                 f.write('\n')

# # Function to generate sell signal
# def generate_sell_signal(symbol, current_price, macd, signal_line):

#     with open(buy_sell_signals_file, 'r') as f:
#         buy_signals = [json.loads(line.strip()) for line in f]

#     for buy_signal in buy_signals:
#         if buy_signal['symbol'] == symbol:
#             if current_price < crypto_data[symbol]['moving_average_200'][-1] and macd > 0 and signal_line > 0 and macd < signal_line:
#                 sell_signal = {
#                     'action': 'sell',
#                     'symbol': symbol,
#                     'sold_price': current_price
#                 }
#                 with open(buy_sell_signals_file, 'a') as f:
#                     json.dump(sell_signal, f)
#                     f.write('\n')

# # Initialize the plot
# plt.figure(figsize=(12, 6))
# plt.xlabel('Change Number')
# plt.ylabel('Price')
# plt.title('Price vs. Predicted Price')

# # Initialize a counter for the collected prices
# price_count = 0

# while True:
#     for symbol in crypto_symbols:
#         # Get data for each cryptocurrency
#         instrument = rh.crypto.get_crypto_quote(symbol)
#         if instrument is not None and 'mark_price' in instrument:
#             latest_price = float(instrument['mark_price'])
#         else:
#             print(f"Error getting price for {symbol}")
#             time.sleep(0.1)
#             continue

#         crypto_data[symbol]['rounded_price'] = round(latest_price, 2)
#         last_logged_price = crypto_data[symbol]['last_logged_price']

#         # Check to not save the same saved value before
#         if crypto_data[symbol]['rounded_price'] != last_logged_price:

#             crypto_data[symbol]['prices'].append(crypto_data[symbol]['rounded_price'])
#             calculate_macd(crypto_data[symbol]['prices'], crypto_data[symbol]['macd'], crypto_data[symbol]['signal_line'])
#             calculate_moving_averages(crypto_data[symbol]['prices'], crypto_data[symbol]['moving_average_200'])

#             # Log data
#             log_data = {
#                 'change_number': crypto_data[symbol]['price_change_count'],
#                 'current_price': crypto_data[symbol]['rounded_price'],
#                 'moving_average_200': crypto_data[symbol]['moving_average_200'][-1] if crypto_data[symbol]['moving_average_200'] else None,
#                 'macd': crypto_data[symbol]['macd'][-1],
#                 'signal_line': crypto_data[symbol]['signal_line'][-1],
#             }

#             with open(f'{symbol}_prices.jsonl', 'a') as f:
#                 json.dump(log_data, f)
#                 f.write('\n')

#             crypto_data[symbol]['price_change_count'] += 1
#             price_count += 1

#             # Update the last_logged_price
#             crypto_data[symbol]['last_logged_price'] = crypto_data[symbol]['rounded_price']

#             # Generate buy and sell signals
#             generate_buy_signal(symbol, log_data['current_price'], log_data['macd'], log_data['signal_line'])
#             generate_sell_signal(symbol, log_data['current_price'], log_data['macd'], log_data['signal_line'])

#         # Graphing start
#         plt.clf()
#         plt.plot(range(1, len(crypto_data[symbol]['prices']) + 1), crypto_data[symbol]['prices'], label='Price', marker='o')

#         if crypto_data[symbol]['moving_average_200']:
#             plt.plot(range(len(crypto_data[symbol]['prices']) - len(crypto_data[symbol]['moving_average_200']) + 1, len(crypto_data[symbol]['prices']) + 1),
#                      crypto_data[symbol]['moving_average_200'], label='Moving Average (200)', marker='o')

#         plt.legend()
#         plt.draw()
#         plt.pause(0.0000001)
#         plt.savefig(graph_file_paths[symbol])

#         # Graphing start
#         plt.clf()
#         plt.plot(range(len(crypto_data[symbol]['prices']) - len(crypto_data[symbol]['macd']) + 1, len(crypto_data[symbol]['prices']) + 1),
#                  crypto_data[symbol]['macd'], label='MACD', marker='o')
#         plt.axhline(y=0, color='k', linestyle='-')
#         plt.plot(range(len(crypto_data[symbol]['prices']) - len(crypto_data[symbol]['signal_line']) + 1, len(crypto_data[symbol]['prices']) + 1),
#                  crypto_data[symbol]['signal_line'], label='Signal Line', marker='o')

#         plt.legend()
#         plt.draw()
#         plt.pause(0.0000001)
#         plt.savefig(macd_file_paths[symbol])
































# Best old working code
# import json
# import time
# import numpy as np
# import robin_stocks.robinhood as rh
# import matplotlib.pyplot as plt
# import pandas as pd

# # Login to Robinhood
# login_response = rh.authentication.login(
#     username="y.orazov2018@gmail.com",
#     password="Turkmenistan123@", 
#     expiresIn=86400,
#     scope='internal',
#     by_sms=True,
#     store_session=True
# )

# last_logged_prices = {symbol: 0 for symbol in ["BTC", "ETH", "BCH", "BNB"]}

# # Initialize variables for each cryptocurrency
# crypto_symbols = ["BTC", "ETH", "BCH", "BNB"]
# crypto_data = {symbol: {'price_change_count': 1, 'prices': [], 'moving_average_200': [], 'macd': [], 'signal_line': [], 'last_logged_price': 0} for symbol in crypto_symbols}

# # Define the file paths to save the graphs
# graph_file_paths = {symbol: f'{symbol}_currentPrice.png' for symbol in crypto_symbols}
# macd_file_paths = {symbol: f'{symbol}_macd.png' for symbol in crypto_symbols}

# # Function to calculate moving averages
# def calculate_moving_averages(prices, moving_average_list):
#     if len(prices) > 199:
#         moving_average_list.append(round(np.mean(prices[-200:]), 2))

# # Function to calculate MACD
# def calculate_macd(prices, macd_list, signal_line_list):
#     prices_df = pd.DataFrame(prices, columns=['current_price'])
#     prices_df['EMA_12'] = prices_df['current_price'].ewm(span=12, adjust=False).mean()
#     prices_df['EMA_26'] = prices_df['current_price'].ewm(span=26, adjust=False).mean()
    
#     prices_df['MACD'] = prices_df['EMA_12'] - prices_df['EMA_26']
#     prices_df['Signal_Line'] = prices_df['MACD'].ewm(span=9, adjust=False).mean()
    
#     macd_list.append(prices_df['MACD'].iloc[-1])
#     signal_line_list.append(prices_df['Signal_Line'].iloc[-1])
    
#     return prices_df

# # Initialize the plot
# plt.figure(figsize=(12, 6))
# plt.xlabel('Change Number')
# plt.ylabel('Price')
# plt.title('Price vs. Predicted Price')

# # Initialize a counter for the collected prices
# price_count = 0

# while True:
#     for symbol in crypto_symbols:
#         # Get data for each cryptocurrency
#         instrument = rh.crypto.get_crypto_quote(symbol)
#         if instrument is not None and 'mark_price' in instrument:
#             latest_price = float(instrument['mark_price'])
#         else:
#             print(f"Error getting price for {symbol}")
#             time.sleep(0.1)
#             continue

#         crypto_data[symbol]['rounded_price'] = round(latest_price, 2)
#         last_logged_price = crypto_data[symbol]['last_logged_price']
        
#         # Check to not save the same saved value before
#         if(crypto_data[symbol]['rounded_price']) !=last_logged_price:

#             crypto_data[symbol]['prices'].append(crypto_data[symbol]['rounded_price'])
#             calculate_macd(crypto_data[symbol]['prices'], crypto_data[symbol]['macd'], crypto_data[symbol]['signal_line'])
#             calculate_moving_averages(crypto_data[symbol]['prices'], crypto_data[symbol]['moving_average_200'])

#             # Log data
#             log_data = {
#                 'change_number': crypto_data[symbol]['price_change_count'],
#                 'current_price': crypto_data[symbol]['rounded_price'],
#                 'moving_average_200': crypto_data[symbol]['moving_average_200'][-1] if crypto_data[symbol]['moving_average_200'] else None,
#                 'macd': crypto_data[symbol]['macd'][-1],
#                 'signal_line': crypto_data[symbol]['signal_line'][-1],
#             }

#             with open(f'{symbol}_prices.jsonl', 'a') as f:
#                 json.dump(log_data, f)
#                 f.write('\n')

#             crypto_data[symbol]['price_change_count'] += 1
#             price_count += 1

#             # Update the last_logged_price
#             crypto_data[symbol]['last_logged_price'] = crypto_data[symbol]['rounded_price']

#         # Graphing start
#         plt.clf()
#         plt.plot(range(1, len(crypto_data[symbol]['prices']) + 1), crypto_data[symbol]['prices'], label='Price', marker='o')
        
#         if crypto_data[symbol]['moving_average_200']:
#             plt.plot(range(len(crypto_data[symbol]['prices']) - len(crypto_data[symbol]['moving_average_200']) + 1, len(crypto_data[symbol]['prices']) + 1),
#                      crypto_data[symbol]['moving_average_200'], label='Moving Average (200)', marker='o')

#         plt.legend()
#         plt.draw()
#         plt.pause(0.0000001)
#         plt.savefig(graph_file_paths[symbol])

#         # Graphing start
#         plt.clf()
#         plt.plot(range(len(crypto_data[symbol]['prices']) - len(crypto_data[symbol]['macd']) + 1, len(crypto_data[symbol]['prices']) + 1),
#                  crypto_data[symbol]['macd'], label='MACD', marker='o')
#         plt.axhline(y=0, color='k', linestyle='-')
#         plt.plot(range(len(crypto_data[symbol]['prices']) - len(crypto_data[symbol]['signal_line']) + 1, len(crypto_data[symbol]['prices']) + 1),
#                  crypto_data[symbol]['signal_line'], label='Signal Line', marker='o')

#         plt.legend()
#         plt.draw()
#         plt.pause(0.0000001)
#         plt.savefig(macd_file_paths[symbol])




















































































































# Old code. Working.

# import json
# import time
# import numpy as np
# import robin_stocks.robinhood as rh
# import matplotlib.pyplot as plt
# import pandas as pd

# # Login to Robinhood
# login_response = rh.authentication.login(
#     username="y.orazov2018@gmail.com",
#     password="Turkmenistan123@", 
#     expiresIn=86400,
#     scope='internal',
#     by_sms=True,
#     store_session=True
# )

# last_logged_prices = {symbol: 0 for symbol in ["BTC", "ETH", "BCH", "BNB"]}

# # Initialize variables for each cryptocurrency
# crypto_symbols = ["BTC", "ETH", "BCH", "BNB"]
# crypto_data = {symbol: {'price_change_count': 1, 'prices': [], 'moving_average_200': [],
#                         'macd': [], 'signal_line': [], 'last_logged_price': 0} for symbol in crypto_symbols}

# # Define the file paths to save the graphs
# graph_file_paths = {symbol: f'{symbol}_currentPrice.png' for symbol in crypto_symbols}
# macd_file_paths = {symbol: f'{symbol}_macd.png' for symbol in crypto_symbols}

# # Function to calculate moving averages
# def calculate_moving_averages(prices, moving_average_list):
#     if len(prices) > 199:
#         moving_average_list.append(round(np.mean(prices[-200:]), 2))

# # Function to calculate MACD
# def calculate_macd(prices, macd_list, signal_line_list):
#     prices_df = pd.DataFrame(prices, columns=['current_price'])
#     prices_df['EMA_12'] = prices_df['current_price'].ewm(span=12, adjust=False).mean()
#     prices_df['EMA_26'] = prices_df['current_price'].ewm(span=26, adjust=False).mean()
    
#     prices_df['MACD'] = prices_df['EMA_12'] - prices_df['EMA_26']
#     prices_df['Signal_Line'] = prices_df['MACD'].ewm(span=9, adjust=False).mean()
    
#     macd_list.append(prices_df['MACD'].iloc[-1])
#     signal_line_list.append(prices_df['Signal_Line'].iloc[-1])
    
#     return prices_df

# # Initialize the plot
# plt.figure(figsize=(12, 6))
# plt.xlabel('Change Number')
# plt.ylabel('Price')
# plt.title('Price vs. Predicted Price')

# # Initialize a counter for the collected prices
# price_count = 0

# # Threshold for significant price change
# # price_change_threshold = 0.01  # You can adjust this threshold based on your requirements

# while True:
#     for symbol in crypto_symbols:
#         # Get data for each cryptocurrency
#         instrument = rh.crypto.get_crypto_quote(symbol)
#         if instrument is not None and 'mark_price' in instrument:
#             latest_price = float(instrument['mark_price'])
#         else:
#             print(f"Error getting price for {symbol}")
#             time.sleep(0.1)
#             continue

#         crypto_data[symbol]['rounded_price'] = round(latest_price, 2)
#         last_logged_price = crypto_data[symbol]['last_logged_price']
        
#         # Check if the absolute difference is greater than the threshold
#         # if abs(crypto_data[symbol]['rounded_price'] - last_logged_price) > price_change_threshold:
#         if(crypto_data[symbol]['rounded_price']) !=last_logged_price:

#             crypto_data[symbol]['prices'].append(crypto_data[symbol]['rounded_price'])
#             calculate_macd(crypto_data[symbol]['prices'], crypto_data[symbol]['macd'], crypto_data[symbol]['signal_line'])
#             calculate_moving_averages(crypto_data[symbol]['prices'], crypto_data[symbol]['moving_average_200'])

#             # Log data
#             log_data = {
#                 'change_number': crypto_data[symbol]['price_change_count'],
#                 'current_price': crypto_data[symbol]['rounded_price'],
#                 'moving_average_200': crypto_data[symbol]['moving_average_200'][-1] if crypto_data[symbol]['moving_average_200'] else None,
#                 'macd': crypto_data[symbol]['macd'][-1],
#                 'signal_line': crypto_data[symbol]['signal_line'][-1],
#             }

#             with open(f'{symbol}_prices.jsonl', 'a') as f:
#                 json.dump(log_data, f)
#                 f.write('\n')

#             crypto_data[symbol]['price_change_count'] += 1
#             price_count += 1

#             # Update the last_logged_price
#             crypto_data[symbol]['last_logged_price'] = crypto_data[symbol]['rounded_price']

#         # Graphing start
#         plt.clf()
#         plt.plot(range(1, len(crypto_data[symbol]['prices']) + 1), crypto_data[symbol]['prices'], label='Price', marker='o')
        
#         if crypto_data[symbol]['moving_average_200']:
#             plt.plot(range(len(crypto_data[symbol]['prices']) - len(crypto_data[symbol]['moving_average_200']) + 1, len(crypto_data[symbol]['prices']) + 1),
#                      crypto_data[symbol]['moving_average_200'], label='Moving Average (200)', marker='o')

#         plt.legend()
#         plt.draw()
#         plt.pause(0.0000001)
#         plt.savefig(graph_file_paths[symbol])

#         # Graphing start
#         plt.clf()
#         plt.plot(range(len(crypto_data[symbol]['prices']) - len(crypto_data[symbol]['macd']) + 1, len(crypto_data[symbol]['prices']) + 1),
#                  crypto_data[symbol]['macd'], label='MACD', marker='o')
#         plt.axhline(y=0, color='k', linestyle='-')
#         plt.plot(range(len(crypto_data[symbol]['prices']) - len(crypto_data[symbol]['signal_line']) + 1, len(crypto_data[symbol]['prices']) + 1),
#                  crypto_data[symbol]['signal_line'], label='Signal Line', marker='o')

#         plt.legend()
#         plt.draw()
#         plt.pause(0.0000001)
#         plt.savefig(macd_file_paths[symbol])









# import json
# import time
# import numpy as np
# import robin_stocks.robinhood as rh
# import matplotlib.pyplot as plt
# import pandas as pd

# # Login to Robinhood
# login_response = rh.authentication.login(
#     username="y.orazov2018@gmail.com",
#     password="Turkmenistan123@", 
#     expiresIn=86400,
#     scope='internal',
#     by_sms=True,
#     store_session=True)

# # Initialize variables
# price_change_count = 1

# # Last saved price
# last_logged_price = 0

# # Define the file path to save the graph
# macd_file_path = 'macd.png'
# current_prices_graph = 'currentPrice.png'

# # Initialize variables for moving averages
# moving_average_200 = []

# # Initialize variables
# prices = []
# macd = []  
# signal_line = []

# # Function to calculate moving averages
# def calculate_moving_averages(prices):
#     if len(prices) > 199:
#         moving_average_200.append(round(np.mean(prices[-200:]), 2))

# # Function to calculate MACD
# def calculate_macd(prices):
#     prices_df = pd.DataFrame(prices, columns=['current_price'])
#     prices_df['EMA_12'] = prices_df['current_price'].ewm(span=12, adjust=False).mean()
#     prices_df['EMA_26'] = prices_df['current_price'].ewm(span=26, adjust=False).mean()
    
#     prices_df['MACD'] = prices_df['EMA_12'] - prices_df['EMA_26']
#     prices_df['Signal_Line'] = prices_df['MACD'].ewm(span=9, adjust=False).mean()
    
#     macd.append(prices_df['MACD'].iloc[-1])
#     signal_line.append(prices_df['Signal_Line'].iloc[-1])
    
#     return prices_df

# # Initialize the plot
# plt.figure(figsize=(12, 6))
# plt.xlabel('Change Number')
# plt.ylabel('Price')
# plt.title('Price vs. Predicted Price')

# # Initialize a counter for the collected prices
# price_count = 0

# while True:

#     # For Crypto
#     instrument = rh.crypto.get_crypto_quote("BTC")
#     # ask_price, mark_price, bid_price
#     if instrument is not None and 'mark_price' in instrument:
#         latest_price = float(instrument['mark_price'])
#     else:
#         print("Error getting price")
#         time.sleep(0.1)
#         continue

#     rounded_price = round(latest_price, 2)

#     if len(prices) > 0:
#         last_logged_price = prices[-1]
        
#     if rounded_price != last_logged_price:
#         prices.append(rounded_price)
#         prices_df = calculate_macd(prices)
#         calculate_moving_averages(prices)  # Calculate moving averages

#         # Log price and moving averages
#         log_data = {
#             'change_number': price_change_count,
#             'current_price': rounded_price,
#             'moving_average_100': moving_average_200[-1] if moving_average_200 else None,
#             'macd': macd[-1],
#             'signal_line': signal_line[-1],
#         }

#         with open('prices.jsonl', 'a') as f:
#             json.dump(log_data, f)
#             f.write('\n')

#         price_change_count += 1
#         price_count += 1

#     time.sleep(0.000000001)

#     # Graphing start
#     plt.clf()
#     plt.plot(range(1, len(prices) + 1), prices, label='Price', marker='o')
    
#     if moving_average_200:
#         plt.plot(range(len(prices) - len(moving_average_200) + 1, len(prices) + 1), moving_average_200, label='Moving Average (200)', marker='o')

#     plt.legend()
#     plt.draw()
#     plt.pause(0.0000001)
#     plt.savefig(current_prices_graph)
#     plt.figtext

#     # Graphing start
#     plt.clf()
#     # plt.plot(range(1, len(prices) + 1), prices, label='Price', marker='o')
#     plt.plot(range(len(prices)-len(macd)+1, len(prices)+1), macd, label='MACD', marker='o')
#     plt.axhline(y=0, color='k', linestyle='-')
#     plt.plot(range(len(prices)-len(signal_line)+1, len(prices)+1), signal_line, label='Signal Line', marker='o') 

#     plt.legend()
#     plt.draw()
#     plt.pause(0.0000001)
#     plt.savefig(macd_file_path)
#     plt.figtext




















































# import json
# import time
# import numpy as np
# import robin_stocks.robinhood as rh
# import matplotlib.pyplot as plt
# from statsmodels.tsa.arima.model import ARIMA
# from pmdarima.arima import auto_arima
# import pandas as pd

# # Login to Robinhood
# login_response = rh.authentication.login(
#     username="y.orazov2018@gmail.com",
#     password="Turkmenistan123@", 
#     expiresIn=86400,
#     scope='internal',
#     by_sms=True,
#     store_session=True)

# # Define action_data with a default value
# # action_data = []

# # Initialize variables
# price_change_count = 1
# last_price = None

# # Initialize p, d, q
# p = 2
# d = 1
# q = 3

# # Last saved price
# last_logged_price = 0

# # Track predicted prices
# predicted_prices = set()

# # Track last saved change_number
# last_saved_change_number = 0

# # Initialize a list to keep track of the most recent 100 future prices
# recent_future_prices = []

# # Define the file path to save the graph
# predicted_prices_graph = 'predictedPrice.png'
# macd_file_path = 'macd.png'
# current_prices_graph = 'currentPrice.png'

# # Initialize variables for moving averages
# moving_average_12 = []
# moving_average_100 = []
# diff_moving_avg = []
# enter_trade = None
# trade_decision = None

# # Initialize variables
# prices = []
# macd = []  
# signal_line = []

# # Function to calculate moving averages
# def calculate_moving_averages(prices):
#     if len(prices) >= 12:
#         moving_average_12.append(round(np.mean(prices[-12:]), 2))
#     if len(prices) >= 100:
#         moving_average_100.append(round(np.mean(prices[-200:]), 2))

# # Define a function to determine the optimal p, d, and q values based on the most recent 40 current_price values
# def find_optimal_p_d_q(prices):
#     recent_prices = prices[-100:]
#     # model = auto_arima(recent_prices, stepwise=True, trace=True, max_p = 15, max_d = 15, max_q = 15)
#     model = auto_arima(recent_prices, stepwise=True, trace=True, seasonal=False, error_action='ignore', suppress_warnings=True)
#     # model = auto_arima(recent_prices, stepwise=True, trace=True, seasonal=True)
#     # model = auto_arima(recent_prices, seasonal=True, m = 12, stepwise=True, trace=True)
#     p, d, q = model.order

#     return p, d, q


# # Function to calculate MACD
# def calculate_macd(prices):
#     prices_df = pd.DataFrame(prices, columns=['current_price'])
#     prices_df['EMA_12'] = prices_df['current_price'].ewm(span=12, adjust=False).mean()
#     prices_df['EMA_26'] = prices_df['current_price'].ewm(span=26, adjust=False).mean()
    
#     prices_df['MACD'] = prices_df['EMA_12'] - prices_df['EMA_26']
#     prices_df['Signal_Line'] = prices_df['MACD'].ewm(span=9, adjust=False).mean()
    
#     macd.append(prices_df['MACD'].iloc[-1])
#     signal_line.append(prices_df['Signal_Line'].iloc[-1])
    
#     return prices_df

# # Initialize the plot
# plt.figure(figsize=(12, 6))
# plt.xlabel('Change Number')
# plt.ylabel('Price')
# plt.title('Price vs. Predicted Price')

# # Initialize a counter for the collected prices
# price_count = 0

# while True:

#     # For Crypto
    
#     instrument = rh.crypto.get_crypto_quote("BTC")
#     # ask_price, mark_price, bid_price
#     if instrument is not None and 'mark_price' in instrument:
#         latest_price = float(instrument['mark_price'])
#     else:
#         print("Error getting price")
#         time.sleep(0.1)
#         continue

#     # instrument = rh.stocks.get_stock_quote_by_symbol("AAPL")
#     # if instrument is not None and 'ask_price' in instrument:
#     #     latest_price = float(instrument['ask_price'])
#     # else:
#     #     print("Error getting price")
#     #     time.sleep(0.000000001)
#     #     continue

#     # 0.035 is the 0.35% buy spread
#     # rounded_price = round(latest_price * 1.035, 2)
#     rounded_price = round(latest_price, 2)

#     if len(prices) > 0:
#         last_logged_price = prices[-1]
        
#     if rounded_price != last_logged_price:
#         prices.append(rounded_price)
#         prices_df = calculate_macd(prices)
#         calculate_moving_averages(prices)  # Calculate moving averages

#         # if len(prices) > 25:
#         #     # Calculate the difference between moving_average_12 and moving_average_26
#         #     diff_moving_avg = round((moving_average_12[-1] - moving_average_25[-1]), 2)

#         # Log price and moving averages
#         log_data = {
#             'change_number': price_change_count,
#             'current_price': rounded_price,
#             # 'moving_average_12': moving_average_12[-1] if moving_average_12 else None,
#             'moving_average_100': moving_average_100[-1] if moving_average_100 else None,
#             'macd': macd[-1],
#             'signal_line': signal_line[-1],
#             # 'diff_moving_avg':diff_moving_avg if diff_moving_avg else None
#         }

#         with open('prices.jsonl', 'a') as f:
#             json.dump(log_data, f)
#             f.write('\n')

#         price_change_count += 1
#         price_count += 1

#     time.sleep(0.000000001)

#     # # Check if you have collected 40 new prices for AUTO ARIMA
#     if price_count == 100:
#         # Call the function to find optimal p, d, q values based on the most recent prices
#         p, d, q = find_optimal_p_d_q(prices)
#         # Reset the price_count
#         price_count = 0

#     # Make a prediction using the updated p, d, q values
#     if len(prices) >= (p + q + d):
#         # print(p, d, q)

#         # print(p, d, q)  This line to check if p, d, q parameter is being upadted.
#         model = ARIMA(prices, order=(p, d, q))

#         model_fit = model.fit()
#         start_index = len(prices)
#         end_index = start_index + 10

#         predictions = model_fit.predict(start=start_index, end=end_index)

#         rounded_predictions = [round(p, 2) for p in predictions]

#         for i, pred in enumerate(rounded_predictions):
#             future_change_number = start_index + i
#             if future_change_number > last_saved_change_number:
#                 predicted_prices.add(pred)

#                 prediction_data = {
#                     'predicted_change_number': future_change_number,
#                     'future_price': round(pred, 2),
#                 }

#                 with open('predictions.jsonl', 'a') as f:
#                     json.dump(prediction_data, f)
#                     f.write('\n')

#                 last_saved_change_number = future_change_number

#                 recent_future_prices.append(pred)

#                 if len(recent_future_prices) == 10:
                    
#                     minimum = min(recent_future_prices)
#                     maximum = max(recent_future_prices)

#                     first_derivative = np.diff(recent_future_prices, 1)
#                     second_derivative = np.diff(first_derivative, 1)

#                     rounded_first_derivative = round(first_derivative[-1], 2)
#                     rounded_second_derivative = round(second_derivative[-1], 2)

#                     if rounded_second_derivative > 0.01:
#                         price_direction = "Concave Up"
#                     else:
#                         price_direction = "Concave Down"
#                     # Read existing data from analyze.jsonl

#                     from_change_number = future_change_number - 9

#                     minPrice_change_number = future_change_number - 9 + recent_future_prices.index(minimum)
#                     maxPrice_change_number = future_change_number - 9 + recent_future_prices.index(maximum)

#                     trade_data = {
#                         'from': from_change_number,
#                         'till': future_change_number,
#                         'Accuracy': enter_trade,
#                         'minimum_price_change_number': minPrice_change_number,
#                         'minimum_predicted_price': minimum,
#                         # 'maximum_predicted_change_number': maxPrice_change_number,
#                         # 'maximum_predicted_price':maximum,
#                         'moving_average_100': moving_average_100[-1] if moving_average_100 else None,
#                         # 'first_derivative': rounded_first_derivative,
#                         # 'second_derivative': rounded_second_derivative,
#                         'price_direction': price_direction,
#                         # 'predictions': recent_future_prices
#                         # 'trade_decision': trade_decision
#                     }

#                     with open('analyze.jsonl', 'a') as trade_file:
#                         json.dump(trade_data, trade_file)
#                         trade_file.write('\n')
#                     recent_future_prices.clear()

#     plt.clf()
#     plt.plot(range(1, len(prices) + 1), prices, label='Price', marker='o')
    
#     if moving_average_100:
#         plt.plot(range(len(prices) - len(moving_average_100) + 1, len(prices) + 1), moving_average_100, label='Moving Average (100)', marker='o')

#     # if moving_average_12:
#     #     plt.plot(range(len(prices) - len(moving_average_12) + 1, len(prices) + 1), moving_average_12, label='MACD Line (12)', marker='o')

#     # if moving_average_25:
#     #     plt.plot(range(len(prices) - len(moving_average_25) + 1, len(prices) + 1), moving_average_25, label='Moving Average (25)', marker='o')
    
#     # with open('predictions.jsonl', 'r') as f:
#         # lines = f.readlines()
#         # prediction_data = [json.loads(line.strip()) for line in lines]
#         # change_numbers = [entry['predicted_change_number'] for entry in prediction_data]
#         # price_predicted = [entry['future_price'] for entry in prediction_data]
#         # plt.plot(change_numbers, price_predicted, label='Price Predicted', marker='x')

#     plt.legend()
#     plt.draw()
#     plt.pause(0.0000001)
#     plt.savefig(current_prices_graph)
#     plt.figtext


#     plt.clf()
#     # plt.plot(range(1, len(prices) + 1), prices, label='Price', marker='o')

#     # if moving_average_12:
#     #     plt.plot(range(len(prices) - len(moving_average_12) + 1, len(prices) + 1), moving_average_12, label='MACD Line (12)', marker='o')

#     # if moving_average_25:
#     #     plt.plot(range(len(prices) - len(moving_average_25) + 1, len(prices) + 1), moving_average_25, label='Moving Average (25)', marker='o')

#     with open('predictions.jsonl', 'r') as f:
#         lines = f.readlines()
#         prediction_data = [json.loads(line.strip()) for line in lines]
#         change_numbers = [entry['predicted_change_number'] for entry in prediction_data]
#         price_predicted = [entry['future_price'] for entry in prediction_data]
#         plt.plot(change_numbers, price_predicted, label='Price Predicted', marker='o')

#     plt.legend()
#     plt.draw()
#     plt.pause(0.0000001)
#     plt.savefig(predicted_prices_graph)
#     plt.figtext



#     plt.clf()
#     # plt.plot(range(1, len(prices) + 1), prices, label='Price', marker='o')
#     plt.plot(range(len(prices)-len(macd)+1, len(prices)+1), macd, label='MACD', marker='o')
#     plt.axhline(y=0, color='k', linestyle='-')
#     plt.plot(range(len(prices)-len(signal_line)+1, len(prices)+1), signal_line, label='Signal Line', marker='o') 

#     plt.legend()
#     plt.draw()
#     plt.pause(0.0000001)
#     plt.savefig(macd_file_path)
#     plt.figtext

#     with open('analyze.jsonl', 'r') as f:
#         analyze_data = [json.loads(line) for line in f]

#     with open('prices.jsonl', 'r') as f:
#         prices_data = [json.loads(line) for line in f]

#     for entry in analyze_data:
#         invest_change_number = entry['minimum_price_change_number']
#         for price_entry in prices_data:
#             if price_entry['change_number'] == invest_change_number:
#                 current_price_min = price_entry['current_price']
#                 entry['current_price'] = current_price_min
#                 minimum_predicted_price = entry['minimum_predicted_price']
#                 current_price = entry['current_price']
#                 # Calculate the difference between minimum_predicted_price and current_price
#                 price_difference_min = abs(minimum_predicted_price - current_price_min)
#                 # Update the "decision" field based on the condition

#                 # if price_difference <= current_price * 0.000745:  #Accuracy: 99.99% 
#                 if price_difference_min <= (0.0001 * current_price):  #Accuracy: 98% 
#                     entry['Accuracy'] = 'Super Accurate'

#                 elif price_difference_min <= (0.0002*current_price):
#                     entry['Accuracy'] = 'Accurate'
#                 else:
#                     entry['Accuracy'] = 'Not Accurate'

                
#     with open('analyze.jsonl', 'w') as f:
#         for entry in analyze_data:
#             json.dump(entry, f)

#             f.write('\n')
