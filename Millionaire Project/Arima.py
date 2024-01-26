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
    username="y.orazov2018@gmail.com",
    password="Turkmenistan123@", 
    expiresIn=86400,
    scope='internal',
    by_sms=True,
    store_session=True)

# Initialize variables
price_change_count = 1
last_price = None

# Initialize p, d, q
p = 0
d = 0
q = 0

# Last saved price
last_logged_price = 0

# Track predicted prices
predicted_prices = set()

# Track last saved change_number
last_saved_change_number = 0

# Initialize a list to keep track of the most recent 100 future prices
recent_future_prices = []

# Define the file path to save the graph
predicted_prices_graph = 'predictedPrice.png'
macd_file_path = 'macd.png'
current_prices_graph = 'currentPrice.png'

# Initialize variables for moving averages
moving_average_12 = []
moving_average_100 = []
diff_moving_avg = []
enter_trade = None
trade_decision = None

# Initialize variables
prices = []
macd = []  
signal_line = []

# Function to calculate moving averages
def calculate_moving_averages(prices):
    if len(prices) >= 12:
        moving_average_12.append(round(np.mean(prices[-12:]), 2))
    if len(prices) >= 200:
        moving_average_100.append(round(np.mean(prices[-200:]), 2))

# Define a function to determine the optimal p, d, and q values based on the most recent 40 current_price values
def find_optimal_p_d_q(prices):
    recent_prices = prices[-200:]
    model = auto_arima(recent_prices, stepwise=True, trace=True, seasonal=False, error_action='ignore', suppress_warnings=True)
    p, d, q = model.order
    return p, d, q

# Function to calculate MACD
# def calculate_macd(prices):
#     prices_df = pd.DataFrame(prices, columns=['current_price'])
#     prices_df['EMA_12'] = prices_df['current_price'].ewm(span=12, adjust=False).mean()
#     prices_df['EMA_26'] = prices_df['current_price'].ewm(span=26, adjust=False).mean()
    
#     prices_df['MACD'] = prices_df['EMA_12'] - prices_df['EMA_26']
#     prices_df['Signal_Line'] = prices_df['MACD'].ewm(span=9, adjust=False).mean()
    
#     macd.append(prices_df['MACD'].iloc[-1])
#     signal_line.append(prices_df['Signal_Line'].iloc[-1])
    
#     return prices_df

def calculate_macd(prices):
    prices_df = pd.DataFrame(prices, columns=['current_price'])
    prices_df['EMA_12'] = prices_df['current_price'].ewm(span=12, adjust=False).mean()
    prices_df['EMA_26'] = prices_df['current_price'].ewm(span=26, adjust=False).mean()
    
    prices_df['MACD'] = prices_df['EMA_12'] - prices_df['EMA_26']
    prices_df['Signal_Line'] = prices_df['MACD'].ewm(span=9, adjust=False).mean()

    prices_df['MACD'] = prices_df['MACD'].round(3)
    prices_df['Signal_Line'] = prices_df['Signal_Line'].round(3)
    
    macd_value = prices_df['MACD'].iloc[-1]
    signal_line_value = prices_df['Signal_Line'].iloc[-1]

    macd_range = np.mean(macd_value)

    # Set MACD and Signal_Line to 0 if abs(MACD) is less than 0.1
    if abs(macd_value) < 1.15 * macd_range:
        macd_value = 0.0
        signal_line_value = 0.0
    
    macd.append(macd_value)
    signal_line.append(signal_line_value)
    
    return prices_df



# Initialize the plot
plt.figure(figsize=(12, 6))
plt.xlabel('Change Number')
plt.ylabel('Price')
plt.title('Price vs. Predicted Price')

# Initialize a counter for the collected prices
price_count = 0

while True:
    # For Crypto
    instrument = rh.crypto.get_crypto_quote("BTC")
    if instrument is not None and 'mark_price' in instrument:
        latest_price = float(instrument['mark_price'])
    else:
        print("Error getting price")
        time.sleep(0.1)
        continue

    rounded_price = round(latest_price, 2)

    if len(prices) > 0:
        last_logged_price = prices[-1]
        
    if rounded_price != last_logged_price:
        prices.append(rounded_price)
        prices_df = calculate_macd(prices)
        calculate_moving_averages(prices)  # Calculate moving averages

        log_data = {
            'change_number': price_change_count,
            'current_price': rounded_price,
            'moving_average_100': moving_average_100[-1] if moving_average_100 else None,
            'macd': macd[-1],
            'signal_line': signal_line[-1],
        }

        with open('prices.jsonl', 'a') as f:
            json.dump(log_data, f)
            f.write('\n')

        price_change_count += 1
        price_count += 1

    time.sleep(0.000000001)

    if price_count == 200:
        p, d, q = find_optimal_p_d_q(prices)
        price_count = 0

    # if len(prices) >= (p + q + d):
    if len(prices) > 200:
        model = ARIMA(prices, order=(p, d, q))
        model_fit = model.fit()
        start_index = len(prices)
        end_index = start_index + 100

        predictions = model_fit.predict(start=start_index, end=end_index)

        rounded_predictions = [round(p, 2) for p in predictions]

        for i, pred in enumerate(rounded_predictions):
            future_change_number = start_index + i
            if future_change_number > last_saved_change_number:
                predicted_prices.add(pred)

                prediction_data = {
                    'predicted_change_number': future_change_number,
                    'future_price': round(pred, 2),
                }

                with open('predictions.jsonl', 'a') as f:
                    json.dump(prediction_data, f)
                    f.write('\n')

                last_saved_change_number = future_change_number

                recent_future_prices.append(pred)

                if len(recent_future_prices) == 100:
                    
                    minimum = min(recent_future_prices)
                    maximum = max(recent_future_prices)

                    first_derivative = np.diff(recent_future_prices, 1)
                    second_derivative = np.diff(first_derivative, 1)

                    rounded_first_derivative = round(first_derivative[-1], 2)
                    rounded_second_derivative = round(second_derivative[-1], 2)

                    if rounded_second_derivative > 0.01:
                        price_direction = "Concave Up"
                    else:
                        price_direction = "Concave Down"

                    from_change_number = future_change_number - 199

                    minPrice_change_number = future_change_number - 199 + recent_future_prices.index(minimum)
                    # maxPrice_change_number = future_change_number - 9 + recent_future_prices.index(maximum)

                    trade_data = {
                        'from': from_change_number,
                        'till': future_change_number,
                        'Accuracy': enter_trade,
                        'minimum_price_change_number': minPrice_change_number,
                        'minimum_predicted_price': minimum,
                        'moving_average_100': moving_average_100[-1] if moving_average_100 else None,
                        'price_direction': price_direction,
                    }

                    with open('analyze.jsonl', 'a') as trade_file:
                        json.dump(trade_data, trade_file)
                        trade_file.write('\n')
                    recent_future_prices.clear()

    plt.clf()
    plt.plot(range(1, len(prices) + 1), prices, label='Price', marker='o')
    
    if moving_average_100:
        plt.plot(range(len(prices) - len(moving_average_100) + 1, len(prices) + 1), moving_average_100, label='Moving Average (100)', marker='o')

    plt.legend()
    plt.draw()
    plt.pause(0.0000001)
    plt.savefig(current_prices_graph)
    plt.figtext

    plt.clf()

    with open('predictions.jsonl', 'r') as f:
        lines = f.readlines()
        prediction_data = [json.loads(line.strip()) for line in lines]
        change_numbers = [entry['predicted_change_number'] for entry in prediction_data]
        price_predicted = [entry['future_price'] for entry in prediction_data]
        plt.plot(change_numbers, price_predicted, label='Price Predicted', marker='o')

    plt.legend()
    plt.draw()
    plt.pause(0.0000001)
    plt.savefig(predicted_prices_graph)
    plt.figtext

    plt.clf()
    plt.plot(range(len(prices)-len(macd)+1, len(prices)+1), macd, label='MACD', marker='o')
    plt.axhline(y=0, color='k', linestyle='-')
    plt.plot(range(len(prices)-len(signal_line)+1, len(prices)+1), signal_line, label='Signal Line', marker='o') 

    plt.legend()
    plt.draw()
    plt.pause(0.0000001)
    plt.savefig(macd_file_path)
    plt.figtext

    with open('analyze.jsonl', 'r') as f:
        analyze_data = [json.loads(line) for line in f]

    with open('prices.jsonl', 'r') as f:
        prices_data = [json.loads(line) for line in f]

    for entry in analyze_data:
        invest_change_number = entry['minimum_price_change_number']
        for price_entry in prices_data:
            if price_entry['change_number'] == invest_change_number:
                current_price_min = price_entry['current_price']
                entry['current_price'] = current_price_min
                minimum_predicted_price = entry['minimum_predicted_price']
                current_price = entry['current_price']
                price_difference_min = abs(minimum_predicted_price - current_price_min)

                if price_difference_min <= (0.0001 * current_price):  # Accuracy: 98% 
                    entry['Accuracy'] = 'Super Accurate'

                elif price_difference_min <= (0.0002*current_price):
                    entry['Accuracy'] = 'Accurate'
                else:
                    entry['Accuracy'] = 'Not Accurate'

                
    with open('analyze.jsonl', 'w') as f:
        for entry in analyze_data:
            json.dump(entry, f)
            f.write('\n')












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

