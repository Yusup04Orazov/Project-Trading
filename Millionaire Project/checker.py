# CHECKER ACCURACY

import json 
from statistics import mean

real_prices = []
predicted_prices = []
errors = []

with open('prices.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        real_prices.append(data['current_price'])

with open('predictions.jsonl', 'r') as f: 
    for line in f:
        data = json.loads(line)
        predicted_prices.append(data['future_price'])

for i in range(len(real_prices)):
    error = predicted_prices[i] - real_prices[i]
    errors.append(error)

print("Mean Absolute Error:", mean(abs(e) for e in errors))
print("Mean Error:", mean(errors))  
print("Min Error:", min(errors))
print("Max Error:", max(errors))

diff_and_date = zip(errors, range(len(real_prices)))
diff_and_date = sorted(diff_and_date)

print("\nLargest Negative Differences:")
for diff, i in diff_and_date[:5]:
    print(i+1, real_prices[i], predicted_prices[i], diff)

print("\nLargest Positive Differences:")  
for diff, i in diff_and_date[-5:]:
    print(i+1, real_prices[i], predicted_prices[i], diff)

# CHECKER ACCURACY