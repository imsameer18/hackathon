import pandas as pd

# Your existing code for fetching results_list

max_price = 1500.0  # Set your maximum price here

results_list = []

for stock_symbol in stocks:
    result = fetch_stock_predictions_below_price(stock_symbol, max_price)
    if result:
        results_list.append(result)

# Convert results_list to DataFrame
df = pd.DataFrame(results_list)

# Print the DataFrame
print("Stock predictions below $1500:")
print(df)
