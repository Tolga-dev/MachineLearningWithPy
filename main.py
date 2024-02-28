import os

import pandas as pd
import random
from datetime import datetime, timedelta

def random_date(start_date, end_date):
    return start_date + timedelta(days=random.randint(0, (end_date - start_date).days))


def generate_sample_data(start_date, end_date, num_records):
    data = []
    for _ in range(num_records):
        date = random_date(start_date, end_date)
        product = "Apples"
        production_quantity = random.randint(500, 2000)
        price_per_unit = round(random.uniform(0.5, 0.8), 2)
        expiration_date = date + timedelta(days=random.randint(1, 10))
        consumer_types = ["Wholesale", "Retail", "Restaurant"]
        for consumer_type in consumer_types:
            consumption_quantity = random.randint(50, 1000)
            data.append([date, product, production_quantity, price_per_unit, expiration_date, consumer_type, consumption_quantity])
    return data

# Define start and end dates
start_date = datetime(2012, 1, 1)
end_date = datetime(2021, 12, 1)

# Generate sample data
sample_data = generate_sample_data(start_date, end_date, num_records=120)

# Create DataFrame
df = pd.DataFrame(sample_data, columns=["Date", "Product", "Production_Quantity", "Price_Per_Unit", "Expiration_Date", "Consumer_Type", "Consumption_Quantity"])

# Save DataFrame to Excel
excel_filename = "sample_data.csv"
df.to_csv(excel_filename, index=False)

full_path = os.path.realpath(excel_filename)
print("File path:", full_path)

print(f"Sample data saved to {excel_filename}")
