from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import random

from pandas._libs.tslibs.offsets import relativedelta


def generate_sample_data(start_date, end_date):
    data = []

    current_date = start_date

    while current_date <= end_date:

        Bread = random.randint(20, 40) + current_date.day / 10
        Fruit = random.randint(10, 20) + current_date.day / 2
        Vegetable = random.randint(10, 20) + current_date.day / 2

        data.append([current_date, Bread, Fruit, Vegetable])
        current_date += timedelta(days=1)

        if current_date.day == 1:
            current_date += relativedelta(months=1)  # Move to the next month

    return data

def CreateMyFakeData():
    # Define start and end dates
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 2, 28)  # February has 28 days in a non-leap year

    # Generate sample data
    sample_data = generate_sample_data(start_date, end_date)

    df = pd.DataFrame(sample_data,
                      columns=["Day", "Bread", "Fruit", "Vegetable"])

    excel_filename = "DemainFood.xlsx"
    df.to_excel(excel_filename, index=False)

    print(f"Sample data saved to {excel_filename}")

def PredictNextTenDay():
    df['Day'] = pd.to_datetime(df['Day'])

    # Extract days from the 'Day' column
    df['Days'] = df['Day'].dt.day

    # Names of categories
    names = ["Bread", "Fruit", "Vegetable"]

    for name in names:
        total_waste_by_day = df.groupby('Days')[name].sum()

        random_values = {}
        for i in range(0, 31):
            random_values[i + 32] = total_waste_by_day.iloc[i] + random.randint(-10,10)

        random_values_series = pd.Series(random_values)
        print(random_values_series)
        total_waste_by_day = pd.concat([total_waste_by_day, random_values_series])

        plt.figure(figsize=(20, 6))
        plt.plot(total_waste_by_day.index, total_waste_by_day.values, linestyle='-', marker='o', label=name)
        plt.title('In Supermarket, Next 30 day amount prediction ' + name)
        plt.xlabel('Day')
        plt.ylabel('Demand')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.legend()
        plt.grid(True)  # Add grid lines
        plt.show()

def ShowLost():
    plt.figure(figsize=(20, 6))

    # Define line styles and markers
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'x', '+']

    # Plot each category with different line styles and markers
    for idx, column in enumerate(df.columns[1:]):
        line_style = line_styles[idx % len(line_styles)]
        marker = markers[idx % len(markers)]
        plt.plot(df['Day'], df[column], linestyle=line_style, marker=marker, label=column)

    # Add title and labels
    plt.xlabel('Day')
    plt.ylabel('Daily Demand tons')
    plt.title('A Super Market Food Demand by Day')
    plt.legend()
    plt.grid(True)  # Add grid lines

    # Show plot
    plt.show()


file_path = '/home/xamblot/PycharmProjects/AIwithPython/DemainFood.xlsx'
df = pd.read_excel(file_path)

PredictNextTenDay()
