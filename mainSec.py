from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import random

from pandas._libs.tslibs.offsets import relativedelta


def generate_sample_data(start_date, end_date):
    data = []

    current_date = start_date
    thresh = random.randint(10, 100) + current_date.year

    while current_date <= end_date:
        thresh += current_date.year

        Production = random.randint(2000+(int)(thresh*3/2), 2300+(int)(thresh*3/2))

        Car = random.randint(5, 20) + thresh/100
        SuperMarket = random.randint(20,40)+ thresh/100
        SmallSuperMarket = random.randint(10,20)+ thresh/100
        CostumerLost = random.randint(10,20)+ thresh/100

        data.append([current_date,Production,Car,SuperMarket, SmallSuperMarket, CostumerLost])
        current_date += relativedelta(months=1)

    return data


def CreateMyFakeData():
    # Define start and end dates
    start_date = datetime(2012, 1, 1)
    end_date = datetime(2022, 12, 1)

    # Generate sample data
    sample_data = generate_sample_data(start_date, end_date)

    df = pd.DataFrame(sample_data,
                      columns=["Year", "Production of Farmer", "Transportation with Car Lost", "Super Market Lost",
                               "Small Super Market Lost", "Costumer Lost"])

    excel_filename = "FoodWastingInChain.xlsx"
    df.to_excel(excel_filename, index=False)

    print(f"Sample data saved to {excel_filename}")

def ShowLost():
    plt.figure(figsize=(10, 6))

    for column in df.columns[2:]:
        plt.plot(df['Year'], df[column], marker='+', label=column)

    plt.xlabel('Year')
    plt.ylabel('Food Lost Ton')
    plt.title('Food Loss by Year')
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()

file_path = '/home/xamblot/PycharmProjects/AIwithPython/FoodWastingInChain.xlsx'
df = pd.read_excel(file_path)

ShowLost()