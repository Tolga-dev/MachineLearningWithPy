from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import random

from pandas._libs.tslibs.offsets import relativedelta


def generate_sample_data(start_date, end_date):
    data = []

    current_date = start_date
    while current_date <= end_date:
        thresh = random.randint(0, 100)
        Shipping = random.randint(2000+thresh, 2300+thresh)
        Car = random.randint(3000+(int)(thresh*3/2), 3300+(int)(thresh*3/2))
        Fly = random.randint(50+thresh, 150+thresh)

        data.append([current_date,Shipping,Car,Fly])
        current_date += relativedelta(months=1)
    return data


def CreateMyFakeData():
    # Define start and end dates
    start_date = datetime(2012, 1, 1)
    end_date = datetime(2022, 12, 1)

    # Generate sample data
    sample_data = generate_sample_data(start_date, end_date)

    df = pd.DataFrame(sample_data,
                      columns=["Year", "Shipping Lost", "Car Lost", "Fly Lost"])

    excel_filename = "FoodWastingInChain.xlsx"
    df.to_excel(excel_filename, index=False)

    print(f"Sample data saved to {excel_filename}")

def PrintAllTypeWastedByYear():
    df['Year'] = pd.to_datetime(df['Year'])

    # Extract year from the 'Date' column
    df['Date'] = df['Year'].dt.year

    names = ["Shipping Lost", "Car Lost", "Fly Lost"]

    for i in names:
        df['Total_Waste'] = df[i]
        total_waste_by_year = df.groupby('Year')['Total_Waste'].sum()

        plt.figure(figsize=(10, 6))
        total_waste_by_year.plot(kind='line', color='skyblue')
        plt.title('Total Wasted Foods by Year ' + i)
        plt.xlabel('Year')
        plt.ylabel('Total Wasted Quantity')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


def PredictLastTenYear():
    df['Year'] = pd.to_datetime(df['Year'])

    # Extract year from the 'Date' column
    df['Date'] = df['Year'].dt.year

    names = ["Shipping Lost", "Car Lost", "Fly Lost"]

    for i in names:
        total_waste_by_year = df.groupby('Date')[i].sum()
        last_third = total_waste_by_year.iloc[len(total_waste_by_year) - 3]
        last_element = total_waste_by_year.iloc[-1]

        random_values = {}
        for year in range(2022, 2027):
            thresh = random.randint(0, 100)*2

            if last_third < last_element:
                random_values[year] = random.randint(last_third, last_element) + thresh
            else:
                random_values[year] = random.randint(last_element, last_third) + thresh

        random_values_series = pd.Series(random_values)

        # Concatenate total_waste_by_year with random_values_series
        total_waste_by_year = pd.concat([total_waste_by_year, random_values_series])

        plt.figure(figsize=(10, 6))
        total_waste_by_year.plot(kind='line', color='skyblue')
        plt.title('Updated Total Wasted Foods by Year ' + i)
        plt.xlabel('Year')
        plt.ylabel('Total Wasted Quantity')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.legend()
        plt.grid(True)
        plt.show()

file_path = '/home/xamblot/PycharmProjects/AIwithPython/MachineLearning/MyTutorials/foodWasting.xlsx'
df = pd.read_excel(file_path)
PredictLastTenYear()