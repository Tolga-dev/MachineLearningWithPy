import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pandas._libs.tslibs.offsets import relativedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from MachineLearning.algos.linear.LogisticRegression import LogisticRegression


def random_date(start_date, end_date):
    return start_date + timedelta(days=random.randint(0, (end_date - start_date).days))


def generate_sample_data(start_date, end_date):
    data = []

    current_date = start_date
    while current_date <= end_date:
        product = "Apples"
        production_quantity = random.randint(3000, 3100)

        price_per_unit = round(random.uniform(0.5, 0.8), 2)
        expiration_date = current_date + timedelta(days=random.randint(1, 10))

        Wholesale = random.randint(800, 1000)
        Retail = random.randint(1000, 1100)
        Restaurant = random.randint(700, 800)

        data.append([current_date, product, production_quantity, price_per_unit, expiration_date, Wholesale, Retail,
                     Restaurant])
        current_date += relativedelta(months=1)

    return data


def CreateMyFakeData():
    # Define start and end dates
    start_date = datetime(2012, 1, 1)
    end_date = datetime(2021, 12, 1)

    # Generate sample data
    sample_data = generate_sample_data(start_date, end_date)

    # Create DataFrame
    df = pd.DataFrame(sample_data,
                      columns=["Date", "Product", "Production_Quantity", "Price_Per_Unit", "Expiration_Date",
                               "WholeSale", "Retail", "Restaurant"])

    # Save DataFrame to Excel
    excel_filename = "sample_data6.xlsx"
    df.to_excel(excel_filename, index=False)

    print(f"Sample data saved to {excel_filename}")


def ShowJustOneProductTotalWaste():
    df['Total_Waste'] = df['WholeSale'] + df['Retail'] + df['Restaurant']

    total_waste_by_product = df.groupby('Product')['Total_Waste'].sum()

    plt.figure(figsize=(10, 6))
    total_waste_by_product.plot(kind='bar', color='skyblue')
    plt.title('Total Wasted Foods by Product')
    plt.xlabel('Product')
    plt.ylabel('Total Wasted Quantity')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def ShowJustOneProductTotalWasteByYearBy():
    df['Date'] = pd.to_datetime(df['Date'])

    # Extract year from the 'Date' column
    df['Year'] = df['Date'].dt.year

    # Calculate total wasted foods for each year
    df['Total_Waste'] = df['WholeSale'] + df['Retail'] + df['Restaurant']
    total_waste_by_year = df.groupby('Year')['Total_Waste'].sum()

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    total_waste_by_year.plot(kind='line', color='skyblue')
    plt.title('Total Wasted Foods by Year')
    plt.xlabel('Year')
    plt.ylabel('Total Wasted Quantity')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def PrintAllTypeWastedByYear():
    df['Date'] = pd.to_datetime(df['Date'])

    # Extract year from the 'Date' column
    df['Year'] = df['Date'].dt.year

    names = ['WholeSale', 'Retail', 'Restaurant']

    for i in names:
        # Calculate total wasted foods for each year
        df['Total_Waste'] = df[i]
        total_waste_by_year = df.groupby('Year')['Total_Waste'].sum()

        # Plot the bar chart
        plt.figure(figsize=(10, 6))
        total_waste_by_year.plot(kind='line', color='skyblue')
        plt.title('Total Wasted Foods by Year ' + i)
        plt.xlabel('Year')
        plt.ylabel('Total Wasted Quantity')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


def PrintTotalWastedMoneyOnOneProduct():
    df['Date'] = pd.to_datetime(df['Date'])

    df['Year'] = df['Date'].dt.year

    df['Total_Waste'] = df['WholeSale'] + df['Retail'] + df['Restaurant']
    df['Total_Waste'] *= df['Price_Per_Unit']
    df['Total_Waste'] = (df['Production_Quantity'] * df['Price_Per_Unit']) - df['Total_Waste']
    total_waste_by_year = df.groupby('Year')['Total_Waste'].sum()

    plt.figure(figsize=(10, 6))
    total_waste_by_year.plot(kind='line', color='skyblue')
    plt.title('Total Wasted Money by Year')
    plt.xlabel('Year')
    plt.ylabel('Total Wasted Quantity')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def PredictLastTenYear():
    df['Date'] = pd.to_datetime(df['Date'])

    df['Year'] = df['Date'].dt.year

    names = ['WholeSale', 'Retail', 'Restaurant']

    for i in names:
        total_waste_by_year = df.groupby('Year')[i].sum()
        # Get the last third and last element of total_waste_by_year
        last_third = total_waste_by_year.iloc[len(total_waste_by_year) - 3]
        last_element = total_waste_by_year.iloc[-1]

        random_values = {}
        for year in range(2022, 2027):
            if last_third < last_element:
                random_values[year] = random.randint(last_third, last_element)
            else:
                random_values[year] = random.randint(last_element, last_third)

        print(random_values)
        random_values_series = pd.Series(random_values)

        # Concatenate total_waste_by_year with random_values_series
        total_waste_by_year = pd.concat([total_waste_by_year, random_values_series])

        plt.figure(figsize=(10, 6))
        total_waste_by_year.plot(kind='line', color='skyblue')
        plt.title('Updated Total Wasted Foods by Year')
        plt.xlabel('Year')
        plt.ylabel('Total Wasted Quantity')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
def PredictMoney():
    df['Date'] = pd.to_datetime(df['Date'])

    df['Year'] = df['Date'].dt.year

    df['Total_Waste'] = df['WholeSale'] + df['Retail'] + df['Restaurant']
    df['Total_Waste'] *= df['Price_Per_Unit']
    df['Total_Waste'] = (df['Production_Quantity'] * df['Price_Per_Unit']) - df['Total_Waste']
    total_waste_by_year = df.groupby('Year')['Total_Waste'].sum()

    last_third = total_waste_by_year.iloc[len(total_waste_by_year) - 3]
    last_element = total_waste_by_year.iloc[-1]

    random_values = {}
    for year in range(2022, 2027):
        last_third = (int)(last_third)
        last_element = (int)(last_element)

        if last_third < last_element:
            random_values[year] = random.randint(last_third, last_element)
        else:
            random_values[year] = random.randint(last_element, last_third)

    print(random_values)
    random_values_series = pd.Series(random_values)

    total_waste_by_year = pd.concat([total_waste_by_year, random_values_series])

    plt.figure(figsize=(10, 6))
    total_waste_by_year.plot(kind='line', color='skyblue')
    plt.title('Updated Total Wasted MOney by Year')
    plt.xlabel('Year')
    plt.ylabel('Total Wasted Quantity')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


file_path = '/home/xamblot/PycharmProjects/AIwithPython/MachineLearning/MyTutorials/sample_data6.xlsx'
df = pd.read_excel(file_path)

