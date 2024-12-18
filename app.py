import csv, json
import pandas as pd

'''

API reference: https://www.zillow.com/research/data/
Data located in data/csv folder
Data file path mapped to HOME_VALUES dictionary

'''

METRO_AND_US_HOME_VALUES = {
    'ZHVI All Homes (SFR, Condo/Co-op) Time Series, Smoothed, Seasonally Adjusted ($)': './data/csv/metro_and_us/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
    'ZHVI All Homes (SFR, Condo/Co-op) Time Series, Raw, Mid-Tier ($)': './data/csv/metro_and_us/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv',
    'ZHVI 1-Bedroom Time Series ($)': './data/csv/metro_and_us/Metro_zhvi_bdrmcnt_1_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
    'ZHVI 2-Bedroom Time Series ($)': './data/csv/metro_and_us/Metro_zhvi_bdrmcnt_2_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
    'ZHVI 3-Bedroom Time Series ($)': './data/csv/metro_and_us/Metro_zhvi_bdrmcnt_3_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
    'ZHVI 4-Bedroom Time Series ($)': './data/csv/metro_and_us/Metro_zhvi_bdrmcnt_4_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
    'ZHVI 5-Bedroom Time Series ($)': './data/csv/metro_and_us/Metro_zhvi_bdrmcnt_5_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
}

CITY_HOME_VALUES = {
    'ZHVI 1-Bedroom Time Series ($)': './data/csv/city/City_zhvi_bdrmcnt_1_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
}

STATE_HOME_VALUES = {
    'ZHVI 1-Bedroom Time Series ($)': './data/csv/state/State_zhvi_bdrmcnt_1_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
}

ZIP_HOME_VALUES = {
    'ZHVI 1-Bedroom Time Series ($)': './data/csv/zip/Zip_zhvi_bdrmcnt_1_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
}

COUNTY_HOME_VALUES = {
    'ZHVI 1-Bedroom Time Series ($)': './data/csv/county/County_zhvi_bdrmcnt_1_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
}

NEIGHBORHOOD_HOME_VALUES = {
    'ZHVI 1-Bedroom Time Series ($)': './data/csv/neighborhood/Neighborhood_zhvi_bdrmcnt_1_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
}

one_bedroom_homes = METRO_AND_US_HOME_VALUES["ZHVI 1-Bedroom Time Series ($)"]
two_bedroom_homes = METRO_AND_US_HOME_VALUES["ZHVI 2-Bedroom Time Series ($)"]
three_bedroom_homes = METRO_AND_US_HOME_VALUES["ZHVI 3-Bedroom Time Series ($)"]
four_bedroom_homes = METRO_AND_US_HOME_VALUES["ZHVI 4-Bedroom Time Series ($)"]
five_bedroom_homes = METRO_AND_US_HOME_VALUES["ZHVI 5-Bedroom Time Series ($)"]

city_one_bedroom_homes = CITY_HOME_VALUES["ZHVI 1-Bedroom Time Series ($)"]

state_one_bedroom_homes = STATE_HOME_VALUES["ZHVI 1-Bedroom Time Series ($)"]

zip_one_bedroom_homes = ZIP_HOME_VALUES["ZHVI 1-Bedroom Time Series ($)"]

county_one_bedroom_homes = COUNTY_HOME_VALUES["ZHVI 1-Bedroom Time Series ($)"]

neighborhood_one_bedroom_homes = NEIGHBORHOOD_HOME_VALUES["ZHVI 1-Bedroom Time Series ($)"]

def load_csv(filename: str) -> list:
    with open(filename, "r") as file:
        reader = csv.reader(file)
        # next(reader)
        data = list(reader)
        return data

def main():
    # data = load_csv(csv_file)
    # print(json.dumps(data, indent=4))
    print("=" * 90)
    print("Metro and US Data\n")
    df = pd.read_csv(one_bedroom_homes)    
    # print(df.head())
    # print(df.tail())  
    # print(df.describe())    
    print(df.sample(20)) 
    
    print("=" * 90)    
    print("City Data\n")
    df = pd.read_csv(city_one_bedroom_homes)    
    # print(df.head())
    # print(df.tail())  
    # print(df.describe())    
    print(df.sample(20))
    print(df[df["RegionName"] == "San Diego"])
    print(df[df["RegionName"] == "Los Angeles"])
    
    print("=" * 90) 
    print("State Data\n")
    df = pd.read_csv(state_one_bedroom_homes)    
    # print(df.head())
    # print(df.tail())  
    # print(df.describe())    
    print(df.sample(20)) 
    print(df[df["RegionName"] == "California"])
    print(df[df["RegionName"] == "Texas"])
    
    print("=" * 90) 
    print("Zip Code Data\n")
    df = pd.read_csv(zip_one_bedroom_homes)    
    # print(df.head())
    # print(df.tail())  
    # print(df.describe())    
    print(df.sample(20)) 
    print(df[df["RegionName"] == 92103])
    print(df[df["RegionName"] == 92109])
    
    print("=" * 90) 
    print("County Data\n")
    df = pd.read_csv(county_one_bedroom_homes)    
    # print(df.head())
    # print(df.tail())  
    # print(df.describe())    
    print(df.sample(20)) 
    print(df[df["RegionName"] == "San Diego County"])
    print(df[df["RegionName"] == "Humboldt County"])
    
    print("=" * 90) 
    print("Neighborhood Data\n")
    df = pd.read_csv(neighborhood_one_bedroom_homes)    
    # print(df.head())
    # print(df.tail())  
    # print(df.describe())    
    print(df.sample(20)) 
    print(df[df["RegionName"] == "Linda Vista"])
    print(df[df["RegionName"] == "Hollywood"])
    
if __name__ == "__main__":
    main()

