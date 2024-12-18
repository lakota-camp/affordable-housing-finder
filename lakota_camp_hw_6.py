import warnings
import sklearn.cluster
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

import pandas as pd
pd.set_option('display.max_columns', None)  # Ensure all columns are displayed
pd.set_option('display.width', 1000)       # Set a large enough width

# Set the column width to avoid truncation
pd.set_option('display.max_colwidth', None)

'''
API reference: https://www.zillow.com/research/data/
Data located in data/csv folder
Data file path mapped to CITY_HOME_VALUES dictionary

Housing Market Cluster Analysis program: Visualizes and analyzes housing market patterns 
across cities using Zillow's Home Value Index (ZHVI) data, allowing users to explore 
relationships between growth rates, volatility, and average prices for different 
bedroom configurations.
'''

# Configure data file paths
CITY_HOME_VALUES = {
    'ZHVI 1-Bedroom Time Series ($)': './data/csv/city/City_zhvi_bdrmcnt_1_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
    'ZHVI 2-Bedroom Time Series ($)': './data/csv/city/City_zhvi_bdrmcnt_2_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
    'ZHVI 3-Bedroom Time Series ($)': './data/csv/city/City_zhvi_bdrmcnt_3_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
    'ZHVI 4-Bedroom Time Series ($)': './data/csv/city/City_zhvi_bdrmcnt_4_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
    'ZHVI 5-Bedroom Time Series ($)': './data/csv/city/City_zhvi_bdrmcnt_5_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
}

CONFIG = {
    "start_year": "2020",
    "number_bedrooms": CITY_HOME_VALUES["ZHVI 1-Bedroom Time Series ($)"],
    "missing_data_threshold_percent": 0.25
}

# Test for missing data
def test_missing_data(df):
    # Group columns by year
    missing_by_year = (
        df[[col for col in df.columns if '-' in col]]  # Select only date columns
        .isnull()
        .mean()
        .groupby(lambda col: col[:4])  # Group by year (first 4 characters of column name)
        .mean()
        * 100
    )
    
    print(missing_by_year)
    missing_by_year.plot(kind='line', figsize=(10, 6), marker='o', title="Missing Data Percentage by Year")
    plt.ylabel("Percentage of Missing Data")
    plt.xlabel("Year")
    plt.grid(True)
    plt.show()

def pd_export_csv(df, filepath="housing_df_output.csv"):
    print("Loading data frame to csv file ")
    try:
        df.to_csv(filepath, index=False)
        print("Data frame successfully loaded to csv file:")
        print(filepath)
        return True
    except Exception as e:
        print("Error loading data frame to csv")
        print(e)
        return False



def preprocess_data(df) -> pd.DataFrame:
    """
    Preprocesses the data frame by extracting date columns, filtering out rows with missing home values, 
    and filling in missing values using interpolation, back fill, and front fill.

    Args:
        df (pd.DataFrame): The data frame to preprocess

    Returns:
        pd.DataFrame: The preprocessed data frame
    """
    # Extract date columns give start_year
    date_columns = [col for col in df.columns if col.startswith("20") and col >= CONFIG["start_year"]]
    
    # Extract missing home values rows (Cities) from data columns
    df["missing_home_values"] = df[date_columns].isnull().sum(axis=1)
    
    # Sort missing home values ascending
    df_sorted = df.sort_values(by="missing_home_values", ascending=False)
    
    # Summarize missing home values by RegionName (city)
    missing_summary = df_sorted[['RegionName', 'missing_home_values']]
    
    # Filter only necessary columns
    # filtered_columns = ["RegionID", "SizeRank", "RegionName", "RegionType", "StateName", "State", "Metro", "CountyName"] + data_columns
    filtered_columns = ["RegionName", "State"] + date_columns

    # Drop rows with more than 50% missing home prices
    threshold = len(date_columns) * CONFIG["missing_data_threshold_percent"]
    
    # Extract rows given threshold
    df_cleaned = df[df["missing_home_values"] <= threshold]
    
    # Clean up data frame   
    df = df_cleaned[filtered_columns]
    
    # Fill remaining missing values using interpolation first, then back fill, then front fill
    df = df.apply(lambda row: row.interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"), axis=1)  
    
    start_year_index = 2  # Index of the first date column
    start_year_column = df.iloc[:, start_year_index]  # Select the first column in the DataFrame
    end_year_column = df.iloc[:, -1]  # Select the last column in the DataFrame
    
    # Calculate growth rate, volatility, and average price
    df["GrowthRate"] = (end_year_column - start_year_column) / start_year_column
    df["Volatility"] = df[date_columns].std(axis=1)
    df["AveragePrice"] = df[date_columns].mean(axis=1)
    
    columns = ["GrowthRate", "RegionName", "State", "Volatility", "AveragePrice"] + date_columns
    df = df[columns]
    return df

def scale_features(features):
    """
    Standardizes numerical features by removing the mean and scaling to unit variance.
    Args:
        features (pd.DataFrame): DataFrame containing numerical features to be scaled
    Returns:
        numpy.ndarray: Scaled feature array with mean=0 and variance=1
    """
    scaler = StandardScaler()
    return scaler.fit_transform(features)
    
def build_k_means_model(scaled_features):
    """
    Builds a K-Means clustering model using the given scaled features and returns the cluster centers, labels, and model.
    Uses silhouette scoring to determine the optimal number of clusters (k) between 2 and 10.

    Args:
        scaled_features (numpy.ndarray): Standardized numerical features to be used for clustering

    Returns:
        dict: {
            'cluster_centers': numpy.ndarray
                Coordinates of cluster centers,
            'labels': numpy.ndarray
                Cluster labels for each data point,
            'k_means': sklearn.cluster.KMeans
                Fitted K-Means model with optimal number of clusters
        }
    """    
    max_sil = 0
    k_value = 0
    
    # Determine optimal number of clusters using silhouette score
    for k in range(2, 11, 1):
        k_means = sklearn.cluster.KMeans(n_clusters=k)
        k_means = k_means.fit(scaled_features)
        sil = sklearn.metrics.silhouette_score(scaled_features, k_means.labels_)
        if sil > max_sil:
            max_sil = sil
            k_value = k
            
    print(f"\nBest silhouette score: {max_sil}")
    print(f"K-Mean value: {k_value}")
    
    k_means = sklearn.cluster.KMeans(n_clusters=k_value, random_state=0)
    k_means = k_means.fit(scaled_features)
    
    return {
        "cluster_centers": k_means.cluster_centers_,
        "labels": k_means.labels_,
        "k_means": k_means
    }

def plot_data(df, feature_one, feature_two, k_means_results, num_bedrooms, city_names=False):
    """
    Creates a scatter plot visualizing housing market clusters based on two selected features.
    Points are color-coded by cluster and can optionally display city names.
    Args:
       df (pd.DataFrame): Housing market data
       feature_one (str): Column name for x-axis feature 
       feature_two (str): Column name for y-axis feature
       k_means_results (dict): Dictionary with clustering results including 'labels'
       num_bedrooms (str): Bedroom count identifier for plot title
       city_names (bool, optional): Show city name labels. Defaults to False

    Returns:
       None: Displays matplotlib scatter plot
    """
    # Create color mapping
    color_map = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'orange'}
    colors = [color_map[label] for label in k_means_results["labels"]]
    # Create the scatter plot
    plt.figure(figsize=(15, 10))
    plt.scatter(df[feature_one], df[feature_two], c=colors, s=10)
    if city_names:
        for index, row in df.iterrows():
            plt.annotate(f"{row["RegionName"]}, {row["State"]}",
                        (row[feature_one], row[feature_two]),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=8)
    
    # Customize plot
    plt.title(f"{num_bedrooms}\nCity Housing Markets Clustered by {feature_one} and {feature_two}")
    plt.xlabel(feature_one)
    plt.ylabel(feature_two)
    plt.grid(True, alpha=0.3)
    
    plt.show()

def prompt_user():
    """
    Prompts user to select bedroom count and visualization features.

    Returns:
        dict: {
            'bedrooms': str
                ZHVI dataset identifier based on bedroom count,
            'features': list[str]
                Two feature names for visualization
        } or None if user exits
    """
    # Valid inputs for features and bedrooms
    valid_inputs = ["GrowthRate", "Volatility", "AveragePrice"]
    valid_bedrooms = {
        "1": "1-Bedroom",
        "2": "2-Bedroom",
        "3": "3-Bedroom",
        "4": "4-Bedroom",
        "5": "5+ Bedroom"
    }
    
    def display_bedroom_options():
        print("\nSelect number of bedrooms:")
        for num, desc in valid_bedrooms.items():
            print(f"{num}. {desc}")
        print("(Enter 'e' to exit)")
    
    def display_feature_options():
        print("\nAvailable variables to visualize:")
        for i, option in enumerate(valid_inputs, 1):
            print(f"{i}. {option}")
        print("(Enter 'e' to exit)")

    def get_bedroom_choice():
        while True:
            display_bedroom_options()
            choice = input("\nEnter number of bedrooms: ").strip()
            
            if choice.lower() == 'e':
                return None
                
            if choice in valid_bedrooms:
                return choice
                
            print("\nInvalid input. Please enter a number between 1-5.")

    def get_feature(prompt_text):
        while True:
            display_feature_options()
            choice = input(prompt_text).strip()
            
            # Check for exit
            if choice.lower() == 'e':
                return None
            
            # Handle numeric input
            if choice.isdigit() and 1 <= int(choice) <= len(valid_inputs):
                return valid_inputs[int(choice) - 1]
            
            # Handle text input
            if choice in valid_inputs:
                return choice
            
            print(f"\nInvalid input. Please enter a number (1-{len(valid_inputs)})" 
                  f" or the exact variable name.")

    # Get bedroom choice first
    bedrooms = get_bedroom_choice()
    if bedrooms is None:
        return None

    # Get first feature
    feature_one = get_feature("\nSelect first variable: ")
    if feature_one is None:
        return None

    # Get second feature
    feature_two = get_feature("\nSelect second variable: ")
    if feature_two is None:
        return None
        
    # Check if same features selected
    if feature_one == feature_two:
        print("\nWarning: You've selected the same variable twice. "
              "This may not provide meaningful insights.")
        if input("Continue anyway? (y/n): ").lower() != 'y':
            return prompt_user()  # Restart selection

    match bedrooms:
        case "1":
            bedrooms = "ZHVI 1-Bedroom Time Series ($)"
        case "2":
            bedrooms = "ZHVI 2-Bedroom Time Series ($)"
        case "3":
            bedrooms = "ZHVI 3-Bedroom Time Series ($)"
        case "4":
            bedrooms = "ZHVI 4-Bedroom Time Series ($)"
        case "5":
            bedrooms = "ZHVI 5-Bedroom Time Series ($)"

    
    return {
        "bedrooms": bedrooms,
        "features": [feature_one, feature_two]
    }

def main():        
    user_input = prompt_user()
    
    if user_input:
        try:
            df = pd.read_csv(CITY_HOME_VALUES[user_input["bedrooms"]])
            
            df = preprocess_data(df)
            
            features = df[["GrowthRate", "Volatility", "AveragePrice"]]
            
            scaled_features = scale_features(features)
            
            k_means_results = build_k_means_model(scaled_features)
            
            plot_data(df=df, 
                    feature_one=user_input["features"][0], 
                    feature_two=user_input["features"][1], 
                    k_means_results=k_means_results, 
                    city_names=False,
                    num_bedrooms=user_input["bedrooms"])   
        except Exception as e:
            print(f"An error occurred: {e}") 
    else:
        print("No user input provided. Exiting program.")
    
if __name__ == "__main__":
    main()