#!/usr/bin/env python
# coding: utf-8

# # 🚗 EV Insights: Trends & Predictions in Washington's Clean Vehicle Movement
# ### Author: Prudhvi Raj Rekula  
# ### Data Source: [WA State EV Data](https://catalog.data.gov/dataset/electric-vehicle-population-data)
# 
# This project explores electric vehicle (EV) adoption across Washington State using public data. The goal is to uncover trends, visualize geospatial patterns, and train a model to predict EV types based on vehicle features.

# ## 🔍 Key Findings
# - **Battery Electric Vehicles (BEVs)** tend to have a longer electric range but a higher MSRP.
# - **Registrations** are heavily concentrated in urban counties like King and Snohomish.
# - **Electric range** has significantly increased for models after 2017.
# - **Utility providers** like PSE and Seattle City Light have high EV coverage.
# - **Random Forest Classifier** achieved high accuracy in classifying EV type from features like make, model, MSRP.

# ## 🎯 Project Goal
# Support the **Washington State Department of Transportation** and EV infrastructure planners by analyzing trends in EV adoption and predicting EV classifications based on vehicle attributes.

# In[4]:


get_ipython().system('pip install keplergl')


# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import patsy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')
from keplergl import KeplerGl


# In[6]:


class DataAnalysis:
    def __init__(self, file_path):
        """
        Initializes a DataAnalysis instance by loading a dataset from a file path.

        Parameters:
            file_path (str): Path to the dataset file (CSV format).
        """
        try:
            self.df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
        except Exception as e:
            print(f"Error: {str(e)}")

    def load_data(self, file_path):
        """
        Loads the dataset from a file path and returns a pandas DataFrame.

        Parameters:
            file_path (str): Path to the dataset file (CSV format).

        Returns:
            pandas.DataFrame: Loaded DataFrame containing the dataset.
        """
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
        except Exception as e:
            print(f"Error: {str(e)}")

    def filter_data(self, column, value):
        """
        Filters the DataFrame based on a column and value.
        Returns a new DataFrame with the filtered rows.
        """
        try:
            return self.df[self.df[column] == value]
        except KeyError:
            print(f"Error: Column '{column}' not found in the DataFrame.")
        except Exception as e:
            print(f"Error: {str(e)}")

    def group_data(self, group_by, aggregate_func=None):
        """
        Groups the DataFrame based on one or more columns.
        Optionally applies an aggregate function to a specified column.
        Returns a grouped DataFrame or a Series if an aggregate function is provided.
        """
        try:
            if aggregate_func is None:
                return self.df.groupby(group_by)
            else:
                column, func = aggregate_func
                return self.df.groupby(group_by)[column].agg(func)
        except KeyError:
            print(f"Error: Column(s) '{group_by}' or '{column}' not found in the DataFrame.")
        except Exception as e:
            print(f"Error: {str(e)}")

    def print_vehicle_info(self, *args, **kwargs):
        """
        Prints vehicle information based on the provided arguments.
        *args: Positional arguments (VIN, model, make, etc.)
        **kwargs: Keyword arguments (verbose=True/False, show_range=True/False, electric_range=value)
        """
        try:
            verbose = kwargs.get('verbose', False)
            show_range = kwargs.get('show_range', False)
            print("Vehicle Information:")
            for arg in args:
                print(f"- {arg}")
            if verbose:
                print("Additional Information:")
                for key, value in kwargs.items():
                    if key != 'verbose' and key != 'show_range':
                        print(f" {key}: {value}")
            if show_range:
                electric_range = kwargs.get('electric_range', None)
                if electric_range is not None:
                    print(f"Electric Range: {electric_range} miles")
                else:
                    print("Error: Electric range not provided")
        except Exception as e:
            print(f"Error: {str(e)}")

    def perform_linear_regression(self):
        """
        Performs linear regression to predict Base MSRP based on Model Year and Electric Range.
        Removes outliers and evaluates using R-squared.
        """
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score

            # Keep only numeric features
            df_reg = self.df[['Model Year', 'Electric Range', 'Base MSRP']].dropna()

            # Remove top 1% MSRP outliers
            q_high = df_reg['Base MSRP'].quantile(0.99)
            df_reg = df_reg[df_reg['Base MSRP'] <= q_high]

            X = df_reg[['Model Year', 'Electric Range']]
            y = df_reg['Base MSRP']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r_squared = r2_score(y_test, y_pred)
            print(f"✅ Linear Regression Completed")
            print(f"R-squared score: {r_squared:.4f}")

        except Exception as e:
            print(f"Error during linear regression: {str(e)}")



    def plot_histogram_for_make(self):
        """
        Plots a histogram for the top 10 makes in the dataset.

        This method counts the frequency of each vehicle make and plots a histogram
        showing the distribution of the top 10 vehicle makes.

        Returns:
            None
        """
        try:
            plt.figure(figsize=(10, 6))
            self.df['Make'].value_counts().head(10).plot(kind='bar', color='skyblue')
            plt.title('Histogram of Top 10 Makes')
            plt.xlabel('Make')
            plt.ylabel('Frequency')
            plt.grid(axis='y')
            plt.show()
        except Exception as e:
            print(f"Error: {str(e)}")

    def plot_pair_plot(self):
        """
        Plots a pair plot for the features in the dataset.

        This method creates a pair plot to visualize pairwise relationships
        between different features in the dataset.

        Returns:
            None
        """
        try:
            sns.pairplot(self.df)
            plt.show()
        except Exception as e:
            print(f"Error: {str(e)}")

    def plot_heatmap(self):
        """
        Plots a heatmap for the correlation between numerical features in the dataset.

        This method calculates the correlation matrix for numerical features
        and displays the correlation heatmap.

        Returns:
            None
        """
        try:
            numeric_df = self.df.select_dtypes(include=['number'])
            correlation_matrix = numeric_df.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Correlation Heatmap of Features')
            plt.show()
        except Exception as e:
            print(f"Error: {str(e)}")

    def plot_scatter_plot(self):
        """
        Plots a scatter plot for Model Year vs. Make.

        This method creates a scatter plot to visualize the relationship
        between model year and vehicle make.

        Returns:
            None
        """
        try:
            plt.figure(figsize=(10, 8))
            plt.scatter(self.df['Model Year'], self.df['Make'])
            plt.show()
        except Exception as e:
            print(f"Error: {str(e)}")

    def plot_vehicle_type_distribution(self):
        """
        Plots a bar chart for the distribution of electric vehicles by type.

        This method counts the distribution of electric vehicle types and
        plots a bar chart to visualize the counts.

        Returns:
            None
        """
        try:
            vehicle_type_counts = self.df['Electric Vehicle Type'].value_counts()
            plt.figure(figsize=(8, 6))
            vehicle_type_counts.plot(kind='bar')
            plt.title('Distribution of Electric Vehicles by Type')
            plt.xlabel('Vehicle Type')
            plt.ylabel('Count')
            plt.show()
        except Exception as e:
            print(f"Error: {str(e)}")

    def plot_model_year_distribution(self):
        """
        Plots a histogram for the distribution of Model Year.

        This method creates a histogram to visualize the distribution
        of vehicle model years in the dataset.

        Returns:
            None
        """
        try:
            plt.figure(figsize=(8, 6))
            plt.hist(self.df['Model Year'], bins=20, alpha=0.7, color='blue', edgecolor='black')
            plt.title('Distribution of Model Year')
            plt.xlabel('Model Year')
            plt.ylabel('Frequency')
            plt.show()
        except KeyError:
            print("Error: 'Model Year' column not found in the DataFrame.")
        except Exception as e:
            print(f"Error: {str(e)}")

# Electric Range Analysis
    def plot_electric_range_distribution(self):
        """
        Plots a histogram for the distribution of Electric Range.

        This method creates a histogram to visualize the distribution of electric range values in the dataset.

        Returns:
            None
        """
        try:
            plt.figure(figsize=(8, 6))
            plt.hist(self.df['Electric Range'], bins=20, alpha=0.7, color='green', edgecolor='black')
            plt.title('Distribution of Electric Range')
            plt.xlabel('Electric Range')
            plt.ylabel('Frequency')
            plt.show()
        except KeyError:
            print("Error: 'Electric Range' column not found in the DataFrame.")
        except Exception as e:
            print(f"Error: {str(e)}")

# Count Analysis
    def plot_cafv_eligibility_count(self):
        """
        Plots a bar chart for the count of Clean Alternative Fuel Vehicle (CAFV) Eligibility.

        This method counts the number of vehicles eligible for CAFV and plots a bar chart to show the counts.

        Returns:
            None
        """
        try:
            self.df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].value_counts().plot(kind='bar')
            plt.title('Count of CAFV Eligibility')
            plt.xlabel('CAFV Eligibility')
            plt.ylabel('Count')
            plt.show()
        except KeyError:
            print("Error: 'Clean Alternative Fuel Vehicle (CAFV) Eligibility' column not found in the DataFrame.")
        except Exception as e:
            print(f"Error: {str(e)}")

# Scatter Plot Analysis between Model Year and Electric Range
    def plot_model_year_vs_electric_range(self):
        """
        Plots a scatter plot for Model Year vs. Electric Range.

        This method creates a scatter plot to visualize the relationship between model year and electric range.

        Returns:
            None
        """
        try:
            plt.figure(figsize=(8, 6))
            plt.scatter(self.df['Model Year'], self.df['Electric Range'])
            plt.title('Model Year vs Electric Range')
            plt.xlabel('Model Year')
            plt.ylabel('Electric Range')
            plt.show()
        except KeyError:
            print("Error: 'Model Year' or 'Electric Range' column not found in the DataFrame.")
        except Exception as e:
            print(f"Error: {str(e)}")

# Vehicle analysis in top 10 counties
    def plot_top_counties(self):
        """
        Plots a bar chart for the count of vehicles by county (top 10).

        This method counts the number of vehicles in each county and plots a bar chart showing the top 10 counties.

        Returns:
            None
        """
        try:
            top_counties = self.df['County'].value_counts().head(10)
            plt.figure(figsize=(10, 6))
            top_counties.plot(kind='bar', color='purple')
            plt.title('Count of Vehicles by County (Top 10)')
            plt.xlabel('County')
            plt.ylabel('Count')
            plt.show()
        except KeyError:
            print("Error: 'County' column not found in the DataFrame.")
        except Exception as e:
            print(f"Error: {str(e)}")

# Vehicle Analysis in top 10 cities
    def plot_top_cities(self):
        """
        Plots a bar chart for the count of vehicles by city (top 10).

        This method counts the number of vehicles in each city and plots a bar chart showing the top 10 cities.

        Returns:
            None
        """
        try:
            top_cities = self.df['City'].value_counts().head(10)
            plt.figure(figsize=(10, 6))
            top_cities.plot(kind='bar', color='brown')
            plt.title('Count of Vehicles by City (Top 10)')
            plt.xlabel('City')
            plt.ylabel('Count')
            plt.show()
        except KeyError:
            print("Error: 'City' column not found in the DataFrame.")
        except Exception as e:
            print(f"Error: {str(e)}")

# Utility Company Analysis
    def plot_top_utility_companies(self):
        """
        Plots a bar chart for the distribution of electric vehicles by top 5 utility companies.

        This method counts the distribution of electric vehicles among the top 5 utility companies and plots a bar chart.

        Returns:
            None
        """
        try:
            utility_company_counts = self.df['Electric Utility'].value_counts().head(5)
            plt.figure(figsize=(12, 6))
            utility_company_counts.plot(kind='bar')
            plt.title('Distribution of Electric Vehicles by Top 5 Utility Companies')
            plt.xlabel('Utility Company')
            plt.ylabel('Count')
            plt.xticks(rotation=90)
            plt.show()
        except KeyError:
            print("Error: 'Electric Utility' column not found in the DataFrame.")
        except Exception as e:
            print(f"Error: {str(e)}")

# Comparision Analysis
    def plot_bev_phev_comparison(self):
        """
        Plots a comparison of Battery Electric Vehicles (BEVs) and Plug-in Hybrid Electric Vehicles (PHEVs)
        based on Electric Range vs. MSRP (Base Manufacturer's Suggested Retail Price).

        This method creates a scatter plot comparing BEVs and PHEVs based on their electric range and MSRP.

        Returns:
            None
        """
        try:
            bev_data = self.df[self.df['Electric Vehicle Type'] == 'Battery Electric Vehicle (BEV)']
            phev_data = self.df[self.df['Electric Vehicle Type'] == 'Plug-in Hybrid Electric Vehicle (PHEV)']

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            sns.scatterplot(x='Electric Range', y='Base MSRP', data=bev_data, alpha=0.5)
            plt.title('BEVs: Electric Range vs. MSRP')
            plt.xlabel('Electric Range (miles)')
            plt.ylabel('MSRP ($)')

            plt.subplot(1, 2, 2)
            sns.scatterplot(x='Electric Range', y='Base MSRP', data=phev_data, alpha=0.5)
            plt.title('PHEVs: Electric Range vs. MSRP')
            plt.xlabel('Electric Range (miles)')
            plt.ylabel('MSRP ($)')

            plt.tight_layout()
            plt.show()
        except KeyError:
            print("Error: 'Electric Vehicle Type' or 'Electric Range' or 'Base MSRP' column not found in the DataFrame.")
        except Exception as e:
            print(f"Error: {str(e)}")

# Multivariate Analysis
    def plot_multivariate_analysis(self):
        """
        Optimized multivariate analysis using sample data and selected features.
        """
        try:
            columns_to_analyze = ['Model Year', 'Electric Range', 'Base MSRP', 'Electric Vehicle Type']
            multivariate_data = self.df[columns_to_analyze].dropna()

            # Sample 1000 rows to avoid performance issues
            sampled_data = multivariate_data.sample(n=1000, random_state=42)

            # Plot using Electric Vehicle Type (fewer categories)
            import seaborn as sns
            import matplotlib.pyplot as plt
            sns.pairplot(sampled_data, hue='Electric Vehicle Type', diag_kind='kde', corner=True)
            plt.show()

        except Exception as e:
            print(f"Error: {str(e)}")

# Merging two DataFrames
    def merge_dataframes(self, other_file_path, on='County', how='inner'):
        """
        Merges the current DataFrame with another DataFrame read from a CSV file.

        This method reads another DataFrame from the specified CSV file path,
        merges it with the current DataFrame based on the specified 'on' column and 'how' merge type,
        and returns the merged DataFrame.

        Args:
            other_file_path (str): Path to the CSV file containing the other DataFrame.
            on (str): Column name to merge on (default is 'County').
            how (str): Type of merge to perform (default is 'inner').

        Returns:
            pd.DataFrame: Merged DataFrame.
        """
        try:
            other_df = pd.read_csv(other_file_path)
            merged_df = pd.merge(self.df, other_df, on=on, how=how)
            return merged_df
        except FileNotFoundError:
            print(f"Error: File not found at {other_file_path}")
        except Exception as e:
            print(f"Error: {str(e)}")

# Slicing Data of a Model Year column value (like year and filtering values depending on the categorical data)
    def filter_by_model_year(self, model_year):
        """
        Filters the DataFrame to include rows with a specific Model Year.

        This method filters the DataFrame to include only rows where the 'Model Year' column matches the specified year.

        Args:
            model_year (int): The model year to filter by.

        Returns:
            pd.DataFrame: Filtered DataFrame based on the specified Model Year.
        """
        try:
            return self.df[self.df['Model Year'] == model_year]
        except KeyError:
            print("Error: 'Model Year' column not found in the DataFrame.")
        except Exception as e:
            print(f"Error: {str(e)}")

# Representing data in matrix form
    def get_data_matrix(self):
        """
        Retrieves the entire DataFrame as a NumPy array.

        This method returns the entire DataFrame represented as a NumPy array.

        Returns:
            numpy.ndarray: The DataFrame converted into a NumPy array.
        """
        try:
            return self.df.values
        except Exception as e:
            print(f"Error: {str(e)}")

    def get_selected_data_matrix(self, selected_columns):
        """
        Retrieves selected columns of the DataFrame as a NumPy array.

        This method returns a subset of the DataFrame, consisting of the specified 'selected_columns',
        represented as a NumPy array.

        Args:
            selected_columns (list): List of column names to select.

        Returns:
            numpy.ndarray: Subset of the DataFrame (selected columns) converted into a NumPy array.
        """
        try:
            selected_data = self.df[selected_columns]
            return selected_data.values
        except KeyError:
            print(f"Error: One or more columns not found in the DataFrame.")
        except Exception as e:
            print(f"Error: {str(e)}")

# Uploading data to Numerical Python (NumPy)
    def get_numpy_array(self):
        """
        Retrieves the DataFrame as a NumPy array.

        This method converts the entire DataFrame into a NumPy array.

        Returns:
            numpy.ndarray: The DataFrame converted into a NumPy array.
        """
        try:
            return np.array(self.df)
        except Exception as e:
            print(f"Error: {str(e)}")

# Selecting data based on a category
    def filter_by_category(self, column, category):
        """
        Filters the DataFrame based on a specific category within a column.

        This method filters the DataFrame to include only rows where the specified 'column' matches the 'category'.

        Args:
            column (str): Name of the column to filter by.
            category (str or int): Category to filter on within the specified column.

        Returns:
            pd.DataFrame: Filtered DataFrame based on the specified category.
        """
        try:
            return self.df[self.df[column] == category]
        except KeyError:
            print(f"Error: Column '{column}' not found in the DataFrame.")
        except Exception as e:
            print(f"Error: {str(e)}")

# Using mathematical and statistical functions using libraries
    def create_brand_model_column(self):
        """
        Creates a new column 'Brand & Model' combining 'Make' and 'Model Year' columns.

        This method concatenates the 'Make' and 'Model Year' columns to create a new column 'Brand & Model'.

        Returns:
            pd.Series: The newly created 'Brand & Model' column.
        """
        try:
            self.df['Brand & Model'] = self.df['Make'] + ' ' + self.df['Model Year'].astype(str)
            return self.df['Brand & Model']
        except KeyError:
            print("Error: 'Make' or 'Model Year' column not found in the DataFrame.")
        except Exception as e:
            print(f"Error: {str(e)}")

    def sort_by_column(self, column, ascending=False):
        """
        Sorts the DataFrame based on a specified column.

        This method sorts the DataFrame based on the specified 'column' in either ascending or descending order.

        Args:
            column (str): Name of the column to sort by.
            ascending (bool, optional): Whether to sort in ascending (True) or descending (False) order.
                                        Default is False (descending order).

        Returns:
            pd.DataFrame: Sorted DataFrame based on the specified column and order.
        """
        try:
            return self.df.sort_values(by=column, ascending=ascending)
        except KeyError:
            print(f"Error: Column '{column}' not found in the DataFrame.")
        except Exception as e:
            print(f"Error: {str(e)}")

    def create_keplergl_map(self):
        """
        Creates an interactive map using KeplerGl for visualizing vehicle locations.

        This method extracts latitude and longitude from the 'Vehicle Location' column,
        samples the data, and displays an interactive map using KeplerGl.

        Returns:
            None
        """
        try:
            # Extract latitude and longitude
            self.df['Latitude'] = self.df['Vehicle Location'].apply(lambda x: float(x.split()[1][1:]) if isinstance(x, str) else None)
            self.df['Longitude'] = self.df['Vehicle Location'].apply(lambda x: float(x.split()[2][:-1]) if isinstance(x, str) else None)

            # Randomly sample 100000 data points
            sampled_df = self.df.sample(n=100000, random_state=42)

            # Create a Kepler.gl map
            map_1 = KeplerGl(height=600)
            map_1.add_data(data=sampled_df, name='vehicle_locations')

            return map_1

        except Exception as e:
            print("An error occurred:", e)
            return None

    def preprocess_data(self):
        # Select relevant features and target variable
        features = ['Model Year', 'Make', 'Model', 'Electric Range', 'Base MSRP', 'Electric Vehicle Type']
        df_cleaned = self.df[features].dropna()  # Drop rows with any missing values

        # Split features and target variable
        X = df_cleaned.drop('Electric Vehicle Type', axis=1)
        y = df_cleaned['Electric Vehicle Type']

        # One-hot encode categorical features
        categorical_features = ['Make', 'Model']
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )

        # Apply preprocessing
        X_encoded = preprocessor.fit_transform(X)

        return X_encoded, y

    def train_random_forest_classifier(self):
        """
        Trains a Random Forest classifier with sampled, stratified, and encoded data.
        Prevents overfitting and improves generalization.
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.compose import ColumnTransformer
            from sklearn.metrics import accuracy_score, classification_report

            # Select features and drop NAs
            df_class = self.df[['Model Year', 'Make', 'Model', 'Electric Range', 'Base MSRP', 'Electric Vehicle Type']].dropna()

            # Sample 10,000 rows for training
            df_sampled = df_class.sample(n=10000, random_state=42)

            # Prepare data
            X = df_sampled.drop('Electric Vehicle Type', axis=1)
            y = df_sampled['Electric Vehicle Type']

            # One-hot encode Make/Model
            cat_features = ['Make', 'Model']
            preprocessor = ColumnTransformer(
                transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)],
                remainder='passthrough'
            )
            X_encoded = preprocessor.fit_transform(X)

            # Stratified split to preserve BEV/PHEV ratio
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

            # Train Random Forest
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            y_pred = rf_model.predict(X_test)

            print("✅ Random Forest Classification Completed")
            print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
            print("Classification Report:")
            print(classification_report(y_test, y_pred))

        except Exception as e:
            print(f"Error during Random Forest training: {str(e)}")


# In[7]:


data_analysis = DataAnalysis("Electric_Vehicle_Population_Data.csv")
data = data_analysis.load_data("Electric_Vehicle_Population_Data.csv")


# In[8]:


# Filter data
filtered_data = data_analysis.filter_data(column='Model Year', value=2020)
filtered_data


# In[14]:


# Filter data
filtered_data = data_analysis.filter_data(column='Model Year', value=2020)
filtered_data


# In[16]:


# Print vehicle info
data_analysis.print_vehicle_info('VIN123', 'Tesla', 'Model S', verbose=True, show_range=True, electric_range=350)


# In[18]:


# Plot histogram for make
data_analysis.plot_histogram_for_make()


# In[19]:


# Sample 1000 rows from your loaded data
sampled_df = data_analysis.df.sample(n=1000, random_state=42)

# Now draw the pairplot with selected columns and simpler hue
import seaborn as sns
cols = ['Model Year', 'Electric Range', 'Base MSRP', 'Electric Vehicle Type']
sns.pairplot(sampled_df[cols], hue='Electric Vehicle Type', diag_kind='kde', corner=True)


# In[20]:


# Plot heatmap
data_analysis.plot_heatmap()


# In[21]:


# Plot scatter plot
data_analysis.plot_scatter_plot()


# In[23]:


# Plot vehicle type distribution
data_analysis.plot_vehicle_type_distribution()


# In[24]:


# Plot model year distribution
data_analysis.plot_model_year_distribution()


# In[25]:


# Plot electric range distribution
data_analysis.plot_electric_range_distribution()


# In[26]:


# Plot CAFV eligibility count
data_analysis.plot_cafv_eligibility_count()


# In[27]:


# Plot model year vs electric range
data_analysis.plot_model_year_vs_electric_range()


# In[28]:


# Plot top counties
data_analysis.plot_top_counties()


# In[29]:


# Plot top cities
data_analysis.plot_top_cities()


# In[30]:


# Plot top utility companies
data_analysis.plot_top_utility_companies()


# In[31]:


# Plot BEV vs PHEV comparison
data_analysis.plot_bev_phev_comparison()


# In[32]:


# Plot multivariate analysis
data_analysis.plot_multivariate_analysis()


# In[33]:


# Filter by model year
filtered_by_model_year = data_analysis.filter_by_model_year(model_year=2020)
filtered_by_model_year


# In[34]:


# Get data matrix
data_matrix = data_analysis.get_data_matrix()
data_matrix


# In[35]:


# Get selected data matrix
selected_data_matrix = data_analysis.get_selected_data_matrix(selected_columns=['Make', 'Model Year', 'Electric Range'])
selected_data_matrix


# In[36]:


# Get numpy array
numpy_array = data_analysis.get_numpy_array()
numpy_array


# In[37]:


# Create brand model column
brand_model_column = data_analysis.create_brand_model_column()
brand_model_column


# In[38]:


# Sort by column
sorted_data = data_analysis.sort_by_column(column='Model Year')
sorted_data


# In[39]:


# Perform linear regression
data_analysis.perform_linear_regression()


# In[40]:


# Call the create_keplergl_map method using the instance
data_analysis.create_keplergl_map()


# In[41]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

# Step 1: Filter and sample the dataset
df_sampled = data_analysis.df[['Model Year', 'Make', 'Electric Range', 'Electric Vehicle Type']].dropna()
df_sampled = df_sampled.sample(n=10000, random_state=42)

# Step 2: Prepare X and y
X = df_sampled[['Model Year', 'Make', 'Electric Range']]
y = df_sampled['Electric Vehicle Type']

# Step 3: Encode categorical variables
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['Make'])],
    remainder='passthrough'
)

X_encoded = preprocessor.fit_transform(X)

# Step 4: Cross-validate the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(rf_model, X_encoded, y, cv=5)

# Step 5: Output results
print("✅ Cross-validated Accuracy Scores:", scores)
print("📊 Mean Accuracy:", scores.mean())


# In[ ]:





# ## 🚀 Model Comparison: Random Forest vs XGBoost

# In[44]:


from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Step 1: Preprocessing
df = data_analysis.df[['Model Year', 'Make', 'Electric Range', 'Electric Vehicle Type']].dropna()
df = df.sample(n=10000, random_state=42)

X = df[['Model Year', 'Make', 'Electric Range']]
y = df['Electric Vehicle Type']

# Step 2: Encode categorical 'Make'
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['Make'])],
    remainder='passthrough'
)

X_encoded = preprocessor.fit_transform(X)

# Step 3: Encode y labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Step 5: XGBoost Training
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Step 6: Evaluate
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Classification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=le.classes_))


# In[45]:


print(df['Electric Vehicle Type'].value_counts())


# In[46]:


X = df[['Model Year', 'Electric Range']].copy()  # make an explicit copy
X['Electric Range'] += np.random.normal(0, 1, size=len(X))


# In[47]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
import numpy as np
import pandas as pd

# Step 1: Filter and prepare data
df_clean = data_analysis.df[['Model Year', 'Electric Range', 'Electric Vehicle Type']].dropna()
df_clean = df_clean.sample(n=10000, random_state=42)

X = df_clean[['Model Year', 'Electric Range']].copy()
X['Electric Range'] += np.random.normal(0, 1, size=len(X))

y = df_clean['Electric Vehicle Type']

# Step 2: Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Step 3: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Cross-validation with XGBoost
xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42)
scores = cross_val_score(xgb_model, X_scaled, y_encoded, cv=5)

# Step 5: Print results
print("✅ Cross-validated Accuracy Scores:", scores)
print("📊 Mean Accuracy:", np.mean(scores))
print("📉 Standard Deviation:", np.std(scores))


# In[48]:


# Fit the model manually
xgb_model.fit(X_scaled, y_encoded)

# Then get feature importances
importances = xgb_model.feature_importances_

# If you're using a preprocessor (e.g. ColumnTransformer), get feature names
# If not, just use X.columns or a manual list
feat_names = ['Model Year', 'Electric Range']  # or use preprocessor.get_feature_names_out() if pipeline used

# Create DataFrame of importances
feat_imp_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'])
plt.title("XGBoost Feature Importances")
plt.gca().invert_yaxis()
plt.show()


# In[ ]:





# ## 📉 Linear Regression on MSRP using Model Year and Electric Range

# In[51]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Predicting Base MSRP from Model Year and Electric Range
try:
    df_reg = data_analysis.df[['Model Year', 'Electric Range', 'Base MSRP']].dropna()
    X = df_reg[['Model Year', 'Electric Range']]
    y = df_reg['Base MSRP']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    print("✅ Linear Regression Completed")
    print("R-squared score:", r2_score(y_test, y_pred))
except Exception as e:
    print("Error during linear regression:", e)


# ## 🌲 Random Forest Classifier (Improved) with Sampled Data

# In[53]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Use a sampled and properly preprocessed dataset
try:
    df_class = data_analysis.df[['Model Year', 'Make', 'Model', 'Electric Range', 'Base MSRP', 'Electric Vehicle Type']].dropna()

    # Sample to reduce memory and overfitting
    df_sampled = df_class.sample(n=10000, random_state=42)

    X = df_sampled.drop('Electric Vehicle Type', axis=1)
    y = df_sampled['Electric Vehicle Type']

    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer

    cat_features = ['Make', 'Model']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ],
        remainder='passthrough'
    )

    X_encoded = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    print("✅ Random Forest Classification Completed")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
except Exception as e:
    print("Error during Random Forest training:", e)


# ## 🤖 Machine Learning Model Comparisons

# In[55]:


# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

# Sample and clean data
df_sampled = data_analysis.df[['Model Year', 'Make', 'Electric Range', 'Electric Vehicle Type']].dropna()
df_sampled = df_sampled.sample(n=10000, random_state=42)

X = df_sampled[['Model Year', 'Make', 'Electric Range']]
y = df_sampled['Electric Vehicle Type']

# One-hot encode 'Make'
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['Make'])],
    remainder='passthrough'
)

X_encoded = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)


# ### 🌲 Random Forest Classifier

# In[57]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Accuracy (Random Forest):", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


# ### 🔁 Logistic Regression

# In[59]:


from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("Accuracy (Logistic Regression):", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))


# ### ⚡ XGBoost Classifier

# In[61]:


from sklearn.preprocessing import LabelEncoder

# Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Now fit the model
xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train_encoded)

# Predict and evaluate
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test_encoded, y_pred_xgb))
print(classification_report(y_test_encoded, y_pred_xgb, target_names=le.classes_))


# ### 📊 Confusion Matrices

# In[63]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Plot confusion matrix with label fix
cm = confusion_matrix(y_test_encoded, y_pred_xgb)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap='Purples', ax=ax)

# Rotate x-axis labels for readability
plt.xticks(rotation=20, ha='right')
plt.yticks(rotation=0)

# Adjust layout to prevent clipping
plt.tight_layout()
plt.show()


# ### 🔍 Feature Importance (Random Forest + XGBoost)

# In[65]:


import numpy as np

feature_names = preprocessor.get_feature_names_out()

# Random Forest Importance
importances_rf = rf_model.feature_importances_
feat_imp_rf = pd.DataFrame({'Feature': feature_names, 'Importance': importances_rf})
feat_imp_rf.sort_values(by='Importance', ascending=False).head(10).plot(
    kind='bar', x='Feature', y='Importance', title='Random Forest - Top 10 Features', legend=False
)
plt.tight_layout()
plt.show()

# XGBoost Importance
importances_xgb = xgb_model.feature_importances_
feat_imp_xgb = pd.DataFrame({'Feature': feature_names, 'Importance': importances_xgb})
feat_imp_xgb.sort_values(by='Importance', ascending=False).head(10).plot(
    kind='bar', x='Feature', y='Importance', title='XGBoost - Top 10 Features', legend=False
)
plt.tight_layout()
plt.show()


# In[ ]:





# 
# ## 🧮 Model Comparison Summary
# 
# | Model                 | Accuracy | Notes |
# |----------------------|----------|-------|
# | Logistic Regression  | ~98.0%   | Performs well, simple and interpretable |
# | Random Forest        | ~99.9%   | Excellent performance, handles non-linearity |
# | XGBoost              | ~99.9%   | Best in class with strong generalization |
# 
# All models perform strongly due to clear separability between BEVs and PHEVs based on Electric Range and Model Year.
# 

# 
# ## 📈 Executive Summary
# 
# This analysis of electric vehicles (EVs) registered in Washington state reveals the following:
# 
# - **BEVs** tend to have significantly higher electric range and MSRP than **PHEVs**.
# - **Vehicle adoption** is concentrated in counties like **King** and **Snohomish**, highlighting urban demand.
# - **Model years post-2017** show a notable improvement in EV range, signaling technological advancement.
# - Machine Learning models, especially **Random Forest and XGBoost**, achieved over **99% accuracy** in classifying EV types based on a small number of features.
# 
# This supports strong confidence in predictive infrastructure planning and targeted EV incentives by region and vehicle type.
# 

# 
# ## ✅ Conclusion & Recommendations
# 
# - The **XGBoost classifier** demonstrated the most consistent and accurate results, making it a reliable model for deployment.
# - Based on vehicle type concentration and registration density, Washington state may prioritize:
#   - **BEV charging infrastructure** in high-density counties like **King**
#   - **Incentives or awareness campaigns** for PHEVs in underserved counties
# - This analysis can be extended by incorporating **charging station data**, **income demographics**, or **utility load forecasts** for deeper urban planning.
# 
# > Overall, this project reflects strong modeling capability, feature engineering, and practical policy insights — ideal for data-driven decision making.
# 

# ## 📌 Conclusion & Business Implications
# - Washington shows promising EV adoption growth, especially in urban and utility-focused regions.
# - BEVs offer superior range but come at a higher upfront cost, informing policy subsidies or incentives.
# - Infrastructure planning can prioritize top utility zones and counties with growing EV density.
# 

# In[ ]:




