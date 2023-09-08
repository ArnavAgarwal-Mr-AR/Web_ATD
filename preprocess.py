import pandas as pd
import warnings
warnings.filterwarnings("ignore")

'''
from google.colab import drive
drive.mount("/content/gdrive")
df = pd.read_csv('/content/gdrive/My Drive/ML Dataset/StrokeData/strokeDataset.csv')
df_original = df
'''

#Checking how the data looks like
df=pd.read_csv("E:\code_self\Github Projects\Abdominal Trauma detection\Dataset\healthcare-dataset-stroke-data.csv")
df_original = df
df.head()

#Checking the number of non-null values in the dataset:
df.info()

#Reverifying the count of null values in all the columns.
df.isna().sum()
#Checking for any duplicate values:
print("The number of duplicated values is ", df.duplicated().sum())

#Removing the ID column from the dataset
df.drop(columns=['id'],inplace=True) # The inplace=True argument changes the df and stores it directly into its updated form (in same variable)
#A function to avoid calling multiple times
def check_vals(df):
    for column in df.columns:
        print(f"Column: {column}")
        print(df[column].value_counts())
        print("-" * 30)
check_vals(df)

#we can observe the number of entries in our data for the "other" category of gender is just 1, so in order to make the process easy, I am removing the entry corresponding to that value.
df = df[df['gender'] != 'Other']

#Transforming categrical values of data into numerical data:
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender']) # encode male as 1 and female as 0
df['ever_married'] = label_encoder.fit_transform(df['ever_married']) # encode yes as 1 and no as 0
df['Residence_type'] = label_encoder.fit_transform(df['Residence_type']) # encode urban as 1 and rural as 0
df = pd.get_dummies(df, columns=['work_type'], prefix=['work_type'], drop_first=False)
df.head()

check_vals(df)

#smoking_status is left to be transformed
df = pd.get_dummies(df, columns=['smoking_status'], prefix=['s_s'], drop_first=False)

#Populating NULL BMI Values in the data using RandomForestRegressor: Imputation

import matplotlib.pyplot as plt

non_null_bmi = df[df['bmi'].notnull()]
null_bmi = df[df['bmi'].isnull()]

plt.scatter(x=non_null_bmi.index, y=non_null_bmi['bmi'], label='Non-null BMI', alpha=0.7)

if not null_bmi.empty:
    plt.scatter(x=null_bmi.index, y=[0] * len(null_bmi), marker='o', color='red', s=10, label='Null BMI')

plt.title('Scatter Plot of BMI Values')
plt.xlabel('Index')
plt.ylabel('BMI')
plt.legend()

plt.show()


#Now Filling up these null values (imputing) the values using random forest regressor, and checking the root mean squared error for the filled up values with given values.
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt

# Copy the DataFrame
df_imputed = df.copy()

# Separate data with missing and non-missing 'bmi' values
data_with_bmi = df_imputed.dropna(subset=['bmi'])
data_to_impute = df_imputed[df_imputed['bmi'].isnull()]

# Split data for training
X_train = data_with_bmi.drop(['bmi'], axis=1)
y_train = data_with_bmi['bmi']

# Create and train the regressor
regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)

# Predict missing 'bmi' values
imputed_bmi = regressor.predict(data_to_impute.drop(['bmi'], axis=1))

# Fill the missing 'bmi' values with the imputed values
df_imputed.loc[df_imputed['bmi'].isnull(), 'bmi'] = imputed_bmi

# Calculate mean, median, and mode for the initial and final 'bmi' values
initial_mean = df['bmi'].mean()
final_mean = df_imputed['bmi'].mean()

initial_median = df['bmi'].median()
final_median = df_imputed['bmi'].median()

initial_mode = df['bmi'].mode().iloc[0]
final_mode = df_imputed['bmi'].mode().iloc[0]

print(f"Initial Mean BMI: {initial_mean:.2f}, Final Mean BMI: {final_mean:.2f}")
print(f"Initial Median BMI: {initial_median:.2f}, Final Median BMI: {final_median:.2f}")
print(f"Initial Mode BMI: {initial_mode:.2f}, Final Mode BMI: {final_mode:.2f}\n\n")

# Create a scatter plot for original BMI values (in blue)
plt.scatter(x=df.index, y=df['bmi'], label='Original BMI', alpha=0.4)

# Create a scatter plot for imputed BMI values (in green)
plt.scatter(x=data_to_impute.index, y=imputed_bmi, label='Imputed BMI', alpha=0.7, c='green')

plt.title('Scatter Plot of Original and Imputed BMI Values')
plt.xlabel('Index')
plt.ylabel('BMI')
plt.legend()

plt.show()

df=df_imputed # updated the variable df with imputed dataframe

#checking the count of NULL values again and duplicated values again:
df.isna().sum()
print("The count of duplicated values is ", df.duplicated().sum())

#Just moving the stroke categorical column (target variable) to extreme right hand side of the dataframe
stroke_column = df.pop('stroke')
df['stroke'] = stroke_column

#Checking the data distribution in Stroke or Non-stroke
from matplotlib import pyplot as plt
import seaborn as sns
y = df['stroke']
plt.figure(figsize=(5, 3))
y.value_counts().plot.pie(autopct="%1.2f%%", colors=sns.color_palette('Set2'), explode=[0, 0.12], title="Stroke (1) or no stroke (0)");
