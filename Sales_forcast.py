# importing necessay files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# Load the data
# get the file
data = pd.read_csv('Walmart_Store_sales.csv')
print(data.head())

# Printing the store that has maximum sales
max_sales =data['Weekly_Sales'].max()
max_sales_store = data[data['Weekly_Sales'] == max_sales]['Store'].values[0]
print("The store with maximum sales of ",max_sales, "is", max_sales_store)

# Printing the store that has maximum standard deviation of sales and also the coefficient of mean to standard deviation
# calculating standard deviation of each store.
std_dev = data.groupby('Store')['Weekly_Sales'].std()
# finding the maximum standard deviation value
std_dev_max = std_dev.max()
print("\nThe standard deviation of sales for each store is :\n",std_dev)
print("The maximum standard deviation is ", std_dev_max)
# fining the store 
std_dev_store = std_dev[std_dev == std_dev_max].index[0]
print("The store with maximum standard deviation of sales of ",std_dev_max, "is", std_dev_store)
mean = data.groupby('Store')['Weekly_Sales'].mean()
print("The mean of sales for each store is :\n", mean)
print("The coefficient of mean to standard deviation is :\n", std_dev/mean)

# Store that have good quarterly growth rate in Q3’2012
# Q3 dates start from july to september
# Sales for third quarterly in 2012
# Convert date to datetime format and show dataset information
print('type of date column', data['Date'].dtype)
data['Date'] = pd.to_datetime(arg=data['Date'], format="%d-%m-%Y", dayfirst=True)
print(data.info())

# Sales for 3rd quater in 2012
Q3 = data[(data['Date'] > '2012-07-01') & (data['Date'] < '2012-09-30')].groupby('Store')['Weekly_Sales'].sum()

# Sales for second quarterly in 2012
Q2 = data[(data['Date'] > '2012-04-01') & (data['Date'] < '2012-06-30')].groupby('Store')['Weekly_Sales'].sum()

#  store/s thst has good quarterly growth rate in Q3’2012
GR = (Q3 - Q2)/Q3
print("The quarterly growth rate for each store is :\n", GR)
best_GR = GR.max()
best_GR_store = GR[GR == best_GR].index[0]
print("Store having the best quarterly growth rate is store number", best_GR_store, "in Q3'2012 with a growth rate of ", best_GR)
print('\n')

# finding holidays which have higher sales than the mean sales in non-holiday season for all stores together
# Super Bowl: 12-Feb-10, 11-Feb-11, 10-Feb-12, 8-Feb-13
# Labour Day: 10-Sep-10, 9-Sep-11, 7-Sep-12, 6-Sep-13
# Thanksgiving: 26-Nov-10, 25-Nov-11, 23-Nov-12, 29-Nov-13
# Christmas: 31-Dec-10, 30-Dec-11, 28-Dec-12, 27-Dec-13

# converting to year 1st than month and day last for all dates for comparision purpose
Super_Bowl = [ '2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08']
Labour_Day = ['2010-09-10', '2011-09-09', '2012-09-07', '2013-09-06']
Thanksgiving = ['2010-11-26', '2011-11-25', '2012-11-23', '2013-11-29']
Christmas = ['2010-12-31', '2011-12-30', '2012-12-28', '2013-12-27']

# Finding mean weekly sales for days with holidays 
SB = data[(data['Date'].isin(Super_Bowl)) & (data['Holiday_Flag'] == 1)]['Weekly_Sales'].mean()
LD = data[(data['Date'].isin(Labour_Day))  & (data['Holiday_Flag'] == 1)]['Weekly_Sales'].mean()
TH = data[(data['Date'].isin(Thanksgiving))  & (data['Holiday_Flag'] == 1)]['Weekly_Sales'].mean()
CH = data[(data['Date'].isin(Christmas))  & (data['Holiday_Flag'] == 1)]['Weekly_Sales'].mean()

print("The mean sales on Super Bowl is ", SB)
print("The mean sales on Labour Day is ", LD)
print("The mean sales on Thanksgiving is ", TH)
print("The mean sales on Christmas is ", CH)

# Mean weekly sales for non-holidays 
Mean_sales_NH = data[data['Holiday_Flag'] == 0]['Weekly_Sales'].mean()
print("The mean sales in non-holiday season for all stores together is ", Mean_sales_NH)

# holidays having higher sales than the mean sales 
print('\n')
print('Holidays having sales higher than mean sales in non-holiday season are:')

# checking if holidays which have higher sales than the mean sales in non-holiday season for all stores together
if SB > Mean_sales_NH:
    print('Super Bowl with average sales of', SB-Mean_sales_NH ,' more than the mean sales in non-holiday season')
if LD > Mean_sales_NH:
    print('Labour Day with average sales of', LD-Mean_sales_NH,' more than the mean sales in non-holiday season')
if TH > Mean_sales_NH:
    print('Thanksgiving with average sales of', TH-Mean_sales_NH, ' more than the mean sales in non-holiday season')
if CH > Mean_sales_NH:
    print('Christmas with average sales of', CH-Mean_sales_NH ,' more than the mean sales in non-holiday season')

if SB < Mean_sales_NH and LD < Mean_sales_NH and TH < Mean_sales_NH and CH < Mean_sales_NH:
    print('None of the holidays have sales higher than the mean sales in non-holiday season')

# a monthly and semester view of sales in units and insights 
# creating a new column for month and year using the date column
data['Month'] = pd.DatetimeIndex(data['Date']).month
data['Year'] = pd.DatetimeIndex(data['Date']).year
print('Datset information after adding month and year columns\n' ,data.head())

# monthly view of slaes for each year
plt.figure(figsize=(10,10))
for i in range(0,3):
    plt.subplot(1,3,i+1)
    plt.scatter(data[data['Year'] == 2010+i]['Month'], data[data['Year'] == 2010+i]['Weekly_Sales'])
    plt.xlabel('Month')
    plt.ylabel('Weekly Sales')
    plt.title('Sales in the year {}'.format(2010+i))
plt.show()

# monthly view of sales for whole dataset
plt.figure(figsize=(10,10))
plt.scatter(data['Month'], data['Weekly_Sales'])
plt.xlabel('Month')
plt.ylabel('Weekly Sales')
plt.title('Sales for each month in the for the whole dataset')
plt.show()

# Utilize variables like date and restructure dates as 1 for 5 Feb 2010 (starting from the earliest date in order).
j = int(1)
m = 0
data= data.assign(Order=pd.Series(np.random.randn(len(data))).values)
for i in range(0,len(data)):
    x = data.iloc[i]['Store']
    if x != m:
        j = 1
    data.loc[i ,'Order']= j
    j = j+1
    m = x
print('Datset information after adding Order column\n',data.head())

# Building a prediction models to forecast demand for store 1 using CPI, unemployment, and fuel price have any impact on sales.
data_store = data # making an instace of original dataset
# checking outliers and correlation with respect to weekly sales
sns.pairplot(data_store[['Weekly_Sales', 'CPI', 'Unemployment', 'Fuel_Price','Temperature']],  diag_kind='kde', kind='reg')
plt.show()
fig, axs = plt.subplots(4,figsize=(6,10))
X = data_store[['Temperature','Fuel_Price','CPI','Unemployment']]
for i,column in enumerate(X):
    plt.title('Variables with outliers in the dataset')
    sns.boxplot(data_store[column], ax=axs[i])
plt.show()

# removing outliers from the dataset
print('dataset shape with outliers',data_store.shape)
data_store = data_store[(data_store['Unemployment'] > 5) & (data_store['Unemployment'] < 10.5) & (data_store['Temperature'] > 10) ]
print('dataset shape without outliers',data_store.shape)

# checking if outliers are removed 
fig, axs = plt.subplots(4,figsize=(6,10))
X = data_store[['Temperature','Fuel_Price','CPI','Unemployment']]
for i,column in enumerate(X):
    plt.title('Variables without outliers in the dataset')
    sns.boxplot(data_store[column], ax=axs[i])
plt.show()

# changing the dates into days
data_store['Day'] = pd.DatetimeIndex(data_store['Date']).day

# creating input and output dataframes for training and testing
input = data_store.drop(['Weekly_Sales', 'Date'], axis=1)
output = data_store['Weekly_Sales']
print('Inputs for training: \n',input) 
print('\n')
print('Outputs for training: \n',output)

# Standardizing the input data set
standardize = StandardScaler() # using Sklearn standarsizing function to create an object stadardize
scaled_X = standardize.fit_transform(input)  # Fit methods calculates the stadardization of input data set, then transform methods applies it to it.
print("The new scaled input datsset is:",scaled_X)

# spliting the datframes for training and testing
X_train, X_test, y_train, y_test = train_test_split(scaled_X, output, test_size=0.2, random_state=42)
print ("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# creating Linear regression model
LR_model = LinearRegression()
history = LR_model.fit(X_train, y_train)
score = history.score(X_test, y_test)
print("The score of the Linera regressor model is:", score*100,"% Accuracy")

# creating a random forest regressor
RF_model = RandomForestRegressor()
RF_model.fit(X_train, y_train)
RF_score = RF_model.score(X_test, y_test)
print("The score of the Random Forest regressor model is:", RF_score*100,"% Accuracy")

# Creating an instace of store 1 for testing the model
data_store_1 = data_store[data['Store'] == 1]
X_store_1 = data_store_1.drop(['Weekly_Sales', 'Date'], axis=1)
y_store_1 = data_store_1['Weekly_Sales']
scaled_test_X = standardize.transform(X_store_1)
print('Inputs for testing: \n',X_store_1) 
print('\n')
print('Outputs for testing: \n',y_store_1)

print("The score of the Linera regressor model for store 1 test data is:", LR_model.score(scaled_test_X, y_store_1)*100,"% Accuracy")
print("The score of the Random Forest regressor model for store 1 test data is:", RF_model.score(scaled_test_X, y_store_1)*100,"% Accuracy")