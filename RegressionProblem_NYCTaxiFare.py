#!/usr/bin/env python
# coding: utf-8

# ## Regression Problem - NYC Taxi Fare Prediction

# In[1]:


#!pip install googlemaps


# In[57]:


import os
import pandas as pd
import matplotlib.pyplot as plt 
#import googlemaps
import random
from math import cos, asin, sqrt
import numpy as np
import datetime
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

#Inorder to calculate the distance we have used the gogale map API as only coordinated are given
#tried to use this api to calculate distance but the dataset is quite large and only 10000 rows are processed at a time
#so dropped this technique to calculate distance
#gmaps = googlemaps.Client(key='AIzaSyAm_TEkXHyBqBAKJXca9N_JrZyy6K7dpoU') 
#TYPE = "


# In[58]:


data = pd.read_csv('C:\\Users\\mudit\\Downloads\\taxi\\data.csv', nrows=50000)


# In[59]:


data.head()


# In[60]:


print('Sum of NaN values for each column')
print(data.isnull().sum())


# In[61]:



data.describe() #take a look at the data
data.isnull().sum() #check for the null values
print("Minimum fare = " , data["fare_amount"].min())
print("Maximum fare = " , data["fare_amount"].max())
print("Negative fare rows = " , sum(data["fare_amount"] < 0))
print("0 fare rows = " , sum(data["fare_amount"] == 0))


# In[62]:


#Data Cleaning

# Currently, minimum taxi fare in NYC is $2.5. Since, we have data from 2009, assuming that minimum fare 
# would be atleast $1, removing rows with fare < $1.5
print("Rows with fare<1 are ", sum(data["fare_amount"] < 1.5)) #Number of rows with fare < $1.5
print("Rows with fare>450 are ", sum(data["fare_amount"] > 450)) #Number of rows with fare > $450

#Maximum fare between two farthest points in NY city is $320 as per the current uber fare. 
#So considering all the data with fare>$450 and fare<1.5 as bad data, we will remove those values
data = data[(data["fare_amount"] > 1.5) & (data["fare_amount"] <= 450)]


# In[63]:


#Data Cleaning

#NY City latitude is between 40.4965, 40.9159 and longitude is between -74.25 , -73.7016
#Identify latitudes and lngitudes which does not belong to NY City, adding some grace distance

# print(train[(train["pickup_latitude"] <39.8) | (train["pickup_latitude"] > 41.3)].shape[0])
# print(train[(train["pickup_longitude"] < -75) | (train["pickup_longitude"] > -71.8)].shape[0])
# print(train[(train["dropoff_latitude"] <39.8) | (train["dropoff_latitude"] > 41.3)].shape[0])
# print(train[(train["dropoff_longitude"] < -75) | (train["dropoff_longitude"] > -71.8)].shape[0])

#Remove rows with bad latitude and longitude values
data = data[(data["pickup_latitude"] >39.8) & (data["pickup_latitude"] < 41.3)]
data = data[(data["pickup_longitude"] > -75) & (data["pickup_longitude"] < -71.8)]
data = data[(data["dropoff_latitude"] >39.8) & (data["dropoff_latitude"] < 41.3)]
data = data[(data["dropoff_longitude"] > -75) & (data["dropoff_longitude"] < -71.8)]

#Removing 195532 rows with passenger count more than 6 and less than 1
# print(data[(data["passenger_count"] > 6 ) | (data["passenger_count"] < 1)].shape[0])

data = data[(data["passenger_count"] <= 6 ) & (data["passenger_count"] >= 1)]


# In[64]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
data.hist(bins=50,figsize=(15,15)) #Look at the histograms
plt.show()


# In[65]:


#Visualizing data
# train_sample = data.sample(n=100000) #take a sample of 100,000 randon rows
data.plot(kind="scatter", x="pickup_longitude", y="pickup_latitude", c="red", alpha=0.1)
data.plot(kind="scatter", x="dropoff_longitude", y="dropoff_latitude", c="blue", alpha=0.1)


# In[66]:


#Look for correlations anmong variables
corr = data.corr()
corr["fare_amount"].sort_values(ascending=False)
from pandas.plotting import scatter_matrix
scatter_matrix(data,figsize=(12,8))


# In[67]:


# Define function to calculate distance between two points by latitude and longitude
# Faced some errors in this function as described below
# 1. lat1, lon1 etc were Panda series and so it won't allow its conversion to float.Typecasted to list when calling the function
# 2.Python won't allow operations between two list, so converted list to np.array
# 3.Math.cos,asin etc functions does not support np.array. So used np.cos, np.asin functions

def distance(lat1, lon1, lat2, lon2):
    lat1 = np.array(lat1)
    lon1 = np.array(lon1)
    lat2 = np.array(lat2)
    lon2 = np.array(lon2)
    p = 0.017453292519943295     #Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return (7917.512 * np.arcsin(np.sqrt(a)))

# Assigning dummy value 1.1 to column "distance" to change its datatype to float64 from object
data["distance"] = 1.1

#Changing datatype of train.Distance from float64 to float16
data["distance"] = data["distance"].astype(np.float16)

#Calculating distance between pickup and drop points using Haversine formula
data['distance'] = distance(data.pickup_latitude.tolist(), data.pickup_longitude.tolist(),
                             data.dropoff_latitude.tolist(), data.dropoff_longitude.tolist())


# In[68]:


data.columns


# In[69]:


#checking null values after distance calculation
#as there maybe data with same pickup and dropoff coordinates
print('Train data: Sum of NaN values for each column')
print(data.isnull().sum())


# In[70]:


#Since we have distance now, let's see its relation with fare amount
# We can see a clear linear relationship between distance and fare. Also, there are some trips with zero
# distance by non-zero fare amount. We can also see some stright lines around fare 40-60, may be fixed fare
# to airports

data.plot(kind="scatter", x="distance", y="fare_amount", c="red", alpha=0.1)
plt.show()


# In[71]:


# Create three new variables for storing year,time and hour of pickup
import datetime
data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])
data["year"] = data["pickup_datetime"].dt.year
data["time"] = data["pickup_datetime"].dt.time
data["hour"] = data["pickup_datetime"].dt.hour


# In[75]:


data.head()


# In[76]:


# Define Function to polulate new column "weekday" where 1 = weekday, 0 = weekend
def weekday(pickup_date):
    weekday = []
    for index,val in pickup_date.iteritems():
        val = pd.to_datetime(val)
        if(val.weekday() == 5 or val.weekday() == 6):
            weekday.append(0)
        else:
            weekday.append(1)
    return weekday

data["weekday"] = 0
data["weekday"] = weekday(data["pickup_datetime"])


# In[77]:


data["hour"] = data["pickup_datetime"].dt.hour
#Histogram on train.hour to see when highest number of cabs are booked
plt.hist(data["hour"], bins=5)
plt.show()


# In[78]:


data.head()


# In[79]:


# Let's categorize time data into morning, afternoon, evening and night in new column "part_of_day"

def time_in_range(start, end, x):
    # Return true if x is in the range [start, end]
    if start <= end:
        return start <= x <= end
    else:
        return start <= x or x <= end
    
def assign_day_part(pickup_date):
    day_part = []
    # Morning = 0600-1000
    mornStart = datetime.time(6, 0, 1)
    mornEnd = datetime.time(10, 0, 0)

    # Midday = 1000-1600
    midStart = datetime.time(10, 0, 1)
    midEnd = datetime.time(16, 0, 0)

    # Evening = 1600-2000
    eveStart = datetime.time(16, 0, 1)
    eveEnd = datetime.time(20, 0, 0)
    
    # Night = 2000-0000
    nightStart = datetime.time(20, 0, 1)
    nightEnd = datetime.time(0, 0, 0)

    # Late Night = 0000-0600
    lateStart = datetime.time(0, 0, 1)
    lateEnd = datetime.time(6, 0, 0)
    
    for index,val in pickup_date.iteritems():
        if time_in_range(mornStart, mornEnd, val.time()):
            day_part.append("morning")
        elif time_in_range(midStart, midEnd, val.time()):
            day_part.append("midday")
        elif time_in_range(eveStart, eveEnd, val.time()):
            day_part.append("evening")
        elif time_in_range(nightStart, nightEnd, val.time()):
            day_part.append("night")
        elif time_in_range(lateStart, lateEnd, val.time()):
            day_part.append("lateNight")

    return day_part

data["part_of_day"] = assign_day_part(data["pickup_datetime"])

# We do not need time and hour variables now. So, dropping them
data = data.drop(["time", "hour"], axis=1)


# In[80]:


data.head()


# In[81]:


#Since we have distance and classification of weekday now, let's see the correlation

data.plot(kind="scatter", x="distance", y="fare_amount", c="red", alpha=0.1)


# In[82]:


# REMOVE COMMENTS

# #Since we have distance and classification of weekday now, let's see the correlation
# # We see that distance is highly related to fare amount positively.

print(data.corr()["fare_amount"])


# In[83]:


data.head()


# In[84]:


# Convert categorical varibale "part_of_day" from to numerical value using sklearn LabelBinarizer
# This will create 5 new columns - evening, lateNight, midday, morning, night

from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
lb_results = lb.fit_transform(data["part_of_day"])
lb_results_df = pd.DataFrame(lb_results, columns=lb.classes_)

data = pd.merge(data, lb_results_df, left_index=True, right_index=True) #Merge output with training set


# In[85]:


data.head()


# In[86]:


data.drop(["part_of_day"], axis=1, inplace=True) # Dropping part_of_day variable


# In[95]:


# Let us also drop pickup_datetime variable as we have included variables like year and part_of_day
data.drop(["pickup_datetime"], axis=1, inplace=True)


# In[96]:


data.drop(["key"], axis=1, inplace=True)


# In[97]:


data.head()


# In[98]:


import seaborn as sn
# Let's check how the features correlate
colormap = plt.cm.RdBu
plt.figure(figsize=(30,30))
#plt.title('Pearson Correlation of Features', y=1.05, size=15)
sn.heatmap(data.corr(), annot=True)


# In[104]:


#when distance is calculated no need for these variables
data.drop(["pickup_longitude"], axis=1, inplace=True)
data.drop(["pickup_latitude"], axis=1, inplace=True)
data.drop(["dropoff_longitude"], axis=1, inplace=True)
data.drop(["dropoff_latitude"], axis=1, inplace=True)


# In[105]:


data.head()


# In[106]:


import seaborn as sn
# Let's check how the features correlate
colormap = plt.cm.RdBu
plt.figure(figsize=(30,30))
#plt.title('Pearson Correlation of Features', y=1.05, size=15)
sn.heatmap(data.corr(), annot=True)

#muticolinearity is removed


# In[107]:


#creating temporary data
new_data=data


# In[108]:


#Let us separate the response variable fare_amount 
train_labels = new_data["fare_amount"]
new_data.drop(["fare_amount"], axis=1, inplace=True)
new_data.head()


# In[110]:


import statsmodels.api as sm


# In[111]:


#assigning independent and reponse variable
X = sm.add_constant( new_data )
Y = train_labels


# In[123]:


X_features = new_data.columns
X_features


# In[224]:


#here we are splitting the dataset
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split( X ,Y,train_size = 0.8,random_state = 42 )


# In[225]:


# Fitting the Model
taxi_model_1 = sm.OLS(train_y, train_X).fit()
taxi_model_1.summary()


# In[131]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[132]:


# calculating Variance inflation factor 
def get_vif_factors( X ):
    X_matrix = X.as_matrix()
    vif = [ variance_inflation_factor( X_matrix, i ) for i in range( X_matrix.shape[1] ) ]
    vif_factors = pd.DataFrame()
    vif_factors['column'] = X.columns
    vif_factors['vif'] = vif
    return vif_factors


# In[133]:


vif_factors = get_vif_factors( X[X_features] )
vif_factors


# In[134]:


# Select the features that have VIF value more than 4
columns_with_large_vif = vif_factors[vif_factors.vif > 4].column


# In[135]:


# Plot the heatmap for features with more than 4
plt.figure( figsize = (12,10) )
sn.heatmap( X[columns_with_large_vif].corr(), annot = True );
plt.title( "Heatmap depicting correlation between features");


# In[138]:


columns_to_be_removed = ['midday','year']
X_new_features = list( set(X_features) - set(columns_to_be_removed) )
#midday show high negative muti-colinearity with other variables
#and after removing midday, variable year was showing high vif, so eliminated that as well


# In[139]:


get_vif_factors( X[X_new_features] )


# In[142]:


# Building a new model after removing multicollinearity
train_X = train_X[['const']+X_new_features]
taxi_model_2 = sm.OLS(train_y, train_X).fit()
taxi_model_2.summary2()


# In[163]:


#from the above model it is found that only distance and weekday variables are significant
#so we will create a new model with only these variables
significant_vars = ['distance', 'weekday']
train_X = train_X[significant_vars]


# In[165]:


taxi_model_3 = sm.OLS(train_y, train_X).fit()
taxi_model_3.summary2()


# In[168]:


### Residual Analysis

# P-P Plot
def draw_pp_plot( model, title ):
    probplot = sm.ProbPlot( model.resid );
    plt.figure( figsize = (8, 6) );
    probplot.ppplot( line='45' );
    plt.title( title );
    plt.show();
    
draw_pp_plot( taxi_model_3,"Normal P-P Plot of Regression Standardized Residuals");


# In[182]:


# Homoscedasticity
def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std()

def plot_resid_fitted( fitted, resid, title):
    plt.scatter( get_standardized_values( fitted ),get_standardized_values( resid ) )
    plt.title( title )
    plt.xlabel( "Standardized predicted values")
    plt.ylabel( "Standardized residual values")
    plt.show()

plot_resid_fitted( taxi_model_3.fittedvalues,ipl_model_3.resid,"Residual Plot")


# In[177]:


# Z-Score
from scipy.stats import zscore


# In[186]:


check_data=data
check_data.head()


# In[185]:


check_data["zscore_fare"] = zscore(train_labels )


# In[187]:


check_data[ (check_data['zscore_fare'] > 3.0) | (check_data['zscore_fare'] < -3.0) ]


# In[202]:


### Making predictions on validation set
pred_y = taxi_model_3.predict( test_X[train_X.columns] )

#Measuring RMSE
from sklearn import metrics
metrics.mean_squared_error(pred_y, test_y)


# In[215]:


pred_y.head()#predicted values


# In[216]:


test_y.head()#actual values


# In[ ]:


#accuaracy of 87.4% is achieved and different transformations were applied but the best model
#achieved was model 3 only because when transformation were performed r-square value got decreased

