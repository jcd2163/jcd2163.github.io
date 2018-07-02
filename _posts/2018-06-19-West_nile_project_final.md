---
title:  "Project: West Nile Virus - Chicago"
date:   2018-06-19 
---
# Project Intro
For this project, we were given the West Nile Virus Predicition Challenge from Kaggle https://www.kaggle.com/c/predict-west-nile-virus/ and tasked with 1) predicting the presence of West Nile over the prediction data set (over 116,000 rows), 2) create a cost benefit analysis on the effectiveness of spraying, and 3) present in class with our findings. This was our first group project and I really enjoyed the experience of collaborating with my classmates to strategize our plan and execute it together. 

Below, I’ve provided code for the first objective around predicting West Nile Virus. Most of the this was written by myself except where I’ve indicated below. This task was especially challenging based on a number of factors, including 1) unbalanced classes (roughly 5% of samples were positive for West Nile Virus, 2) data availability (we were provided the number of mosquitos per trap and spraying information only for the training samples), and 3) low correlation between features and target. Below, I’ve detailed our approach to dealing with each of these factors.

Some parts of this project that I really enjoyed were the advanced feature engineering tasks and designing features based on background research, dealing with time and location data, resampling to deal with the unbalanced classes, imputing missing data with fancyimpute (wasn’t really necessary but I wanted to gain experience with this package), implementing a pipeline to deal with both continuous and categorical features, and the advanced models used. I also got to play around with Uber's new, open-source Kepler.gl tool for plotting some of the time series data. I've included a screenshot of one of the maps I made below and can send you the file upon request (unfortuately, no way to embed the map yet but you can upload a copy of the ones I've made to see get a feel for the data yourself).

I hope you enjoy reading through my code and explanations as much as I did working on this project! As always, feel free to reach out with any questions or comments.


![gif](/images/West_nile_project_final_files/movie3.gif)

![png](/images/West_nile_project_final_files/map.png)


# Data Exploration

Training and prediction data sets contain information from mosquito traps around Chicago, with data, location, and species data for each. Training data set has two additional features compared with the prediction data set - a count of mosquistos and whether West Nile Virus is present. Training data set contains odd years from 2007 to 2013 while prediction data set is even years from 2008 to 2014. Months included are summer and fall months. Weather data has data for all years from two weather stations in Chicago. Spray data is pretty limited, containing only two years (2011 & 2013) and ten days of spraying. No indication that this is the complete spray data for each year. 


```python
train_data.describe(include='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Address</th>
      <th>Species</th>
      <th>Block</th>
      <th>Street</th>
      <th>Trap</th>
      <th>AddressNumberAndStreet</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>AddressAccuracy</th>
      <th>NumMosquitos</th>
      <th>WnvPresent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10506</td>
      <td>10506</td>
      <td>10506</td>
      <td>10506.000000</td>
      <td>10506</td>
      <td>10506</td>
      <td>10506</td>
      <td>10506.000000</td>
      <td>10506.000000</td>
      <td>10506.000000</td>
      <td>10506.000000</td>
      <td>10506.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>95</td>
      <td>138</td>
      <td>7</td>
      <td>NaN</td>
      <td>128</td>
      <td>136</td>
      <td>138</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>2007-08-01</td>
      <td>ORD Terminal 5, O'Hare International Airport, ...</td>
      <td>CULEX PIPIENS/RESTUANS</td>
      <td>NaN</td>
      <td>W OHARE AIRPORT</td>
      <td>T900</td>
      <td>1000  W OHARE AIRPORT, Chicago, IL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>551</td>
      <td>750</td>
      <td>4752</td>
      <td>NaN</td>
      <td>750</td>
      <td>750</td>
      <td>750</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>35.687797</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.841139</td>
      <td>-87.699908</td>
      <td>7.819532</td>
      <td>12.853512</td>
      <td>0.052446</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24.339468</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.112742</td>
      <td>0.096514</td>
      <td>1.452921</td>
      <td>16.133816</td>
      <td>0.222936</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.644612</td>
      <td>-87.930995</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.732984</td>
      <td>-87.760070</td>
      <td>8.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>33.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.846283</td>
      <td>-87.694991</td>
      <td>8.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>52.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.954690</td>
      <td>-87.627796</td>
      <td>9.000000</td>
      <td>17.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>98.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>42.017430</td>
      <td>-87.531635</td>
      <td>9.000000</td>
      <td>50.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
predict_data.describe(include='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Date</th>
      <th>Address</th>
      <th>Species</th>
      <th>Block</th>
      <th>Street</th>
      <th>Trap</th>
      <th>AddressNumberAndStreet</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>AddressAccuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>116293.000000</td>
      <td>116293</td>
      <td>116293</td>
      <td>116293</td>
      <td>116293.000000</td>
      <td>116293</td>
      <td>116293</td>
      <td>116293</td>
      <td>116293.000000</td>
      <td>116293.000000</td>
      <td>116293.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>95</td>
      <td>151</td>
      <td>8</td>
      <td>NaN</td>
      <td>139</td>
      <td>149</td>
      <td>151</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>2012-07-09</td>
      <td>ORD Terminal 5, O'Hare International Airport, ...</td>
      <td>CULEX PIPIENS/RESTUANS</td>
      <td>NaN</td>
      <td>N OAK PARK AVE</td>
      <td>T009</td>
      <td>1000  W OHARE AIRPORT, Chicago, IL</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>1293</td>
      <td>1468</td>
      <td>15359</td>
      <td>NaN</td>
      <td>1617</td>
      <td>1528</td>
      <td>1468</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>58147.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.131100</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.849389</td>
      <td>-87.693658</td>
      <td>7.954357</td>
    </tr>
    <tr>
      <th>std</th>
      <td>33571.041765</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24.864726</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.106593</td>
      <td>0.080699</td>
      <td>1.252733</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.644612</td>
      <td>-87.930995</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>29074.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.753411</td>
      <td>-87.750938</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>58147.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>39.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.862292</td>
      <td>-87.694991</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>87220.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>61.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.951866</td>
      <td>-87.648860</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>116293.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>98.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>42.017430</td>
      <td>-87.531635</td>
      <td>9.000000</td>
    </tr>
  </tbody>
</table>
</div>



# Clean data and feature engineering - weather data

Weather data has some null values (represented by 'M') and some features that aren't relevant and can be dropped (e.g. 'SnowFall'). Also has many columns as object data types that will need to be converted to float. Weather data represents the bulk to the data available for both training and predicition data sets so will be the focus of my feature engineering. I will explore features such as rolling averages, time-lagged data, and categorizing light and heavy rain days. I will create these features in the original data set before merging, as after merging some weather data will be removed because the training and prediction data sets have readings for only specific days. 


```python
weather_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2944 entries, 0 to 2943
    Data columns (total 22 columns):
    Station        2944 non-null int64
    Date           2944 non-null object
    Tmax           2944 non-null int64
    Tmin           2944 non-null int64
    Tavg           2944 non-null object
    Depart         2944 non-null object
    DewPoint       2944 non-null int64
    WetBulb        2944 non-null object
    Heat           2944 non-null object
    Cool           2944 non-null object
    Sunrise        2944 non-null object
    Sunset         2944 non-null object
    CodeSum        2944 non-null object
    Depth          2944 non-null object
    Water1         2944 non-null object
    SnowFall       2944 non-null object
    PrecipTotal    2944 non-null object
    StnPressure    2944 non-null object
    SeaLevel       2944 non-null object
    ResultSpeed    2944 non-null float64
    ResultDir      2944 non-null int64
    AvgSpeed       2944 non-null object
    dtypes: float64(1), int64(5), object(16)
    memory usage: 506.1+ KB



```python
for i in weather_data.columns: #iterate through each column
    if i not in ['Date', 'CodeSum', 'Station']: #ignore these columns for now (want to convert the rest to floats)
        weather_data[i].replace('M', np.nan, inplace=True) #'M' means missing
        weather_data[i].replace('-', 0, inplace=True) #'-' is 0
        weather_data[i].replace('  T', 0.005, inplace=True) #'T' is trace 
        weather_data[i] = weather_data[i].astype('float') #covert to float
```


```python
weather_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2944 entries, 0 to 2943
    Data columns (total 22 columns):
    Station        2944 non-null int64
    Date           2944 non-null object
    Tmax           2944 non-null float64
    Tmin           2944 non-null float64
    Tavg           2933 non-null float64
    Depart         1472 non-null float64
    DewPoint       2944 non-null float64
    WetBulb        2940 non-null float64
    Heat           2933 non-null float64
    Cool           2933 non-null float64
    Sunrise        2944 non-null float64
    Sunset         2944 non-null float64
    CodeSum        2944 non-null object
    Depth          1472 non-null float64
    Water1         0 non-null float64
    SnowFall       1472 non-null float64
    PrecipTotal    2942 non-null float64
    StnPressure    2940 non-null float64
    SeaLevel       2935 non-null float64
    ResultSpeed    2944 non-null float64
    ResultDir      2944 non-null float64
    AvgSpeed       2941 non-null float64
    dtypes: float64(19), int64(1), object(2)
    memory usage: 506.1+ KB



```python
weather_data.isnull().sum().sort_values(ascending=False) #check for null values
```




    Water1         2944
    Depart         1472
    SnowFall       1472
    Depth          1472
    Tavg             11
    Cool             11
    Heat             11
    SeaLevel          9
    StnPressure       4
    WetBulb           4
    AvgSpeed          3
    PrecipTotal       2
    Date              0
    Tmax              0
    Tmin              0
    Sunrise           0
    DewPoint          0
    ResultDir         0
    Sunset            0
    CodeSum           0
    ResultSpeed       0
    Station           0
    dtype: int64




```python
weather_data.drop(['Depart','CodeSum','Depth','Water1','SnowFall'], axis=1, inplace=True) #drop irrelevant and/or columns with a lot of missing data
```


```python
#try fancyimput to fill missing values (not really necessary, but good to know how to do this)
mask = weather_data.isnull().any(axis=1) #set up mask for rows with missing data
cols = [cols for cols in weather_data.columns if weather_data[cols].isnull().any() == True] #get columns with missing data
nanvalues = KNN(k=3, verbose=False).complete(weather_data.loc[mask, cols]) #use fancyimpute with KNN to generate replacements for missing values
weather_data.loc[mask, cols] = nanvalues #fill in missing data with new values
```


```python
assert weather_data.isnull().sum().sum() == 0 #check to make sure there is no more missing data
```


```python
#generate a feature for humidity (function written by a teammate)

def humidity(Tf, Tdf):

    # convert temp to celsius
    Tc=5.0/9.0*(Tf-32.0)
    # convert dewpoint temp to celsius
    Tdc=5.0/9.0*(Tdf-32.0)
    # saturation vapor pressure
    Es=6.11*10.0**(7.5*Tc/(237.7+Tc))
    # actual vapor pressure
    E=6.11*10.0**(7.5*Tdc/(237.7+Tdc))
    #relative humidity
    RH =(E/Es)*100
        
    return RH

weather_data["RelHum"] = humidity(weather_data['Tavg'],weather_data['DewPoint'])
```


```python
#create datetime object for date
weather_data['Date'] = pd.to_datetime(weather_data['Date'])

#calculate day length
weather_data['day_length'] = weather_data['Sunset'] - weather_data['Sunrise']

#create list of features that are not date or codesum
weather_feature_cols = [cols for cols in weather_data.columns if weather_data[cols].dtype in [float, int]]

#computer rolling averages 
for i in [3,5,10,14,21,30]: #set intervals for rolling average calculation
    for j in weather_feature_cols:
        weather_data[j + '_rolling_' + str(i)] = weather_data[j].rolling(window=i).mean() #create column with interval for each computation                                                             

#create shifted columns
for i in range(1,15):
    for j in weather_feature_cols:
        weather_data[j + '_shift_' + str(i)] = weather_data[j].shift(periods=i)

#create columns for light and heavy rain days based on quantile of rainy days
precip_50quantile = weather_data[weather_data['PrecipTotal'] > 0]['PrecipTotal'].quantile(.5) #heavy rain day defined as 50th quantile or above of rainy days
precip_10quantile = weather_data[weather_data['PrecipTotal'] > 0]['PrecipTotal'].quantile(.1) #light rain day defined as 50th quantile or above of rainy days

weather_data['heavy_rain'] = (weather_data['PrecipTotal']>precip_50quantile).astype(int)
weather_data['light_rain'] = ((weather_data['PrecipTotal']>0) & (weather_data['PrecipTotal']<precip_10quantile)).astype(int)

light_rain_col_list = []
heavy_rain_col_list = []
for i in range(1,15):
    weather_data['light_rain_' + str(i) + '_days_ago'] = weather_data['light_rain'].shift(periods=i)
    weather_data['heavy_rain_'+ str(i) + '_days_ago'] = weather_data['heavy_rain'].shift(periods=i)
    light_rain_col_list.append('light_rain_' + str(i) + '_days_ago')
    heavy_rain_col_list.append('heavy_rain_' + str(i) + '_days_ago')

#create column if it rained in the past 14 days 
weather_data['heavy_rain_last_14_days'] = np.where(weather_data[heavy_rain_col_list].any(axis=1), 1, 0)
weather_data['light_rain_last_14_days'] = np.where(weather_data[light_rain_col_list].any(axis=1), 1, 0)
weather_data['heavy_rain_last_14_days_count'] = weather_data[heavy_rain_col_list].sum(axis=1)
weather_data['light_rain_last_14_days_count'] = weather_data[light_rain_col_list].sum(axis=1)
```

# Train & Predict data sets clean up, feature engineering, and combine with weather


```python
#fix NumMosquitos column
train_data['NumMosquitos_sum'] = np.nan #create sum column
#fill sum column with sum of matching rows
train_data['NumMosquitos_sum'].fillna(train_data.groupby(['Date','Trap','Species'])['NumMosquitos'].transform('sum'), inplace=True)
train_data.drop(['NumMosquitos'], axis=1, inplace=True) #drop old column
train_data.drop_duplicates(inplace=True) #drop duplicated rows 
train_data.reset_index(inplace=True) #reset index 
train_data.drop(['index'], axis=1, inplace=True) #drop old index
```


```python
assert train_data.duplicated().sum() == 0
```


```python
def clean_data_1(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['week'] = df['Date'].dt.week
    df['day'] = df['Date'].dt.day
    df['Trap'] = df['Trap'].str.extract('(\d\d\d)').astype(int)
    return df

train_data = clean_data_1(train_data)
predict_data= clean_data_1(predict_data)
```



```python
hot_spots = {'ord_location':(41.993851, -87.937940), 'hegewich':(41.655528, -87.570488), 
             'mdw':(41.785116, -87.752466), 'riverforest':(41.911294, -87.827725)}
```


```python
def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h
```


```python
def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))
```


```python
def add_distances_to_hotspots(df):
    for i,j in hot_spots.items():
        df['distance_to_' + i] = haversine_array(df['Latitude'], df['Longitude'], j[0], j[1])
        df['bearing_to_' + i] = bearing_array(df['Latitude'], df['Longitude'], j[0], j[1])
    return df 

train_data = add_distances_to_hotspots(train_data)
predict_data = add_distances_to_hotspots(predict_data)
```


```python
max_day = train_data.groupby(by='day_of_year')['WnvPresent'].sum().idxmax()

def days_since_max_wnvpresent(df):
    df['time_since_max_wnvpresent'] = abs(max_day-df['day_of_year'])
    return df

train_data = days_since_max_wnvpresent(train_data)
predict_data = days_since_max_wnvpresent(predict_data)
```


```python
def combine_with_weather(df):
    mask1 = df['distance_to_ord_location'] > df['distance_to_mdw']
    mask2 = df['distance_to_ord_location'] < df['distance_to_mdw']
    df.loc[mask1,'weather_station'] = 1
    df.loc[mask2,'weather_station'] = 2
    df = df.join(weather_data.set_index(['Date','Station']), on=(['Date','weather_station']), how='left')
    df.drop(['weather_station'], axis=1, inplace=True)
    return df

train_data = combine_with_weather(train_data)
predict_data = combine_with_weather(predict_data)
```

# clustering


```python
mapdata = np.loadtxt('mapdata_copyright_openstreetmap_contributors.txt')  
aspect = mapdata.shape[0] * 1.0 / mapdata.shape[1]
lon_lat_box = (train_data['Longitude'].min()-.1, train_data['Longitude'].max()+.1, train_data['Latitude'].min()-.1, train_data['Latitude'].max()+.1)

for j in train_data['year'].drop_duplicates().values:
    for i in train_data['month'].drop_duplicates().values:
        fig = plt.figure(figsize=(5,8))
        plt.imshow(mapdata, cmap=plt.get_cmap('gray'), aspect=aspect, extent=lon_lat_box)
        plt.title('month_' + str(i) + ', '+ str(j))
        plt.scatter(train_data[(train_data['month'] == i) & (train_data['year'] == j) & (train_data['WnvPresent'] == 1)]['Longitude'], 
                    train_data[(train_data['month'] == i) & (train_data['year'] == j) & (train_data['WnvPresent'] == 1)]['Latitude'], 
                    marker='o')
        plt.savefig(f'plot_{i,j}.png')
        plt.close(fig);
```


```python
!convert -delay 50 plot*.png movie.gif
```

!['Wnv Present by Month'](movie.gif)


```python
spray_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Time</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-08-29</td>
      <td>6:56:58 PM</td>
      <td>42.391623</td>
      <td>-88.089163</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-08-29</td>
      <td>6:57:08 PM</td>
      <td>42.391348</td>
      <td>-88.089163</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-08-29</td>
      <td>6:57:18 PM</td>
      <td>42.391022</td>
      <td>-88.089157</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-08-29</td>
      <td>6:57:28 PM</td>
      <td>42.390637</td>
      <td>-88.089158</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-08-29</td>
      <td>6:57:38 PM</td>
      <td>42.390410</td>
      <td>-88.088858</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_spray = spray_data[['Latitude', 'Longitude']]
spray_cluster_model = DBSCAN(eps=.01, min_samples=20).fit(X_spray)
spray_clusters = spray_cluster_model.labels_
X_spray['cluster'] = spray_clusters

lon_lat_box_spray = (X_spray['Longitude'].min()-.1, X_spray['Longitude'].max()+.1, X_spray['Latitude'].min()-.1,X_spray['Latitude'].max()+.1)

plt.figure(figsize=(5,8))
plt.imshow(mapdata, cmap=plt.get_cmap('gray'), aspect=aspect, extent=lon_lat_box_spray) 

plt.scatter(X_spray['Longitude'], X_spray['Latitude'], c=X_spray['cluster'], marker='o');
```

    /anaconda3/lib/python3.6/site-packages/ipykernel/__main__.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy



![png](/images/West_nile_project_final_files/West_nile_project_final_35_1.png)



```python
X_clusters = train_data[train_data['WnvPresent'] == 1][['Latitude', 'Longitude']]

cluster_model = KMeans(n_clusters=12).fit(X_clusters)
clusters = cluster_model.labels_
X_clusters['cluster'] = clusters

mapdata = np.loadtxt('mapdata_copyright_openstreetmap_contributors.txt')              

aspect = mapdata.shape[0] * 1.0 / mapdata.shape[1]
lon_lat_box = (X_clusters['Longitude'].min()-.1, X_clusters['Longitude'].max()+.1, X_clusters['Latitude'].min()-.1,X_clusters['Latitude'].max()+.1)

plt.figure(figsize=(5,8))
plt.imshow(mapdata, cmap=plt.get_cmap('gray'), aspect=aspect, extent=lon_lat_box)
plt.scatter(X_clusters['Longitude'], X_clusters['Latitude'], c=X_clusters['cluster'], 
                marker='o');
```


![png](/images/West_nile_project_final_files/West_nile_project_final_36_0.png)



```python
X_clusters_month = train_data[train_data['WnvPresent'] == 1][['month','Latitude', 'Longitude']]

cluster_month_model = KMeans(n_clusters=20).fit(X_clusters_month)
clusters_month = cluster_month_model.labels_
X_clusters_month['cluster'] = clusters_month

mapdata = np.loadtxt('mapdata_copyright_openstreetmap_contributors.txt')              

aspect = mapdata.shape[0] * 1.0 / mapdata.shape[1]
lon_lat_box = (X_clusters_month['Longitude'].min()-.1, X_clusters_month['Longitude'].max()+.1, X_clusters_month['Latitude'].min()-.1,X_clusters_month['Latitude'].max()+.1)

plt.figure(figsize=(5,8))
plt.imshow(mapdata, cmap=plt.get_cmap('gray'), aspect=aspect, extent=lon_lat_box)
plt.scatter(X_clusters_month['Longitude'], X_clusters_month['Latitude'], c=X_clusters_month['cluster'], 
                marker='o');
```


![png](/images/West_nile_project_final_files/West_nile_project_final_37_0.png)



```python
%matplotlib notebook
plt.figure(figsize=(7,7))
ax = plt.subplot(111, projection='3d')
ax.scatter(
    xs=X_clusters_month['Latitude'], 
    ys=X_clusters_month['Longitude'],
    zs=X_clusters_month['month'], 
    c=X_clusters_month['cluster'],
    cmap='bwr')
```



```python
for i in train_data['month'].drop_duplicates().values:
    fig = plt.figure(figsize=(5,8))
    plt.imshow(mapdata, cmap=plt.get_cmap('gray'), aspect=aspect, extent=lon_lat_box)
    plt.title(i)
    plt.scatter(X_clusters_month[X_clusters_month['month'] == i]['Longitude'], 
                X_clusters_month[X_clusters_month['month'] == i]['Latitude'], 
                c=X_clusters_month[X_clusters_month['month'] == i]['cluster'], 
                marker='o')
    plt.savefig(f'month_{i}.png')
    plt.close(fig);
```

```python
!convert -delay 50 month*.png movie2.gif
```

![](movie2.gif)


```python
def add_clusters(df):
    df['cluster'] = cluster_model.predict(df[['Latitude', 'Longitude']])
    df['month_cluster'] = cluster_month_model.predict(df[['month','Latitude','Longitude']])

    X = X_spray[['Latitude', 'Longitude']]
    y = X_spray['cluster']
    rf1 = RandomForestClassifier()
    fit1 = rf1.fit(X, y)
    df['spray_cluster'] = fit1.predict(df[['Latitude','Longitude']])

    return df

train_data = add_clusters(train_data)
predict_data = add_clusters(predict_data)
```


```python
cols_to_dummy = ['Species','Block','Street','Trap','month','week','cluster','spray_cluster','month_cluster']

def final_clean(df):
    for i in cols_to_dummy:
        df[i] = df[i].map((lambda x: i + '_' + str(x).replace(' ', '_')))
    return df

train_data = final_clean(train_data)
predict_data = final_clean(predict_data)
```


```python
train_data.shape
```




    (8610, 440)




```python
train_data['WnvPresent'].value_counts(normalize=True)
```




    0    0.946922
    1    0.053078
    Name: WnvPresent, dtype: float64




```python
predict_data.set_index('Id',inplace=True)
predict_data.shape
```




    (116293, 438)




```python
assert [cols for cols in train_data.columns if cols not in predict_data.columns] == ['WnvPresent','NumMosquitos_sum']
```

# Create pipeline and model


```python
train_data.drop(['Date', 'Address','AddressNumberAndStreet', 'NumMosquitos_sum'], axis=1, inplace=True)
```


```python
class ModelTransformer(BaseEstimator,TransformerMixin):

    def __init__(self, model=None):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        return self.model.transform(X)
    
class SampleExtractor(BaseEstimator, TransformerMixin):
    """Takes in varaible names as a **list**"""

    def __init__(self, vars):
        self.vars = vars  # e.g. pass in a column names to extract

    def transform(self, X, y=None):
        if len(self.vars) > 1:
            return pd.DataFrame(X[self.vars]) # where the actual feature extraction happens
        else:
            return pd.Series(X[self.vars[0]])

    def fit(self, X, y=None):
        return self  # generally does nothing
    
    
class DenseTransformer(BaseEstimator,TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self
```


```python
species = Pipeline([
              ('text',SampleExtractor(['Species'])),
              ('dummify', CountVectorizer(binary=True)),
              ('densify', DenseTransformer()),
             ])
block = Pipeline([
              ('text',SampleExtractor(['Block'])),
              ('dummify', CountVectorizer(binary=True)),
              ('densify', DenseTransformer()),
             ])
street = Pipeline([
              ('text',SampleExtractor(['Street'])),
              ('dummify', CountVectorizer(binary=True)),
              ('densify', DenseTransformer()),
             ])
trap = Pipeline([
              ('text',SampleExtractor(['Trap'])),
              ('dummify', CountVectorizer(binary=True)),
              ('densify', DenseTransformer()),
             ])
month = Pipeline([
              ('text',SampleExtractor(['month'])),
              ('dummify', CountVectorizer(binary=True)),
              ('densify', DenseTransformer()),
             ])
week = Pipeline([
              ('text',SampleExtractor(['week'])),
              ('dummify', CountVectorizer(binary=True)),
              ('densify', DenseTransformer()),
             ])
cluster = Pipeline([
              ('text',SampleExtractor(['cluster'])),
              ('dummify', CountVectorizer(binary=True)),
              ('densify', DenseTransformer()),
             ])
spray_cluster = Pipeline([
              ('text',SampleExtractor(['spray_cluster'])),
              ('dummify', CountVectorizer(binary=True)),
              ('densify', DenseTransformer()),
             ])
month_cluster = Pipeline([
              ('text',SampleExtractor(['month_cluster'])),
              ('dummify', CountVectorizer(binary=True)),
              ('densify', DenseTransformer()),
             ])
```


```python
X = train_data[[cols for cols in train_data.columns if cols != 'WnvPresent']]
y = train_data['WnvPresent']

cont_col_list = [cols for cols in X.columns if cols not in cols_to_dummy]

X_train, X_test, y_train, y_test, = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
```


```python
pipeline_logreg = Pipeline([
    ('features', FeatureUnion([
        ('species', species),
        ('block', block),
        ('street', street),
        ('trap', trap),
        ('month', month),
        ('week', week),
        ('cluster',cluster),
        ('spray_cluster',spray_cluster),
        ('month_cluster',month_cluster),
        ('cont_features', Pipeline([
                      ('continuous', SampleExtractor(cont_col_list)),
                      ])),
        ])),
        ('scale', ModelTransformer()),
        ('sm', SMOTE(ratio='minority', random_state=240)),
        ('lr', LogisticRegression()),
])


params_logreg = {
    'scale__model': [StandardScaler(), MinMaxScaler()],
    'lr__penalty':['l1','l2'],
    'lr__C':[.01,0.03,.05],
    'lr__class_weight':[None,'balanced']
}
    

gs1 = GridSearchCV(pipeline_logreg,param_grid=params_logreg, scoring='roc_auc', n_jobs=-1)
gs1.fit(X_train,y_train)
print('best cv score', gs1.best_score_)
print('best paramas', gs1.best_params_)
print('test score', gs1.score(X_test, y_test))
```

    best cv score 0.8126109108641119
    best paramas {'lr__C': 0.03, 'lr__class_weight': None, 'lr__penalty': 'l1', 'scale__model': MinMaxScaler(copy=True, feature_range=(0, 1))}
    test score 0.8654536756927927



```python
model_logreg = gs1.best_estimator_
model_logreg.fit(X, y)
test_pred = model_logreg.predict_proba(predict_data)
test_pred = pd.DataFrame(test_pred)
test_pred['Id'] = [i for i in range(1,116294)]
test_pred.rename({1:'WnvPresent'}, axis=1, inplace=True)
test_pred.drop([0],axis=1,inplace=True)
```


```python
feature_names = []
for i in [species, block, street,trap,month, week, cluster, spray_cluster,month_cluster]:
    feature_names.append(i.fit(train_data).steps[1][1].get_feature_names())

columns_list = [i for i in X.columns if i not in cols_to_dummy]

feature_names.append(columns_list)

feature_names = [i for j in feature_names for i in j]

coef = pd.DataFrame(feature_names, model_logreg.steps[3][1].coef_.tolist(),columns=['Feature'])
coef.reset_index(inplace=True)

coef.rename({'level_0':'Coefficient'}, axis=1, inplace=True)

coef['abs'] = coef['Coefficient'].abs()
coef50 = coef.sort_values(by='abs',ascending=False)[:50]
coef50
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coefficient</th>
      <th>Feature</th>
      <th>abs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>412</th>
      <td>-4.217283</td>
      <td>time_since_max_wnvpresent</td>
      <td>4.217283</td>
    </tr>
    <tr>
      <th>402</th>
      <td>1.163883</td>
      <td>year</td>
      <td>1.163883</td>
    </tr>
    <tr>
      <th>658</th>
      <td>1.152250</td>
      <td>SeaLevel_shift_7</td>
      <td>1.152250</td>
    </tr>
    <tr>
      <th>353</th>
      <td>1.115628</td>
      <td>week_36</td>
      <td>1.115628</td>
    </tr>
    <tr>
      <th>354</th>
      <td>1.103759</td>
      <td>week_37</td>
      <td>1.103759</td>
    </tr>
    <tr>
      <th>399</th>
      <td>-1.046141</td>
      <td>Longitude</td>
      <td>1.046141</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.033282</td>
      <td>species_culex_pipiens</td>
      <td>1.033282</td>
    </tr>
    <tr>
      <th>665</th>
      <td>-0.985192</td>
      <td>Tmax_shift_8</td>
      <td>0.985192</td>
    </tr>
    <tr>
      <th>785</th>
      <td>-0.881603</td>
      <td>ResultSpeed_shift_14</td>
      <td>0.881603</td>
    </tr>
    <tr>
      <th>714</th>
      <td>-0.730839</td>
      <td>ResultDir_shift_10</td>
      <td>0.730839</td>
    </tr>
    <tr>
      <th>501</th>
      <td>-0.688991</td>
      <td>day_length_rolling_14</td>
      <td>0.688991</td>
    </tr>
    <tr>
      <th>409</th>
      <td>0.636571</td>
      <td>bearing_to_mdw</td>
      <td>0.636571</td>
    </tr>
    <tr>
      <th>130</th>
      <td>0.584938</td>
      <td>street__s_doty_ave</td>
      <td>0.584938</td>
    </tr>
    <tr>
      <th>807</th>
      <td>0.580135</td>
      <td>heavy_rain_8_days_ago</td>
      <td>0.580135</td>
    </tr>
    <tr>
      <th>563</th>
      <td>0.476769</td>
      <td>Cool_shift_2</td>
      <td>0.476769</td>
    </tr>
    <tr>
      <th>337</th>
      <td>0.441473</td>
      <td>month_8</td>
      <td>0.441473</td>
    </tr>
    <tr>
      <th>419</th>
      <td>0.397492</td>
      <td>Cool</td>
      <td>0.397492</td>
    </tr>
    <tr>
      <th>394</th>
      <td>-0.391246</td>
      <td>month_cluster_6</td>
      <td>0.391246</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.378700</td>
      <td>species_culex_territans</td>
      <td>0.378700</td>
    </tr>
    <tr>
      <th>483</th>
      <td>-0.362233</td>
      <td>day_length_rolling_10</td>
      <td>0.362233</td>
    </tr>
    <tr>
      <th>707</th>
      <td>0.358990</td>
      <td>Cool_shift_10</td>
      <td>0.358990</td>
    </tr>
    <tr>
      <th>695</th>
      <td>-0.341275</td>
      <td>ResultSpeed_shift_9</td>
      <td>0.341275</td>
    </tr>
    <tr>
      <th>138</th>
      <td>0.321791</td>
      <td>street__s_kostner_ave</td>
      <td>0.321791</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.307184</td>
      <td>block_22</td>
      <td>0.307184</td>
    </tr>
    <tr>
      <th>599</th>
      <td>0.306923</td>
      <td>Cool_shift_4</td>
      <td>0.306923</td>
    </tr>
    <tr>
      <th>19</th>
      <td>-0.301181</td>
      <td>block_25</td>
      <td>0.301181</td>
    </tr>
    <tr>
      <th>790</th>
      <td>-0.288806</td>
      <td>heavy_rain</td>
      <td>0.288806</td>
    </tr>
    <tr>
      <th>587</th>
      <td>-0.274918</td>
      <td>ResultSpeed_shift_3</td>
      <td>0.274918</td>
    </tr>
    <tr>
      <th>680</th>
      <td>0.269050</td>
      <td>RelHum_shift_8</td>
      <td>0.269050</td>
    </tr>
    <tr>
      <th>742</th>
      <td>-0.259680</td>
      <td>Heat_shift_12</td>
      <td>0.259680</td>
    </tr>
    <tr>
      <th>640</th>
      <td>0.246913</td>
      <td>SeaLevel_shift_6</td>
      <td>0.246913</td>
    </tr>
    <tr>
      <th>793</th>
      <td>0.235962</td>
      <td>heavy_rain_1_days_ago</td>
      <td>0.235962</td>
    </tr>
    <tr>
      <th>512</th>
      <td>0.220491</td>
      <td>PrecipTotal_rolling_21</td>
      <td>0.220491</td>
    </tr>
    <tr>
      <th>398</th>
      <td>-0.204311</td>
      <td>Latitude</td>
      <td>0.204311</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.195590</td>
      <td>block_10</td>
      <td>0.195590</td>
    </tr>
    <tr>
      <th>552</th>
      <td>0.173549</td>
      <td>ResultDir_shift_1</td>
      <td>0.173549</td>
    </tr>
    <tr>
      <th>372</th>
      <td>-0.171298</td>
      <td>spray_cluster_1</td>
      <td>0.171298</td>
    </tr>
    <tr>
      <th>515</th>
      <td>-0.167893</td>
      <td>ResultSpeed_rolling_21</td>
      <td>0.167893</td>
    </tr>
    <tr>
      <th>254</th>
      <td>0.167723</td>
      <td>trap_225</td>
      <td>0.167723</td>
    </tr>
    <tr>
      <th>803</th>
      <td>0.164948</td>
      <td>heavy_rain_6_days_ago</td>
      <td>0.164948</td>
    </tr>
    <tr>
      <th>363</th>
      <td>-0.161601</td>
      <td>cluster_2</td>
      <td>0.161601</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.150232</td>
      <td>block_11</td>
      <td>0.150232</td>
    </tr>
    <tr>
      <th>359</th>
      <td>-0.145105</td>
      <td>cluster_0</td>
      <td>0.145105</td>
    </tr>
    <tr>
      <th>677</th>
      <td>0.143506</td>
      <td>ResultSpeed_shift_8</td>
      <td>0.143506</td>
    </tr>
    <tr>
      <th>767</th>
      <td>-0.137883</td>
      <td>ResultSpeed_shift_13</td>
      <td>0.137883</td>
    </tr>
    <tr>
      <th>639</th>
      <td>0.135556</td>
      <td>StnPressure_shift_6</td>
      <td>0.135556</td>
    </tr>
    <tr>
      <th>0</th>
      <td>-0.134610</td>
      <td>restuans</td>
      <td>0.134610</td>
    </tr>
    <tr>
      <th>713</th>
      <td>-0.111554</td>
      <td>ResultSpeed_shift_10</td>
      <td>0.111554</td>
    </tr>
    <tr>
      <th>811</th>
      <td>0.106675</td>
      <td>heavy_rain_10_days_ago</td>
      <td>0.106675</td>
    </tr>
    <tr>
      <th>381</th>
      <td>0.104742</td>
      <td>month_cluster_11</td>
      <td>0.104742</td>
    </tr>
  </tbody>
</table>
</div>




```python
%matplotlib inline
coef50.sort_values(by='abs').plot(y='Coefficient',x='Feature',kind='barh', figsize=(10,10), color='b', 
                                  title='Coefficients for Top 50 Features - Logistic Regression');
```


![png](/images/West_nile_project_final_files/West_nile_project_final_56_0.png)


```python
pipeline_rf = Pipeline([
    ('features', FeatureUnion([
        ('species', species),
        ('block', block),
        ('street', street),
        ('trap', trap),
        ('month', month),
        ('week', week),
        ('cluster',cluster),
        ('spray_cluster',spray_cluster),
        ('month_cluster',month_cluster),
        ('cont_features', Pipeline([
                      ('continuous', SampleExtractor(cont_col_list)),
                      ])),
        ])),
        ('scale', ModelTransformer()),
        ('sm', SMOTE(ratio='minority',random_state=240)),
        ('rf', RandomForestClassifier()),
])


params_rf = {
    'scale__model': [StandardScaler(), MinMaxScaler()],
    'rf__n_estimators':[50, 100, 150],
    'rf__max_features':['auto', 'log2', None],
    'rf__max_depth':[10,30,None]
}
    

gs2 = GridSearchCV(pipeline_rf,param_grid=params_rf, scoring='roc_auc')
gs2.fit(X_train,y_train)
print('best cv score', gs2.best_score_)
print('best paramas', gs2.best_params_)
print('test score', gs2.score(X_test, y_test))
```

    best cv score 0.8095833270494316
    best paramas {'rf__max_depth': 10, 'rf__max_features': 'auto', 'rf__n_estimators': 100, 'scale__model': MinMaxScaler(copy=True, feature_range=(0, 1))}
    test score 0.8663261937326927



```python
model_rf = gs2.best_estimator_
model_rf.fit(X, y)
test_pred2 = model_rf.predict_proba(predict_data)
test_pred2 = pd.DataFrame(test_pred2)
test_pred2['Id'] = [i for i in range(1,116294)]
test_pred2.rename({1:'WnvPresent'}, axis=1, inplace=True)
test_pred2.drop([0],axis=1,inplace=True)
test_pred2.to_csv('test_pred_50.csv',index=False)
```


```python
feature_importances = pd.DataFrame(feature_names, model_rf.steps[3][1].feature_importances_.tolist(),columns=['Feature'])
feature_importances.reset_index(inplace=True)
feature_importances.rename({'index':'Feature Importance'}, axis=1, inplace=True)

feature_importances.sort_values(by='Feature Importance', ascending=False)[:50]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature Importance</th>
      <th>Feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>412</th>
      <td>0.034035</td>
      <td>time_since_max_wnvpresent</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.027067</td>
      <td>species_culex_pipiens</td>
    </tr>
    <tr>
      <th>401</th>
      <td>0.023069</td>
      <td>day_of_year</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.019823</td>
      <td>species_culex_restuans</td>
    </tr>
    <tr>
      <th>529</th>
      <td>0.014650</td>
      <td>Sunset_rolling_30</td>
    </tr>
    <tr>
      <th>483</th>
      <td>0.014350</td>
      <td>day_length_rolling_10</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.013294</td>
      <td>restuans</td>
    </tr>
    <tr>
      <th>492</th>
      <td>0.012848</td>
      <td>Sunrise_rolling_14</td>
    </tr>
    <tr>
      <th>493</th>
      <td>0.012846</td>
      <td>Sunset_rolling_14</td>
    </tr>
    <tr>
      <th>475</th>
      <td>0.012584</td>
      <td>Sunset_rolling_10</td>
    </tr>
    <tr>
      <th>474</th>
      <td>0.011776</td>
      <td>Sunrise_rolling_10</td>
    </tr>
    <tr>
      <th>523</th>
      <td>0.011266</td>
      <td>Tavg_rolling_30</td>
    </tr>
    <tr>
      <th>429</th>
      <td>0.010870</td>
      <td>day_length</td>
    </tr>
    <tr>
      <th>505</th>
      <td>0.010711</td>
      <td>Tavg_rolling_21</td>
    </tr>
    <tr>
      <th>439</th>
      <td>0.010655</td>
      <td>Sunset_rolling_3</td>
    </tr>
    <tr>
      <th>501</th>
      <td>0.010277</td>
      <td>day_length_rolling_14</td>
    </tr>
    <tr>
      <th>535</th>
      <td>0.009478</td>
      <td>AvgSpeed_rolling_30</td>
    </tr>
    <tr>
      <th>601</th>
      <td>0.009427</td>
      <td>Sunset_shift_4</td>
    </tr>
    <tr>
      <th>528</th>
      <td>0.009351</td>
      <td>Sunrise_rolling_30</td>
    </tr>
    <tr>
      <th>537</th>
      <td>0.008969</td>
      <td>day_length_rolling_30</td>
    </tr>
    <tr>
      <th>465</th>
      <td>0.008815</td>
      <td>day_length_rolling_5</td>
    </tr>
    <tr>
      <th>508</th>
      <td>0.008596</td>
      <td>Heat_rolling_21</td>
    </tr>
    <tr>
      <th>457</th>
      <td>0.008467</td>
      <td>Sunset_rolling_5</td>
    </tr>
    <tr>
      <th>447</th>
      <td>0.008378</td>
      <td>day_length_rolling_3</td>
    </tr>
    <tr>
      <th>506</th>
      <td>0.008326</td>
      <td>DewPoint_rolling_21</td>
    </tr>
    <tr>
      <th>522</th>
      <td>0.008144</td>
      <td>Tmin_rolling_30</td>
    </tr>
    <tr>
      <th>525</th>
      <td>0.008080</td>
      <td>WetBulb_rolling_30</td>
    </tr>
    <tr>
      <th>681</th>
      <td>0.007550</td>
      <td>day_length_shift_8</td>
    </tr>
    <tr>
      <th>489</th>
      <td>0.007481</td>
      <td>WetBulb_rolling_14</td>
    </tr>
    <tr>
      <th>561</th>
      <td>0.007240</td>
      <td>WetBulb_shift_2</td>
    </tr>
    <tr>
      <th>495</th>
      <td>0.006654</td>
      <td>StnPressure_rolling_14</td>
    </tr>
    <tr>
      <th>565</th>
      <td>0.006422</td>
      <td>Sunset_shift_2</td>
    </tr>
    <tr>
      <th>434</th>
      <td>0.006197</td>
      <td>DewPoint_rolling_3</td>
    </tr>
    <tr>
      <th>407</th>
      <td>0.006093</td>
      <td>bearing_to_hegewich</td>
    </tr>
    <tr>
      <th>527</th>
      <td>0.005751</td>
      <td>Cool_rolling_30</td>
    </tr>
    <tr>
      <th>531</th>
      <td>0.005747</td>
      <td>StnPressure_rolling_30</td>
    </tr>
    <tr>
      <th>637</th>
      <td>0.005328</td>
      <td>Sunset_shift_6</td>
    </tr>
    <tr>
      <th>532</th>
      <td>0.005328</td>
      <td>SeaLevel_rolling_30</td>
    </tr>
    <tr>
      <th>519</th>
      <td>0.005210</td>
      <td>day_length_rolling_21</td>
    </tr>
    <tr>
      <th>789</th>
      <td>0.005083</td>
      <td>day_length_shift_14</td>
    </tr>
    <tr>
      <th>409</th>
      <td>0.005079</td>
      <td>bearing_to_mdw</td>
    </tr>
    <tr>
      <th>753</th>
      <td>0.005030</td>
      <td>day_length_shift_12</td>
    </tr>
    <tr>
      <th>404</th>
      <td>0.004594</td>
      <td>distance_to_ord_location</td>
    </tr>
    <tr>
      <th>400</th>
      <td>0.004470</td>
      <td>AddressAccuracy</td>
    </tr>
    <tr>
      <th>709</th>
      <td>0.004409</td>
      <td>Sunset_shift_10</td>
    </tr>
    <tr>
      <th>521</th>
      <td>0.004384</td>
      <td>Tmax_rolling_30</td>
    </tr>
    <tr>
      <th>536</th>
      <td>0.004203</td>
      <td>RelHum_rolling_30</td>
    </tr>
    <tr>
      <th>731</th>
      <td>0.004166</td>
      <td>ResultSpeed_shift_11</td>
    </tr>
    <tr>
      <th>421</th>
      <td>0.003945</td>
      <td>Sunset</td>
    </tr>
    <tr>
      <th>695</th>
      <td>0.003921</td>
      <td>ResultSpeed_shift_9</td>
    </tr>
  </tbody>
</table>
</div>




```python
feature_importances.sort_values(by='Feature Importance', ascending=False)[:50].plot(y='Feature Importance',x='Feature',kind='barh', figsize=(10,10), color='b', 
                                  title='Coefficients for Top 50 Features - Random Forest');
```


![png](/images/West_nile_project_final_files/West_nile_project_final_61_0.png)

