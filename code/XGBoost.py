#importing libraries
import numpy as np
import pandas as pd
import os
from statsmodels import api as sm
import pylab as py
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from datetime import datetime
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import seaborn as sns
import squarify
from scipy.stats import kstest,norm
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
from sklearn.utils import resample
from sklearn import metrics
from scipy.stats import chi2_contingency

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

df = pd.read_csv('~/Documents/Georgetown/ANLY512/project/2019-Oct.csv')
df = reduce_mem_usage(df)

df.info()
df = df.dropna()
df.shape

#No of visitors by date
data = df.loc[:, ['event_time', 'user_id']]
#Extracting only dates
data['event_time'] = data['event_time'].apply(lambda s: str(s)[0:10])
visitor_by_date = data.drop_duplicates().groupby(['event_time'])['user_id'].agg(['count']).sort_values(by=['event_time'], ascending=True)
x = pd.Series(visitor_by_date.index.values).apply(lambda s: datetime.strptime(s, '%Y-%m-%d').date())
y = visitor_by_date['count']
plt.rcParams['figure.figsize'] = (20,8)

plt.plot(x,y)
plt.show()

# most bought brands
df['brand'].value_counts()
df['event_type'].value_counts()

title_type = df.groupby('brand').agg('count')
print(title_type)
type_labels = title_type.user_id.sort_values().index
type_counts = title_type.user_id.sort_values()
plt.figure(1,figsize =(20,10))
the_grid = GridSpec(2,2)
cmap = plt.get_cmap('Spectral')
colors = [cmap(i) for i in np.linspace(0,1,8)]
plt.subplot(the_grid[0,1],aspect=1,title = 'Brand titles')
type_show_ids = plt.pie(type_counts,labels = type_labels,autopct = '%1.1f%%',shadow = True,colors = colors)
plt.show()
# popular prouct categories
top_category_n = 30
top_category = df.loc[:,'category_code'].value_counts()[:top_category_n].sort_values(ascending=False)
squarify.plot(sizes=top_category, label=top_category.index.array, color=["red","cyan","green","orange","blue","grey"],
              alpha=.7)
plt.axis('off')
plt.show()

labels = ['view', 'cart', 'purchase']
size = df['event_type'].value_counts()
colors = ['yellowgreen', 'lightskyblue','lightcoral']
explode = [0, 0.1, 0.1]

plt.rcParams['figure.figsize'] = (8, 8)
plt.pie(size, colors=colors, explode=explode, labels=labels, shadow=True, autopct='%.2f%%')
plt.title('Event_Type', fontsize=20)
plt.axis('off')
plt.legend()
plt.show()
# conversion rate
view_count = df['event_type'].value_counts()[0]
cart_count = df['event_type'].value_counts()[1]
purchase_count = df['event_type'].value_counts()[2]
print("Rate of conversion between view and purchase events" +str((purchase_count/view_count)*100) +'%')
print("Rate of conversion between view and add to cart events" +str((cart_count/view_count)*100) +'%')
print("Rate of conversion between add to cart and purchase events" +str((purchase_count/cart_count)*100) +'%')

#Brandwise sales of all event types
df['brand'].value_counts().head(50).plot.bar(figsize = (18,7))
plt.title('Top brand',fontsize = 20)
plt.xlabel('Names of brand')
plt.ylabel('Count')
plt.show()

d = df.loc[df['event_type'].isin(['purchase'])].drop_duplicates()
print(d['brand'].value_counts())
d['brand'].value_counts().head(70).plot.bar(figsize =(18,7))
plt.xlabel('Names of brand')
plt.ylabel('Count')
plt.show()

top_player = df['brand'].value_counts()[0]
second_player = df['brand'].value_counts()[1]
last_player = df['brand'].value_counts()[-1]
print("Top brand saw " +str((top_player/second_player)*100)+"%more sales than second_player in the market")
print("Top brand saw " +str((top_player/last_player)*100)+"%more sales than bottom player in the market")
# preparing the data
#List of people who has bought or added products to the cart
cart_purchase_users = df.loc[df["event_type"].isin(["cart", "purchase"])].drop_duplicates(subset=['user_id'])
cart_purchase_users.dropna(how='any', inplace=True)
print(cart_purchase_users)

cart_purchase_users_all_activity = df.loc[df['user_id'].isin(cart_purchase_users['user_id'])]
print(cart_purchase_users_all_activity)
activity_in_session = cart_purchase_users_all_activity.groupby(['user_session'])['event_type'].count().reset_index()
activity_in_session = activity_in_session.rename(columns={"event_type": "activity_count"})
print(activity_in_session)

# Extract event date from event_time column and find on which date the activity occurs

def convert_time_to_date(utc_timestamp):
  utc_date = datetime.strptime(utc_timestamp[0:10],'%Y-%m-%d').date()
  return utc_date


df['event_date'] = df['event_time'].apply(lambda s:convert_time_to_date(s))

# Splitting of category and sub category is done by string handling
df_targets = df.loc[df["event_type"].isin(["cart","purchase"])].drop_duplicates(subset = ['event_type',
                                                                    'product_id','price','user_id','user_session'])
df_targets["is_purchased"] = np.where(df_targets["event_type"]=="purchase",1,0)
df_targets["is_purchased"] = df_targets.groupby(["user_session","product_id"])["is_purchased"].transform("max")
df_targets = df_targets.loc[df_targets["event_type"]=='cart'].drop_duplicates(["user_session","product_id","is_purchased"])
df_targets['event_weekday'] = df_targets['event_date'].apply(lambda s: s.weekday())
df_targets.dropna(how = 'any',inplace = True)
df_targets["category_code_level1"] = df_targets["category_code"].str.split(".",expand = True)[0].astype('category')
df_targets["category_code_level2"] = df_targets["category_code"].str.split(".",expand = True)[1].astype('category')

df_targets = df_targets.merge(activity_in_session,on = 'user_session',how ='left')
df_targets['activity_count'] = df_targets['activity_count'].fillna(0)
df_targets.head()

df_targets.info()
df_targets['event_weekday'] = df_targets.event_weekday.astype(object)
df_targets.info()

#Saving a copy of preprocessed data
df_targets.to_csv('training_data.csv')
df_targets = pd.read_csv('training_data.csv')
df_targets.head()

#Resampling data to have equal no of purchased and not purchased items¶
#no.of rows when the item was purchased was around 5 lakh and not purchased was around 8 lakh.

#To balance data resampling is done
is_purchase_set = df_targets[df_targets['is_purchased'] == 1]
is_purchase_set.shape[0]

not_purchase_set = df_targets[df_targets['is_purchased'] == 0]
not_purchase_set.shape[0]

n_samples = 270000
is_purchase_downsampled = resample(is_purchase_set,replace = False,n_samples = n_samples,random_state = 27)
not_purchase_set_downsampled = resample(not_purchase_set,replace = False,n_samples = n_samples,random_state = 27)

downsampled = pd.concat([is_purchase_downsampled,not_purchase_set_downsampled])
downsampled['is_purchased'].value_counts()

features = downsampled[['brand','price','event_weekday','category_code_level1','category_code_level2','activity_count']]

# encoding categorical arrtibutes
features.loc[:,'brand'] =LabelEncoder().fit_transform(downsampled.loc[:,'brand'].copy())
features.loc[:,'event_weekday'] = LabelEncoder().fit_transform(downsampled.loc[:,'event_weekday'].copy())
features.loc[:,'category_code_level1'] = LabelEncoder().fit_transform(downsampled.loc[:,'category_code_level1'].copy())
features.loc[:,'category_code_level2'] = LabelEncoder().fit_transform(downsampled.loc[:,'category_code_level2'].copy())
is_purchased = LabelEncoder().fit_transform(downsampled['is_purchased'])
features.head()

print(list(features.columns))
features['brand'] = features['brand'].astype("category")
features['event_weekday'] = features['event_weekday'].astype("category")
features['category_code_level1'] = features['category_code_level1'].astype("category")
features['category_code_level2'] = features['category_code_level2'].astype("category")
features['category_code_level1'] = features['category_code_level1'].astype("category")


features.info()
str(features)
# spilt the data
X_train, X_test, y_train, y_test = train_test_split(features,
                                                    is_purchased,
                                                    test_size = 0.25,
                                                    random_state = 0)

from xgboost import XGBClassifier
import matplotlib.pyplot as plt
xg_model = XGBClassifier(learningrate=0.1)
xg_model.fit(X_train,y_train)
y_pred = xg_model.predict(X_test)

print("Accuracy", metrics.accuracy_score(y_test, y_pred))
print("Precision", metrics.precision_score(y_test, y_pred))
print("Recall", metrics.recall_score(y_test, y_pred))
print("fbeta", metrics.fbeta_score(y_test, y_pred, average='weighted', beta=0.5))

# feature importance
ax = plot_importance(xg_model, max_num_features=10, importance_type = 'gain')
fig = ax.figure
#fig.set_size_inches(15, 25)
#fig, ax = plt.subplots(figsize=(20, 15))
#plot_importance(xg_model, ax = ax)
#plt.figure(figsize=(10,5))
plt.rcParams['figure.figsize'] = (27, 10)
plt.title('Feature Importance', fontsize=20, fontname="ITC Officina Sans", fontweight="bold",
          color="#726abb")
plt.ylabel('Features', fontname="ITC Officina Sans", fontweight="bold", color="#726abb",
           fontsize=18)
plt.xlabel('F Score', fontname="ITC Officina Sans", fontweight="bold",
           color="#726abb", fontsize=18)
#plt.tight_layout()
plt.xticks(fontname="ITC Officina Sans", color="#726abb", fontsize=18)
plt.yticks(fontname="ITC Officina Sans", color="#726abb", fontsize=18)
plt.savefig('Feature Importance2.png')
plt.show()

# confusion matrix
from numpy import array,array_equal
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

matrix = confusion_matrix(y_test, y_pred, normalize='all')
df_cm = pd.DataFrame(matrix, index=['Not Purchase', 'Purchased'],
                     columns=['Not Purchase', 'Purchased'])
plt.figure(figsize=(10, 7))
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
sns.heatmap(df_cm, annot=True, cmap="BuPu", fmt="d")
plt.title('Confusion matrix of XGBoost', fontsize=16, fontname="ITC Officina Sans", fontweight="bold",
          color="#726abb")
plt.xlabel('Predicted', fontname="ITC Officina Sans", fontweight="bold", color="#726abb",
           fontsize=14)
plt.ylabel('True', fontname="ITC Officina Sans", fontweight="bold",
           color="#726abb", fontsize=14)
plt.tight_layout()
plt.xticks(fontname="ITC Officina Sans", color="#726abb", fontsize=13)
plt.yticks(fontname="ITC Officina Sans", color="#726abb", fontsize=13)
plt.savefig('XGBoost_confusionmatrix.png')
plt.show()

