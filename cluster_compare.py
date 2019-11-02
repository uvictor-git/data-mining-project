import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import time as time
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_swiss_roll
from sklearn import preprocessing

# #############################################################################
# Reading and preparing the data
df = pd.read_csv('online_retail.csv60001.csv')
df_tr = df
df_tr.columns =['InvoiceNo', 'StockCode','Description', 'Quantity', 'InvoiceDate','UnitPrice', 'CustomerID',
         'Country']
df_tr['InvoiceNo'].str.lstrip('+-').str.rstrip('cC')

#Converting categorical data to numeric
df_tr['Country'] = df_tr['Country'].astype('category')
df_tr['InvoiceDate'] = df_tr['InvoiceDate'].astype('category')
df_tr["Country"] = df_tr["Country"].cat.codes

df_tr["InvoiceDate"] = df_tr["InvoiceDate"].cat.codes


df_tr = pd.get_dummies(df_tr, columns=['Description'])

#############################################################################
#Normalizing the data
clmns = ['Quantity', 'InvoiceDate','UnitPrice', 'Country']
clmns2 = ['Quantity', 'InvoiceDate','UnitPrice', 'CustomerID', 'Country']

df_tr_std = stats.zscore(df_tr[clmns])


# Create x, where x the 'scores' column's values as floats
df_tr = df_tr[clmns2].values.astype(float)

# Create a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()

# Create an object to transform the data to fit minmax processor
x_scaled = min_max_scaler.fit_transform(df_tr)

# Run the normalizer on the dataframe
df_tr = pd.DataFrame(x_scaled)
df_tr.columns =['Quantity', 'InvoiceDate','UnitPrice', 'CustomerID', 'Country']



# #############################################################################
#Cluster the data KMEANS
print("Compute KMEANS Clustering...")
ct = time.time()
kmeans = KMeans(n_clusters=3, random_state=0).fit(df_tr_std)
labels_KM = kmeans.labels_
elapsed_time_KM = time.time() - ct
print("Elapsed time: %.2fs" % elapsed_time_KM)
print("Number of points: %i" % labels_KM.size)

#Glue back to original data
df_tr['clusters_KM'] = labels_KM

#Add the column into our list
clmns.extend(['clusters_KM'])


# #############################################################################
# Compute clustering hierarchical
print("Compute hierarchical clustering...")
st = time.time()
ward = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(df_tr_std)
elapsed_time_HC = time.time() - st
label_HC = ward.labels_
print("Elapsed time: %.2fs" % elapsed_time_HC)
print("Number of points: %i" % label_HC.size)

#Glue back to originaal data
df_tr['clusters_HC'] = label_HC

#Add the column into our list
clmns.extend(['clusters_HC'])

# #############################################################################
#Lets analyze the clusters
print (df_tr[clmns].groupby(['clusters_KM']).mean())

print (df_tr[clmns].groupby(['clusters_HC']).mean())

# #############################################################################
# Visualize the clusters

# Visualize the 
sns.lmplot('InvoiceDate', 'CustomerID', 
           data=df_tr, 
           fit_reg=False, 
             
           scatter_kws={"marker": "D", "s": 10})
plt.title( 'StockCode vs UnitPrice')
plt.xlabel('InvoiceDate')
plt.ylabel('CustomerID')


#Scatter plot of KMEANS clustering
sns.lmplot('InvoiceDate', 'CustomerID', 
           data=df_tr, 
           fit_reg=False, 
           hue="clusters_KM",  
           scatter_kws={"marker": "D", "s": 10})
plt.title('Clusters_KM StockCode vs UnitPrice')
plt.xlabel('InvoiceDate')
plt.ylabel('CustomerID')

plt.show()

#Scatter plot of Heirachical clustering
sns.lmplot('InvoiceDate', 'CustomerID', 
           data=df_tr, 
           fit_reg=False, 
           hue="clusters_HC",  
           scatter_kws={"marker": "D", "s": 10})
plt.title('Clusters_HC StockCode vs UnitPrice')
plt.xlabel('InvoiceDate')
plt.ylabel('CustomerID')

plt.show()


