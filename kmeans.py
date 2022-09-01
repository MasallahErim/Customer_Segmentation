# library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# data read
df = pd.read_csv("./data/rfm_df.csv")
df = df.iloc[:,1:4]


# scaler processing
scaler = StandardScaler()
df_scaled_ss = scaler.fit_transform(df)
df_scaled_ss = pd.DataFrame(df_scaled_ss,columns=df.columns)


plt.figure(figsize=(8,6))
#  WCSS, bir kümedeki her nokta ile merkez nokta arasındaki uzaklığın karesinin toplamıdır.
wcss = [] 

# elbow method
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,
                    init="k-means++",
                    n_init=10,
                    max_iter=300,
                    random_state = 42 ).fit(df_scaled_ss)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title("The Elbow Methot")
plt.xlabel("number of cluster")
plt.ylabel("wcss")
plt.show()




# Kmeans 
kmeans = KMeans(n_clusters=5,
                    init="k-means++",
                    n_init=10,
                    max_iter=300,
                    random_state = 0).fit(df_scaled_ss)

# prediction process
pred = kmeans.predict(df_scaled_ss)

df_last_rfm = pd.DataFrame(df)
df_last_rfm["cluster"] = pred
df_last_rfm["cluster"].value_counts()


# General statistics for each cluster
df_last_rfm.groupby("cluster").agg({"mean","std", "count"}).round(2)







































































































































































