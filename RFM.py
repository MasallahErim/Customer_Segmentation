import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np

# Read data
df = pd.read_csv("./data/data.csv",encoding= 'unicode_escape')
df.head()


# data understanding
def v_counts(df):
   return [df[col].value_counts() for col in df.columns]
print(v_counts(df))
df.info()


# I didn't do any padding with NaN values in the CustomerID column
# CustomerID kolonunda NaN değerler olduğu herhangi bir doldurma işlemi yapmadım
df.dropna(inplace = True)
df.isna().sum()

# we will not take returns
# iade edilenleri almayacağız
df = df[~df["InvoiceNo"].str.contains("C", na=False)]

#? The prices of some products are entered as 0 in the UnitPrice column.
# We found the index numbers of 0 values in the UnitPrice column
#? UnitPrice kolonunda bazı ürünlerin fiyatları 0 olarak girilmiş 
# UnitPrice kolonundaki 0 değerlerin index numaralarını bulduk
unitprice_0_index = df[df["UnitPrice"]==0].index

# We found the description values of the values with 0
# 0 olan değerlerin description değerlerini bulduk 
unitPrice_0 = []
for i in df[df["UnitPrice"]==0]["Description"]:
    unitPrice_0.append(i)

# We added UnitPrice value of those with the same description value to the list
# aynı description değerine sahip olanların UnitPrice değerini listeye ekledik
new_value=  []
for i in unitPrice_0:
    new_value.append(list(df[df["Description"] ==i]["UnitPrice"])[1])

# We replaced the UnitPrice value we found with values that are 0
# bulduğumuz UnitPrice değerini 0 olan değerler ile değiştirdik
x = 0
for i in unitprice_0_index:
    df.loc[i,"UnitPrice"] = new_value[x]
    if len(new_value) == x:
        break
    x+=1


#  we examine descriptive statistics for outliers
#  betimsel istatikleri inceliyoruz aykırı değerler için 
df.describe([0.05,0.10,0.25,0.50,0.80,0.95,0.99]).T
#? aykırı değerler var

# we detect and delete these outliers with the 3std method
# bu aykırı değerler 3std yöntemi ile  tespit edip siliyoruz
for column in ["Quantity", "UnitPrice"]:
    for group in df["Country"].unique():
        selected_group = df[df["Country"] == group]
        selected_colomn = selected_group[column]

        std = selected_colomn.std()
        avg = selected_colomn.mean() 

        three_sigma_plus  = avg + (3*std)
        three_sigma_minus = avg - (3*std)
        outliers = selected_colomn[((selected_group[column] > three_sigma_plus) | (selected_group[column] < three_sigma_minus))].index
        df.drop(index=outliers,axis=0,inplace=True)
        # print(column , group, outliers) 




# total money spent on a product
# bir ürün için harcanan toplam para 
df["total_price"] = df["UnitPrice"] * df["Quantity"]

# converted to date format
# tarih formatına çevirdik
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# We set today's date for recency value
# Recency değeri için bu günün tarihini belirledik
date_of_today = dt.datetime(2011,12,31)
print(date_of_today)





#! Recency (R) – Innovation – Time from last purchase to date
#! Frequency (F) –Frequency – Total number of sales
#! Monetary (M) – Monetary value – Monetary value of all purchases


# find the total money Recency (R) spent by each CustomerID
# we calculate the days from the last purchase of each CustomerID to the analysis day
# her CustomerID'nin harcağıdı toplam para Recency (R) buluyoruz
# her CustomerID'nin son alışverişinden analiz gününe kadar geçen günü hesaplıyoruz
df_R_M = df.groupby("CustomerID").agg({"total_price":lambda x : x.sum(),
                                      "InvoiceDate": lambda x : (date_of_today - x.max()).days})
#? Recency ve Monetary değerlerinini bulduk


# Do gruopby by CustomerID and InvoiceNo and sum the total_price values
# find the length of the total_price value
# CustomerID'ye  ve InvoiceNo'ya göre gruopby yap ve total_price değerlerini topla
# total_price değerinin uzunluğunu bul
df_F = df.groupby(["CustomerID", "InvoiceNo"]).agg({"total_price":lambda x : x.sum()})
df_F = df_F.groupby("CustomerID").agg({"total_price":lambda x : len(x)})

# Merge based on CustomerID
# CustomerID'e baz alarak birleştirme
rfm_df = pd.merge(df_R_M, df_F, on="CustomerID")

# rename operation
# yeniden adlandırma işlemi
rfm_df.rename(columns={"total_price_x":"Monetary",
                          "total_price_y":"Frequency",
                          "InvoiceDate":"Recency" }, inplace=True)


rfm_df = rfm_df[(rfm_df["Monetary"] > 0) & (rfm_df["Frequency"] > 0)]

# divide the columns into 5 parts
# kolonları 5 parçaya ayırdık
def FScore(x, p, d):
    if x <= d[p][0.20]:
        return 1
    
    if x <= d[p][0.40]:
        return 2
    
    if x <= d[p][0.60]:
        return 3

    if x <= d[p][0.80]:
        return 4
    else:
        return 5  


def creat_tile(df,name_score_list,function):
    quantiles = df.quantile(q = [0.20, 0.40, 0.60, 0.80])
    quantiles = quantiles.to_dict()
    for i in name_score_list:
        new_name= i+"_score"
        df[new_name] = df[i].apply(function, args=(i,quantiles,))

# Columns to be divided into 5 equal parts
# 5 eşit parçaya bölünecek kolonlar
name_score_list = ["Recency","Frequency", "Monetary"]
creat_tile(rfm_df,name_score_list,FScore)

# We combined the R FM M columns for the RFM score
# RFM skoru için R F M kolonlarını birleştirdik
rfm_df["RFM_Score"] =(rfm_df["Recency_score"].astype(str) +
                     rfm_df["Frequency_score"].astype(str) +
                     rfm_df["Monetary_score"].astype(str))


# we calculated ["min", "max", "mean", "count"] for a brief statistical interpretation
# kısa bir istatiksel yorumlama için ["min", "max", "mean", "count"] değerlerini hesapladık
rfm_df.groupby("RFM_Score").agg({"Recency" : ["mean", "min", "max", "count"],
                                 "Frequency" : ["mean", "min", "max", "count"],
                                 "Monetary" : ["mean", "min", "max", "count"]}).round(2)



# We used regex to cluster customers by RFM_Score column
# RFM_Score kolonuna göre  müşterileri kümelemek için regex kullandık
seg_map = {
    r'[1-2][1-2]': 'Hibernating',
    r'[1-2][3-4]': 'At_Risk',
    r'[1-2]5': 'Cant_Loose',
    r'3[1-2]': 'About_to_Sleep',
    r'33': 'Need_Attention',
    r'[3-4][4-5]': 'Loyal_Customers',
    r'41': 'Promising',
    r'51': 'New_Customers',
    r'[4-5][2-3]': 'Potential_Loyalists',
    r'5[4-5]': 'Champions'}
rfm_df['cluster_of_customer'] = rfm_df['Recency_score'].astype(str) + rfm_df['Frequency_score'].astype(str)
rfm_df['cluster_of_customer'] = rfm_df['cluster_of_customer'].replace(seg_map, regex=True)



# General statistics for each cluster_of_customer
# her bir benzersiz cluster_of_customer değeri için genel istatistikleri görüntüledik
rfm_df[["cluster_of_customer", "Recency", "Frequency", "Monetary"]].groupby("cluster_of_customer").agg(["mean", 
                                                                                                        "count"])
rfm_df[["cluster_of_customer", "Recency", "Frequency", "Monetary"]].groupby("cluster_of_customer").agg(["mean", 
                                                                                                        "count", 
                                                                                                        "min", ])
                 

# pasta grafiği 
colors = ['#C2C2C2', '#5BADFF', '#33FF99', '#FFAB4B', 
          '#C184FF', '#AD6F33','#FFFF5B','#85BBB2',
          '#FF8484','#33FFFF']
x=0
fig, ax = plt.subplots(6 ,figsize=(20,25))
for i in ["Recency", "Frequency", "Monetary"]:
    for j in ["count", "mean"]:
        figure_df= rfm_df[["cluster_of_customer","Recency","Frequency","Monetary"]].groupby("cluster_of_customer").agg(["mean", "median","count"])
        ax[x].pie(figure_df[i][j] ,labels=rfm_df["cluster_of_customer"].unique() ,colors=colors, 
                explode=[0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05] ,autopct='%1.1f%%')
        ax[x].set_title(i+"_"+j, fontsize=18)
        x+=1
plt.show()



# save final version of df
rfm_df.to_csv("./data/rfm_df.csv")



