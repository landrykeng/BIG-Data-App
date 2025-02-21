
import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px 
import os
import warnings
import datetime
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.offline as py
import plotly.tools as tls
import plotly
import time
import datetime as dt

from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import add_changepoints_to_plot, plot_cross_validation_metric

from pyspark.ml.stat import Correlation
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import classification_report, confusion_matrix

# Import Sparksession
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import *
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

import warnings
warnings.filterwarnings('ignore')
spark=SparkSession.builder.appName("Data_Wrangling").getOrCreate()
warnings.filterwarnings('ignore')


st.markdown(
    """
    <style>
    body {
        font-family: sans-serif;
    }
    .title {
        text-align: center;
        color: #306609; /* Couleur du titre */
    }
    .subtitle {
        text-align: center;
        color: #6699CC; /* Couleur du sous-titre */
    }
    .section-header {
        background-color: #1864B8; /* Couleur de fond des sections */
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

st.title("PROJET BIG DATA AVEC STREAMLIT")
st.subheader('GROUPE 1')
st.subheader("Enseignant: Mr Serge Ndoumin")
st.markdown('<div class="section-header"><center><h2>Analyse avec des données du e-commerce au Pakistan</h2></center></div>', unsafe_allow_html=True)



#==========================================================
#==== Base et Traitement ==================================
#==========================================================
#@st.cache_data
#def load_data():
file_location = 'Pakistan_Dataset.csv'
file_type = "csv"
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

df = spark.read.format(file_type)\
.option("inferSchema", infer_schema)\
.option("header", first_row_is_header)\
.option("sep", delimiter)\
.load(file_location)


#dfp=spark.createDataFrame(load_data())
#dfp=load_data()
df_brut=df

drop_columns_list=["_c21","_c22","_c23","_c24","_c25"]
df_brut = df_brut.drop(*drop_columns_list)

st.sidebar.image("Logo.png") # Remplacez par le chemin de votre logo
st.sidebar.title("Membre du GROUPE")
st.sidebar.text("ASSADICK IBNI Oumar Ali")
st.sidebar.text("ATANGANA TSIMI Arsène Joël")
st.sidebar.text("HUSKEN TIAWE Alphonse")
st.sidebar.text("KENGNE Bienvenu Landry")
st.sidebar.text("MAGUETSWET Rivalien")
st.sidebar.text("MIKOUIZA BOUNGOUDI Jeanstel Hurvinel")
st.sidebar.text("NOFOZO YIMFOU Sylvain")
st.sidebar.text("YAKAWOU Komlanvi Eyram")
st.sidebar.text("YALIGAZA Edson Belkrys De-Valor")
#==========================================================
#==== FONCTION DE VISUALISATION ===========================
#==========================================================

def make_bar(data,x_val,y_val,text_val,color,titre="",titre_x="",titre_y=""):
    fig = px.bar(data, x=x_val, y=y_val, text=text_val,  
    color_discrete_sequence=color)
    fig.update_traces(texttemplate='%{text}',  textposition='outside')
    fig.update_layout(title=titre,xaxis_title=titre_x,yaxis_title=titre_y)
    st.plotly_chart(fig)

def make_bar_polar(data,value,group,color,titre=""):
    fig = px.bar_polar(data, 
                   r=value, 
                   theta=group, 
                   color_discrete_sequence=color,
                   title=titre)

    fig.update_traces(
        text=data[value],  # Ajoute les valeurs sur les barres
        marker=dict(line=dict(color='white', width=1.5))  # Bordures blanches pour démarquer les barres
    )

    fig.update_layout(
        title_font_size=20,  # Taille du titre
        title_x=0.5,  # Centrage du titre
        polar=dict(
            radialaxis=dict(showticklabels=True, ticks=""),  # Affiche les étiquettes sur l'axe radial
            bgcolor='rgba(0,0,0,0)'  # Fond transparent
        ),
        paper_bgcolor='rgba(0,0,0,0)',  # Fond transparent du cadre
        plot_bgcolor='rgba(0,0,0,0)'  # Fond transparent du graphique
    )

    st.plotly_chart(fig)

def make_tremap(data, value,group,titre=""):
    figtr = px.treemap(data, 
                 path=[group], 
                 values=value,
                 color=value,  # Ajoute une gradation de couleur en fonction des valeurs
                 color_continuous_scale="Viridis",  # Palette plus moderne et esthétique
                 title=titre)

    figtr.update_traces(
    textinfo="label+value",  # Affiche à la fois le nom et la valeur dans chaque case
    textfont_size=18,  # Augmente la taille du texte pour plus de lisibilité
    marker=dict(
        line=dict(color='white', width=2)  # Bordures blanches pour mieux démarquer les catégories
    )
)

    figtr.update_layout(
    title_font_size=20,  # Taille du titre
    title_x=0.5,  # Centrage du titre
    paper_bgcolor='rgba(0,0,0,0)',  # Fond transparent du cadre
    plot_bgcolor='rgba(0,0,0,0)'  # Fond transparent du graphique
)
    st.plotly_chart(figtr)


def make_combined_map(input_geodf, pib_column, population_column, width=800, height=600):
    pass

 
#st.sidebar.write(f"Téléphone : {telephone}")
tables = st.tabs(["Données Brutes", "Petite observation des donnée brutes", "Données Traitées","Visualisation Indicateur","Modélisation en Bonus"])
with tables[0]:
    st.text("Ici se trouve la base brute sans traitement avec les valeurs atipiques")
    st.dataframe(df_brut.toPandas())
    
with tables[1]:
    st.text("Une description des imperfections des données brutes")
    df_desc1=df_brut.describe().toPandas().T
    df_desc1.columns = df_desc1.iloc[0]  
    df_desc1 = df_desc1[1:]
    st.text("Statistiques descriptives de la base")
    st.dataframe(df_desc1)
    
    price_data = [row['price'] for row in df_brut.select('price').collect()]
    fig1 = px.histogram(price_data, nbins=50, color_discrete_sequence=['green'])
    st.plotly_chart(fig1)
    
    df_Nan=df_brut.select([count(when((col(c)=='') | col(c).isNull() |isnan(c), c)).alias(c) for c in df_brut.columns])
    pd_df_na = pd.DataFrame(list(df_Nan.collect()[0].asDict().items()), columns=['Column', 'Missing Values'])

    st.text("Visualisation des Valeurs manquantes")
    fig_na=px.bar(pd_df_na,x="Column",y="Missing Values")
    st.plotly_chart(fig_na)
    
    
    # TRAITEMENT DE LA BASE
    drop_columns_list=["_c21","_c22","_c23","_c24","_c25"]
    df_brut = df_brut.drop(*drop_columns_list)
    df_brut = df_brut.na.drop(how = "all")
    mode_status = df_brut.groupby("status").count().orderBy("count", ascending=False).first()[0]
    df_brut = df_brut.fillna(mode_status, subset=['status'])
    mode_category_name_1 = df_brut.groupby("category_name_1").count().orderBy("count", ascending=False).first()

    mode_category_name_1 = df_brut.groupby("category_name_1").count().orderBy("count", ascending=False).first()[0]
    df_brut = df_brut.fillna(mode_category_name_1, subset=['category_name_1'])
    if mode_category_name_1 is not None and 'category_name_1' in mode_category_name_1:
        mode_category_name_1 = mode_category_name_1['category_name_1'] # Access the value
    else:
        mode_category_name_1 = 'Unknown'  # Or any suitable default value
    df_brut = df_brut.fillna(mode_category_name_1, subset=['category_name_1'])
    
    df_brut = df_brut.na.drop(subset=['Working Date', 'sku', 'Customer ID'])
    
    df_brut = df_brut.withColumn('created_at', F.to_date(F.unix_timestamp('created_at', 'M/d/y').cast('timestamp')))
    df_brut = df_brut.withColumn('Working Date', F.to_date(F.unix_timestamp('Working Date', 'M/d/y').cast('timestamp')))
    df_brut = df_brut.withColumn("qty_ordered", df_brut["qty_ordered"].cast(IntegerType()))
    df_brut = df_brut.withColumn("price", df_brut["price"].cast(IntegerType()))
    df_brut = df_brut.withColumn("grand_total", df_brut["grand_total"].cast(IntegerType()))
    df_brut = df_brut.withColumn("discount_amount", df_brut["discount_amount"].cast(IntegerType()))
    df_brut = df_brut.withColumn("Month", df_brut["Month"].cast(IntegerType()))
    df_brut = df_brut.withColumn("Year", df_brut["Year"].cast(IntegerType()))
    df_brut = df_brut.withColumnRenamed(" MV ", "MV")
    df_brut = df_brut.withColumnRenamed("created_at", "order_date")
    good_df = df_brut.drop_duplicates()
        
#==============================================================
#==============Tableau de Bord Proprement dit==================
#==============================================================
with tables[2]:
    st.text("La base de donnée présentée ici est la Base appurée, elle a été traitée")
    st.dataframe(good_df.toPandas())

with tables[3]:
    st.text("Visualisation et calcul des indicateurs avec la base appurées")
    df_desc2=good_df.describe().toPandas().T
    df_desc2.columns = df_desc2.iloc[0]  
    df_desc2 = df_desc2[1:]
    st.text("Statistique descriptive de la base")
    st.dataframe(df_desc2)
    
    best_category = good_df.groupby("category_name_1").count().sort(col("count").desc()).toPandas()
    col1, col2 =st.columns([7,3])
    
    st.text("Categorie de produit les plus vendus")
    with col1:
        make_bar(best_category,x_val="category_name_1",y_val="count",text_val="count",color=px.colors.qualitative.Pastel1_r)
    with col2:
        st.dataframe(best_category,
                 column_order=("category_name_1", "count"),
                 hide_index=True,
                 width=None,
                 column_config={
                    "category_name_1": st.column_config.TextColumn(
                        "category_name_1",
                    ),
                    "count": st.column_config.ProgressColumn(
                        "count",
                        format="%f",
                        min_value=0,
                        max_value=115874,
                     )})
        
    status_count = good_df.groupby("status").count().sort(col("count").desc()).toPandas()
    
    cl01, cl02 =st.columns([3,7])
    
    st.text("Satut des paiements")
    with cl01:
        st.dataframe(status_count,
                 column_order=("status", "count"),
                 hide_index=True,
                 width=None,
                 column_config={
                    "status": st.column_config.TextColumn(
                        "status",
                    ),
                    "count": st.column_config.ProgressColumn(
                        "count",
                        format="%f",
                        min_value=0,
                        max_value=233695,
                     )})
    with cl02:
        make_bar(status_count,x_val="status",y_val="count",text_val="count",color=px.colors.qualitative.Light24)
    
    
    clm1, clm2, clm3 =st.columns(3)
    df_16= good_df.filter(F.col("Year") == 2016)
    df_16=df_16.groupby("status").count().sort(col("count").desc()).toPandas()
    
    df_17= good_df.filter(F.col("Year") == 2017)
    df_17=df_17.groupby("status").count().sort(col("count").desc()).toPandas()
    
    df_18= good_df.filter(F.col("Year") == 2018)
    df_18=df_18.groupby("status").count().sort(col("count").desc()).toPandas()
    with clm1:
        make_bar(df_16,x_val="status",y_val="count",text_val="count",color=px.colors.qualitative.Antique_r,titre="Répartition des statut en 2016")
    
    with clm2:
        make_bar(df_17,x_val="status",y_val="count",text_val="count",color=px.colors.qualitative.Set1_r,titre="Répartition des statut en 2017")
        
    with clm3:
        make_bar(df_18,x_val="status",y_val="count",text_val="count",color=px.colors.qualitative.Prism,titre="Répartition des statut en 2018")
    
    
    payment_method_count = good_df.groupby('payment_method').count().sort(col("count").desc()).toPandas()
    clp1, clp2=st.columns([6.5,2.5])
    with clp1:
        make_bar(payment_method_count,x_val="payment_method",y_val="count",text_val="count",titre="Classement des mode de payement",color=px.colors.qualitative.Alphabet_r)
    with clp2:
        st.dataframe(payment_method_count,
                 column_order=("payment_method", "count"),
                 hide_index=True,
                 width=None,
                 column_config={
                    "payment_method": st.column_config.TextColumn(
                        "payment_method",
                    ),
                    "count": st.column_config.ProgressColumn(
                        "count",
                        format="%f",
                        min_value=0,
                        max_value=271933,
                     )})
    
    
    
    order_per_month_year = good_df.groupby('M-Y').count().sort(col("count").desc()).toPandas()
    order_per_month_year["M-Y"] = pd.to_datetime(order_per_month_year["M-Y"], format="%m-%Y").dt.strftime('%Y-%m-%d')
    order_per_month_year = order_per_month_year.sort_values(by="M-Y")
    figline=px.line(order_per_month_year,x="M-Y",y="count",title="Évolution des ventes par mois")
    st.plotly_chart(figline)
    
    
    oreder_completed = good_df.filter(F.col('status').isin(['complete','paid', 'received', 'cash_on_delivery']))
    oreder_completed_by_category = oreder_completed.groupby('category_name_1').count()\
    .sort(col("count").desc()).toPandas()
    
    oreder_not_completed = good_df.filter(~F.col('status').isin(['complete','paid', 'received', 'cash_on_delivery']))
    oreder_not_completed_by_category = oreder_not_completed.groupby('category_name_1').count()\
    .sort(col("count").desc()).toPandas()

    clf1, clf2 =st.columns(2)
    
    with clf1:
        make_bar_polar(oreder_completed_by_category,value="count",group="category_name_1",color=px.colors.qualitative.Vivid_r)
    with clf2:
        make_tremap(oreder_not_completed_by_category,value="count",group="category_name_1")
    
    
