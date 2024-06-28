# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 18:57:25 2024

@author: torug
"""

#%% Instalando os pacotes

!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install seaborn
!pip install plotly
!pip install scipy
!pip install scikit-learn
!pip install pingouin

#%% Importando os pacotes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import scipy.stats as stats
from scipy.stats import zscore
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pingouin as pg
import plotly.express as px 
import plotly.io as pio
pio.renderers.default='browser'

#%%

dados_cartao = pd.read_csv('cartao_credito.csv')

dados_cartao.info()

cartao_cluster = dados_cartao.drop(columns=['Sl_No', 'Customer Key'])

tab_describe = cartao_cluster.describe()

cartao_pad = cartao_cluster.apply(zscore, ddof=1)

#%%

#Descobrindo números de Cluster

elbow = []
K = range(1,11) 
for k in K:
    kmeanElbow = KMeans(n_clusters=k, init='random', random_state=100).fit(cartao_pad)
    elbow.append(kmeanElbow.inertia_)

    
plt.figure(figsize=(16,8))
plt.plot(K, elbow, marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.xticks(range(1,11))
plt.ylabel('WCSS', fontsize=16)
plt.title('Método de Elbow', fontsize=16)
plt.show()

#%%

#Aplicando Kmeans

cartao_kmeans = KMeans(n_clusters=3,init='random',random_state=100).fit(cartao_pad)

kmeans_cluters = cartao_kmeans.labels_
cartao_cluster['cluster_kmeans'] = kmeans_cluters
cartao_pad['cluster_kmeans'] = kmeans_cluters
cartao_cluster['cluster_kmeans'] = cartao_cluster['cluster_kmeans'].astype('category')
cartao_pad['cluster_kmeans'] = cartao_pad['cluster_kmeans'].astype('category')

#%%

#Gerando os anovas

# Avg_Credit_Limit
pg.anova(dv='Avg_Credit_Limit', 
         between='cluster_kmeans', 
         data=cartao_pad,
         detailed=True).T

# Total_Credit_Cards
pg.anova(dv='Total_Credit_Cards', 
         between='cluster_kmeans', 
         data=cartao_pad,
         detailed=True).T

# Total_visits_bank
pg.anova(dv='Total_visits_bank', 
         between='cluster_kmeans', 
         data=cartao_pad,
         detailed=True).T

# Total_visits_online
pg.anova(dv='Total_visits_online', 
         between='cluster_kmeans', 
         data=cartao_pad,
         detailed=True).T

# Total_calls_made
pg.anova(dv='Total_calls_made', 
         between='cluster_kmeans', 
         data=cartao_pad,
         detailed=True).T

#%%

#Gerando Gráfico
cartao_cluster.info()
fig = px.scatter_3d(cartao_cluster,
                    x='Avg_Credit_Limit',
                    y='Total_Credit_Cards',
                    z='Total_visits_online',
                    color='cluster_kmeans')
fig.show()

#%%

#Entendendo as caracteristicas dos cluster

cartao_grupo = cartao_cluster.groupby(by='cluster_kmeans')

tab_desc_grupo = cartao_grupo.describe().T