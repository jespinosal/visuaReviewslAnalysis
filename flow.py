# enviroment py36



from GraphicsLibrary import *
from PreprocessingLibrary import *

import pandas as pd
import gzip
import graficosv1
from matplotlib.pyplot import cm
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates




path = "/home/espinosa/TFM_Project/reviews_Video_Games_5.json.gz"
# Read data
df = getDF(path)
# ScoreSummaryVG = pd.read_pickle('/home/espinosa/TFM_Project/ScoreSummaryVG.pkl')
dfVideoGames = pd.read_pickle('/home/espinosa/TFM_Project/dfVideoGames.pkl')
# Summarize & prepare the data
d00_1 = Preprocesamiento()
d00_1.summarizeProductTopics(dfVideoGames)
d00_1.addScore(5, 'overall')
d00_1.concatenateCharacter('T', 'maximos' )

# Bucketizer overviews by product in labels
d01_5 = Preprocesamiento()
d01_5.discretizacion('overall','0700099867',df,5)
d02_3 = Preprocesamiento()
d02_3.discretizacion('overall','0700099867',df,3)

# Bucketizer several products overviews in  a array
d03_array_3_5 = Preprocesamiento()
d03_array_3_5.discretizacionArray(['B000006P0K','6050036071','0700099867'],df,5)


######################## PIECHAR DEMO ###########################
# Unique product overall
pieChart(d01_5.frec.index,d01_5.frec,d01_5.titulo)
pieChart(d02_3.frec.index,d02_3.frec,d02_3.titulo)
# Unique product topics
pieChart(d00_1.dfSummarized[d00_1.dfSummarized.asin=='0700099867'].maximos.values,d00_1.dfSummarized[d00_1.dfSummarized.asin=='0700099867'].summary, 'Tópicos producto 0700099867 número de opiniones')
pieChart(d00_1.dfSummarized[d00_1.dfSummarized.asin=='0700099867'].maximos.values,d00_1.dfSummarized[d00_1.dfSummarized.asin=='0700099867'].overall, 'Tópicos producto 0700099867 promedio de valoración')
# Unique product topics Vs levels by summary
pieChart2(d00_1.dfSummarized[d00_1.dfSummarized.asin=='0700099867'].maximos.values,d00_1.dfSummarized[d00_1.dfSummarized.asin=='0700099867'].summary,d00_1.dfSummarized[d00_1.dfSummarized.asin=='0700099867'].levels, 'Tópicos y valoración producto 0700099867 ')
pieChart2(d00_1.dfSummarized[d00_1.dfSummarized.asin=='B000006P0K'].maximos.values,d00_1.dfSummarized[d00_1.dfSummarized.asin=='B000006P0K'].summary,d00_1.dfSummarized[d00_1.dfSummarized.asin=='B000006P0K'].levels, 'Tópicos y valoración producto B000006P0K ')


######################## BARCHAR DEMO ###########################
barChart(d01_5.frec.index.categories,d01_5.frec, column='asin', ylabel= 'Valoración', Title='Producto:'+d01_5.configuracion['filtro'])
barChart(d02_3.frec.index.categories,d02_3.frec, column='asin',ylabel= 'Valoración', Title='Producto:'+d02_3.configuracion['filtro'] )
# Topics Vs OVerall unique product
barChart(d00_1.dfSummarized[d00_1.dfSummarized.asin=='0700099867'].maximos.values,d00_1.dfSummarized[d00_1.dfSummarized.asin=='0700099867'],'overall','Valoración','Valoración por tópico', )
barChart(d00_1.dfSummarized[d00_1.dfSummarized.asin=='0700099867'].maximos.values,d00_1.dfSummarized[d00_1.dfSummarized.asin=='0700099867'],'summary', 'Opiniones','Opiniones por tópico')
# Overall several products
barChartVs(d03_array_3_5.inputs, d03_array_3_5.labels, ylabel='Opiniones',Title= 'Comparación productos: '+str(d03_array_3_5.configuracion['listaProductos']), productNames = d03_array_3_5.configuracion['listaProductos'])
# Topics Vs Overall several products
topicsByClient =  fillEmptyTopics(d00_1.dfSummarized,nTopics=10)
summarize2 = pd.merge(topicsByClient,d00_1.dfSummarized,on=['asin','maximos'],how='left')
summarize2['overall'] = summarize2['overall'].fillna(0)
summarize2['summary']= summarize2['summary'].fillna(0)
d04_array_5 = Preprocesamiento()
d04_array_5.discretizacionArray(['B000006P0K','6050036071','0700099867'],df = summarize2, mode='overall',n =5)
barChartVs(d04_array_5.inputs, d04_array_5.labels,  ylabel='Valoraciones',Title= 'Comparación productos: '+str(d03_array_3_5.configuracion['listaProductos']), productNames = d03_array_3_5.configuracion['listaProductos'])


##################### POLARCHAR #####################
polarChar(d00_1.dfSummarized,product='0700099867', Title= 'Score by Topic')

################### TREE MAP #####################
treeMap(d00_1.dfSummarized,'0700099867')

################### STACKPLOT
products = ['B000006P0K','6050036071','0700099867']
d03_array_3_5.discretizacionArray(products,df = summarize2, mode='summary',n =5) # over summarize2 to get impute data
stackplots(d03_array_3_5.inputs,d03_array_3_5.labels,products, Title = 'Tópicos por producto :'+ str(products), ylabel = 'Opiniones')



############## Time series
# Example by products user evaluation

plotSeries(df= dfVideoGames, productos = ["B00000DMAQ","B00002CF96"],target ='overall',modo = "asin", topics=[], ylabel= 'Valoración Promedio',xlabel = 'Años')
plotSeries(df= dfVideoGames, productos = ["B00000DMAQ","B00002CF96"],target ='summary',modo = "asin", topics=[], ylabel= 'Opiniones',xlabel = 'Años')
#plot.show()

# Excample by topics
plotSeries(df= dfVideoGames, productos = [],target ='overall',modo = "maximos", topics=[0,1,2,3], ylabel= 'Valoración Promedio',xlabel = 'Años')
plotSeries(df= dfVideoGames, productos = [],target ='summary',modo = "maximos", topics=[0,1,2,3], ylabel= 'Opiniones',xlabel = 'Años')
#plt.show()

#@todo re asrignar atibutos de libreria graficos para que lean el objeto y no atributos
#@todo git y youtube
############## Paralllel plot
parallelPlot(dfVideoGames,2007,2014,target = 'Años',xlabel = 'Tópicos', ylabel = 'Valoración promedio', title = 'Valoración por año')


################## BAR CHAR 2 OPINION OBSERVER
dfSummary0Pos0, dfSummary0Neg0 = polaritySummary(dfVideoGames)
products = ['B000006P0K','6050036071','0700099867']
inputs,labels=ProductsToArray(products, dfSummary0Pos0, dfSummary0Neg0)
barChartVs2(inputs,labels, ylabel='Valoración',Title= 'Opiniones por producto' )



############# BAR CHAR3 (Relacion competitiva)
# @todo comparar overall*summary
products = ['B000006P0K','6050036071']
dfSummay0Filter0, dfSummay0Filter1 = compititiveSummary(dfVideoGames,products)
inputs,labels=ProductsToArray(products, dfSummay0Filter0, dfSummay0Filter1)
horizontalBar(inputs,labels)
barChartVs3(inputs,labels, ylabel='Valoración',Title= 'Opiniones por producto')
