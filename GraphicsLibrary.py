# This library include the some of the main plots to represent products reviews
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 18:10:14 2017

@author: espinosa
"""

import pandas as pd
from matplotlib.pyplot import cm
import numpy as np
import matplotlib.pyplot as plt
import squarify # pip install squarify (algorithm for treemap)
from pandas.plotting import parallel_coordinates


########################### Pie chart, where the slices will be ordered and plotted counter-clockwise
# Original

def pieChart(lab,frec,Titulo = 'Pie Chart'):
    #explode = (0, 0.1, 0)
    fig1, ax1 = plt.subplots()
    ax1.set_title(Titulo)
    #ax1.set_ylabel('ylabel')
    ax1.pie(frec,labels=lab, autopct='%1.1f%%',shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()
    
    
def pieChart2(lab,frec,lab2, Titulo = 'Pie Chart'):
    #explode = (0, 0.1, 0)
    labels = lab
    sizes = frec
    #colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    patches, texts, autotexts = plt.pie(sizes,labels=lab2, startangle=90, autopct='%1.1f%%')
    plt.legend(patches, labels, loc="best")
    plt.title(Titulo)
    # Set aspect ratio to be equal so that pie is drawn as a circle.
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    

########################   Bar chart

def barChart(lab,frec, column='reviewerID',ylabel='scores', Title= 'Score by Group'):
    fig3, ax3= plt.subplots()  
    ax3.set_title(Title)
    ax3.set_ylabel(ylabel)
    ax3.bar(range(len(frec)), frec[column].tolist())
    ind = np.arange(len(frec))
    plt.xticks(ind, lab.tolist())
    plt.show()


########################  Several Bar chart horizontal
# @todo compare several products add labels
def barChartVs(inputs,labels, ylabel='scores',Title= 'Score by Group', productNames=[]):
    rects = []
    k = []
    n = len(inputs[0])
    m = len(inputs)
    ax = plt.subplot()
    ax.set_title(Title)
    ax.set_ylabel(ylabel)
    array = np.array(range(n))
    colorRB=cm.rainbow(np.linspace(0,1,m)) # https://matplotlib.org/api/colors_api.html#matplotlib.colors.Normalize
    delta = 1/(m+1)
    plt.xticks(array, labels)
    for i in range(m):
        rects.append(ax.bar(array+delta*i, inputs[i],width=delta,color=colorRB[i],align='center') )
        k.append('P'+str(i))
    if productNames==[]:
        ax.legend((rects),(k))
    else:
        ax.legend((rects), (productNames))
    plt.show()



# @todo comeentar el codigo y mejorar etiquetas graficos e implementar uno nuevo.
# Diagrama de radar
# Grafico en el tiempo con regresion lineal
# TreeMap

def polarChar(summarize, product ,Title= 'Top 4 Tópicos producto: '):
    if Title=='Top 4 Tópicos producto: ':
        Title+= product
    dfPolar= summarize[summarize.asin==product]
    N = len(dfPolar)
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    radii = np.array(dfPolar.summary)
    width = np.pi / 4 * np.array(dfPolar.overall)
    ax = plt.subplot(111, projection='polar')
    ax.set_title(Title)
    if N==4:
        ax.set_xticklabels([dfPolar.maximos[0], '',dfPolar.maximos[1], '', dfPolar.maximos[2], '', dfPolar.maximos[3]])
    bars = ax.bar(theta, radii, width=width, bottom=0.0)
    # Use custom colors and opacity
    for r, bar in zip(radii, bars):
        bar.set_facecolor(plt.cm.viridis(r / 10.))
        bar.set_alpha(0.5)

    plt.show()
    

################### TREE MAP

#libraries


def treeMap(summarize,product, Title ='Tree Map' ):
    # Change color
    dfPolar= summarize[summarize.asin==product]
    colorRB= plt.cm.rainbow(np.linspace(0,1,len(dfPolar)))
    squarify.plot(sizes=list(dfPolar.summary), label=list(dfPolar.levels), color=colorRB, alpha=.4 )
    plt.title(Title +' product :'+product)
    plt.axis('off')
    plt.show()


################### STACKPLOT 

def stackplots(inputs,labels,products, Title = 'Stack plot', ylabel = 'Opiniones'):
    #https://matplotlib.org/gallery/lines_bars_and_markers/stackplot_demo.html#sphx-glr-gallery-lines-bars-and-markers-stackplot-demo-py
    fig, ax = plt.subplots()
    ax.stackplot(labels, inputs, labels=products)
    ax.set_title(Title)
    ax.set_ylabel(ylabel)
    ax.legend(loc=2)
    plt.show()
    

############### Stem Plot

def stemPlot(df,product,topic):
    # Historicos de Reviews por topic
    df=df.sort_values(by='reviewTime')
    if topic:
        Overall = df[(df.asin==product) & (df.maximos==topic)]
    else:
        Overall = df[(df.asin==product)]
    x = range(len(list(Overall.index)))
    y = list(Overall.overall-2.5)
    markerline, stemlines, baseline = plt.stem(x,y, '-.')
    plt.setp(baseline, 'color', 'r', 'linewidth', 2)
    plt.show()
    

############## Time series
def expandReviewTime(df):
    date = (df.reviewTime).str.split(',', expand=True).rename(columns={0:'dia_mes',1:'año'})
    date2 = date.dia_mes.str.split(' ', expand=True).rename(columns={0:'dia',1:'mes'})
    df['año'] = date.año;df['mes'] = date2.mes;df['dia'] = date2.dia
    #https://stackoverflow.com/questions/12680754/split-explode-pandas-dataframe-string-entry-to-separate-rows
    return df


def plotSeries(df, productos, target, modo, topics, xlabel = 'Años', ylabel = 'Magnitud',titulo='Titulo'):
    df2 = expandReviewTime(df)
    df3 = df2.groupby([modo,'año'],as_index=False,sort=True).agg({'overall':'mean','summary':'count'})
    if productos:
        for producto in productos:
            dfProducto =df3[df3.asin==producto]
            ts = pd.Series(dfProducto[target].values,dfProducto['año'].values)
            ts.plot(legend=True,label=producto, title ="Serie Anual por producto: "+target+" "+titulo, fontsize = 10,x= 'año')
    else:
        for topic in topics:
            df4= df3[df3.maximos==topic]
            ts = pd.Series(df4[target].values,df4['año'].values)
            ts.plot(legend=True,label="Topic"+str(topic), title ="Serie Anual por topic: "+target+" "+titulo, fontsize = 10, x = 'año')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()


############## Paralllel plot
def parallelPlot(df,yearI,yearF,target = 'Años',xlabel = 'Topics', ylabel = 'Valoración promedio', title = 'Valoración por año'):
    data = expandReviewTime(df)
    dfVideoGamesSummary1=pd.pivot_table(data, values='overall', index=['maximos'], columns = ['año'], aggfunc=np.average)
    dfVideoGamesSummary1 = dfVideoGamesSummary1.transpose()
    dfVideoGamesSummary1[target] = dfVideoGamesSummary1.index
    Years = [" "+str(i) for i in range(yearI,yearF)]#; Years.append('Years')
    dfVideoGamesSummary1 = dfVideoGamesSummary1.drop(Years)
    #dfVideoGamesSummary2 = dfVideoGamesSummary1[Years]
    plt.figure()
    plt.ylabel(ylabel);plt.xlabel(xlabel);plt.title(title)
    parallel_coordinates(dfVideoGamesSummary1, target, colormap='gist_rainbow')
    plt.show()

################## BAR CHAR 2 OPINION OBSERVER

def barChartVs2(inputs,labels, ylabel='scores',Title= 'Score by Group' ):
    inputsPos=inputs[0:int(len(inputs)/2)]
    inputsNeg=inputs[int(len(inputs)/2):len(inputs)]
    rects = []
    j = []
    n = len(inputsPos[0])
    m = len(inputsPos)
    ax = plt.subplot()
    ax.set_title(Title)
    ax.set_ylabel(ylabel)
    array = np.array(range(n))
    colorRB=cm.rainbow(np.linspace(0,1,m)) # https://matplotlib.org/api/colors_api.html#matplotlib.colors.Normalize
    delta = 1/(m+1)
    plt.xticks(array, labels)
    for i in range(m):
        rects.append(ax.bar(array+delta*i, inputsNeg[i],width=delta,color=colorRB[i],align='center') )
    for i in range(m):
        rects.append(ax.bar(array+delta*i, inputsPos[i],width=delta,color=colorRB[i],align='center') )
        j.append('P'+str(i))
    ax.legend((rects),(j))
    plt.show()
    
def ProductsToArray(listaProductos,dfp,dfn):
    arrayProductos = []
    for df in [dfp,dfn]:
        for i in listaProductos:
            r = df[df.asin==i][['maximos','overall']]
            if r.empty:
                print('empty')
            else:
                arrayProductos.append(list(r.overall))
        indices = r.maximos
    return np.array(arrayProductos), indices
# todo: imputar valores con topics faltantes para completar la secuencia 

############# BAR CHAR3 Relacion competitiva
# @todo comparar overall*summary
def barChartVs3(inputs,labels, ylabel='scores',Title= 'Score by Group' ):
    inputsPos=inputs[0]
    inputsNeg=inputs[1]
    rects = []
    j = []
    n = len(labels)
    m = 2
    ax = plt.subplot()
    ax.set_title(Title)
    ax.set_ylabel(ylabel)
    array = np.array(range(n))
    colorRB=cm.rainbow(np.linspace(0,1,m)) # https://matplotlib.org/api/colors_api.html#matplotlib.colors.Normalize
    plt.xticks(array, labels)
    rects.append(ax.bar(array, inputsNeg,color=colorRB[0],align='center') )
    rects.append(ax.bar(array, inputsPos,color=colorRB[1],align='center') )
    j.append('P'+str(0))
    j.append('P'+str(1))
    ax.legend((rects),(j))
    plt.show()



def horizontalBar(inputs,labels):
    inputsPos=inputs[0]
    inputsNeg=inputs[1]
    fig, ax = plt.subplots()
    y_pos = np.arange(len(labels))
    rects = []
    rects.append(ax.barh(y_pos, inputsPos, align='center',
            color='green', ecolor='black'))
    rects.append(ax.barh(y_pos, inputsNeg, align='center',
            color='red', ecolor='black'))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Valoración')
    ax.set_ylabel('Tópicos')
    ax.set_title('P1 Vs P2')
    ax.legend((rects),(['P1','P2']))
    plt.show()
    
    

    
'''
############## Paralllel plot
def parallelPlot(df,yearI,yearF,target = 'Topics',xlabel = 'Años', ylabel = 'Valoración promedio', title = 'Valoración por año'):
    data = expandReviewTime(df)
    dfVideoGamesSummary1=pd.pivot_table(data, values='overall', index=['maximos'], columns = ['año'], aggfunc=np.average)
    #dfVideoGamesSummary1 = dfVideoGamesSummary1.transpose()
    dfVideoGamesSummary1[target] = dfVideoGamesSummary1.index
    Years = [" "+str(i) for i in range(yearI,yearF)]; Years.append('Topics')
    dfVideoGamesSummary2 = dfVideoGamesSummary1[Years]
    plt.figure()
    plt.ylabel(ylabel);plt.xlabel(xlabel);plt.title(title)
    parallel_coordinates(dfVideoGamesSummary2, 'Topics', colormap='gist_rainbow')
    plt.show()
'''

if __name__ == "__main__":
    # Read dataFrame
    path = "/home/espinosa/TFM_Project/reviews_Video_Games_5.json.gz"



