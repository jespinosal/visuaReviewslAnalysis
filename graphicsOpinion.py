# This library include the some of the main plots to represent products reviews
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 18:10:14 2017

@author: espinosa
"""

import pandas as pd
from matplotlib.pyplot import cm
import numpy as np
import matplotlib.pyplot as plt
import squarify  # pip install squarify (algorithm for treemap)
import pandas as pd
import gzip
import numpy as np
from pandas.plotting import parallel_coordinates


class Preprocesamiento:

    def summarizeProductTopics(self,dfTopics, product=False):
        dfSummarized = dfTopics.groupby(['asin', 'maximos'], as_index=False, sort=True).agg(
            {'overall': 'mean', 'summary': 'count'})
        if product:
            dfSummarized = dfSummarized.where(dfSummarized.asin == product).dropna()
        self.configuracion = {'product':product,}
        self.dfSummarized = dfSummarized

    def addScore(self, levels, objetivo):
        if levels == 3:
            levels = pd.cut( self.dfSummarized[objetivo], 3, labels=["Negativo", "Neutral", "Positivo"])
        elif levels == 5:
            levels = pd.cut( self.dfSummarized[objetivo], 5, labels=["Muy Malo", "Malo", "Regular", "Bueno", "Exelente"])
        self.dfSummarized['levels'] = levels


    def concatenateCharacter(self,character, column ):
        self.dfSummarized[column] = 'T' + self.dfSummarized[column].astype(str)



    def discretizacion(self,objetivo, filtro, df, niveles):
        '''Clase que discretiza la variable objetivo de df en niveles
        input:df = dataframe de referencia
        objetivo = vatiable de df sobre la que se hace el analisis
        niveles = 3 ("Negativo", "Neutral", "Positivo") ó 5 ("Muy Malo", "Malo", "Regular", "Bueno", "Exelente"), para otro valor se devulven las etiquetas originales de la variable filtrada
        outout = configuracion: resumen de parametros de entrada, titulo: titulo del gráfico, frec: tabla de frecuencias por etiqueta'''
        self.configuracion = {'objetivo': objetivo, 'filtro': filtro, 'niveles': niveles}
        self.titulo = "Opiniones asociadas al producto: " + filtro

        if niveles == 3:
            '''Esta funcion discretiza los valores de la variable objetivo en n niveles y extrae las etiquetas y frecuencias de una variable dado un data frame'''
            levels = pd.cut(df[objetivo], 3, labels=["Negativo", "Neutral", "Positivo"])
            df['levels'] = levels
            dfSubset = df
            if filtro:
                dfSubset = df[df.asin == filtro]
            frec = dfSubset.groupby('levels').count()[['asin']]
            # lab = frec.index
        elif niveles == 5:
            levels = pd.cut(df[objetivo], 5, labels=["Muy Malo", "Malo", "Regular", "Bueno", "Exelente"])
            df['levels'] = levels
            dfSubset = df
            if filtro:
                dfSubset = df[df.asin == filtro]
            frec = dfSubset.groupby('levels').count()[['asin']]
            # lab = frec.index
        else:
            dfSubset = df
            if filtro:
                dfSubset = df[df.asin == filtro]
            frec = dfSubset.groupby(objetivo).count()[['asin']]
            # lab = frec.index
        return frec

    def discretizacionArray(self,listaProductos, df, n=5, mode='levels'):
        '''Esta función agrupa elementos de df según mode'''
        arrayProductos = []
        if mode == 'levels':
            for i in listaProductos:
                r = self.discretizacion('overall', i, df, n)
                arrayProductos.append(list(r.asin))
            indices = r.index.categories
        elif mode == 'overall':
            for i in listaProductos:
                r = df[df.asin == i][['maximos', 'overall']]
                arrayProductos.append(list(r.overall))
            indices = list(r.maximos)
        else:
            for i in listaProductos:
                r = df[df.asin == i][['maximos', 'summary']]
                arrayProductos.append(list(r.summary))
            indices = list(r.maximos)

        self.configuracion  = {'listaProductos': listaProductos, 'n': n, 'levels':mode}
        self.inputs = np.array(arrayProductos)
        self.labels = indices

    def fillEmptyTopics(self,summarize, nTopics=10):
        maximos = ['T' + str(i) for i in range(0, nTopics)]
        # maximos = pd.DataFrame({'maximos':maximos})
        asin = list(pd.unique(summarize.asin))
        return pd.DataFrame([(x, y) for x in asin for y in maximos], columns=['asin', 'maximos'])

    def summarize(self):
        topicsByClient = self.fillEmptyTopics(self.dfSummarized, nTopics=10)
        self.summarize2 = pd.merge(topicsByClient, self.dfSummarized, on=['asin', 'maximos'], how='left')
        self.summarize2['overall'] = self.summarize2['overall'].fillna(0)
        self.summarize2['summary'] = self.summarize2['summary'].fillna(0)

    def polaritySummary(self,dfVideoGames, n=10):
        dfVideoGamesNorm = dfVideoGames.copy()
        dfVideoGamesNorm.maximos = 'T' + dfVideoGamesNorm.maximos.astype(str)
        dfVideoGamesNorm['overall'] = dfVideoGamesNorm['overall'] - 2.5
        dfVideoGamesNorm['polarity'] = np.where(dfVideoGamesNorm['overall'] > 0, 'p', 'n')
        dfSummary0 = dfVideoGamesNorm.groupby(['asin', 'polarity', 'maximos'], as_index=False, sort=True).agg(
            {'overall': 'mean', 'summary': 'count'})
        dfSummary0Pos = dfSummary0[
            dfSummary0.polarity == 'p']  # ; PosList = dfSummary0Pos.summary*dfSummary0Pos.overall
        dfSummary0Neg = dfSummary0[
            dfSummary0.polarity == 'n']  # ; NegList = dfSummary0Neg.summary*dfSummary0Neg.overall
        dfSummary0Pos['overall'] = dfSummary0Pos.summary * dfSummary0Pos.overall
        dfSummary0Neg['overall'] = dfSummary0Neg.summary * dfSummary0Neg.overall

        topicsByClient = self.fillEmptyTopics(dfVideoGames, nTopics=n)

        dfSummary0Pos0 = pd.merge(topicsByClient, dfSummary0Pos, on=['asin', 'maximos'], how='left')
        dfSummary0Pos0['overall'] = dfSummary0Pos0['overall'].fillna(0)
        dfSummary0Pos0['summary'] = dfSummary0Pos0['summary'].fillna(0)
        dfSummary0Pos0['polarity'] = dfSummary0Pos0['polarity'].fillna('p')

        dfSummary0Neg0 = pd.merge(topicsByClient, dfSummary0Neg, on=['asin', 'maximos'], how='left')
        dfSummary0Neg0['overall'] = dfSummary0Neg0['overall'].fillna(0)
        dfSummary0Neg0['summary'] = dfSummary0Neg0['summary'].fillna(0)
        dfSummary0Neg0['polarity'] = dfSummary0Neg0['polarity'].fillna('n')
        self.dfSummary0Pos0=dfSummary0Pos0
        self.dfSummary0Neg0=dfSummary0Neg0

    def compititiveSummary(self,dfVideoGames, products):
        dfVideoGamesNorm = dfVideoGames.copy()
        dfVideoGamesNorm.maximos = 'T' + dfVideoGamesNorm.maximos.astype(str)
        dfVideoGamesNorm['overall'] = dfVideoGamesNorm['overall']
        dfSummary0 = dfVideoGamesNorm.groupby(['asin', 'maximos'], as_index=False, sort=True).agg(
            {'overall': 'mean', 'summary': 'count'})
        topicsByClient = self.fillEmptyTopics(dfVideoGames, nTopics=10)
        dfSummary0 = pd.merge(topicsByClient, dfSummary0, on=['asin', 'maximos'], how='left')
        dfSummary0['overall'] = dfSummary0['overall'].fillna(0)
        dfSummary0['summary'] = dfSummary0['summary'].fillna(0)

        dfSummay0Filter0 = dfSummary0.where(dfSummary0.asin == products[0]).dropna()
        dfSummay0Filter1 = dfSummary0.where(dfSummary0.asin == products[1]).dropna();
        dfSummay0Filter1['overall'] = dfSummay0Filter1['overall'] * -1
        dfSummay0Filter = dfSummay0Filter0.append(pd.DataFrame(data=dfSummay0Filter1))

        self.dfSummary0Pos0 = dfSummay0Filter0
        self.dfSummary0Neg0 = dfSummay0Filter1

    def productsToArray(self,listaProductos):
        arrayProductos = []
        for df in [self.dfSummary0Pos0, self.dfSummary0Neg0]:
            for i in listaProductos:
                r = df[df.asin == i][['maximos', 'overall']]
                if r.empty:
                    print('empty')
                else:
                    arrayProductos.append(list(r.overall))
            indices = r.maximos
        self.inputs = np.array(arrayProductos)
        self.labels = indices



class graphicsSummary(Preprocesamiento):

    def getFrecLab(self,product):
        self.lab = self.dfSummarized[self.dfSummarized.asin == product].maximos.values
        self.frec = self.dfSummarized[self.dfSummarized.asin == product].summary
        self.lab2 = self.dfSummarized[self.dfSummarized.asin==product].levels
        self.frec2 = self.dfSummarized[self.dfSummarized.asin == product]

    def pieChart(self,Titulo='Pie Chart'):
        # explode = (0, 0.1, 0)
        fig1, ax1 = plt.subplots()
        ax1.set_title(Titulo)
        # ax1.set_ylabel('ylabel')
        ax1.pie(self.frec, labels=self.lab, autopct='%1.1f%%', shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.show()

    def pieChart2(self, Titulo='Pie Chart'):
        # explode = (0, 0.1, 0)
        labels = self.lab
        sizes = self.frec
        # colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
        patches, texts, autotexts = plt.pie(sizes, labels=self.lab2, startangle=90, autopct='%1.1f%%')
        plt.legend(patches, labels, loc="best")
        plt.title(Titulo)
        # Set aspect ratio to be equal so that pie is drawn as a circle.
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    def barChart(self,column='reviewerID', ylabel='scores', Title='Score by Group'):
        fig3, ax3 = plt.subplots()
        ax3.set_title(Title)
        ax3.set_ylabel(ylabel)
        ax3.bar(range(len(self.frec2)), self.frec2[column].tolist())
        ind = np.arange(len(self.frec2))
        plt.xticks(ind, self.lab.tolist())
        plt.show()



class graphicsCompare(Preprocesamiento):

    ########################  Several Bar chart horizontal
    # @todo compare several products add labels
    def barChartVs(self,ylabel='scores', Title='Score by Group'):

        productNames = self.configuracion['listaProductos']
        rects = []
        k = []
        n = len(self.inputs[0])
        m = len(self.inputs)
        ax = plt.subplot()
        ax.set_title(Title)
        ax.set_ylabel(ylabel)
        array = np.array(range(n))
        colorRB = cm.rainbow(
            np.linspace(0, 1, m))  # https://matplotlib.org/api/colors_api.html#matplotlib.colors.Normalize
        delta = 1 / (m + 1)
        plt.xticks(array, self.labels)
        for i in range(m):
            rects.append(ax.bar(array + delta * i, self.inputs[i], width=delta, color=colorRB[i], align='center'))
            k.append('P' + str(i))
        if productNames == []:
            ax.legend((rects), (k))
        else:
            ax.legend((rects), (productNames))
        plt.show()

    ################### STACKPLOT

    def stackplots(self,Title='Stack plot', ylabel='Opiniones'):
        # https://matplotlib.org/gallery/lines_bars_and_markers/stackplot_demo.html#sphx-glr-gallery-lines-bars-and-markers-stackplot-demo-py
        products = self.configuracion['listaProductos']
        fig, ax = plt.subplots()
        ax.stackplot(self.labels, self.inputs, labels=products)
        ax.set_title(Title)
        ax.set_ylabel(ylabel)
        ax.legend(loc=2)
        plt.show()

    ################## BAR CHAR 2 OPINION OBSERVER

    def barChartVs2(self, ylabel='scores', Title='Score by Group'):
        inputsPos = self.inputs[0:int(len(self.inputs) / 2)]
        inputsNeg = self.inputs[int(len(self.inputs) / 2):len(self.inputs)]
        rects = []
        j = []
        n = len(inputsPos[0])
        m = len(inputsPos)
        ax = plt.subplot()
        ax.set_title(Title)
        ax.set_ylabel(ylabel)
        array = np.array(range(n))
        colorRB = cm.rainbow(
            np.linspace(0, 1, m))  # https://matplotlib.org/api/colors_api.html#matplotlib.colors.Normalize
        delta = 1 / (m + 1)
        plt.xticks(array, self.labels)
        for i in range(m):
            rects.append(ax.bar(array + delta * i, inputsNeg[i], width=delta, color=colorRB[i], align='center'))
        for i in range(m):
            rects.append(ax.bar(array + delta * i, inputsPos[i], width=delta, color=colorRB[i], align='center'))
            j.append('P' + str(i))
        ax.legend((rects), (j))
        plt.show()

    ############# BAR CHAR3 Relacion competitiva
    def barChartVs3(self, ylabel='scores', Title='Score by Group'):
        inputsPos = self.inputs[0]
        inputsNeg = self.inputs[1]
        rects = []
        j = []
        n = len(self.labels)
        m = 2
        ax = plt.subplot()
        ax.set_title(Title)
        ax.set_ylabel(ylabel)
        array = np.array(range(n))
        colorRB = cm.rainbow(
            np.linspace(0, 1, m))  # https://matplotlib.org/api/colors_api.html#matplotlib.colors.Normalize
        plt.xticks(array, self.labels)
        rects.append(ax.bar(array, inputsNeg, color=colorRB[0], align='center'))
        rects.append(ax.bar(array, inputsPos, color=colorRB[1], align='center'))
        j.append('P' + str(0))
        j.append('P' + str(1))
        ax.legend((rects), (j))
        plt.show()

    def horizontalBar(self,title='P1 Vs P2'):
        inputsPos = self.inputs[0]
        inputsNeg = self.inputs[1]
        fig, ax = plt.subplots()
        y_pos = np.arange(len(self.labels))
        rects = []
        rects.append(ax.barh(y_pos, inputsPos, align='center',
                             color='green', ecolor='black'))
        rects.append(ax.barh(y_pos, inputsNeg, align='center',
                             color='red', ecolor='black'))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(self.labels)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Valoración')
        ax.set_ylabel('Tópicos')
        ax.set_title(title)
        ax.legend((rects), (['P1', 'P2']))
        plt.show()




class graphicsTime:

    def __init__(self,df):
        self.df = df


    def stemPlot(df, product, topic):
        # Historicos de Reviews por topic
        df = df.sort_values(by='reviewTime')
        if topic:
            Overall = df[(df.asin == product) & (df.maximos == topic)]
        else:
            Overall = df[(df.asin == product)]
        x = range(len(list(Overall.index)))
        y = list(Overall.overall - 2.5)
        markerline, stemlines, baseline = plt.stem(x, y, '-.')
        plt.setp(baseline, 'color', 'r', 'linewidth', 2)
        plt.show()

    ############## Time series
    def expandReviewTime(self,df):
        date = (df.reviewTime).str.split(',', expand=True).rename(columns={0: 'dia_mes', 1: 'año'})
        date2 = date.dia_mes.str.split(' ', expand=True).rename(columns={0: 'dia', 1: 'mes'})
        df['año'] = date.año;
        df['mes'] = date2.mes;
        df['dia'] = date2.dia
        # https://stackoverflow.com/questions/12680754/split-explode-pandas-dataframe-string-entry-to-separate-rows
        return df

    def plotSeries(self, productos, target, modo, topics, xlabel='Años', ylabel='Magnitud', titulo='Titulo'):
        df2 = self.expandReviewTime(self.df)
        df3 = df2.groupby([modo, 'año'], as_index=False, sort=True).agg({'overall': 'mean', 'summary': 'count'})
        if productos:
            for producto in productos:
                dfProducto = df3[df3.asin == producto]
                ts = pd.Series(dfProducto[target].values, dfProducto['año'].values)
                ts.plot(legend=True, label=producto, title="Serie Anual por producto: " + target + " " + titulo,
                        fontsize=10, x='año')
        else:
            for topic in topics:
                df4 = df3[df3.maximos == topic]
                ts = pd.Series(df4[target].values, df4['año'].values)
                ts.plot(legend=True, label="Topic" + str(topic),
                        title="Serie Anual por topic: " + target + " " + titulo, fontsize=10, x='año')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.show()

    ############## Paralllel plot
    def parallelPlot(self,yearI, yearF, target='Años', xlabel='Topics', ylabel='Valoración promedio',
                     title='Valoración por año'):
        data = self.expandReviewTime(self.df)
        dfVideoGamesSummary1 = pd.pivot_table(data, values='overall', index=['maximos'], columns=['año'],
                                              aggfunc=np.average)
        dfVideoGamesSummary1 = dfVideoGamesSummary1.transpose()
        dfVideoGamesSummary1[target] = dfVideoGamesSummary1.index
        Years = [" " + str(i) for i in range(yearI, yearF)]  # ; Years.append('Years')
        dfVideoGamesSummary1 = dfVideoGamesSummary1.drop(Years)
        # dfVideoGamesSummary2 = dfVideoGamesSummary1[Years]
        plt.figure()
        plt.ylabel(ylabel);
        plt.xlabel(xlabel);
        plt.title(title)
        parallel_coordinates(dfVideoGamesSummary1, target, colormap='gist_rainbow')
        plt.show()




class graphicsGeneralSummary(Preprocesamiento):

    def polarChar(self, product, Title='Top 4 Tópicos producto: '):
        if Title == 'Top 4 Tópicos producto: ':
            Title += product
        dfPolar =  self.dfSummarized[ self.dfSummarized.asin == product]
        N = len(dfPolar)
        theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
        radii = np.array(dfPolar.summary)
        width = np.pi / 4 * np.array(dfPolar.overall)
        ax = plt.subplot(111, projection='polar')
        ax.set_title(Title)
        if N == 4:
            ax.set_xticklabels(
                [dfPolar.maximos[0], '', dfPolar.maximos[1], '', dfPolar.maximos[2], '', dfPolar.maximos[3]])
        bars = ax.bar(theta, radii, width=width, bottom=0.0)
        # Use custom colors and opacity
        for r, bar in zip(radii, bars):
            bar.set_facecolor(plt.cm.viridis(r / 10.))
            bar.set_alpha(0.5)

        plt.show()

    def treeMap(self,product, Title='Tree Map'):
        # Change color
        dfPolar = self.dfSummarized[self.dfSummarized.asin == product]
        colorRB = plt.cm.rainbow(np.linspace(0, 1, len(dfPolar)))
        squarify.plot(sizes=list(dfPolar.summary), label=list(dfPolar.levels), color=colorRB, alpha=.4)
        plt.title(Title + ' product :' + product)
        plt.axis('off')
        plt.show()



def parse(path0):
    g = gzip.open(path0, 'rb')
    for l in g:
        yield eval(l)

def getDF(path1):
    i = 0
    df = {}
    for d in parse(path1):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')



if __name__ == "__main__":


    path = "/home/espinosa/TFM_Project/reviews_Video_Games_5.json.gz"
    df = getDF(path)
    dfVideoGames = pd.read_pickle('/home/espinosa/TFM_Project/dfVideoGames.pkl')


    ##########################################
    graficosResumen = graphicsSummary()
    graficosResumen.summarizeProductTopics(dfVideoGames)
    graficosResumen.addScore(5, 'overall')
    graficosResumen.concatenateCharacter('T', 'maximos' )
    graficosResumen.getFrecLab('0700099867')
    graficosResumen.pieChart(Titulo='Pie Chart')
    graficosResumen.pieChart2(Titulo='Pie Chart')
    graficosResumen.barChart('overall','Valoración','Valoración por tópico')
    ###########################################
    products = ['B000006P0K','6050036071','0700099867']
    graficosComparacion = graphicsCompare()
    graficosComparacion.discretizacionArray(['B000006P0K','6050036071','0700099867'],df,5)
    graficosComparacion.barChartVs(ylabel='Opiniones',Title= 'Comparación productos')
    ############################################
    graficosComparacion = graphicsCompare()
    graficosComparacion.summarizeProductTopics(dfVideoGames)
    graficosComparacion.addScore(5, 'overall')
    graficosComparacion.concatenateCharacter('T', 'maximos' )
    graficosComparacion.summarize()
    graficosComparacion.discretizacionArray(['B000006P0K','6050036071','0700099867'],graficosComparacion.summarize2,5,mode='summary')
    graficosComparacion.stackplots(Title='Stack plot', ylabel='Opiniones')
    ############################################
    products = ['B000006P0K','6050036071','0700099867']
    graficosComparacion = graphicsCompare()
    graficosComparacion.polaritySummary(dfVideoGames)
    graficosComparacion.productsToArray(products)
    graficosComparacion.barChartVs2(ylabel='Valoración',Title= 'Opiniones por producto')
    ############################################
    products = ['B000006P0K','6050036071']
    graficosComparacion = graphicsCompare()
    graficosComparacion.compititiveSummary(dfVideoGames,products)
    graficosComparacion.productsToArray(products)
    graficosComparacion.horizontalBar('Producto 1 Vs Producto 2')
    graficosComparacion.barChartVs3(ylabel='Valoración',Title= 'Opiniones por producto')
    #############################################
    # Example Topics in time
    graficosTiempo= graphicsTime(dfVideoGames)
    graficosTiempo.parallelPlot(2007,2014,target = 'Años',xlabel = 'Tópicos', ylabel = 'Valoración promedio', title = 'Valoración por año')
    # Example by products user evaluation
    graficosTiempo= graphicsTime(dfVideoGames)
    graficosTiempo.plotSeries(productos = ["B00000DMAQ","B00002CF96"],target ='overall',modo = "asin", topics=[], ylabel= 'Valoración Promedio',xlabel = 'Años')
    graficosTiempo.plotSeries(productos = ["B00000DMAQ","B00002CF96"],target ='summary',modo = "asin", topics=[], ylabel= 'Opiniones',xlabel = 'Años')
    # Excample by topics
    graficosTiempo.plotSeries(productos = [],target ='overall',modo = "maximos", topics=[0,1,2,3], ylabel= 'Valoración Promedio',xlabel = 'Años')
    graficosTiempo.plotSeries(productos = [],target ='summary',modo = "maximos", topics=[0,1,2,3], ylabel= 'Opiniones',xlabel = 'Años')
    ##############################################
    ##################### POLARCHAR ##############
    graficoResumenGeneral=graphicsGeneralSummary()
    graficoResumenGeneral.summarizeProductTopics(dfVideoGames)
    graficoResumenGeneral.polarChar(product='0700099867', Title= 'Score by Topic')

    ################### TREE MAP ##################
    graficoResumenGeneral=graphicsGeneralSummary()
    graficoResumenGeneral.summarizeProductTopics(dfVideoGames)
    graficoResumenGeneral.addScore(5, 'overall')
    graficoResumenGeneral.concatenateCharacter('T', 'maximos' )
    graficoResumenGeneral.treeMap(product='0700099867', Title= 'Score by Topic')
