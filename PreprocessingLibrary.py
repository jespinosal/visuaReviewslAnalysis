#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This library include all the data methods to prepair the amazon review data,
# the tocics extraction and features extraction data
"""
Created on Wed Dec 27 18:16:08 2017

@author: espinosa
"""

import pandas as pd
import gzip 
import numpy as np

class Preprocesamiento:

    def discretizacion(self, objetivo, filtro, df, niveles):
        '''Clase que discretiza la variable objetivo de df en niveles
        input:
        df = dataframe de referencia
        objetivo = vatiable de df sobre la que se hace el analisis
        niveles = 3 ("Negativo", "Neutral", "Positivo") ó 5 ("Muy Malo", "Malo", "Regular", "Bueno", "Exelente"), para otro valor se devulven las etiquetas originales de la variable filtrada
        outout = configuracion: resumen de parametros de entrada, titulo: titulo del gráfico, frec: tabla de frecuencias por etiqueta

        '''
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
        self.frec = frec

    def discretizacionArray(self,listaProductos, df, n=5, mode='levels'):
        '''Esta función agrupa elementos de df según mode'''
        arrayProductos = []
        if mode == 'levels':
            for i in listaProductos:
                r = discretizacion('overall', i, df, n)
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
        return  self.dfSummarized

    def concatenateCharacter(self,character, column ):
        self.dfSummarized[column] = 'T' + self.dfSummarized[column].astype(str)




path = "/home/espinosa/TFM_Project/reviews_Video_Games_5.json.gz"
# Read as dictionay
def parse(path0): 
    g = gzip.open(path0, 'rb') 
    for l in g: 
        yield eval(l)


# Import as Panda dataFrame
def getDF(path1): 
    i = 0 
    df = {} 
    for d in parse(path1): 
        df[i] = d 
        i += 1 
    return pd.DataFrame.from_dict(df, orient='index') 
 


#@todo montar un ejemplo con el analisis de un producto ('asin') para obtener un resumen y el overall por tema
def summarizeProductTopics(dfTopics, product=False):
    dfSummarized = dfTopics.groupby(['asin','maximos'],as_index=False,sort=True).agg({'overall':'mean','summary':'count'})
    if product:
        dfSummarized  = dfSummarized.where(dfSummarized.asin==product).dropna()
    return (dfSummarized)


def AddScore(df, levels, objetivo):
        if levels == 3:
            levels = pd.cut(df[objetivo], 3,labels=["Negativo","Neutral","Positivo"])
        elif levels==5:
            levels = pd.cut(df[objetivo], 5,labels=["Muy Malo","Malo","Regular","Bueno","Exelente"]) 
        df['levels'] = levels
        return df

def discretizacion(objetivo,filtro,df,niveles):
    if niveles == 3:
        '''Esta funcion discretiza los valores de la variable objetivo en n niveles y extrae las etiquetas y frecuencias de una variable dado un data frame'''
        levels = pd.cut(df[objetivo], 3,labels=["Negativo","Neutral","Positivo"])
        df['levels'] = levels
        dfSubset=df
        if filtro:
            dfSubset = df[df.asin==filtro]
        frec = dfSubset.groupby('levels').count()[['asin']]
        #lab = frec.index
    elif niveles==5:
        levels = pd.cut(df[objetivo], 5,labels=["Muy Malo","Malo","Regular","Bueno","Exelente"]) 
        df['levels'] = levels
        dfSubset=df
        if filtro:
            dfSubset = df[df.asin==filtro]
        frec = dfSubset.groupby('levels').count()[['asin']]
        #lab = frec.index
    else:
        dfSubset=df
        if filtro:
            dfSubset = df[df.asin==filtro]
        frec = dfSubset.groupby(objetivo).count()[['asin']]
        #lab = frec.index
    return frec


def discretizacionArray(listaProductos,df,n=5, mode = 'levels'):
    '''Esta función agrupa elementos de df según mode'''
    arrayProductos = []
    if mode == 'levels':
        for i in listaProductos:
            r = discretizacion('overall',i,df,n)
            arrayProductos.append(list(r.asin))
        indices = r.index.categories
    elif mode =='overall':
        for i in listaProductos:
            r = df[df.asin==i][['maximos','overall']]
            arrayProductos.append(list(r.overall))
        indices = list(r.maximos)
    else:
        for i in listaProductos:
            r = df[df.asin==i][['maximos','summary']]
            arrayProductos.append(list(r.summary))
        indices = list(r.maximos)
        
    return np.array(arrayProductos), indices

# ....countainue desing discretizacionArray to summarize impte TN = 0 whenis neccesary
def fillEmptyTopics(summarize,nTopics=10):
    maximos = ['T'+str(i) for i in range(0,nTopics)]
    #maximos = pd.DataFrame({'maximos':maximos})
    asin = list(pd.unique(summarize.asin))
    return pd.DataFrame([(x, y) for x in asin for y in maximos], columns= ['asin','maximos'])


def polaritySummary(dfVideoGames,n=10):
    dfVideoGamesNorm= dfVideoGames.copy()
    dfVideoGamesNorm.maximos = 'T'+dfVideoGamesNorm.maximos.astype(str)
    dfVideoGamesNorm['overall'] = dfVideoGamesNorm['overall']-2.5
    dfVideoGamesNorm['polarity'] = np.where(dfVideoGamesNorm['overall']>0,'p','n')
    dfSummary0 = dfVideoGamesNorm.groupby(['asin','polarity','maximos'],as_index=False,sort=True).agg({'overall':'mean','summary':'count'})
    dfSummary0Pos = dfSummary0[dfSummary0.polarity=='p']#; PosList = dfSummary0Pos.summary*dfSummary0Pos.overall
    dfSummary0Neg = dfSummary0[dfSummary0.polarity=='n']#; NegList = dfSummary0Neg.summary*dfSummary0Neg.overall
    dfSummary0Pos['overall'] = dfSummary0Pos.summary*dfSummary0Pos.overall
    dfSummary0Neg['overall'] = dfSummary0Neg.summary*dfSummary0Neg.overall
    
    topicsByClient =  fillEmptyTopics(dfVideoGames,nTopics=n)
    
    dfSummary0Pos0 = pd.merge(topicsByClient,dfSummary0Pos,on=['asin','maximos'],how='left')
    dfSummary0Pos0['overall'] = dfSummary0Pos0['overall'].fillna(0)
    dfSummary0Pos0['summary']= dfSummary0Pos0['summary'].fillna(0)
    dfSummary0Pos0['polarity']= dfSummary0Pos0['polarity'].fillna('p')
    
    dfSummary0Neg0 = pd.merge(topicsByClient,dfSummary0Neg,on=['asin','maximos'],how='left')
    dfSummary0Neg0['overall'] = dfSummary0Neg0['overall'].fillna(0)
    dfSummary0Neg0['summary']= dfSummary0Neg0['summary'].fillna(0)
    dfSummary0Neg0['polarity']= dfSummary0Neg0['polarity'].fillna('n')
    return dfSummary0Pos0, dfSummary0Neg0


def compititiveSummary(dfVideoGames,products):
    dfVideoGamesNorm= dfVideoGames.copy()
    dfVideoGamesNorm.maximos = 'T'+dfVideoGamesNorm.maximos.astype(str)
    dfVideoGamesNorm['overall'] = dfVideoGamesNorm['overall']
    dfSummary0 = dfVideoGamesNorm.groupby(['asin','maximos'],as_index=False,sort=True).agg({'overall':'mean','summary':'count'})
    topicsByClient =  fillEmptyTopics(dfVideoGames,nTopics=10)
    dfSummary0 = pd.merge(topicsByClient,dfSummary0,on=['asin','maximos'],how='left')
    dfSummary0['overall'] = dfSummary0['overall'].fillna(0)
    dfSummary0['summary']= dfSummary0['summary'].fillna(0)
    
    dfSummay0Filter0 = dfSummary0.where(dfSummary0.asin==products[0]).dropna()
    dfSummay0Filter1 = dfSummary0.where(dfSummary0.asin==products[1]).dropna();dfSummay0Filter1['overall']=dfSummay0Filter1['overall']*-1
    dfSummay0Filter = dfSummay0Filter0.append(pd.DataFrame(data = dfSummay0Filter1))
    
    return dfSummay0Filter0, dfSummay0Filter1