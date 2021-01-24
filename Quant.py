# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta 

def preprocessing_path(path) :
    ############################
    ## 재무 데이터 전처리     ##
    ############################
    df = pd.read_excel(path, header = [0,1], index_col = 0)
    return df.round(4)

def preprocessing_price(path_price) :
    ############################
    ## 가격 데이터 전처리     ##
    ############################
    x = pd.DataFrame([])
    for i in pd.read_csv(path_price, chunksize = 5000, parse_dates = ['Symbol'], index_col = 0) :
        x = pd.concat([x,i], axis = 0)
    return x.astype(np.float64)

def preprocessing_mktdata(path_mkt_data) :    
    ############################
    ## 시가총액 데이터 전처리 ##
    ############################
    mkt_data = pd.DataFrame([])
    for i in pd.read_csv(path_mkt_data , index_col = 0, chunksize = 12000, engine = 'python') :
        mkt_data = pd.concat([mkt_data, i], axis = 0)
    
    Common = mkt_data[mkt_data.columns[::2]].iloc[1:].applymap(lambda x : x.replace(',','') if type(x) == str else x).astype(np.float64)
    Prefer = mkt_data[mkt_data.columns[1::2]].iloc[1:].applymap(lambda x : x.replace(',','') if type(x) == str else x).astype(np.float64)
    
    mkt_data = pd.DataFrame(np.array(Common)/100 + np.array(Prefer.fillna(0))/100, columns = Common.columns, index = Common.index)
    mkt_data.index = pd.to_datetime(mkt_data.index)
    return mkt_data.resample('D').last().fillna(method = 'ffill', limit = 10)

def preprocessing_kospiyn(path_kospiyn, mkt='유가증권시장') :
    ##########################################
    ### 거래소(코스피,코스닥)데이터 전처리  ##
    #########################################
    data = pd.DataFrame([])
    for i in pd.read_csv(path_kospiyn, index_col = 0 , chunksize = 6000 , engine = 'python') :
        data = pd.concat([data,i], axis = 0)
    data.index = pd.to_datetime(data.index)
    if mkt == '유가증권시장' or mkt == '코스닥':
        data = data.astype(str).applymap(lambda x : int(mkt in x)).resample('D').last()
    elif mkt == 'both' :
        data = data.astype(str).applymap(lambda x : int('유가증권시장' in x or '코스닥' in x)).resample('D').last()
    data = data.fillna(method = 'ffill', limit = 30)
    return data

def preprocessing_stop_and_delist(path_delist_and_stop) :
    #########################################
    ## 거래정지 및 상장폐지 데이터 전처리  ##
    #########################################    
    data = pd.DataFrame([])
    for i in pd.read_csv(path_delist_and_stop, index_col = 0 , chunksize = 6000, engine = 'python') :
        data = pd.concat([data,i], axis = 0)
    delist_data = data[[data.columns[1]]].iloc[1:]
    delist_data = delist_data[delist_data['상장폐지일자'].isna() == False]
    delist_data['상장폐지일자'] = delist_data['상장폐지일자'].apply(lambda x : pd.to_datetime(x))
    small_col = list(data[data.columns[2:]].iloc[0])
    small_col = pd.to_datetime(small_col)
    big_col = pd.Series(data.columns[2:]).apply(lambda x : x.split('.')[0])    
    stop_data = data[data.columns[2:]].iloc[1:]
    stop_data.columns = [big_col,small_col]
    stop_data = stop_data.applymap(lambda x : 1 if x in ['TRUE' , True] else 0)        
    return  delist_data, stop_data

preprocessing_period = lambda path_endmonth : pd.read_csv(path_endmonth,index_col = 0, parse_dates=['Symbol']).fillna(method = 'ffill').fillna(method = 'bfill').fillna(12)

def Value(cleaned_data, cleaned_mkt, data_date, today, n = 50,cleaned_price = '') : 
    mvalue = cleaned_mkt.loc[:pd.to_datetime(today) - relativedelta(days = 1)].iloc[-1]
    Earning = cleaned_data['당기순이익'][data_date]
    Earning_Plus_index = Earning[Earning>0].index
    PER = mvalue[Earning_Plus_index]/Earning[Earning_Plus_index]
    
    Book = cleaned_data['총자본'][data_date] - cleaned_data['무형자산'][data_date]
    PBR = mvalue[Book>0]/Book[Book>0]
    
    Sales = cleaned_data['매출액'][data_date]
    PSR = mvalue[Sales>0] / Sales[Sales>0]
    
    CF = cleaned_data['영업활동으로인한현금흐름'][data_date]
    PCR = mvalue[CF>0] /CF[CF>0] 
    
    EV = mvalue + cleaned_data['총부채'][data_date]
    EBITDA = cleaned_data['EBITDA'][data_date]
    EVEBITDA = EV[EBITDA>0]/ EBITDA[EBITDA>0]
    
    Value_Rank = pd.concat([PER.rank(), PBR.rank(), PSR.rank(), PCR.rank(), EVEBITDA.rank()],axis = 1)
    Value_Rank.columns = ['PER','PBR','PSR','PCR','EVEBITDA']
    Value_Rank['Total_Rank'] = Value_Rank.mean(axis = 1, skipna = True).round(2)
    number_nan = Value_Rank.isna().sum(1)
    under_2_nan = number_nan[number_nan<=2].index
    my_index = Earning_Plus_index.intersection(under_2_nan)
    result = Value_Rank.loc[my_index]
    return result.sort_values(by = ['Total_Rank']).iloc[:n]

def Quality(cleaned_data, data_date, today , n = 50, cleaned_price = '',cleaned_mkt='') :
    GP = cleaned_data['매출총이익'][data_date] / cleaned_data['매출액'][data_date]
    OP = cleaned_data['영업이익'][data_date] / cleaned_data['매출액'][data_date]
    ROE = cleaned_data['당기순이익'][data_date] / cleaned_data['총자본'][data_date]
    ROA = cleaned_data['당기순이익'][data_date] / cleaned_data['총자산'][data_date]
    df = pd.concat([GP.rank(ascending = False) , OP.rank(ascending = False) ,
                    ROE.rank(ascending = False), ROA.rank(ascending = False)],axis = 1)
    df.columns = ['GP','OP','ROE','ROA']
    number_nan = df.isna().sum(1, skipna = True)
    under_1_nan = number_nan[number_nan<=1].index
    df = df.loc[under_1_nan]
    df['Total_Rank'] = df.mean(axis = 1 , skipna = True)
    return df.sort_values(by = ['Total_Rank']).iloc[:n]

def Value_Quality(cleaned_data, cleaned_mkt, data_date, today, n = 50, cleaned_price = '') :
    V = Value(cleaned_data, cleaned_mkt, data_date, today, n = 5000,cleaned_price = '')
    Q = Quality(cleaned_data, data_date, today , n = 5000, cleaned_price = '',cleaned_mkt='')
    data = pd.concat([V[V.columns[:-1]],Q[Q.columns[:-1]]],axis = 1)
    number_nan = data.isna().sum(1, skipna = True)
    under_3_nan = number_nan[number_nan<=3].index
    Data  = data.loc[under_3_nan]
    Data['Total_Rank'] = Data.mean(1, skipna = True).round(2)
    return Data.sort_values(by = ['Total_Rank']).iloc[:n]

def Growth(cleaned_data, data_date, n = 50, today = '', cleaned_mkt = '',cleaned_price = '') :
    Data_Date = pd.to_datetime(data_date)
    if Data_Date >= pd.to_datetime('2001-12-31') :
        Data_bDate = pd.to_datetime(str(Data_Date.year -1) +'-'+ str(Data_Date.month)+'-'+str(Data_Date.day))
    else :
        if Data_Date.month < 5 :
            Data_bDate = pd.to_datetime(str(Data_Date.year -2) + '-12-31')
        else :
            Data_bDate = pd.to_datetime(str(Data_Date.year -1) +'-12-31')
    GP = (cleaned_data['매출총이익'][Data_Date]/cleaned_data['매출액'][Data_Date]).round(2)
    OP = (cleaned_data['영업이익'][Data_Date]/cleaned_data['매출액'][Data_Date]).round(2)
    ROE = (cleaned_data['당기순이익'][Data_Date]/cleaned_data['총자산'][Data_Date]).round(2)
    ROA = (cleaned_data['당기순이익'][Data_Date]/cleaned_data['총자본'][Data_Date]).round(2)
    LEV = (cleaned_data['총부채'][Data_Date]/cleaned_data['총자본'][Data_Date]).round(2)
    
    bGP = (cleaned_data['매출총이익'][Data_bDate]/cleaned_data['매출액'][Data_bDate]).round(2)
    bOP = (cleaned_data['영업이익'][Data_bDate]/cleaned_data['매출액'][Data_bDate]).round(2)
    bROE = (cleaned_data['당기순이익'][Data_bDate]/cleaned_data['총자산'][Data_bDate]).round(2)
    bROA = (cleaned_data['당기순이익'][Data_bDate]/cleaned_data['총자본'][Data_bDate]).round(2)
    bLEV = (cleaned_data['총부채'][Data_bDate]/cleaned_data['총자본'][Data_bDate]).round(2)
        
    dGP = GP - bGP
    dOP = OP - bOP
    dROE = ROE - bROE
    dROA = ROA - bROA
    dLEV = LEV - bLEV

    df = pd.concat([dGP.rank(ascending = False), dOP.rank(ascending = False), 
                    dROE.rank(ascending = False), dROA.rank(ascending = False),
                    dLEV.rank(ascending = True)], axis = 1)
    df.columns = ['dGP','dOP','dROE','dROA','dLEV']
    number_nan = df.isna().sum(1, skipna = True)
    under_2_nan = number_nan[number_nan<=2].index
    Data  = df.loc[under_2_nan]    
    Data['Total_Rank'] = Data.mean(1,skipna = True).round(2)
    return Data.sort_values(by = ['Total_Rank']).iloc[:n]

def Value_Quality_Growth(cleaned_data, cleaned_mkt, data_date, today, n = 50, cleaned_price = '') :
    V = Value(cleaned_data, cleaned_mkt, data_date, today, n = 5000,cleaned_price = '')['Total_Rank']
    Q = Quality(cleaned_data, data_date, today , n = 5000, cleaned_price = '',cleaned_mkt='')['Total_Rank']
    G = Growth(cleaned_data, data_date, n = 5000, today = '', cleaned_mkt = '',cleaned_price = '')['Total_Rank']
    DF = pd.concat([V,Q,G],axis = 1)
    DF.columns = ['Value_Rank','Quality_Rank','Growth_Rank']
    DF['Total_Rank'] = (DF['Value_Rank'] * 0.4 + DF['Quality_Rank'] * 0.4 +
                        DF['Growth_Rank'] * 0.2)
    return DF.sort_values(by = ['Total_Rank']).dropna().iloc[:n]

def momentum_screen(cleaned_price, today,cleaned_mkt = '', cleaned_data = '', n = '', data_date = '') :
    before_24 = cleaned_price[:pd.to_datetime(today) - relativedelta(months = 24)].iloc[-1]
    before_6 = cleaned_price[:pd.to_datetime(today) - relativedelta(months = 6)].iloc[-1]
    before_1 = cleaned_price[:pd.to_datetime(today) - relativedelta(months = 1)].iloc[-1]
    current_price = cleaned_price[:today].iloc[-1]
    index1 = current_price[current_price<before_24].index
    index2 = current_price[current_price>before_6].index
    index3 = current_price[current_price<before_1].index
    index = index1.intersection(index2).intersection(index3)
    return cleaned_price[index]

