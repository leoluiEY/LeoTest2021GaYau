import pymssql
import pandas as pd
from pandas import DataFrame
from pandas import json_normalize
import urllib
import sqlalchemy as sa
import time
import re
import json 
import numpy as np
import os
from dotenv import load_dotenv
from pathlib import Path
from threading import Thread
import tracemalloc
from pyxirr import xirr
from datetime import date
from datetime import datetime
from scipy import optimize


dotenv_path = Path('dev.env')
load_dotenv(dotenv_path=dotenv_path)
DATABASE_HOST = os.getenv('DATABASE_HOST')
DATABASE_ACC = os.getenv('DATABASE_ACC')
DATABASE_PW = os.getenv('DATABASE_PW')
DATABASE_NAME = os.getenv('DATABASE_NAME')

def SelectFromDB(host0 = DATABASE_HOST , account = DATABASE_ACC, pw = DATABASE_PW ,DBNAME = DATABASE_NAME, sql=[]):            
    #connect to DB
    conn = pymssql.connect(
    host = host0,
    user = account,
    password = pw,
    database = DBNAME
)
    c2 = conn.cursor(as_dict = True)
    c2.execute(sql[0])
    c2_list = c2.fetchall()
    df_raws=pd.DataFrame(c2_list)
    c2.close()
    conn.close()
    c2_list_jsondata = json.dumps(c2_list)
    return df_raws

# In[2]:
def get_data(request_data):
    global ProjectItemData 
    global HeadroomItemData 
    global debt_to_ebitda_ratio
    global budget
    global budget_array
    global nribudget
    global naebudget
    global filter_dict
    global cagr_filter_dict
    global unique_hprojectitem_id
    global unique_hproject_id 
    global IRR_raw
    global NAEData
    global NRIData
    
    must_not_include_projects = []
    for i in range(len(request_data['project_option'])):
        if (request_data['project_option'][i]['must_not_include'] == True):
            must_not_include_projects.append(request_data['project_option'][i]['project_id'])
    must_not_include_projects1 = tuple(must_not_include_projects)
    if len(must_not_include_projects1) > 0:
        must_not_include_sql = " AND a.[hproject_id]" + ("!= " + str(must_not_include_projects[0]) if len(must_not_include_projects1) == 1 else "NOT IN" + str(must_not_include_projects1))
        #must_not_include_sql1 = " AND [hproject_id]" + ("!= " + str(must_not_include_projects[0]) if len(must_not_include_projects1) == 1 else "NOT IN" + str(must_not_include_projects1))
    else:
        must_not_include_sql = ''
        #must_not_include_sql1 = ''

    headroom_id = request_data['headroom_id']
    include_dummy_project = request_data['include_dummy_project']
    
    ProjectItemData_sql = """SELECT a.[hproject_id], a.[hprojectitem_id] ,b.[region_code]  ,a.[sector_code]  ,a.[segment_code]  ,a.[hproject_ownership]  
    ,a.[hprojectitem_npv]  ,a.[hprojectitem_irr] , b.hproject_name, b.hproject_type, b.[dummy_proj],
    b.commitment_timing, b.jv_sub , b.headroom_id, c.*
    FROM [dbo].[bga_hrj_HProjectItem] a 
    left join [dbo].[bga_hrj_HProject] b 
    ON a.hproject_id = b.hproject_id
    right join [dbo].[bga_hrj_HProjectItemValue] c
    ON a.hprojectitem_id = c.hprojectitem_id
    where a.[sector_code] is not null and a.[segment_code] is not null 
    and b.hproject_type in ('Probable', 'Possible')
    and a.is_enable = 1
     """
    headroom_id_condition = " AND b.[headroom_id] = " + str(headroom_id)
    dummy_proj_condition = " AND b.[dummy_proj] IN " + ("(0, 1)" if include_dummy_project else "(0)")
    ProjectItemData_final_sql = ProjectItemData_sql + str(headroom_id_condition) + str(dummy_proj_condition) + str(must_not_include_sql)
    ProjectItemData = SelectFromDB(sql = [ProjectItemData_final_sql])
    
    NRIData_sql = """SELECT a.[hproject_id], a.[hprojectitem_id] , b.[hproject_name] ,b.[region_code] ,a.[sector_code]  ,a.[segment_code]   
    ,a.[hprojectitem_npv], a.[hprojectitem_irr] , b.[hproject_irr], b.[hproject_type],  a.[hproject_ownership], 
    b.[dummy_proj], b.commitment_timing, b.jv_sub 
    , c.[t1]* a.[hproject_ownership] as t1, c.[t2]* a.[hproject_ownership] as t2, c.[t3]* a.[hproject_ownership] as t3
    , c.[t4]* a.[hproject_ownership] as t4, c.[t5]* a.[hproject_ownership] as t5, c.[t6]* a.[hproject_ownership] as t6
    , c.[t7]* a.[hproject_ownership] as t7, c.[t8]* a.[hproject_ownership] as t8, c.[t9]* a.[hproject_ownership] as t9
    , c.[t10]* a.[hproject_ownership] as t10, c.[t11]* a.[hproject_ownership] as t11
    FROM [dbo].[bga_hrj_HProjectItem] a 
      left join [dbo].[bga_hrj_HProject] b 
      ON a.hproject_id = b.hproject_id
      right join [dbo].[bga_hrj_HProjectItemValue] c
      ON a.hprojectitem_id = c.hprojectitem_id where a.[sector_code] is not null 
      and a.[segment_code] is not null and b.hproject_type in ('Probable', 'Possible')
    and c.hprojectitemvalue_type = 'nri_include_vat_impact'  and a.segment_code = 'ip'
    and a.is_enable = 1
     """
    #and a.[hproject_id] < 1000619 for fire's testing
    #headroom_id_condition = " AND b.[headroom_id] = " + str(headroom_id)
    #dummy_proj_condition = " AND b.[dummy_proj] IN " + ("(0, 1)" if include_dummy_project else "(0)")
    NRIData_final_sql = NRIData_sql + str(headroom_id_condition) + str(dummy_proj_condition)  + str(must_not_include_sql)
    NRIData = SelectFromDB(sql = [NRIData_final_sql])
    
    NAEData_sql = """select [hproject_id], [hprojectitem_id] ,[hproject_ownership],[region_code], [sector_code]  ,[segment_code] , [dummy_proj],
    t1 as at1, t1+t2 as at2, t1+t2+t3 as at3,
    t1+t2+t3+t4 as at4, t1+t2+t3+t4+t5 as at5, t1+t2+t3+t4+t5+t6 as at6, t1+t2+t3+t4+t5+t6+t7 as at7,
    t1+t2+t3+t4+t5+t6+t7+t8 as at8, t1+t2+t3+t4+t5+t6+t7+t8+t9 as at9,
    t1+t2+t3+t4+t5+t6+t7+t8+t9+t10 as at10, t1+t2+t3+t4+t5+t6+t7+t8+t9+t10+t11 as at11
    from ( 
        SELECT a.[hproject_id], a.[hprojectitem_id] ,a.[hproject_ownership]  ,a.[sector_code]  ,a.[segment_code] ,b.[dummy_proj] ,b.[region_code],
      sum (
    CASE WHEN c.hprojectitemvalue_type = 'free_cashflow_exclude_finance_charge'
    then c.[t1]* a.[hproject_ownership]*-1
    else c.[t1]* a.[hproject_ownership]
    end ) as t1,
      sum (
    CASE WHEN c.hprojectitemvalue_type = 'free_cashflow_exclude_finance_charge'
    then c.[t2]* a.[hproject_ownership]*-1
    else c.[t2]* a.[hproject_ownership]
    end ) as t2,
      sum (
    CASE WHEN c.hprojectitemvalue_type = 'free_cashflow_exclude_finance_charge'
    then c.[t3]* a.[hproject_ownership]*-1
    else c.[t3]* a.[hproject_ownership]
    end ) as t3,
      sum (
    CASE WHEN c.hprojectitemvalue_type = 'free_cashflow_exclude_finance_charge'
    then c.[t4]* a.[hproject_ownership]*-1
    else c.[t4]* a.[hproject_ownership]
    end ) as t4,
      sum (
    CASE WHEN c.hprojectitemvalue_type = 'free_cashflow_exclude_finance_charge'
    then c.[t5]* a.[hproject_ownership]*-1
    else c.[t5]* a.[hproject_ownership]
    end ) as t5,
      sum (
    CASE WHEN c.hprojectitemvalue_type = 'free_cashflow_exclude_finance_charge'
    then c.[t6]* a.[hproject_ownership]*-1
    else c.[t6]* a.[hproject_ownership]
    end ) as t6,
      sum (
    CASE WHEN c.hprojectitemvalue_type = 'free_cashflow_exclude_finance_charge'
    then c.[t7]* a.[hproject_ownership]*-1
    else c.[t7]* a.[hproject_ownership]
    end ) as t7,
      sum (
    CASE WHEN c.hprojectitemvalue_type = 'free_cashflow_exclude_finance_charge'
    then c.[t8]* a.[hproject_ownership]*-1
    else c.[t8]* a.[hproject_ownership]
    end ) as t8,
      sum (
    CASE WHEN c.hprojectitemvalue_type = 'free_cashflow_exclude_finance_charge'
    then c.[t9]* a.[hproject_ownership]*-1
    else c.[t9]* a.[hproject_ownership]
    end ) as t9,
      sum (
    CASE WHEN c.hprojectitemvalue_type = 'free_cashflow_exclude_finance_charge'
    then c.[t10]* a.[hproject_ownership]*-1
    else c.[t10]* a.[hproject_ownership]
    end ) as t10,
      sum (
    CASE WHEN c.hprojectitemvalue_type = 'free_cashflow_exclude_finance_charge'
    then c.[t11]* a.[hproject_ownership]*-1
    else c.[t11]* a.[hproject_ownership]
    end ) as t11
    FROM [dbo].[bga_hrj_HProjectItem] a 
      left join [dbo].[bga_hrj_HProject] b 
      ON a.hproject_id = b.hproject_id
      right join [dbo].[bga_hrj_HProjectItemValue] c
      ON a.hprojectitem_id = c.hprojectitem_id where a.[sector_code] is not null 
      and a.[segment_code] is not null and b.hproject_type in ('Probable', 'Possible')
      and a.is_enable = 1
    """
    
    NAE_end_sql = """  and c.hprojectitemvalue_type in ('nri_include_vat_impact', 'gross_profit_from_tp',
      'operating_profit_from_hotel_exclude_depreciation', 'free_cashflow_exclude_finance_charge')
        group by a.[hproject_id] , a.[hprojectitem_id] ,a.[hproject_ownership], a.[sector_code]  ,a.[segment_code] ,b.[dummy_proj], b.[region_code]
        ) final_data
    """
    #headroom_id_condition = " AND b.[headroom_id] = " + str(headroom_id)
    #dummy_proj_condition = " AND b.[dummy_proj] IN " + ("(0, 1)" if include_dummy_project else "(0)")
    NAEData_final_sql = NAEData_sql + str(headroom_id_condition) + str(dummy_proj_condition) + str(must_not_include_sql)+ NAE_end_sql
    NAEData = SelectFromDB(sql = [NAEData_final_sql])
    
    HeadroomItemData_sql = """SELECT [headroomitem_id] ,[headroom_id] ,[headroomitem_type] ,[region_code] ,[sector_code]
    ,[segment_code] ,[region_id] ,[sector_id] ,[segment_id] ,[t1] ,[t2] ,[t3] ,[t4] ,[t5] 
    ,[t6] ,[t7] ,[t8] ,[t9] ,[t10] ,[t11] FROM [dbo].[bga_hrj_HeadroomItem] """
    headroom_id_condition1 = " WHERE [headroom_id] = " + str(headroom_id)
    headroomItemData_final_sql = HeadroomItemData_sql + headroom_id_condition1
    HeadroomItemData = SelectFromDB(sql = [headroomItemData_final_sql])
    IRR_raw = ProjectItemData.loc[ProjectItemData['hprojectitemvalue_type'] == 'free_cashflow_exclude_finance_charge']
    ProjectItemData = ProjectItemData[['hproject_id', 'hprojectitem_id', 'hproject_name','region_code', 'sector_code', 'segment_code', 'hprojectitem_npv', 'hproject_type', 'jv_sub', 'hproject_ownership' ,'hprojectitemvalue_type', 'commitment_timing', 't1', 't2', 't3','t4', 't5', 't6','t7', 't8', 't9','t10','t11']]
    debt_to_ebitda_ratio = HeadroomItemData[HeadroomItemData.headroomitem_type == "debt_to_ebitda_ratio"].fillna(0)[['t1', 't2', 't3','t4', 't5', 't6','t7', 't8', 't9','t10','t11']].values[0].tolist()
    budget = HeadroomItemData[HeadroomItemData.headroomitem_type == "budget_headroom"].fillna(0)[['t1', 't2', 't3','t4', 't5', 't6','t7', 't8', 't9','t10','t11']].values[0].tolist()
    budget_array = np.array(budget)
    
    filter_dict = {}
    for idx, val in request_data['filter'].items():
        if val['active']:
            _ = np.array(sorted(zip(val['key'], val['value']), key=lambda x: val['key'].index(x[0]))).T
            _ = _[:,_[0].argsort(0)]
            filter_dict[idx] = _[0], _[1].astype(float) - val['diff'], _[1].astype(float) + val['diff']    
    cagr_filter_dict = {}
    for idx, val in request_data['cagr_filter'].items():
        if val['active']:
            _ = np.array(sorted(zip(val['key'], val['value']), key=lambda x: val['key'].index(x[0]))).T
            _ = _[:,_[0].argsort(0)]
            cagr_filter_dict[idx] = _[0], _[1].astype(float) - val['diff'], _[1].astype(float) + val['diff']
    
    nribudget = HeadroomItemData[(HeadroomItemData.headroomitem_type == "nri") & (HeadroomItemData.segment_code == "ip")][['region_code','segment_code' ,'sector_code','t1', 't2', 't3','t4', 't5', 't6','t7', 't8', 't9','t10','t11']]
    #nribudget = HeadroomItemData[HeadroomItemData.headroomitem_type == "nri"][['region_code','segment_code','t1', 't2', 't3','t4', 't5', 't6','t7', 't8', 't9','t10','t11']]
    naebudget = HeadroomItemData[HeadroomItemData.headroomitem_type == "nae"][['region_code','segment_code','sector_code','t1', 't2', 't3','t4', 't5', 't6','t7', 't8', 't9','t10','t11']]

    unique_hprojectitem_id = ProjectItemData['hprojectitem_id'].unique()
    unique_hproject_id = ProjectItemData['hproject_id'].unique()
    
# In[3]:
def get_basic_combos(hprojectitem):
    data_new = ProjectItemData[ProjectItemData['hprojectitem_id'] == hprojectitem].reset_index(drop = True)
    commitment_timing = data_new.loc[0, 'commitment_timing']
    hproject_type = data_new.loc[0, 'hproject_type']
    jv_sub = data_new.loc[0, 'jv_sub']
    ownership = data_new.loc[0, 'hproject_ownership']
    npv = data_new.loc[0, 'hprojectitem_npv']*ownership
    debt_to_ebitda_ratio_array  =np.array(debt_to_ebitda_ratio)
    investment_cost = sum(data_new.loc[data_new['hprojectitemvalue_type'] == 'investment_cost_equity_funded'][['t1', 't2', 't3','t4', 't5', 't6','t7', 't8', 't9','t10','t11']].reset_index(drop = True).values[0] +                    data_new.loc[data_new['hprojectitemvalue_type'] == 'investment_cost_debt_funded'][['t1', 't2', 't3','t4', 't5', 't6','t7', 't8', 't9','t10','t11']].reset_index(drop = True).values[0]) * ownership
    #total_trading_profit = sum(data_new.loc[data_new['hprojectitemvalue_type'] == 'gross_profit_from_tp'][['t1', 't2', 't3','t4', 't5', 't6','t7', 't8', 't9','t10','t11']].reset_index(drop = True).values[0])*ownership
    try:
        total_trading_profit = sum(data_new.loc[data_new['hprojectitemvalue_type'] == 'gross_profit_from_tp'][['t1', 't2', 't3','t4', 't5', 't6','t7', 't8', 't9','t10','t11']].reset_index(drop = True).values[0])*ownership
    except IndexError: 
        total_trading_profit = 0
    net_finance_charge = data_new.loc[data_new['hprojectitemvalue_type'] == 'net_finance_charge'][['t1', 't2', 't3','t4', 't5', 't6','t7', 't8', 't9','t10','t11']].reset_index(drop = True).values[0]
    cum_net_finance_charge = np.add.accumulate(data_new.loc[data_new['hprojectitemvalue_type'] == 'net_finance_charge'][['t1', 't2', 't3','t4', 't5', 't6','t7', 't8', 't9','t10','t11']].reset_index(drop = True).values[0])
    freecashflow = data_new.loc[data_new['hprojectitemvalue_type'] == 'free_cashflow_exclude_finance_charge'][['t1', 't2', 't3','t4', 't5', 't6','t7', 't8', 't9','t10','t11']].reset_index(drop = True).values[0]
    ebitda = data_new.loc[data_new['hprojectitemvalue_type'].isin(['gross_profit_from_tp', 'operating_profit_from_hotel_exclude_depreciation', 'nri_include_vat_impact'])][['t1', 't2', 't3','t4', 't5', 't6','t7', 't8', 't9','t10','t11']].reset_index(drop = True).values
    ebitda1 = ebitda.sum(axis=0)
    gri = sum(data_new.loc[data_new['hprojectitemvalue_type'] == 'gri'][['t1', 't2', 't3','t4', 't5', 't6','t7', 't8', 't9','t10','t11']].reset_index(drop = True).values[0])*ownership 
    aup = sum(data_new.loc[(data_new['hprojectitemvalue_type'] == 'underlying_profit') | (data_new['hprojectitemvalue_type'] == 'aup')][['t1', 't2', 't3','t4', 't5', 't6','t7', 't8', 't9','t10','t11']].reset_index(drop = True).values[0])*ownership 

    if (jv_sub == "JV"):
        cum_free_cashflow_with_ownership = np.add.accumulate(freecashflow)*ownership
        try:
            ebitda_with_ownership = data_new.loc[data_new['hprojectitemvalue_type'] == 'gross_profit_from_tp'][['t1', 't2', 't3','t4', 't5', 't6','t7', 't8', 't9','t10','t11']].reset_index(drop = True).values[0]*data_new.loc[0, 'hproject_ownership']
        except IndexError: 
            ebitda_with_ownership = 0
        #ebitda_with_ownership = data_new.loc[data_new['hprojectitemvalue_type'] == 'gross_profit_from_tp'][['t1', 't2', 't3','t4', 't5', 't6','t7', 't8', 't9','t10','t11']].reset_index(drop = True).values[0]*data_new.loc[0, 'hproject_ownership']
        final_ebitda_with_ownership = ebitda_with_ownership * debt_to_ebitda_ratio_array
        remaining_budget = cum_net_finance_charge*ownership + cum_free_cashflow_with_ownership + final_ebitda_with_ownership
        
    elif ((jv_sub == "Sub") & (ownership < 1)):
        cum_free_cashflow = np.add.accumulate(freecashflow)
        final_ebitda = ebitda1 * debt_to_ebitda_ratio_array
        cum_equity_portion = np.add.accumulate(data_new.loc[data_new['hprojectitemvalue_type'] == 'investment_cost_equity_funded'][['t1', 't2', 't3','t4', 't5', 't6','t7', 't8', 't9','t10','t11']].reset_index(drop = True).values[0])*round(1 - data_new.loc[0, 'hproject_ownership'], 2)
        remaining_budget = cum_net_finance_charge + cum_free_cashflow + final_ebitda - cum_equity_portion
    elif ((jv_sub == "Sub") & (ownership >= 1)):
        cum_free_cashflow = np.add.accumulate(freecashflow)
        final_ebitda = ebitda1 * debt_to_ebitda_ratio_array
        remaining_budget = cum_net_finance_charge + cum_free_cashflow + final_ebitda
    order_info = {}
    order_info = {
        'npv': npv, 
        'total_trading_profit': total_trading_profit,
        'investment_cost' : investment_cost,
        'net_finance_charge':  net_finance_charge,
        'aup': aup,
        'gri':gri,
        'commitment_timing' : commitment_timing,
        'probable_proj' : hproject_type
    }
    return remaining_budget, order_info

def get_all_combos1_v0(combos, must_include_projects, exceed_headroom_a_new):
    project_size = len(combos[1]) #14 here, num of projects
    project_list = sorted([i[0] for i, x in combos[1].items()]) #project_list
    for combo_size in range(2, project_size+1): # 2,3,4,5....15 , so loop till 14
        combos[combo_size] = {}
        if combo_size == 2:
            for key, row in combos[1].items():
                project_list.remove(key[0])
                row['candidates'] = project_list.copy()
                if not row['candidates']:
                    continue
                for candidate in row['candidates'].copy():
                    project_set = key + tuple([candidate])
                    row['candidates'].remove(candidate)
                    min_cum_remainf = filter_in_budget(row['min_cum_remain'], combos[1][tuple([candidate])]['min_cum_remain'])
                    remain_budget = min_cum_remainf + budget_array
                    combos[combo_size][key + tuple([candidate])] = {'candidates': row['candidates'].copy(),
                                                                    'project_set' : project_set,
                                                                    'min_cum_remain': min_cum_remainf,
                                                                    'remaining_headroom': remain_budget
                                                               }
        if combo_size > 2:
            for key, row in dict(combos[combo_size-1]).items():
                if not row['candidates']:
                    continue
                for candidate in row['candidates'].copy():
                    project_set = key + tuple([candidate])
                    row['candidates'].remove(candidate)
                    min_cum_remainf = filter_in_budget(row['min_cum_remain'], combos[1][tuple([candidate])]['min_cum_remain'])
                    remain_budget = min_cum_remainf + budget_array
                    combos[combo_size][key + tuple([candidate])] = {'candidates': row['candidates'].copy(),
                                                                    'project_set' : project_set,
                                                                    'min_cum_remain': min_cum_remainf,
                                                                    'remaining_headroom': remain_budget
                                                                   }
#                 if not filter_must(combos[combo_size-1][key]['project_set'], must_include_projects): # if not all project must, continue
#                     #combos[combo_size-1].pop(key)
#                     del combos[combo_size-1][key]
#                     #print(key, 'must')
#                     continue
                min_headroom = filter_headroom(combos[combo_size-1][key]['remaining_headroom'], exceed_headroom_a_new)
                if min_headroom is False:
                    #combos[combo_size-1].pop(key)
                    del combos[combo_size-1][key]
                    #print(key, 'Min')
                    continue
                combos[combo_size-1][key].pop('candidates') 
                combos[combo_size-1][key].pop('project_set') 
                combos[combo_size-1][key].pop('min_cum_remain')             
    return combos


def filter_in_budget(remain, additional):
    new_remain = remain + additional
    return new_remain

def filter_must(combo_key, must_projects):
    return len(set(must_projects) - set(combo_key)) == 0

def filter_headroom(remain, additional):
    new_remain = remain + additional
    if new_remain.min() < 0:
        return False
    else:
        return True

def filter_prop(hprojectitem_idx, combo_detail, accept_types):
    for filter_key, accepts in accept_types.items():
        if combo_detail[filter_key][hprojectitem_idx] not in accepts:
            return False
    return True

def get_proportion(keys, values):
    tt = pd.DataFrame({'key': keys, 'value': values})
    if tt['value'].sum() == 0:
        total=1
    else:
        total=tt['value'].sum() 
    proportion = tt.groupby('key').agg({'value': lambda x: x.sum() / total }).reset_index().values.T
    return proportion[0], proportion[1].astype(float)

def get_accept_types(combo_detail, budget_data):
    accept_types = {}
    for filter_key in filter_dict.keys():
        prop_key, prop_val = get_proportion(
            np.concatenate([budget_data[filter_key], combo_detail[filter_key]]),
            np.concatenate([budget_data.iloc[:, -1],
                            combo_detail.iloc[:, -1]  ]))
        max_value = filter_dict[filter_key][2] > prop_val
        min_value = filter_dict[filter_key][1] < prop_val
        accept_types[filter_key] = prop_key[np.logical_and(max_value,min_value)]
    return accept_types

def get_accept_types_cagr(combo_detail, budget_data):
    accept_types = {}
    for filter_key in cagr_filter_dict.keys():
        accept_types[filter_key] = []
        for key in range(0, len(cagr_filter_dict[filter_key][0])):
            testcombo1 = combo_detail.loc[combo_detail[filter_key] == cagr_filter_dict[filter_key][0][key]]
            nribudget1 = budget_data.loc[budget_data[filter_key] == cagr_filter_dict[filter_key][0][key]]
            A= np.concatenate([nribudget1.iloc[:, -1],  testcombo1.iloc[:, -1]  ]).sum()
            B = np.concatenate([nribudget1.iloc[:, -11],  testcombo1.iloc[:, -11]  ]).sum()
            #print(key, A, B)           
            if ((A <= 0) or (B <= 0)):
                #CAGR = 0
                CAGR = -999
            else:
                C = (A/B)
                CAGR = np.float_power(C, 1/10) - 1
#             if np.isnan(CAGR):
#                 CAGR = 0
#             print(key)
#             print(CAGR)
#             print(cagr_filter_dict[filter_key][1][key])
            if (cagr_filter_dict[filter_key][1][key] < CAGR):
                accept_types[filter_key] += [cagr_filter_dict[filter_key][0][key]]
    return accept_types

def xirr1(cashflow, guess=0.1):
    residual = 1.0
    step = 0.05
    epsilon = 0.0001
    limit = 10000
    while abs(residual) > epsilon and limit > 0:
        limit -= 1
        residual = 0.0
        for i, trans in enumerate(cashflow[0]):
            # residual += trans[1] / pow(guess, years[i])
            residual += trans / pow(guess, i)
        if abs(residual) > epsilon:
            if residual > 0:
                guess += step
            else:
                guess -= step
                step /= 2.0
    return guess - 1

def new_scoring_xirr(subprojects_fcf, guess=0.1):
    project_fcf_sum = np.sum(subprojects_fcf, axis = 0)    
    project_fcf_sum = np.trim_zeros(project_fcf_sum, 'b')
    project_fcf_sum = np.expand_dims(project_fcf_sum, axis = 0)
    return xirr1(project_fcf_sum, guess=guess)

def get_accept_combo2(accepted_combos, request_data, dictN, must_include_projects,NRIData, NAEData, naebudget, nribudget,exceed_headroom_a_new):     
    for keys in dictN.keys():
        if not filter_must(keys, must_include_projects): # if not all project must, continue
            continue
        min_headroom = filter_headroom(dictN[keys]['remaining_headroom'], exceed_headroom_a_new)
        if min_headroom is False:
            continue
        if request_data['use_nri_nae'].lower() == "nae":
            testcombo = NAEData[(NAEData["hproject_id"].isin(keys))]
            testnrinaebudget=naebudget

        elif request_data['use_nri_nae'].lower() == "nri":
            testcombo = NRIData[(NRIData["hproject_id"].isin(keys))]
            testnrinaebudget=nribudget      
        
        combo_filter_prop = True
        if request_data['mix_or_cagr'].lower() == "mix":
            accept_types = get_accept_types(testcombo, testnrinaebudget, 1)
            for idx, hprojectitem_id in testcombo.iterrows():
                if filter_prop(idx, testcombo, accept_types) is False:
                    combo_filter_prop = False
            if combo_filter_prop is False:
                continue

        elif request_data['mix_or_cagr'].lower() == "cagr":
            accept_types = get_accept_types_cagr(testcombo, testnrinaebudget)
            if set(cagr_filter_dict['region_code'][0]) != set(accept_types['region_code']):
                combo_filter_prop = False
            if combo_filter_prop is False:
                 continue
        accepted_combos[keys] =  dictN[keys]

def get_accept_combo2_mix(accepted_combos, dictN, NRI_NAEData, nrinaebudget):     
    for keys in dictN.keys():          
        testcombo = NRI_NAEData[(NRI_NAEData["hproject_id"].isin(keys))]
        testnrinaebudget=nrinaebudget   
        
        combo_filter_prop = True        
        accept_types = get_accept_types(testcombo, testnrinaebudget)
        for idx, hprojectitem_id in testcombo.iterrows():
            if filter_prop(idx, testcombo, accept_types) is False:
                combo_filter_prop = False
        if combo_filter_prop is False:
            continue
        accepted_combos[keys] =  dictN[keys]
        
def get_accept_combo2_cagr(accepted_combos, dictN, NRI_NAEData, nrinaebudget, cagr_idx):     
    for keys in dictN.keys():
        testcombo = NRI_NAEData[(NRI_NAEData["hproject_id"].isin(keys))]
        testnrinaebudget=nribudget      
        
        combo_filter_prop = True
        #print(testcombo, testnrinaebudget)
        accept_types = get_accept_types_cagr(testcombo, testnrinaebudget)
#         if set(cagr_filter_dict['region_code'][0]) != set(accept_types['region_code']):
#             combo_filter_prop = False
#         if combo_filter_prop is False:
#              continue
        #print(accept_types)
        for idx in cagr_idx: 
            if set(cagr_filter_dict[idx][0]) == set(accept_types[idx]):
                pass
            elif set(cagr_filter_dict[idx][0]) != set(accept_types[idx]):
                combo_filter_prop = False
                continue
        if combo_filter_prop is False:
             continue
                
        accepted_combos[keys] =  dictN[keys]
        
        
        
#20210902 full report

def get_accept_types_final(combo_detail, budget_data):
    region_code_prop_key, region_code_prop_val = get_proportion(
        np.concatenate([budget_data['region_code'], combo_detail['region_code']]),
        np.concatenate([budget_data.iloc[:, -1],
                        combo_detail.iloc[:, -1]  ]))
    
    segment_code_key, segment_code_prop_val = get_proportion(
        np.concatenate([budget_data['segment_code'], combo_detail['segment_code']]),
        np.concatenate([budget_data.iloc[:, -1],
                        combo_detail.iloc[:, -1]  ]))
    return region_code_prop_val, segment_code_prop_val

def get_accept_types_final_NAE(combo_detail, budget_data):
    region_code_prop_key, region_code_prop_val = get_proportion(
        np.concatenate([budget_data['region_code'], combo_detail['region_code']]),
        np.concatenate([budget_data.iloc[:, -1],
                        combo_detail.iloc[:, -1]  ]))
    
    segment_code_key, segment_code_prop_val = get_proportion(
        np.concatenate([budget_data['segment_code'], combo_detail['segment_code']]),
        np.concatenate([budget_data.iloc[:, -1],
                        combo_detail.iloc[:, -1]  ]))
    return region_code_prop_val, segment_code_prop_val

def get_accept_types_final_NRI(combo_detail, budget_data):
    region_code_prop_key, region_code_prop_val = get_proportion(
        np.concatenate([budget_data['region_code'], combo_detail['region_code']]),
        np.concatenate([budget_data.iloc[:, -1],
                        combo_detail.iloc[:, -1]  ]))
    
    sector_code_key, sector_code_prop_val = get_proportion(
        np.concatenate([budget_data['sector_code'], combo_detail['sector_code']]),
        np.concatenate([budget_data.iloc[:, -1],
                        combo_detail.iloc[:, -1]  ]))
    return region_code_prop_val, sector_code_prop_val

def get_accept_types_cagr_final(combo_detail, budget_data):
    accept_types = {}
    for filter_key in cagr_filter_dict.keys():
        accept_types[filter_key] = []
        for key in range(0, len(cagr_filter_dict[filter_key][0])):
            testcombo1 = combo_detail.loc[combo_detail[filter_key] == cagr_filter_dict[filter_key][0][key]]
            nribudget1 = budget_data.loc[budget_data[filter_key] == cagr_filter_dict[filter_key][0][key]]
            A= np.concatenate([nribudget1.iloc[:, -1],  testcombo1.iloc[:, -1]  ]).sum()
            B = np.concatenate([nribudget1.iloc[:, -11],  testcombo1.iloc[:, -11]  ]).sum()
            if ((A <= 0) or (B <= 0)):
                #print(A,B)
                #CAGR = 'NA'
                CAGR = -999
                accept_types[filter_key] += [CAGR]
                continue 
            C = (A/B)
            CAGR = np.float_power(C, 1/10) - 1
            accept_types[filter_key] += [CAGR]
    return accept_types


def get_conn(
    host0=DATABASE_HOST,
    account=DATABASE_ACC,
    pw=DATABASE_PW,
    DBNAME=DATABASE_NAME, sql=[]
    ):
    conn = pymssql.connect(host=host0, user=account, password=pw,
                           database=DBNAME)
    cursor = conn.cursor()
    return conn,cursor

def trunc_table(tables):
    conn,cursor = get_conn()
    for table in tables:
            cursor.execute("truncate table %s;" % table)
            print("truncated table: %s" % table)
            conn.commit()
    cursor.close()

def get_column_list(table_name):
    conn,cursor = get_conn()
    cursor.execute("select * from %s;" % table_name)
    field_list = [i[0] for i in cursor.description]
    field_list = field_list[1:] # Remove the ID Field
    cursor.close()
    return field_list

def gen_sql_dict(pj_set, srt_no_pj,srt_no_pb_pj, srt_ttl_rm_hr, srt_cmt_tim,srt_nri,
                srt_nae,srt_trd_pft,srt_gri,srt_aup,cb_irr,npv,pj_trd_pft,
                invest_cost,rm_hr_t1,rm_hr_t2,rm_hr_t3,rm_hr_t4,rm_hr_t5,rm_hr_t6,
                rm_hr_t7,rm_hr_t8,rm_hr_t9,rm_hr_t10,rm_hr_t11,nri_t2,nri_t6,nri_t11,
                nae_t2,nae_t6,nae_t11,fc_t2,fc_t6,fc_t11,field_list):

    field_list = field_list
    
    values_list = [pj_set, srt_no_pj,srt_no_pb_pj, srt_ttl_rm_hr, srt_cmt_tim,srt_nri,
                srt_nae,srt_trd_pft,srt_gri,srt_aup,cb_irr,npv,pj_trd_pft,
                invest_cost,rm_hr_t1,rm_hr_t2,rm_hr_t3,rm_hr_t4,rm_hr_t5,rm_hr_t6,
                rm_hr_t7,rm_hr_t8,rm_hr_t9,rm_hr_t10,rm_hr_t11,nri_t2,nri_t6,nri_t11,
                nae_t2,nae_t6,nae_t11,fc_t2,fc_t6,fc_t11 ]
    
    res = {field_list[i]: values_list[i] for i in range(len(field_list))}

    return res

def gen_insert_union_sql(table_name,field_list,data_list):
    
    insertTitleStr=f"Insert into {table_name} (" + ','.join(field_list)+")\n"

    i=0        
    conn,cursor = get_conn()
    for row_dict in data_list:

        insertRowStrList=[]
        for columnName in field_list:
            columValue= row_dict[columnName]       
            queryStr=columnName+" = "+str(columValue)
            insertRowStrList.append(queryStr)
        insertRowStr=','.join(insertRowStrList)
        if i==0:
            insertStr="Select "+insertRowStr
        else:
            insertStr+="\n union all \nSelect "+insertRowStr
        i+=1
        if i%3000==0: # Commit Size
            sql=(insertTitleStr+'\n'+insertStr)
            cursor.execute(sql)
            conn.commit()
            i=0
    sql=(insertTitleStr+'\n'+insertStr)
    cursor.execute(sql)
    conn.commit()
    cursor.close()


def gen_insert_sql(table_name, item):
    #Obsoleted
    sql = "insert into %s(%s) values(%s)"
    keys = item.keys()
    key_str = ",".join(keys)
    value_str = ",".join(["%s" % v for k, v in item.items()])
    return sql % (table_name, key_str, value_str)



#import datetime
from scipy import optimize

def xnpv(rate,cashflows):
    chron_order = sorted(cashflows, key = lambda x: x[0])
    t0 = 0
    return sum([cf/(1+rate)**((t-t0)) for (t,cf) in enumerate(chron_order[0])])

def xirr_raw(cashflows,guess=0.1):
    return optimize.newton(lambda r: xnpv(r,cashflows),guess)

def get_all_full_report(request_data):
    
    truntbl=[]
    truntbl.append('[dbo].[bga_hrj_CalculationResult]')
    trunc_table(tables=truntbl)

    insrt_tbl_name = '[dbo].[bga_hrj_CalculationResult]'
    field_list = get_column_list(insrt_tbl_name)
    insertTitleStr=f"Insert into {insrt_tbl_name} (" + ','.join(field_list)+")\n"
    print(insertTitleStr)

    must_include_projects = []
    must_not_include_projects = []
    for i in range(len(request_data['project_option'])):
        if (request_data['project_option'][i]['must_include'] == True):
            must_include_projects.append(request_data['project_option'][i]['project_id'])
        if (request_data['project_option'][i]['must_not_include'] == True):
            must_not_include_projects.append(request_data['project_option'][i]['project_id'])
        if (request_data['project_option'][i]['must_not_include'] == True) & (request_data['project_option'][i]['must_include'] == True):
            print('Must_include and Must_not_include can not exist concurrently')
        #return("message" + ":" + " Error, Must_include and Must_not_include exist concurrently")
    must_not_include_projects1 = tuple(must_not_include_projects)




    start_time0 = time.time()
    
    start_time = time.time()

    tracemalloc.start()
    request_data = request_data
    get_data(request_data)

    exceed_headroom = HeadroomItemData[HeadroomItemData.headroomitem_type == "allow_headroom_to_be_exceed"].fillna(0)[['t1', 't2', 't3','t4', 't5', 't6','t7', 't8', 't9','t10','t11']].values[0].tolist()
    exceed_headroom_a = np.array(exceed_headroom)
    exceed_headroom_a_new = [100000000000 if x==1 else x for x in exceed_headroom_a]

    # project_priority = {}
    # for i in range(len(request_data['project_option'])):
    #     project_priority[request_data['project_option'][i]['project_id']]= request_data['project_option'][i]['priority']   

    ProjectItemDataDict = ProjectItemData.groupby('hproject_id')['hprojectitem_id'].apply(list).to_dict()
    for key,value in ProjectItemDataDict.items():
         ProjectItemDataDict[key] = list(dict.fromkeys(ProjectItemDataDict[key]))

    elapsed_time = time.time() - start_time
    print("Get data need time:", elapsed_time)
    current, peak = tracemalloc.get_traced_memory()
    get_data_time = elapsed_time
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    get_data_mem_current =  current
    get_data_mem_peak =  peak
    tracemalloc.stop()
    
    f = open("running_log.txt", "a")
    now = datetime.now()
    f.write("\n")
    f.write(str(now))
    f.write("Now finished step 1")
    f.write("__")
    #f.write(str(must_in))
    f.close()

    tracemalloc.start()
    start_time = time.time()

    budget_remain = {}
    order_info = {}
    for i in unique_hprojectitem_id:
        npv_p = 0
        total_trading_profit_p = 0
        investment_cost_p = 0
        aup_p = 0
        gri_p = 0
        net_finance_charge_p = np.zeros(11)
        data_new_0 = ProjectItemData[ProjectItemData['hprojectitem_id'] == i].reset_index(drop = True)
        commitment_timing = data_new_0.loc[0, 'commitment_timing']
        hproject_type = data_new_0.loc[0, 'hproject_type']

        order_info_project_0 = {
            'npv':  npv_p,
            'total_trading_profit' : total_trading_profit_p,
            'investment_cost' : investment_cost_p,
            'net_finance_charge' : net_finance_charge_p,
            'aup':aup_p,
            'gri':gri_p,
            'commitment_timing' : commitment_timing,
            'probable_proj' : hproject_type
        }
        budget_remain[i] ,order_info[i]= get_basic_combos(i)

    elapsed_time = time.time() - start_time
    print("Get basic combos need time:", elapsed_time)
    current, peak = tracemalloc.get_traced_memory()
    get_basic_combos_time = elapsed_time
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    get_basic_combos_current =  current
    get_basic_combos_peak =  peak
    tracemalloc.stop()
                      
            
            
            
            
    tracemalloc.start()
    start_time = time.time()

    budget_remain_project = {}
    for i in list(ProjectItemDataDict.keys()):
        multi_budget_remain_project = np.zeros(11)
        if (len(ProjectItemDataDict[i]) == 1):
            budget_remain_project[tuple([i])] = {'min_cum_remain':  budget_remain[ProjectItemDataDict[i][0]], 
                                                 'remaining_headroom': budget_remain[ProjectItemDataDict[i][0]] +budget_array
                                                }
        else:
            lenSubProject = len(ProjectItemDataDict[i])
            for j in range(0, lenSubProject):
                multi_budget_remain_project += budget_remain[ProjectItemDataDict[i][j]]
            budget_remain_project[tuple([i])] = {'min_cum_remain':  multi_budget_remain_project, 
                                                 'remaining_headroom': multi_budget_remain_project +budget_array
                                                }
    order_info_project = {}
    for i in list(ProjectItemDataDict.keys()):
        npv_p = 0
        total_trading_profit_p = 0
        investment_cost_p = 0
        aup_p = 0
        gri_p = 0
        net_finance_charge_p = np.zeros(11)

        if (len(ProjectItemDataDict[i]) == 1):
            #budget_remain_project[i] = budget_remain[ProjectItemDataDict[i][0]]
            order_info_project[i] = order_info[ProjectItemDataDict[i][0]]

        elif (len(ProjectItemDataDict[i]) > 1):
            lenSubProject = len(ProjectItemDataDict[i])
            for j in range(0, lenSubProject):
                #print(i,j)
                npv_p += order_info[ProjectItemDataDict[i][j]]['npv']
                total_trading_profit_p += order_info[ProjectItemDataDict[i][j]]['total_trading_profit']
                investment_cost_p += order_info[ProjectItemDataDict[i][j]]['investment_cost']
                #NRI_p += order_info[ProjectItemDataDict[i][j]]['NRI']
                #NAE_p += order_info[ProjectItemDataDict[i][j]]['NAE']
                net_finance_charge_p += order_info[ProjectItemDataDict[i][j]]['net_finance_charge']
                aup_p += order_info[ProjectItemDataDict[i][j]]['aup']
                gri_p += order_info[ProjectItemDataDict[i][j]]['gri']
                #print(i,j)
            #budget_remain_project[i] = multi_budget_remain_project
            order_info_project[i] = {
                'npv':  npv_p, 
                'total_trading_profit' : total_trading_profit_p,
                'investment_cost' : investment_cost_p,
                'net_finance_charge' : net_finance_charge_p,
                'aup':aup_p,
                'gri':gri_p,
                'commitment_timing' : order_info[ProjectItemDataDict[i][j]]['commitment_timing'],
                'probable_proj' : order_info[ProjectItemDataDict[i][j]]['probable_proj']
            }
    #     try: 
    #         if project_priority[i] >= 0:
    #             order_info_project[i]['combination_priority'] = project_priority[i]
    #     except KeyError:
    #         order_info_project[i]['combination_priority'] = 999

    elapsed_time = time.time() - start_time
    print("Group Data to project level need time:", elapsed_time)
    current, peak = tracemalloc.get_traced_memory()
    project_level_data = elapsed_time
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    project_level_data_current =  current
    project_level_data_peak =  peak
    tracemalloc.stop()
        
        
        
        
    tracemalloc.start()
    start_time = time.time()


    combos_raw = {}
    combos_raw[1] = budget_remain_project
    #combos_raw1 = get_all_combos1(combos_raw)
    combos_raw1 = get_all_combos1_v0(combos_raw,must_include_projects, exceed_headroom_a_new)

    elapsed_time = time.time() - start_time
    print("Get All Combos need time:", elapsed_time)
    current, peak = tracemalloc.get_traced_memory()
    get_all_combos_time = elapsed_time
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    get_all_combos_current =  current
    get_all_combos_peak =  peak
    tracemalloc.stop()

    f = open("running_log.txt", "a")
    f.write("Now finished generate all combo:")
    f.write(str(get_all_combos_peak))
    f.write("__")
    f.write(str(get_all_combos_time))
    f.close()

    tracemalloc.start()
    start_time = time.time()
    

    combos = {}
    for i in combos_raw1.keys():
        for j in combos_raw1[i].keys():
#             if not filter_must(j, must_include_projects): # if not all project must, continue
#                 continue
            min_headroom = filter_headroom(combos_raw1[i][j]['remaining_headroom'], exceed_headroom_a_new)
            if min_headroom is False:
                continue
            combos[j] = combos_raw1[i][j]

    filter_active = False
    cagr_idx = []
    for idx, val in request_data['filter'].items():
        if val['active']:
            filter_active = True
    for idx, val in request_data['cagr_filter'].items():
        if val['active']:
            filter_active = True
            cagr_idx.append(idx)

    accepted_combos = combos
    print('nice')

    elapsed_time = time.time() - start_time
    print("Get Accepted Combos need time:", elapsed_time)
    current, peak = tracemalloc.get_traced_memory()
    get_accepted_combos_time = elapsed_time
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    get_accepted_combos_current =  current
    get_accepted_combos_peak =  peak
    tracemalloc.stop()

    f = open("running_log.txt", "a")
    f.write("Now finished filtering all combo:")
    f.write(str(get_accepted_combos_peak))
    f.write("_No of Accepted Combo:_")
    f.write(str(get_accepted_combos_time))
    f.write("__")
    f.write(str(len(accepted_combos)))
    f.write("\n")
    f.close()

    tracemalloc.start()
    start_timeB = time.time()
    print("Start_Final_Loop",start_timeB)
    
    res = []
    combination_output = {}
    list_ouput = []
    index = 1
    print("No. of accepted_combos:", len(accepted_combos))
    for key in accepted_combos.keys():
        npv_c = 0
        total_trading_profit_c = 0
        investment_cost_c = 0
        aup_c = 0
        gri_c = 0
        net_finance_charge_c = np.zeros(11)

        if (len(key) == 1):
            combination1 = order_info_project[key[0]]
            if combination1['probable_proj'] == 'Probable':
                probable_proj_count = 1
            elif combination1['probable_proj'] == 'Possible':
                probable_proj_count = 0
            try:
                IRR = IRR_raw[IRR_raw['hproject_id'] == key].reset_index(drop = True).iloc[:, -91:]*IRR_raw[IRR_raw['hproject_id'] == key].reset_index(drop = True)['hproject_ownership'][0]
                project_fcf_sum = np.sum(IRR, axis = 0)    
                project_fcf_sum = np.trim_zeros(project_fcf_sum, 'b')
                project_fcf_sum = np.expand_dims(project_fcf_sum, axis = 0)
                #combination_irr = xirr(np.array(dates[:len(project_fcf_sum.tolist()[0])]), np.array(project_fcf_sum))
                combination_irr = xirr_raw(project_fcf_sum,guess=0.1)
            except Exception as e:
                combination_irr = 0 
                f = open("running_log.txt", "a")
                f.write("IRR Error:")
                f.write(str(e))
                f.write("key: ")
                f.write(str(key))
                f.write("_")
                f.close()
                continue
            npv_c = combination1['npv']
            total_trading_profit_c = combination1['total_trading_profit']
            investment_cost_c = combination1['investment_cost']
            net_finance_charge_c = combination1['net_finance_charge']
            aup_c = combination1['aup']
            gri_c = combination1['gri']
            commitment_timing = int(combination1['commitment_timing'])

        elif (len(key) > 1):
            try:    
                TestM = []
                lenProject = len(key)
                probable_proj_count = 0
                commitment_timing = 0
                for j in range(0, lenProject):
                    npv_c += order_info_project[key[j]]['npv']
                    total_trading_profit_c += order_info_project[key[j]]['total_trading_profit']
                    investment_cost_c += order_info_project[key[j]]['investment_cost']
                    net_finance_charge_c += order_info_project[key[j]]['net_finance_charge']
                    aup_c += order_info_project[key[j]]['aup']
                    gri_c += order_info_project[key[j]]['gri']
                    #combination_priority += order_info_project[key[j]]['combination_priority']
                    if order_info_project[key[j]]['probable_proj'] == 'Probable':
                        probable_proj_count += 1
                    commitment_timing += int(order_info_project[key[j]]['commitment_timing'])
                    IRR_array = IRR_raw[IRR_raw['hproject_id'] == key[j]].reset_index(drop = True).iloc[:, -91:]*IRR_raw[IRR_raw['hproject_id'] == key[j]].reset_index(drop = True)['hproject_ownership'][0]
                    TestM.append(IRR_array)
                M1= np.concatenate(TestM)
                project_fcf_sum = np.sum(M1, axis = 0)    
                project_fcf_sum = np.trim_zeros(project_fcf_sum, 'b')
                project_fcf_sum = np.expand_dims(project_fcf_sum, axis = 0)
                #combination_irr = xirr(np.array(dates[:len(project_fcf_sum.tolist()[0])]), np.array(project_fcf_sum))
                combination_irr = xirr_raw(project_fcf_sum,guess=0.1)
                commitment_timing = round(commitment_timing/lenProject, 2)
            except Exception as e:
                f = open("running_log.txt", "a")
                f.write("IRR Error:")
                f.write(str(e))
                f.write("key: ")
                f.write(str(key))
                f.write("_")
                f.close()
                continue
        try:  
             project_set =  accepted_combos[key]['project_set']
        except KeyError:
             project_set =  key
        project_set = list(project_set)
        for project in range(len(project_set)):
            project_set[project] = str(project_set[project])
        project_set_string = re.sub('[\[\]\']', '', str(project_set))
        NRI_c0 = NRIData[(NRIData["hproject_id"].isin(key))]
        NRI_c = np.array(NRI_c0.iloc[:,-11:].sum())
        NAE_c0 = NAEData[(NAEData["hproject_id"].isin(key))]
        NAE_c = np.array(NAE_c0.iloc[:,-11:].sum())
        
        
        
        
#         #Combination_into_db
#         pj_set_insert ="'" + ",".join(project_set)  + "'"
#         rm_hdrm_insert = list(accepted_combos[key]['remaining_headroom'])
#         nri_insert=list(NRI_c)
#         nae_insert=list(NAE_c)
#         net_fc_insert=list(net_finance_charge_c)
#          # the Insertion
#         result = gen_sql_dict(pj_set=pj_set_insert, 
#                        srt_no_pj=len(project_set),
#                        srt_no_pb_pj=probable_proj_count,
#                        srt_ttl_rm_hr=sum(list(accepted_combos[key]['remaining_headroom'])[:5])*-1, 
#                        srt_cmt_tim=commitment_timing*-1,
#                        srt_nri=NRI_c.sum(),
#                        srt_nae=NAE_c.sum(),
#                        srt_trd_pft=total_trading_profit_c,
#                        srt_gri=gri_c,
#                        srt_aup=aup_c,
#                        cb_irr=combination_irr,
#                        npv=npv_c,pj_trd_pft=total_trading_profit_c,
#                        invest_cost=investment_cost_c,
#                        rm_hr_t1=rm_hdrm_insert[0],
#                        rm_hr_t2=rm_hdrm_insert[1],
#                        rm_hr_t3=rm_hdrm_insert[2],
#                        rm_hr_t4=rm_hdrm_insert[3],
#                        rm_hr_t5=rm_hdrm_insert[4],
#                        rm_hr_t6=rm_hdrm_insert[5],
#                        rm_hr_t7=rm_hdrm_insert[6],
#                        rm_hr_t8=rm_hdrm_insert[7],
#                        rm_hr_t9=rm_hdrm_insert[8],
#                        rm_hr_t10=rm_hdrm_insert[9],
#                        rm_hr_t11=rm_hdrm_insert[10],
#                        nri_t2=nri_insert[1],
#                        nri_t6=nri_insert[5],
#                        nri_t11=nri_insert[10],
#                        nae_t2=nae_insert[1],
#                        nae_t6=nae_insert[5],
#                        nae_t11=nae_insert[10],
#                        fc_t2=net_fc_insert[1],
#                        fc_t6=net_fc_insert[5],
#                        fc_t11=net_fc_insert[10],
#                        field_list=field_list)
#         res.append(result)
#         gen_insert_union_sql(insrt_tbl_name,field_list,res)
        #print('done')
        
#         print(key)
#         print(NRI_c0)
#         print(nribudget)
        
        NRI_CAGR = get_accept_types_cagr_final(NRI_c0, nribudget)
        NAE_CAGR = get_accept_types_cagr_final(NAE_c0, naebudget)
        #print(NRI_CAGR)
        
        #NRI_PerMix_region, NRI_PerMix_segment = get_accept_types_final_NRI(NRI_c0, nribudget)
        NRI_PerMix_region, NRI_PerMix_sector = get_accept_types_final_NRI(NRI_c0, nribudget)
        NAE_PerMix_region, NAE_PerMix_segment = get_accept_types_final_NAE(NAE_c0, naebudget)
        
        if (index == int(round(len(accepted_combos)/4))):
            time_now = time.time()
            print("Yeah, 1/4 done", time.asctime( time.localtime(time_now)))
        elif (index == int(round(len(accepted_combos)/2))):
            time_now = time.time()
            print("Yeah, half done", time.asctime( time.localtime(time_now)))            
        elif (index == int(round(len(accepted_combos)*3/4))):
            time_now = time.time()
            print("Yeah, 3/4 done", time.asctime( time.localtime(time_now)))     
        elif (index == int(round(len(accepted_combos)*9/10))):
            time_now = time.time()
            print("Yeah, 90% done", time.asctime( time.localtime(time_now)))               
        #project_set_string = re.sub('[\[\]\']', '', project_set)
        
        combination={
            "combination_id": index,
            "project_set": project_set_string,#project_set,
            "Sorting": {
                "num_of_proj": len(project_set),
                "num_of_probable_proj": probable_proj_count,
                "total_remaining_headroom": sum(list(accepted_combos[key]['remaining_headroom'])[1:6]),
                #"total_remaining_headroom": sum(list(accepted_combos[key]['remaining_headroom'])[:5])*-1,
                #"commitment_timing" : commitment_timing*-1,
                "commitment_timing" : commitment_timing,
                "nri": NRI_c.sum(),
                "nae": NAE_c.sum(),
                "trading_profit": total_trading_profit_c,
                "gri": gri_c,
                "aup" : aup_c
            },
            "combination_irr": combination_irr,
            "NPV": npv_c,
            "project_trading_profit" : total_trading_profit_c,
            "investment_cost" : investment_cost_c,
            "remaining_headroom": list(accepted_combos[key]['remaining_headroom']),
            "NRI": list(NRI_c),
            "NAE": list(NAE_c),
            "finance_charge": list(net_finance_charge_c),
            "NRI_CAGR_region": list(NRI_CAGR['region_code']),
            "NRI_CAGR_sector": list(NRI_CAGR['sector_code']),
            "NAE_CAGR_region": list(NAE_CAGR['region_code']),
            "NAE_CAGR_segment": list(NAE_CAGR['segment_code']),
            "NRI_PerMix_region": list(NRI_PerMix_region),
            "NRI_PerMix_sector": list(NRI_PerMix_sector),
            "NAE_PerMix_region": list(NAE_PerMix_region),
            "NAE_PerMix_segment": list(NAE_PerMix_segment),
        }    
        index+=1
        list_ouput.append(combination)

    combination_output = {"combinations": list_ouput}
    
    result0 = combination_output['combinations']
    resultoutput = json_normalize(result0)

    resultoutput[['rh_y1','rh_y2','rh_y3','rh_y4','rh_y5','rh_y6','rh_y7','rh_y8','rh_y9','rh_y10','rh_y11']] = resultoutput['remaining_headroom'].apply(pd.Series)
    resultoutput[['nri_y1','nri_y2','nri_y3','nri_y4','nri_y5','nri_y6','nri_y7','nri_y8','nri_y9','nri_y10','nri_y11']] = resultoutput['NRI'].apply(pd.Series)
    resultoutput[['nae_y1','nae_y2','nae_y3','nae_y4','nae_y5','nae_y6','nae_y7','nae_y8','nae_y9','nae_y10','nae_y11']] = resultoutput['NAE'].apply(pd.Series)
    resultoutput[['fc_y1','fc_y2','fc_y3','fc_y4','fc_y5','fc_y6','fc_y7','fc_y8','fc_y9','fc_y10','fc_y11']] = resultoutput['finance_charge'].apply(pd.Series)

    resultoutput[['cm_nri_cagr','hk_nri_cagr','others_nri_cagr','us_nri_cagr']] = resultoutput['NRI_CAGR_region'].apply(pd.Series)
    resultoutput[['office_nri_cagr','residential_nri_cagr','retail_nri_cagr']] = resultoutput['NRI_CAGR_sector'].apply(pd.Series)
    resultoutput[['cm_nae_cagr','hk_nae_cagr','others_nae_cagr','us_nae_cagr']] = resultoutput['NAE_CAGR_region'].apply(pd.Series)
    resultoutput[['hotel_nae_cagr','ip_nae_cagr','tp_nae_cagr']] = resultoutput['NAE_CAGR_segment'].apply(pd.Series)

    resultoutput[['cm_nri_pct','hk_nri_pct','others_nri_pct','us_nri_pct']] = resultoutput['NRI_PerMix_region'].apply(pd.Series)
    resultoutput[['office_nri_pct','residential_nri_pct','retail_nri_pct']] = resultoutput['NRI_PerMix_sector'].apply(pd.Series)
    resultoutput[['cm_nae_pct','hk_nae_pct','others_nae_pct','us_nae_pct']] = resultoutput['NAE_PerMix_region'].apply(pd.Series)
    resultoutput[['hotel_nae_pct','ip_nae_pct','tp_nae_pct']] = resultoutput['NAE_PerMix_segment'].apply(pd.Series)

    resultoutput[['num_of_proj',  'num_of_probable_proj', 'total_remaining_headroom',
           'commitment_timing', 'nri', 'nae',
           'trading_profit', 'gri', 'aup']]   = resultoutput[['Sorting.num_of_proj',  'Sorting.num_of_probable_proj', 'Sorting.total_remaining_headroom',
           'Sorting.commitment_timing', 'Sorting.nri', 'Sorting.nae',
           'Sorting.trading_profit', 'Sorting.gri', 'Sorting.aup']] 
    #drop too slow
    #resultoutput.drop(['remaining_headroom', 'NRI', 'NAE','finance_charge', 'NRI_CAGR_region', 'NRI_CAGR_sector', 'NAE_CAGR_region', 'NAE_CAGR_segment', 'NRI_PerMix_region', 'NRI_PerMix_sector', 'NAE_PerMix_region', 'NAE_PerMix_segment'], axis=1, inplace=True)
    resultoutput1 = resultoutput[['combination_id', 'project_set', 'combination_irr', 'NPV',
           'project_trading_profit', 'investment_cost', 'num_of_proj',
           'num_of_probable_proj', 'total_remaining_headroom',
           'commitment_timing', 'nri', 'nae',
           'trading_profit', 'gri', 'aup', 'rh_y1',
           'rh_y2', 'rh_y3', 'rh_y4', 'rh_y5', 'rh_y6', 'rh_y7', 'rh_y8', 'rh_y9',
           'rh_y10', 'rh_y11', 'nri_y1', 'nri_y2', 'nri_y3', 'nri_y4', 'nri_y5',
           'nri_y6', 'nri_y7', 'nri_y8', 'nri_y9', 'nri_y10', 'nri_y11',
           'nae_y1', 'nae_y2', 'nae_y3', 'nae_y4', 'nae_y5', 'nae_y6',
           'nae_y7', 'nae_y8', 'nae_y9', 'nae_y10', 'nae_y11', 'fc_y1',
           'fc_y2', 'fc_y3', 'fc_y4', 'fc_y5', 'fc_y6', 'fc_y7', 'fc_y8',
           'fc_y9', 'fc_y10', 'fc_y11', 'cm_nri_cagr', 'hk_nri_cagr',
           'others_nri_cagr', 'us_nri_cagr', 'office_nri_cagr',
           'residential_nri_cagr', 'retail_nri_cagr', 'cm_nae_cagr',
           'hk_nae_cagr', 'others_nae_cagr', 'us_nae_cagr',
           'hotel_nae_cagr', 'ip_nae_cagr', 'tp_nae_cagr',
           'cm_nri_pct', 'hk_nri_pct', 'others_nri_pct',
           'us_nri_pct', 'office_nri_pct', 'residential_nri_pct',
           'retail_nri_pct', 'cm_nae_pct', 'hk_nae_pct',
           'others_nae_pct', 'us_nae_pct', 'hotel_nae_pct',
           'ip_nae_pct', 'tp_nae_pct']]
    
    #'DRIVER={FreeTDS};'
    connection_string = (
    'Driver={ODBC Driver 17 for SQL Server};'
    'SERVER=tcp:aibg-dev-dbserver.database.windows.net,1433;'
    'Database=aibg-dev-db;'
    'UID=aibgdevadmin;'
    'PWD={Jh8*E\WJ;TcSMM4E};'
    'Trusted_Connection=no;'
    )
    connection_uri = f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(connection_string)}"
    engine = sa.create_engine(connection_uri, fast_executemany=True)

    with engine.begin() as conn:
        conn.execute(sa.text("SELECT TOP(50) * FROM [aibg-dev-db].dbo.t20210904"))
    conn
    resultoutput1.to_sql("bga_hrj_CalculationResult_0909", engine, schema="dbo", if_exists="append", index=False)
        
    elapsed_time = time.time() - start_timeB
    print("Get Combos Info need time:", elapsed_time)
    current, peak = tracemalloc.get_traced_memory()
    get_combos_info_time = elapsed_time
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    get_combos_info_current =  current
    get_combos_info_peak =  peak
    tracemalloc.stop()

    f = open("running_log.txt", "a")
    f.write("Now finished all combo info and sorting:")
    f.write(str(get_combos_info_peak))
    f.write("__")
    f.write(str(get_combos_info_time))
    f.close()

    endtime0 = time.time() - start_time0
    return combination_output, endtime0, get_data_time, get_basic_combos_time, project_level_data, get_all_combos_time, get_accepted_combos_time, get_combos_info_time, get_data_mem_peak,  get_basic_combos_peak, project_level_data_peak,  get_all_combos_peak, get_accepted_combos_peak,  get_combos_info_peak

#Edit here:
#import json
#f = open('RealInput.json')
#request_data = json.load(f)
#combination_output, endtime0, get_data_time, get_basic_combos_time, project_level_data, get_all_combos_time, get_accepted_combos_time, get_combos_info_time, get_data_mem_peak,  get_basic_combos_peak, project_level_data_peak,  get_all_combos_peak, get_accepted_combos_peak,  get_combos_info_peak = get_all4_a(request_data)
#combination_output
#endtime

