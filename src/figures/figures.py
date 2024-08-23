import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import networkx as nx
import geopandas as gpd
import os
import glob

import itertools

from src.optimization_model.optimization import OptimizationModel
from src.others.variable_declaration import (inorganic_chemical_list, position_dict_all, product_list_ordered,
                                             other_intermediate_list_up, other_intermediate_list_down,
                                             btx_list, sector_subsector_dict, final_product_list,
                                             co2_feedstock_list, co2_emission_list, other_raw_material_list,
                                             carbon_content_dict, polymer_source_list1, polymer_source_list2,
                                             residue_list_code
                                             )
from src.data_preparation.master_file import (sensitivity_demand, regional_results, sensitivity_ele_demand,
                                              different_scenarios, MasterFile, system_contribution_analysis
                                              )

colors_4 = ['#838BC0', '#BDCE37', '#F9EBA2', '#53833B']
colors_5 = ['#B84426', '#838BC0', '#BDCE37', '#F9EBA2', '#53833B']
colors_5_with_grey = ['#838BC0', '#BDCE37', '#F9EBA2', '#53833B', '#d8d8d8']
colors_6 = ['#838BC0', '#BDCE37', '#F9EBA2', '#53833B', '#40419A', '#B84426']
colors_7 = ['#838BC0', '#BDCE37', '#F9EBA2', '#53833B', '#40419A', '#B84426', '#d8d8d8']
colors_26 = ['#838bc0', '#9ace37', '#dff9a2', '#3b8355', '#92409a', '#84b826', '#c083b1',
             '#37cea5', '#a2f9de', '#3b4d83', '#9a5b40', '#26b893', '#c0aa83', '#4837ce',
             '#a8a2f9', '#833b7a', '#5d9a40', '#3926b8', '#83c084', '#ce3783', '#f9a2d1',
             '#83603b', '#40949a', '#b8266d', '#83a7c0', '#cebc37']


def cmp_yellow_orange():
    vals = np.ones((256, 4))
    vals[:, 0] = np.concatenate((np.linspace(252 / 256, 226 / 256, 100),
                                 np.linspace(226 / 256, 184 / 256, 156)))
    vals[:, 1] = np.concatenate((np.linspace(243 / 256, 181 / 256, 100),
                                 np.linspace(181 / 256, 68 / 256, 156)))
    vals[:, 2] = np.concatenate((np.linspace(202 / 256, 151 / 256, 100),
                                 np.linspace(151 / 256, 38 / 256, 156)))

    newcmp = ListedColormap(vals)
    return newcmp


def cmp_purple():
    vals = np.ones((256, 4))
    vals[:, 0] = (np.linspace(39 / 256, 230 / 256, 256))
    vals[:, 1] = (np.linspace(67 / 256, 231 / 256, 256))
    vals[:, 2] = (np.linspace(146 / 256, 242 / 256, 256))
    newcmp = ListedColormap(vals)
    return newcmp

def cmp_heatmap():
    # Define your colors in hex
    colors_hex = ["#FCF1C1", "#F9EBA2", "#B84426", "#17183A"]

    # Convert hex to RGB and normalize
    colors_rgb = [tuple(int(color[i:i + 2], 16) for i in (1, 3, 5)) for color in colors_hex]
    colors_normalized = [(r / 255, g / 255, b / 255) for r, g, b in colors_rgb]

    # Create the colormap
    custom_colormap = LinearSegmentedColormap.from_list("my_custom_cmap", colors_normalized, N=256)
    custom_colormap.set_under('#E5ECBB')
    return custom_colormap

def df_sankey_flow(model, objective):
    df0, df_flow_result = model.single_objective_optimization_full_outputs(objective)
    df = df0.copy()
    columns_to_modify = ['product_from', 'product_to']
    substrings_to_replace = ['_bs', '_bl', '_f', '_liquid', '_low', '_high']
    for col in columns_to_modify:
        df.loc[df[col].isin(inorganic_chemical_list), col] = "other_inorganic_chemicals"
        df.loc[df[col].isin(other_intermediate_list_up), col] = "other_intermediates_upstream"
        df.loc[df[col].isin(other_intermediate_list_down), col] = "other_intermediates_downstream"
        for substring in substrings_to_replace:
            pattern = rf'{substring}$'
            df.loc[(df[col].str.contains(substring, na=False)) &
                   (~df[col].str.contains('co2', na=False)) &
                   (~df[col].str.contains('flexible', na=False)), col] = df[col].str.replace(pattern, '', regex=True)
    df = df.loc[df['product_from'] != df['product_to']].copy()
    df = df[df.product_name != "cooling_water_kg"].copy()
    df = df[df.product_name != "water"].copy()
    df["flowxcc"] = df["flow_amount"] * df["cc_product"]
    df['flowxbdv'] = df['flow_amount'] * df['bdv_product']
    df['flowxcarbon'] = df['flow_amount'] * df['carbon_content']
    df = pd.pivot_table(df, index=['product_from', 'product_to', 'unit'],
                        values=['flow_amount', 'flowxcc', 'flowxbdv', 'flowxcarbon'],
                        aggfunc='sum').reset_index()
    df1 = df[df.flowxcc > 0.01].copy()
    dfc = df[df.flowxcarbon >= 0].copy()
    return df1, dfc


def sankey_flow_impacts(model, objective):
    df0 = df_sankey_flow(model, objective)[0]
    df = df0.copy()
    df_from = df.groupby(by=['product_from', 'unit']).sum(numeric_only=True).reset_index()
    sorter_dict = dict(zip(product_list_ordered, range(len(product_list_ordered))))
    df['product_from_rank'] = df['product_from'].map(sorter_dict)
    df['product_to_rank'] = df['product_to'].map(sorter_dict)
    df.sort_values(['product_from_rank', 'product_to_rank'], ascending=[True, True], inplace=True)
    df.drop(['product_from_rank', 'product_to_rank'], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    product_list = [item for item in product_list_ordered if item in df['product_from'].unique()
                    or item in df["product_to"].unique()]
    product_index = list(range(len(product_list)))
    product_node_dict = dict(zip(product_list, product_index))
    x_position_dict = position_dict_all()[0]
    y_position_dict = position_dict_all()[1]
    df_product = pd.DataFrame(product_list, columns=['product_name'])

    df_product['x_position'] = df_product['product_name'].map(x_position_dict)
    df_product['y_position'] = df_product['product_name'].map(y_position_dict)
    df_product['product_code'] = df_product['product_name'].map(product_node_dict)
    df['product_code_from'] = df['product_from'].map(product_node_dict)
    df['product_code_to'] = df['product_to'].map(product_node_dict)
    node = dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=list(df_product.product_name),
        x=list(df_product.x_position),
        y=list(df_product.y_position),
        color="blue"
    )
    link_ghg = dict(
        source=df['product_code_from'],
        target=df['product_code_to'],
        value=df['flowxcc']
    )
    link_bdv = dict(
        source=df['product_code_from'],
        target=df['product_code_to'],
        value=df['flowxbdv']
    )
    fig = go.Figure(data=[go.Sankey(
        node=node,
        link=link_ghg,
    )])
    fig.update_layout(title_text=f"Objective: {objective}, climate change impact flows", font_size=10)
    fig.show()
    fig = go.Figure(data=[go.Sankey(
        node=node,
        link=link_bdv,
    )])
    fig.update_layout(title_text=f"Objective: {objective}, biodiversity loss impact flows", font_size=10)
    fig.show()
    return df


def sankey_flow_carbon(model, objective):
    df0 = df_sankey_flow(model, objective)[1]
    df = df0.copy()
    for col in ['product_from', 'product_to']:
        # df.loc[df[col].isin(final_product_list), col] = "plastics"
        # df.loc[df[col].isin(agricultural_residue_list), col] = "agricultural_residue"
        df.loc[df[col].isin(['other_intermediates_downstream', 'other_intermediates_upstream',
                             'butadiene']), col] = "other_intermediates"
        df.loc[df[col].isin(co2_feedstock_list), col] = "co2_feedstock"
        df.loc[df[col].isin(co2_emission_list), col] = "co2_emission"
        df.loc[df[col].isin(other_raw_material_list), col] = "other_raw_materials"
        df.loc[df[col].isin(btx_list), col] = "btx"

    for sector in sector_subsector_dict.keys():
        subsector_list = sector_subsector_dict[sector]
        for subsector in subsector_list:
            df.loc[df['product_from'].str.contains(subsector), 'product_from'] = sector
            df.loc[df['product_to'].str.contains(subsector), 'product_to'] = sector
    df = df[df['product_from'] != df['product_to']].copy()
    df = df[df.flowxcarbon > 0.001].copy()
    df = df.groupby(by=['product_from', 'product_to', 'unit']).sum(numeric_only=True).reset_index()
    df.to_csv('data/intermediate/sankey_flow_carbon.csv', index=False)
    product_list_ordered_1 = list(set(df['product_from'].unique()).union(set(df['product_to'].unique())))
    sorter_dict = dict(zip(product_list_ordered_1, range(len(product_list_ordered_1))))
    df['product_from_rank'] = df['product_from'].map(sorter_dict)
    df['product_to_rank'] = df['product_to'].map(sorter_dict)
    df.sort_values(['product_from_rank', 'product_to_rank'], ascending=[True, True], inplace=True)
    df.drop(['product_from_rank', 'product_to_rank'], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    product_list = [item for item in product_list_ordered_1 if item in df['product_from'].unique()
                    or item in df["product_to"].unique()]
    product_index = list(range(len(product_list)))
    product_node_dict = dict(zip(product_list, product_index))
    x_position_dict = position_dict_all()[0]
    y_position_dict = position_dict_all()[1]
    df_product = pd.DataFrame(product_list, columns=['product_name'])

    df_product['x_position'] = df_product['product_name'].map(x_position_dict)
    df_product['y_position'] = df_product['product_name'].map(y_position_dict)
    df_product['product_code'] = df_product['product_name'].map(product_node_dict)
    df['product_code_from'] = df['product_from'].map(product_node_dict)
    df['product_code_to'] = df['product_to'].map(product_node_dict)
    node = dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=list(df_product.product_name),
        # x=list(df_product.x_position),
        # y=list(df_product.y_position),
        color="blue"
    )
    link_carbon = dict(
        source=df['product_code_from'],
        target=df['product_code_to'],
        value=df['flowxcarbon']
    )
    fig = go.Figure(data=[go.Sankey(
        node=node,
        link=link_carbon,
    )])
    fig.update_layout(title_text=f"Objective: {objective}, climate change impact flows", font_size=10)
    fig.show()
    return df


def product_impact_plot(model, objective):
    df_ghg = model.calculate_product_impacts(objective)[1]
    df_ghg.loc[df_ghg.flow_amount < 0.001, 'flow_amount'] = 0
    # df_ghg = sankey_flow_impacts(year, scenario, 'GHG', country, file_path)
    # df_bdv = sankey_flow_impacts(year, scenario, 'Biodiversity', country, file_path)
    df_ghg_product = df_ghg.loc[df_ghg.value == 1].copy()
    # df_bdv_product = df_bdv.loc[df_bdv.value == 1].copy()
    product_list_to_check = ['hydrogen', 'methanol_bs', 'methanol_bl', 'methanol_f', 'ethanol_bs', 'ethanol_bl',
                             'ethylene', 'propylene']
    product_list_to_check = ['hdpe', 'ldpe', 'pp', 'pvc', 'gpps', 'hips', 'pet', 'pur_flexible', 'pur_rigid',
                             'pla', 'pbs']

    for product in product_list_to_check:
        df_temp = df_ghg_product.loc[df_ghg_product.product_name.isin([f'{product}{suffix}' for suffix in
                                     polymer_source_list2])].copy()
        df_temp1 = pd.DataFrame()
        for suffix in polymer_source_list1:
            df_temp2 = df_temp.loc[df_temp.product_name == f'{product}{suffix}'].copy()
            if df_temp.loc[df_temp.product_name == f'{product}{suffix}', 'flow_amount'].sum() == 0:
                df_temp2 = df_temp2.loc[df_temp2.cc_process == df_temp2.cc_process.min()].copy()
            else:
                df_temp2 = df_temp2.loc[df_temp2.flow_amount > 0].copy()
            df_temp1 = pd.concat([df_temp1, df_temp2], ignore_index=True)
        df_temp2 = df_temp.loc[df_temp.product_name.str.contains('_mr')].copy()
        cc_impact = (df_temp2['flow_amount'] * df_temp2['cc_process']).sum() / df_temp2['flow_amount'].sum()
        flow = df_temp2['flow_amount'].sum()
        mr_row = pd.DataFrame({'product_name': [f'{product}_mr'], 'flow_amount': [flow], 'cc_process': [cc_impact]})
        df_temp1 = pd.concat([df_temp1, mr_row], ignore_index=True)
        fig, ax = plt.subplots(1, 1, figsize=(16, 8), squeeze=True)
        sns.barplot(x='product_name', y='cc_process', data=df_temp1, hue='flow_amount')
        handles, labels = ax.get_legend_handles_labels()
        # Format labels to have no decimal places and set them as the new labels
        new_labels = ['{:.0f}'.format(float(label)) for label in labels]
        # Create a new legend with the updated labels
        ax.legend(handles, new_labels, title="flow amount")
        plt.title(f'GHG impact of {product}')
        plt.show()
    df_ghg_plot = df_ghg_product.loc[df_ghg_product.product_name.isin(product_list_to_check)].copy()
    df_ghg_plot = df_ghg_plot.loc[(df_ghg_plot.flow_amount > 0.001) |
                                  (df_ghg_plot.process == "PHB, from glucose_bs fermentation") |
                                  (df_ghg_plot.process == "pef, from fdca") |
                                  (df_ghg_plot.process == "pbat, from aliphatic aromatic copolyester process")].copy()
    # df_bdv_plot = df_bdv_product.loc[df_bdv_product.product_name.isin(product_list_to_check)].copy()
    # order df according to product_list_to_check
    sorter_dict = dict(zip(product_list_to_check, range(len(product_list_to_check))))
    df_ghg_plot['product_rank'] = df_ghg_plot['product_name'].map(sorter_dict)
    df_ghg_plot.sort_values('product_rank', inplace=True)
    df_ghg_plot.drop('product_rank', axis=1, inplace=True)
    # plot GHG of products
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_ghg_plot['product_name'],
        y=df_ghg_plot['cc_process'],
        name='CC',
        marker_color='indianred'
    ))
    fig.show()

    return df_ghg_product


def product_consumption_analysis(model, objective, product_name):
    df_flow_result = model.single_objective_optimization_simple_outputs(objective)[1]
    df = df_flow_result[df_flow_result.product_name == product_name].copy()
    df = df[df.flow_amount > 0.001].copy()
    df = df[df['type'] != 'PRODUCT'].copy()
    df['product_flow'] = df['flow_amount'] * df['value']
    flow_sum = df['product_flow'].sum()
    cut_off = abs(flow_sum * 0.025)
    df1 = df[['process', 'product_flow']].copy()
    df2 = df1[(df1.product_flow <= cut_off) & (df1.product_flow >= -cut_off)].copy()
    df1 = df1[(df1.product_flow < -cut_off) | (df1.product_flow > cut_off)].copy()
    other_amount = df2.product_flow.sum()
    df_temp = pd.DataFrame({'process': ['other'], 'product_flow': [other_amount]})
    df1 = pd.concat([df1, df_temp], ignore_index=True)
    df1['product_flow'] *= -1
    df1 = df1.sort_values('product_flow', ascending=False).reset_index(drop=True)
    df1['cumulative'] = df1['product_flow'].cumsum()
    bottom = [0]
    height = [df1['product_flow'][0]]
    for i in range(1, len(df1)):
        product_flow = df1.loc[i, 'product_flow']
        if product_flow >= 0:
            bottom.append(df1.loc[i - 1, 'cumulative'])
            height.append(product_flow)
        else:
            bottom.append(df1.loc[i - 1, 'cumulative'] + product_flow)
            height.append(-product_flow)
    df1['bottom'] = bottom
    df1['height'] = height
    df1['process_short'] = df1['process'].str.split(',').str[0]
    df1.loc[df1.process.str.contains('mechanical recycling'), 'process_short'] += '_r'
    df1['color'] = ['red' if product_flow < 0 else 'dodgerblue' for product_flow in df1['product_flow']]
    fig, ax = plt.subplots(1, 1, figsize=(12, 5), squeeze=True)
    ax.bar(df1['process_short'], df1['height'], bottom=df1['bottom'], color=df1['color'])
    if product_name == 'electricity':
        label = f'{product_name} consumption (TWh)'
    else:
        label = f'{product_name} consumption (Mt)'
    plt.ylabel(label)
    plt.show()
    return df1


def feedstock_analysis(model, objective):
    df_flow_result = model.single_objective_optimization_full_outputs(objective)[1]
    df_agri = df_flow_result[df_flow_result.product_name.isin(['agricultural_residue'])].copy()
    df_agri = df_agri[(df_agri.value < 0) & (df_agri.flow_amount > 0.0001)].copy()
    df_forest = df_flow_result[df_flow_result.product_name.isin(['forest_residue'])].copy()
    df_forest = df_forest[(df_forest.value < 0) & (df_forest.flow_amount > 0.0001)].copy()
    df_co2 = df_flow_result[df_flow_result.product_name.str.contains('co2_feedstock')].copy()
    df_co2 = df_co2[df_co2.flow_amount > 0.0001].copy()
    df_plastics = df_flow_result[df_flow_result.process.str.contains('mechanical recycling')].copy()
    df_plastics = df_plastics[df_plastics.type == 'PRODUCT'].copy()
    c_agri = -(df_agri['value'] * df_agri['flow_amount']).sum() * 0.494
    c_forest = -(df_forest['value'] * df_forest['flow_amount']).sum() * 0.521
    c_co2 = -(df_co2['value'] * df_co2['flow_amount']).sum() * 12 / 44
    df_plastics['carbon_content'] = df_plastics['product_name'].map(carbon_content_dict)
    c_plastics = (df_plastics['flow_amount'] * df_plastics['carbon_content']).sum()
    df = pd.DataFrame(
        {'feedstock': ['plastics_mechanical_recycled', 'agricultural_residue', 'forest_residue', 'co2_feedstock'],
         'carbon_input': [c_plastics, c_agri, c_forest, c_co2]})
    df_sorted = df.sort_values('carbon_input', ascending=True).reset_index(drop=True)
    df_neg = df_sorted[df_sorted['carbon_input'] < 0].copy()
    df_pos = df_sorted[df_sorted['carbon_input'] > 0].copy()
    fig, ax = plt.subplots(1, 1, figsize=(9, 9), squeeze=True)
    bottom = 0
    bars = []
    for index, row in df_neg.iterrows():
        bars.append(ax.bar('1', row['carbon_input'], bottom=bottom, label=row['feedstock']))
        bottom += row['carbon_input']
    bottom = 0
    for index, row in df_pos.iterrows():
        bars.append(ax.bar('1', row['carbon_input'], bottom=bottom, label=row['feedstock']))
        bottom += row['carbon_input']
    ax.legend()
    plt.ylabel('Feedstock (Mt C)')
    plt.show()
    return 0


def plot_ele_sensitivity(user_input):
    df = user_input.sensitivity_ele_impact('GHG')
    # plot carbon input
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), squeeze=True)
    ax.stackplot(df['ele_impact'], df['c_agri'], df['c_forest'],
                 df['c_co2'],
                 df['c_fossil'], labels=['agricultural residue', 'forest residue', 'co2', 'fossil fuel'],
                 colors=colors_5[::-1])
    ax.set_xlim(0.001, 0.4)
    ax.set_xlabel('Electricity impact (kg CO2eq/kWh)')
    ax.set_ylabel('Carbon input (Mt)')
    ax.legend()
    plt.title('Carbon Source')
    plt.grid()
    plt.show()

    # plot biomass use
    dfp = df[[x for x in df.columns if 'agricultural_residue_to' in x or 'forest_residue_to' in x]].copy()
    dfp['agricultural_residue_other'] = df['c_agri'] / 0.494 - dfp[[x for x in dfp.columns if 'agricultural_residue_to' in x]].sum(axis=1)
    dfp['forest_residue_other'] = df['c_forest'] / 0.521 - dfp[[x for x in dfp.columns if 'forest_residue_to' in x]].sum(axis=1)
    dfp['ele_impact'] = df['ele_impact']
    dfp2 = pd.DataFrame()
    dfp2['to_ethanol'] = dfp['agricultural_residue_to_ethanol'] + dfp['forest_residue_to_ethanol'] + \
                         dfp['forest_residue_to_glucose'] + dfp['agricultural_residue_to_glucose']
    dfp2['to_methanol'] = dfp['agricultural_residue_to_methanol'] + dfp['forest_residue_to_methanol']
    dfp2['to_lactic_acid'] = dfp['agricultural_residue_to_lactic acid'] + dfp['forest_residue_to_lactic acid']
    dfp2['to_syngas'] = dfp['forest_residue_to_syngas'] + dfp['agricultural_residue_to_syngas']
    dfp2['to_heat'] = dfp['agricultural_residue_to_heat'] + dfp['forest_residue_to_heat']
    dfp2['to_electricity'] = dfp['agricultural_residue_to_electricity'] + dfp['forest_residue_to_electricity']
    dfp2['ele_impact'] = df['ele_impact']
    dfp = dfp.loc[:, (dfp > 0.01).any(axis=0)].copy()
    dfp2 = dfp2.loc[:, (dfp2 > 0.01).any(axis=0)].copy()
    dfp = dfp2.copy()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), squeeze=True)
    ax.stackplot(dfp['ele_impact'], dfp.drop('ele_impact', axis=1).T, labels=dfp.drop('ele_impact', axis=1).columns,
                 colors=colors_7)
    ax.set_xlim(0.001, 0.4)
    ax.set_xlabel('Electricity impact (kg CO2eq/kWh)')
    ax.set_ylabel('Biomass use (Mt)')
    ax.legend(loc='lower left')
    plt.title('Biomass use')
    plt.grid()
    plt.show()

    # plot methanol source
    dfp = df[[x for x in df.columns if 'methanol,' in x or 'methanol_from_waste' in x]].copy()
    dfp['ele_impact'] = df['ele_impact']
    dfp2 = pd.DataFrame()
    dfp2['waste gasification'] = dfp['methanol_from_waste_gasi']
    dfp2['biomass gasification'] = dfp['methanol, from agricultural residue gasification, beccs_biogenic_short'] \
                                   #+ dfp['methanol, from forest residue gasification, beccs_biogenic_long']
    dfp2['co2 hydrogenation (process co2)'] = dfp['methanol, from co2 hydrogenation_biogenic_long'] + \
                                              dfp['methanol, from co2 hydrogenation_biogenic_short']
    dfp2['co2 hydrogenation (additional co2)'] = dfp['methanol, from co2 hydrogenation_co2']
    dfp2['ele_impact'] = df['ele_impact']
    dfp = dfp2.copy()
    dfp = dfp.loc[:, (dfp > 0.01).any(axis=0)].copy()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), squeeze=True)
    ax.stackplot(dfp['ele_impact'], dfp.drop('ele_impact', axis=1).T, labels=dfp.drop('ele_impact', axis=1).columns,
                 colors=colors_6)
    ax.set_xlim(0.001, 0.4)
    ax.set_xlabel('Electricity impact (kg CO2eq/kWh)')
    ax.set_ylabel('Methanol production (Mt)')
    ax.legend(loc='lower right')
    plt.title('Methanol source')
    plt.grid()
    plt.show()

    # plot plastics source
    dfp = df[[x for x in df.columns if 'plastics_' in x]].copy()
    dfp['ele_impact'] = df['ele_impact']
    dfp = dfp.loc[:, (dfp > 0.01).any(axis=0)].copy()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), squeeze=True)
    ax.stackplot(dfp['ele_impact'], dfp.drop('ele_impact', axis=1).T, labels=dfp.drop('ele_impact', axis=1).columns,
                 colors=colors_5)
    ax.set_xlim(0.001, 0.4)
    ax.set_xlabel('Electricity impact (kg CO2eq/kWh)')
    ax.set_ylabel('Plastics production (Mt)')
    ax.legend()
    plt.title('Plastics source')
    plt.grid()
    plt.show()

    # plot alternative plastics
    dfp = df[[x for x in df.columns if '_total' in x]].copy()
    dfp2 = pd.DataFrame()
    dfp2['pe'] = dfp['hdpe_total'] + dfp['ldpe_total']
    dfp2['pp'] = dfp['pp_total']
    dfp2['gpps'] = dfp['gpps_total']
    dfp2['pet'] = dfp['pet_total']
    dfp2['pla'] = dfp['pla_total']
    dfp2['pbs'] = dfp['pbs_total']
    dfp2['other'] = dfp.sum(axis=1) - dfp2.sum(axis=1)
    dfp2['ele_impact'] = df['ele_impact']
    dfp = dfp2.copy()
    dfp['ele_impact'] = df['ele_impact']
    dfp2['pla_percentage'] = dfp['pla'] / dfp.sum(axis=1)
    dfp = dfp.loc[:, (dfp > 0.01).any(axis=0)].copy()
    fig, ax = plt.subplots(1, 1, figsize=(6, 8), squeeze=True)
    ax.stackplot(dfp['ele_impact'], dfp.drop('ele_impact', axis=1).T, labels=dfp.drop('ele_impact', axis=1).columns,
                    colors=colors_7)
    ax.set_xlim(0.001, 0.4)
    ax.set_xlabel('Electricity impact (kg CO2eq/kWh)')
    ax.set_ylabel('Plastics production (Mt)')
    ax.legend(loc='lower left')
    plt.title('Alternative plastics')
    plt.grid()
    plt.show()

    # plot ghg impact
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), squeeze=True)
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=0.25)
    sns.lineplot(x=df['ele_impact'], y=df['ghg'], ax=ax, color=colors_7[4], linewidth=0.75)
    ax.legend()
    #plt.savefig(r'figure/biomass_sensitivity_ghg_no_ccs.pdf')
    plt.show()
    '''
    p1 = sns.scatterplot(data=df, x='ele_impact', y='ghg', ax=ax, color=colors_4[0])
    for line in df.index:
        label = f"{df.ghg[line]:.0f}"
        p1.text(df.ele_impact[line] + 0.003, df.ghg[line]-8, label, horizontalalignment='left', size='small')
    '''
    ax.set_xlabel('Electricity impact (kg CO2eq/kWh)')
    ax.set_ylabel('GHG emissions (Mt CO2eq)')
    ax.set_xlim(0, 0.4)
    plt.title('GHG impact')
    #plt.grid()
    plt.show()
    a=0


def plot_demand_sensitivity(master_file_path, plastics_file_path):
    df = sensitivity_demand(master_file_path, plastics_file_path)
    df['ghg_intensity'] = df['ghg'] / df['plastic_production']
    # plot ghg impact
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), squeeze=True)
    ax.scatter(df['plastic_production'], df['ghg'], color=colors_4[0])
    ax.set_xlabel('Plastics production (Mt)')
    ax.set_ylabel('GHG emissions (Mt CO2eq)')
    ax2 = ax.twinx()
    ax2.bar(df['plastic_production'], df['ghg_intensity'], width=80, color=colors_4[3], alpha=0.5)
    ax2.set_ylabel('GHG intensity (kg CO2eq/kg plastics)')
    plt.show()

    # plot ele consumption
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), squeeze=True)
    ax.scatter(df['plastic_production'], df['electricity_non_biomass'], color=colors_4[0])
    ax.set_xlabel('Plastics production (Mt)')
    ax.set_ylabel('Electricity consumption (TWh)')
    plt.show()

    # plot carbon input
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), squeeze=True)
    ax.stackplot(df['plastic_production'], df['c_agri'], df['c_forest'],
                 df['c_fossil'], labels=['agricultural residue', 'forest residue', 'fossil fuel'],
                 colors=colors_6)
    ax.set_xlabel('Plastics production (Mt)')
    ax.set_ylabel('Carbon input (Mt)')
    ax.legend(loc='upper left')
    ax.set_xlim(df['plastic_production'].min(), df['plastic_production'].max())
    plt.title('Carbon Source')
    plt.show()

    # plot biomass use
    dfp = df[[x for x in df.columns if 'agricultural_residue_to' in x or 'forest_residue_to' in x]].copy()
    dfp['agricultural_residue_other'] = df['c_agri'] / 0.494 - dfp[[x for x in dfp.columns if 'agricultural_residue_to' in x]].sum(axis=1)
    dfp['forest_residue_other'] = df['c_forest'] / 0.521 - dfp[[x for x in dfp.columns if 'forest_residue_to' in x]].sum(axis=1)
    dfp['plastic_production'] = df['plastic_production']
    dfp2 = pd.DataFrame()
    dfp2['to_ethanol'] = dfp['agricultural_residue_to_ethanol'] + dfp['forest_residue_to_ethanol'] + \
                         dfp['forest_residue_to_glucose'] + dfp['agricultural_residue_to_glucose']
    dfp2['to_methanol'] = dfp['agricultural_residue_to_methanol'] + dfp['forest_residue_to_methanol']
    dfp2['to_lactic_acid'] = dfp['agricultural_residue_to_lactic acid'] + dfp['forest_residue_to_lactic acid']
    dfp2['to_syngas'] = dfp['forest_residue_to_syngas'] + dfp['agricultural_residue_to_syngas']
    dfp2['to_heat'] = dfp['agricultural_residue_to_heat'] + dfp['forest_residue_to_heat']
    dfp2['to_electricity'] = dfp['agricultural_residue_to_electricity'] + dfp['forest_residue_to_electricity']
    dfp2['plastic_production'] = df['plastic_production']
    dfp = dfp.loc[:, (dfp > 0.01).any(axis=0)].copy()
    dfp2 = dfp2.loc[:, (dfp2 > 0.01).any(axis=0)].copy()
    dfp = dfp2.copy()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), squeeze=True)
    ax.stackplot(dfp['plastic_production'], dfp.drop('plastic_production', axis=1).T,
                 labels=dfp.drop('plastic_production', axis=1).columns,
                 colors=colors_6)
    ax.set_xlim(df['plastic_production'].min(), df['plastic_production'].max())
    ax.set_xlabel('Plastics production (Mt)')
    ax.set_ylabel('Biomass use (Mt)')
    ax.legend(loc='upper left')
    plt.title('Biomass use')
    plt.show()

    # plot methanol source
    dfp = df[[x for x in df.columns if 'methanol,' in x or 'methanol_from_waste' in x]].copy()
    dfp['plastic_production'] = df['plastic_production']
    dfp2 = pd.DataFrame()
    dfp2['waste gasification'] = dfp['methanol_from_waste_gasi']
    dfp2['biomass gasification'] = dfp['methanol, from agricultural residue gasification, beccs_biogenic_short'] + \
                                   dfp['methanol, from forest residue gasification, beccs_biogenic_long']
    dfp2['co2 hydrogenation (process co2)'] = dfp['methanol, from co2 hydrogenation_biogenic_long'] + \
                                              dfp['methanol, from co2 hydrogenation_biogenic_short']
    dfp2['co2 hydrogenation (additional co2)'] = dfp['methanol, from co2 hydrogenation_co2']
    dfp2['plastic_production'] = df['plastic_production']
    dfp = dfp2.copy()
    dfp = dfp.loc[:, (dfp > 0.01).any(axis=0)].copy()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), squeeze=True)
    ax.stackplot(dfp['plastic_production'], dfp.drop('plastic_production', axis=1).T,
                 labels=dfp.drop('plastic_production', axis=1).columns,
                 colors=colors_6)
    ax.set_xlim(df['plastic_production'].min(), df['plastic_production'].max())
    ax.set_xlabel('Plastics production (Mt)')
    ax.set_ylabel('Methanol production (Mt)')
    ax.legend(loc='upper left')
    plt.title('Methanol source')
    plt.show()

    # plot plastics source
    dfp = df[[x for x in df.columns if 'plastics_' in x]].copy()
    dfp['plastic_production'] = df['plastic_production']
    dfp = dfp.loc[:, (dfp > 0.01).any(axis=0)].copy()
    fig, ax = plt.subplots(1, 1, figsize=(6, 8), squeeze=True)
    ax.stackplot(dfp['plastic_production'], dfp.drop('plastic_production', axis=1).T,
                 labels=dfp.drop('plastic_production', axis=1).columns,
                 colors=colors_6)
    ax.set_xlim(df['plastic_production'].min(), df['plastic_production'].max())
    ax.set_xlabel('Plastics production (Mt)')
    ax.set_ylabel('Plastics production (Mt)')
    ax.legend(loc='upper left')
    plt.title('Plastics source')
    plt.show()

    # plot alternative plastics
    dfp = df[[x for x in df.columns if '_total' in x]].copy()
    dfp2 = pd.DataFrame()
    dfp2['pe'] = dfp['hdpe_total'] + dfp['ldpe_total']
    dfp2['pp'] = dfp['pp_total']
    dfp2['gpps'] = dfp['gpps_total']
    dfp2['pet'] = dfp['pet_total']
    dfp2['pla'] = dfp['pla_total']
    dfp2['pbs'] = dfp['pbs_total']
    dfp2['other'] = dfp.sum(axis=1) - dfp2.sum(axis=1)
    dfp2['plastic_production'] = df['plastic_production']
    dfp = dfp2.copy()
    dfp = dfp.loc[:, (dfp > 0.01).any(axis=0)].copy()
    fig, ax = plt.subplots(1, 1, figsize=(6, 8), squeeze=True)
    ax.stackplot(dfp['plastic_production'], dfp.drop('plastic_production', axis=1).T,
                 labels=dfp.drop('plastic_production', axis=1).columns,
                 colors=colors_7)
    ax.set_xlim(df['plastic_production'].min(), df['plastic_production'].max())
    ax.set_xlabel('Plastics production (Mt)')
    ax.set_ylabel('Plastics production (Mt)')
    ax.legend(loc='upper left')
    plt.title('Alternative plastics')
    plt.show()

    a=0
def plot_network(user_input):
    df0 = user_input.model_results('GHG')[1]
    df = df0.copy()
    df['product_from'] = df['product_name']
    for p in df['process'].unique():
        df_temp = df[df['process'] == p].copy()
        if 'PRODUCT' in df_temp['type'].values:
            product_name = df_temp[df_temp['type'] == 'PRODUCT']['product_name'].values[0]
            df.loc[df['process'] == p, 'product_to'] = product_name
    df.dropna(subset=['product_to'], inplace=True)
    df1 = df[df.type.isin(['RAW MATERIALS'])].copy()
    df1 = df1[~df1['process'].str.startswith(tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))].copy()
    df2 = df[df['process'].str.startswith(tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))].copy()
    df2 = df2[df2['type'].isin(['RAW MATERIALS', 'WASTE'])].copy()
    for waste in df2.loc[df2['type'] == 'WASTE', 'product_name'].unique():
        process = df2.loc[df2['product_name'] == waste, 'process'].values[0]
        product_from = df2.loc[(df2['process'] == process) & (df2['type'] == 'RAW MATERIALS'), 'product_name'].values[0]
        df2.loc[df2['product_name'] == waste, 'product_from'] = product_from
        df2.loc[df2['product_name'] == waste, 'product_to'] = waste
    df2 = df2.loc[df2['type'] == 'WASTE'].copy()
    df2 = df2.groupby(['product_from', 'product_to']).sum(numeric_only=True).reset_index()
    df1 = pd.concat([df1, df2], ignore_index=True)
    suffix_list = ['_biogenic_short', '_biogenic_long', '_fossil', '_co2']
    for suffix in suffix_list:
        df1.loc[df1['product_from'].str.contains(suffix), 'product_from'] = df1['product_from'].str.replace(suffix, '')
        df1.loc[df1['product_to'].str.contains(suffix), 'product_to'] = df1['product_to'].str.replace(suffix, '')
        df.loc[df['product_name'].str.contains(suffix), 'product_name'] = df['product_name'].str.replace(suffix, '')
    df1 = df1.groupby(['product_from', 'product_to']).sum(numeric_only=True).reset_index()
    #df.loc[df['product_type'] == 'raw_material', 'color'] = colors_5[3]
    df.loc[df['product_type'] == 'intermediate', 'color'] = colors_5[1]
    df.loc[df['product_type'] == 'waste', 'color'] = colors_5[4]
    df1 = df1.loc[df1.product_from != 'water']
    df1 = df1.loc[df1.product_to != 'electricity']
    df1 = df1.loc[df1.product_to != 'heat_high']
    #df1 = df1.loc[df1.product_to != 'agricultural_residue']
    df1 = df1.loc[abs(df1.flowxvalue) > 0.01]
    df1.reset_index(inplace=True, drop=True)

    node_list = list(set(list(df1['product_from'].unique()) + list(df1['product_to'].unique())))
    G = nx.Graph()
    for node in node_list:
        G.add_node(node)
    for index, row in df1.iterrows():
        G.add_edge(row['product_from'], row['product_to'])
    pos = nx.spring_layout(G)
    fig = go.Figure()
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        try:
            weight = abs(df1.loc[(df1['product_from'] == edge[0]) & (df1['product_to'] == edge[1]), 'flowxvalue'].values[0])
        except:
            weight = abs(df1.loc[(df1['product_from'] == edge[1]) & (df1['product_to'] == edge[0]), 'flowxvalue'].values[0])
        weight = (weight / 15) ** 0.4
        if weight < 0.5:
            weight = 0.5
        if weight > 6:
            weight = 6
        edge_trace = go.Scatter(x=[x0, x1], y=[y0, y1], line=dict(width=weight, color='#888'), hoverinfo='none',
                                mode='lines')
        fig.add_trace(edge_trace)

    # Create node trace
    node_x = []
    node_y = []
    node_name = []
    node_size = []
    node_color = []
    for node in pos:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_name.append(node)
        if node in final_product_list:
            node_size.append(15)
            node_color.append(colors_5[0])
        elif df.loc[df['product_name'] == node, 'product_type'].values[0] == 'raw_material':
            node_size.append(15)
            node_color.append(colors_5[3])
        else:
            node_size.append(5)
            color = df.loc[df['product_name'] == node, 'color'].values[0]
            node_color.append(color)
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',  # Add text to display labels
        text=node_name,  # Labels are the node names
        textposition='top center',  # Position labels above nodes
        hoverinfo='text',
        marker=dict(showscale=False, size=node_size, color=node_color, line=dict(color='black', width=2)))

    node_trace.text = node_name
    fig.add_trace(node_trace)
    # Create network graph
    fig.update_layout(
                        showlegend=False,
                        paper_bgcolor='white',  # Set background color of the figure to white
                        plot_bgcolor='white',
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    )
    fig.show()
    a=0


def plot_region(master_file_path, plastics_file_path):
    df = regional_results(master_file_path, plastics_file_path)

    world_ghg = df.loc[df.country.str.contains('World'), 'ghg'].values[0]
    world_ghg_r = df.loc[~df.country.str.contains('World'), 'ghg'].sum()
    world_bdv = df.loc[df.country.str.contains('World'), 'bdv'].values[0]
    world_bdv_r = df.loc[~df.country.str.contains('World'), 'bdv'].sum()
    world_health = df.loc[df.country.str.contains('World'), 'health'].values[0]
    world_health_r = df.loc[~df.country.str.contains('World'), 'health'].sum()
    world_production_r = df.loc[~df.country.str.contains('World'), 'plastic_production'].sum()
    df['plastics_health_intensity'] = df['health'] / df['plastic_production']
    df['country'] = df['country'].str.replace('_', '')
    df['c_plastics_waste'] = df['c_plastics'] + df['c_plastics_gasi']
    df1 = df[['country', 'plastic_production', 'ghg', 'bdv', 'health', 'plastics_ghg_intensity', 'plastics_bdv_intensity',
              'plastics_health_intensity', 'c_plastics_waste', 'c_agri', 'c_forest', 'c_co2', 'c_fossil',
              'c_plastics', 'c_plastics_gasi',
              'agri_usage', 'forest_usage', 'electricity_usage']].copy()
    df1.to_excel(r'data/figure/regional_optimization_results.xlsx')
    df1 = df1[df1.country != 'World'].copy()
    # plot ghg_intensity vs bdv_intensity
    df1['size'] = df1['plastic_production']
    df1.loc[df1['size'] < 10, 'size'] = 10
    df1.loc[df1['size'] > 100, 'size'] = 100
    fig, ax = plt.subplots(1, 1, figsize=(11.5, 8), squeeze=True)
    p1 = sns.scatterplot(data=df1, x='plastics_ghg_intensity', y='plastics_bdv_intensity', size='size',
                         sizes=(300, 3000), hue='plastics_health_intensity', ax=ax, palette=cmp_purple().reversed(),
                         alpha=0.8)
    for line in df1.index:
        p1.text(df1.plastics_ghg_intensity[line], df1.plastics_bdv_intensity[line], df1.country[line],
                horizontalalignment='center', verticalalignment='center', size='small')
    plt.xlabel('GHG intensity (kg CO2eq/kg)')
    plt.ylabel('Biodiversity loss intensity (x10^-15 PDF/kg)')
    ax.legend(loc='upper right', frameon=False, bbox_to_anchor=(1.6, 1.05))
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-0.5, 14.5)
    plt.subplots_adjust(top=0.95, bottom=0.12, right=0.673, left=0.12)
    plt.savefig(r'figure/region_impact_intensity_bubble.pdf')
    plt.show()

    # stacked bar ghg
    dfp = df1[['country', 'ghg']].sort_values(by='ghg').copy()
    other_neg = dfp.loc[(dfp.ghg < 0) & (dfp.ghg >= -80), 'ghg'].sum()
    df_other_neg = pd.DataFrame({'country': 'other_neg', 'ghg': other_neg}, index=[0])
    other_pos = dfp.loc[(dfp.ghg > 0) & (dfp.ghg <= 80), 'ghg'].sum()
    df_other_pos = pd.DataFrame({'country': 'other_pos', 'ghg': other_pos}, index=[0])
    dfp = dfp.loc[(dfp.ghg < -80) | (dfp.ghg > 80)].copy()
    dfp = pd.concat([dfp, df_other_neg, df_other_pos], ignore_index=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 8), squeeze=True)
    neg_b = 0
    pos_b = 0
    i = 0
    for x in dfp.index:
        value = dfp.loc[x, 'ghg']
        if value < 0:
            ax.bar('regional optimization', value, bottom=neg_b, color=colors_5[i])
            neg_b += value
        else:
            ax.bar('regional optimization', value, bottom=pos_b, color=colors_5[i])
            pos_b += value
        i += 1
    ax.bar('global optimization', world_ghg, color=colors_7[4])
    ax.scatter('regional optimization', world_ghg_r, color=colors_7[4], edgecolor='white', s=80)
    ax.scatter('global optimization', world_ghg, color=colors_7[4], edgecolor='white', s=80)
    plt.savefig(r'figure/ghg_global_vs_regional_sum.pdf')
    plt.show()

    # plot feedstock use percentage
    fig, ax = plt.subplots(1, 1, figsize=(12, 4), squeeze=True)
    sns.barplot(data=df1, x='country', y=1, ax=ax, color='lightgrey')
    sns.barplot(data=df1, x='country', y='agri_usage', ax=ax, color=colors_4[0])
    plt.xlabel('')
    plt.ylabel('x100%')
    plt.xticks(rotation=90)
    plt.title('Agricultural residue usage')
    plt.show()
    fig, ax = plt.subplots(1, 1, figsize=(12, 4), squeeze=True)
    sns.barplot(data=df1, x='country', y=1, ax=ax, color='lightgrey')
    sns.barplot(data=df1, x='country', y='forest_usage', ax=ax, color=colors_4[1])
    plt.xlabel('')
    plt.ylabel('x100%')
    plt.xticks(rotation=90)
    plt.title('Forest residue usage')
    plt.show()
    fig, ax = plt.subplots(1, 1, figsize=(12, 4), squeeze=True)
    sns.barplot(data=df1, x='country', y=1, ax=ax, color='lightgrey')
    sns.barplot(data=df1, x='country', y='electricity_usage', ax=ax, color=colors_4[3])
    plt.xlabel(' ')
    plt.ylabel('x100%')
    plt.xticks(rotation=90)
    plt.title('Electricity usage')
    plt.show()

    # plot feedstock composition by country (stacked bar plot)
    #df3 = df1[['country', 'c_plastics_waste', 'c_agri', 'c_forest', 'c_co2', 'c_fossil']].copy().set_index('country')
    # add ceu and weu to be eur
    #df3.loc['EUR'] = df3.loc['CEU'] + df3.loc['WEU']
    #df3 = df3[df3.index.isin(['BRA', 'EUR', 'CHN', 'JPN', 'ME', 'USA'])]
    #df3_percentage = df3.div(df3.sum(axis=1), axis=0) * 100
    # with gasification
    mpl.rcParams['hatch.linewidth'] = 1.5  # Default is 1.0
    mpl.rcParams['hatch.color'] = 'white'
    df3 = df1[['country', 'c_plastics', 'c_plastics_gasi', 'c_agri', 'c_forest', 'c_co2', 'c_fossil']].copy().set_index(
        'country')
    df3.loc['EUR'] = df3.loc['CEU'] + df3.loc['WEU']
    df3 = df3[df3.index.isin(['BRA', 'EUR', 'CHN', 'JPN', 'ME', 'USA'])]
    df3_percentage = df3.div(df3.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), squeeze=True)
    plot_colors = [colors_5[-1], colors_5[-1], colors_5[-2], colors_5[-3], colors_5[-4], '#58595b']

    # Define hatch patterns
    hatch_patterns = ['//', '\\\\', '', '', '', '']
    bottom = np.zeros(len(df3_percentage))
    # Plot each column individually
    for idx, (column, hatch) in enumerate(zip(df3_percentage.columns, hatch_patterns)):
        ax.bar(df3_percentage.index, df3_percentage[column], bottom=bottom, label=column,
               color=plot_colors[idx % len(plot_colors)], hatch=hatch, edgecolor='white', linewidth=0.5, width=0.4)
        bottom += df3_percentage[column].values
    plt.xlabel('')
    plt.ylabel('Carbon input source (%)')
    plt.subplots_adjust(top=0.95, bottom=0.2, right=0.85, left=0.08)
    ax.legend(loc='upper right', frameon=False, bbox_to_anchor=(1.255, 0.99))
    plt.savefig(r'figure/carbon_input_source_key_countries.pdf')
    plt.show()
    mpl.rcParams['hatch.linewidth'] = 1.0
    mpl.rcParams['hatch.color'] = 'black'

    # as map
    world_shape = gpd.read_file("data/external/World_Countries_(Generalized)/"
                                "World_Countries__Generalized_.shp")
    world_shape.rename(columns={'ISO': 'ISO2'}, inplace=True)
    world_shape = world_shape.loc[(world_shape.COUNTRY != 'Canarias') &
                                  (world_shape.COUNTRY != 'Azores') &
                                  (world_shape.COUNTRY != 'Madeira')]
    world_shape = world_shape.loc[world_shape.COUNTRY != 'Antarctica'].copy()
    df_country = pd.read_excel(r'data/raw/Country.xlsx', engine='openpyxl', sheet_name='Sheet1')
    df_country.loc[df_country.Country == "Namibia", "ISO2"] = "NA"
    df = pd.merge(df_country, world_shape, on='ISO2', how='right')
    df = gpd.GeoDataFrame(df, geometry=df.geometry)
    df_image = df[['IMAGE_region', 'geometry']]
    df_image = df_image.dissolve(by='IMAGE_region').reset_index()
    df_image.loc[df_image.IMAGE_region == 'JAP', 'IMAGE_region'] = 'JPN'
    df_image.loc[df_image.IMAGE_region == 'INDO', 'IMAGE_region'] = 'IDN'
    df_image.loc[df_image.IMAGE_region == 'INDIA', 'IMAGE_region'] = 'IND'
    df_image.loc[df_image.IMAGE_region == 'SAF', 'IMAGE_region'] = 'ZAF'
    df_image.loc[df_image.IMAGE_region == 'RUKR', 'IMAGE_region'] = 'UKR'
    import matplotlib.lines as mlines
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    df1['centroids'] = df1.geometry.centroid
    df2 = df1.copy()
    df2.geometry = df1['centroids']
    df2 = df2.loc[df2.IMAGE_region != 'SEAS']
    labels = [10, 100, 200]
    sizes = [x * 2 for x in labels]
    handles = [
        mlines.Line2D([], [], color=colors_5[1], marker='o', markersize=np.sqrt(size), linestyle='None', alpha=0.8,
                      label=label)
        for size, label in zip(sizes, labels)]
    fig, ax = plt.subplots(1, 1, figsize=(15, 7))
    df1.plot(column='plastics_ghg_intensity', ax=ax, legend=True,
             linewidth=.5, edgecolor='grey', cmap=cmp_heatmap(), vmin=0,
             missing_kwds={'color': 'grey', 'label': 'Missing values'})
    df2.plot(ax=ax, marker='o', markersize=df1['plastic_production'] * 2, color=colors_5[1], alpha=0.8,
             edgecolor='white')
    ax.axis('off')
    plt.legend(handles=handles, title="Circle Size")
    plt.savefig(r'figure/map_ghg_intensity.svg')
    plt.show()
    a=0


def plot_image_map(master_file_path, plastics_file_path):
    world_shape = gpd.read_file("data/external/World_Countries_(Generalized)/"
                                "World_Countries__Generalized_.shp")
    world_shape.rename(columns={'ISO': 'ISO2'}, inplace=True)
    world_shape = world_shape.loc[(world_shape.COUNTRY != 'Canarias') &
                                  (world_shape.COUNTRY != 'Azores') &
                                  (world_shape.COUNTRY != 'Madeira')]
    world_shape = world_shape.loc[world_shape.COUNTRY != 'Antarctica'].copy()
    df_country = pd.read_excel(r'data/raw/Country.xlsx', engine='openpyxl', sheet_name='Sheet1')
    df_country.loc[df_country.Country == "Namibia", "ISO2"] = "NA"
    df = pd.merge(df_country, world_shape, on='ISO2', how='right')
    df = gpd.GeoDataFrame(df, geometry=df.geometry)
    df_image = df[['IMAGE_region', 'geometry']]
    df_image = df_image.dissolve(by='IMAGE_region').reset_index()
    df_image.loc[df_image.IMAGE_region == 'JAP', 'IMAGE_region'] = 'JPN'
    df_image.loc[df_image.IMAGE_region == 'INDO', 'IMAGE_region'] = 'IDN'
    df_image.loc[df_image.IMAGE_region == 'INDIA', 'IMAGE_region'] = 'IND'
    df_image.loc[df_image.IMAGE_region == 'SAF', 'IMAGE_region'] = 'ZAF'
    df_image.loc[df_image.IMAGE_region == 'RUKR', 'IMAGE_region'] = 'UKR'
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", colors_26)
    fig, ax = plt.subplots(1, 1, figsize=(18/2.54, 10/2.54))
    df_image.plot(column='IMAGE_region',
                  ax=ax,
                  legend=True,
                  linewidth=.5,
                  edgecolor='white',
                  cmap=cmap,
                  missing_kwds={'color': 'white', 'label': 'Missing values'},
                  legend_kwds={'loc': 'lower center',
                               'bbox_to_anchor': (0.5, -0.1),
                               'ncol': 9,  # Adjust this value as needed
                               'frameon': False,
                               'markerscale': 0.7,
                               'fontsize': 6})
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(r'figure/image_region_map.png', bbox_inches='tight', dpi=300)
    plt.show()
    return df


def plot_demand_sensitivity_2(user_input_file, master_file_path, plastics_file_path):
    ele_impact_list = [-999]
    biomass_ratio_list = [1]
    demand_list = [round(i, 2) for i in np.arange(0.1, 1.501, 0.025)]
    #demand_list = [0.4638, 1]
    base_path = os.path.join("data", "raw", "user_inputs_ccs_no_ccs")
    files = glob.glob(os.path.join(base_path, "*.xlsx"))
    df0 = pd.DataFrame()
    for user_input_file in files:
        user_input = MasterFile(user_input_file, master_file_path, plastics_file_path)
        if 'default' in user_input_file:
            scenario = 'with_ccs'
        else:
            scenario = 'without_ccs'
        df = user_input.sensitivity_demand_ele_biomass(ele_impact_list, demand_list, biomass_ratio_list)
        df['scenario'] = scenario
        df0 = pd.concat([df, df0], ignore_index=True)
    df0['GHG'] /= 1000
    df0['ghg_intensity'] = df0['GHG'] / (1.007 * df0['demand_ratio'])
    df0['fossil_fuel'] = (df0['natural_gas'] + df0['petroleum']) / 1000  # Gt
    df0['biomass'] = (df0['agricultural_residue'] + df0['forest_residue']) / 1000  # Gt
    df0['electricity'] = df0['ele_use_total'] / 1000  # PWh
    demand_list2 = [round(i, 2) for i in np.arange(0.1, 1.501, 0.1)]
    # ghg and ghg intensity
    dfy = df0.loc[df0.scenario == 'with_ccs', :].copy()
    dfn = df0.loc[df0.scenario == 'without_ccs', :].copy()
    dfy1 = dfy.loc[dfy['demand_ratio'].isin(demand_list2), :]
    dfn1 = dfn.loc[dfn['demand_ratio'].isin(demand_list2), :]
    df = df0.loc[df0['demand_ratio'].isin(demand_list2), :]

    # all
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    # ax1.axvline(x=8, color='grey')
    # ax1.axvline(x=0.4638 * 10 - 2, color='grey')
    ax1.axvline(x=1, color='grey')
    ax1.axvline(x=0.4638, color='grey')
    ax2.axvline(x=1, color='grey')
    ax2.axvline(x=0.4638, color='grey')
    ax3.axvline(x=1, color='grey')
    ax3.axvline(x=0.4638, color='grey')
    ax4.axvline(x=1, color='grey')
    ax4.axvline(x=0.4638, color='grey')
    ax1.axhline(y=0, color='grey')
    ax1.set_xlim(0.05, 1.55)
    ax2.set_xlim(0.05, 1.55)
    ax3.set_xlim(0.05, 1.55)
    ax4.set_xlim(0.05, 1.55)
    ax1r = ax1.twinx()
    ax1.set_ylim(-0.9, 1.98)
    ax1r.set_ylim(-2.2, 4.84)
    width = 0.035
    ax1r.bar(dfn1['demand_ratio'] - width / 2, dfn1['ghg_intensity'], width, label='GHG intensity', color=colors_4[0],
             alpha=0.6)
    ax1r.bar(dfy1['demand_ratio'] + width / 2, dfy1['ghg_intensity'], width, label='GHG intensity', color=colors_4[1],
             alpha=0.6)
    ax1r.set_ylabel('unit climate-change impacts (kg CO2eq/kg plastics)')
    sns.lineplot(data=df, x='demand_ratio', y='GHG', hue='scenario', ax=ax1, palette=colors_4)
    ax1.set_xlabel('production ratio')
    ax1.set_ylabel('climate-change impacts (Gt CO2eq)')
    sns.lineplot(data=df, x='demand_ratio', y='fossil_fuel', hue='scenario', ax=ax2, palette=colors_4)
    ax2.set_xlabel('production ratio')
    ax2.set_ylabel('Fossil fuel use (Gt)')
    ax2.get_legend().set_visible(False)
    sns.lineplot(data=df, x='demand_ratio', y='biomass', hue='scenario', ax=ax3, palette=colors_4)
    ax3.set_xlabel('production ratio')
    ax3.set_ylabel('Biomass use (Gt)')
    ax3.get_legend().set_visible(False)
    sns.lineplot(data=df, x='demand_ratio', y='electricity', hue='scenario', ax=ax4, palette=colors_4)
    ax4.set_xlabel('production ratio')
    ax4.set_ylabel('Electricity use (PWh)')
    ax4.get_legend().set_visible(False)
    ax2.set_box_aspect(1)
    ax1.set_box_aspect(1)
    ax3.set_box_aspect(1)
    ax4.set_box_aspect(1)
    figname = 'figure/demand_sensitivity_2.pdf'
    plt.savefig(figname)
    plt.show()
    df.to_excel('data/figure/demand_sensitivity_2.xlsx')

    df0 = pd.DataFrame({'demand_ratio': [0], 'fossil_fuel': [0], 'biomass': [0], 'electricity': [0]})
    df3 = pd.concat([df0, df3], ignore_index=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(df3['demand_ratio'], df3['fossil_fuel'], label='Fossil fuel', color=colors_5[0])
    ax.plot(df3['demand_ratio'], df3['biomass'], label='Biomass', color=colors_5[1])
    ax.plot(df3['demand_ratio'], df3['electricity'], label='Electricity', color=colors_5[2])
    ax.plot([0, 1.2], [0, 1.2], color='grey', linestyle='--')
    ax.set_xlabel('Demand ratio')
    ax.set_ylabel('Normalized resource use')
    ax.axvline(x=1, color='grey')
    ax.axvline(x=0.4638, color='grey')
    ax.legend()
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 2)
    plt.show()

def plot_ele_demand_sensitivity(user_input_file, master_file_path, plastics_file_path):
    ele_impact_list = [round(i, 2) for i in np.arange(0, 0.401, 0.01)]
    #ele_impact_list = [-999]
    biomass_ratio_list = [1]
    biomass_ratio_list = [round(i, 2) for i in np.arange(0.0, 2.001, 0.05)]
    demand_list = [round(i, 2) for i in np.arange(0.2, 1.201, 0.025)]
    demand_list = [1]
    base_path = os.path.join("data", "raw", "user_inputs_fossil_impacts_no_ccs")
    files = glob.glob(os.path.join(base_path, "*.xlsx"))
    #ele_impact_list = [0, 0.001, 0.005] + [round(i, 3) for i in np.arange(0.01, 0.41, 0.01)]
    #ele_impact_list = [0.2]
    demand_list = [1]
    df = pd.DataFrame()
    for user_input_file in files:
        user_input = MasterFile(user_input_file, master_file_path, plastics_file_path)
        scenario = user_input_file.split('_')[-1].split('.')[0]
        df1 = user_input.sensitivity_demand_ele_biomass(ele_impact_list, demand_list, biomass_ratio_list)
        df1['scenario'] = scenario
        df = pd.concat([df, df1], ignore_index=True)
    df.loc[df.scenario == 'in', 'scenario'] = 'fossil_lock_in'
    df.loc[df.scenario == 'fossil', 'scenario'] = 'no_fossil'
    df['fossil'] = df['natural_gas'] + df['petroleum']
    #df = sensitivity_ele_demand(master_file_path, plastics_file_path)
    cmap = 'magma_r'
    #cmap = cmp_purple()
    #cmap = cmp_heatmap()
    df1 = df.loc[df.scenario == 'default']
    df2 = df.loc[df.scenario == 'no_fossil']
    df3 = df.loc[df.scenario == 'fossil_lock_in']
    #df1 = df.loc[df.biomass_ratio == 1]
    #df2 = df.loc[df.biomass_ratio == 0.5]
    # heatmap, total ghg
    table1 = pd.pivot_table(df1, index='ele_impact', columns='biomass_ratio',  values='GHG', aggfunc='mean')
    table2 = pd.pivot_table(df2, index='ele_impact', columns='biomass_ratio', values='GHG', aggfunc='mean')
    for x in biomass_ratio_list:
        if x not in table2.columns:
            table2[x] = np.nan
    table2 = table2[biomass_ratio_list]
    table3 = pd.pivot_table(df3, index='ele_impact', columns='biomass_ratio', values='GHG', aggfunc='mean')
    table1 = table1.applymap(lambda x: np.nan if x < 1 else x)
    table22 = table2.applymap(lambda x: np.nan if x < 1 else x)
    table3 = table3.applymap(lambda x: np.nan if x < 1 else x)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 4), squeeze=True, subplot_kw={'aspect': 1})
    sns.heatmap(
        np.where(table1.isna(), 0, np.nan),
        ax=ax1, vmin=0, vmax=0,
        cbar=False, cmap=ListedColormap([colors_4[1]]), linewidth=0)
    sns.heatmap(table1, square=True, annot=False, fmt=".0f", vmin=0, vmax=3000,
            linewidth=0, cbar=False, ax=ax1, cmap=cmap, cbar_kws={"shrink": 0.5},
            annot_kws={"size": 10, "va": "center_baseline", "color": "black"}, )
    sns.heatmap(
        np.where(table22.isna(), 0, np.nan),
        ax=ax2, vmin=0, vmax=0,
        cbar=False, cmap=ListedColormap([colors_4[1]]), linewidth=0)
    sns.heatmap(
        np.where(table2.isna(), 0, np.nan),
        ax=ax2, vmin=0, vmax=0,
        cbar=False, cmap=ListedColormap(['#d8d8d8']), linewidth=0)
    sns.heatmap(table22, square=True, annot=False, fmt=".0f", vmin=0, vmax=3000,
            linewidth=0, cbar=False, ax=ax2, cmap=cmap, cbar_kws={"shrink": 0.5},
            annot_kws={"size": 10, "va": "center_baseline", "color": "black"}, )
    sns.heatmap(
        np.where(table3.isna(), 0, np.nan),
        ax=ax3, vmin=0, vmax=0,
        cbar=False, cmap=ListedColormap([colors_4[1]]), linewidth=0)
    sns.heatmap(table3, square=True, annot=False, fmt=".0f", vmin=0, vmax=3000,
            linewidth=0, cbar=False, ax=ax3, cmap=cmap, cbar_kws={"shrink": 0.5},
            annot_kws={"size": 10, "va": "center_baseline", "color": "black"}, )
    sns.heatmap(table3, square=True, annot=False, fmt=".0f", vmin=0, vmax=3000,
                linewidth=0, cbar=True, ax=ax4, cmap=cmap, cbar_kws={"shrink": 0.5},
                annot_kws={"size": 10, "va": "center_baseline", "color": "black"}, )

    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    figname = f'figure/heatmap_ghg_biomass_fossil.pdf'
    plt.savefig(figname, bbox_inches='tight')
    fig.show()

    # heatmap, fossil use
    table1 = pd.pivot_table(df1, index='ele_impact', columns='biomass_ratio', values='fossil', aggfunc='mean')
    table2 = pd.pivot_table(df2, index='ele_impact', columns='biomass_ratio', values='fossil', aggfunc='mean')
    for x in biomass_ratio_list:
        if x not in table2.columns:
            table2[x] = np.nan
    table2 = table2[biomass_ratio_list]
    table3 = pd.pivot_table(df3, index='ele_impact', columns='biomass_ratio', values='fossil', aggfunc='mean')
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 4), squeeze=True, subplot_kw={'aspect': 1})
    sns.heatmap(table1, square=True, annot=False, fmt=".0f", vmin=0, vmax=3000,
            linewidth=0.5, cbar=False, ax=ax1, cmap=cmap, cbar_kws={"shrink": 0.5},
            annot_kws={"size": 10, "va": "center_baseline", "color": "black"}, )
    sns.heatmap(
        np.where(table2.isna(), 0, np.nan),
        ax=ax2, vmin=0, vmax=0,
        cbar=False, cmap=ListedColormap(['#d8d8d8']), linewidth=0)

    sns.heatmap(table2, square=True, annot=False, fmt=".0f", vmin=0, vmax=3000,
            linewidth=0.5, cbar=False, ax=ax2, cmap=cmap, cbar_kws={"shrink": 0.5},
            annot_kws={"size": 10, "va": "center_baseline", "color": "black"}, )
    sns.heatmap(table3, square=True, annot=False, fmt=".0f", vmin=0, vmax=3000,
            linewidth=0.5, cbar=False, ax=ax3, cmap=cmap, cbar_kws={"shrink": 0.5},
            annot_kws={"size": 10, "va": "center_baseline", "color": "black"}, )
    sns.heatmap(table3, square=True, annot=False, fmt=".0f", vmin=0, vmax=3000,
                linewidth=0.5, cbar=True, ax=ax4, cmap=cmap, cbar_kws={"shrink": 0.5},
                annot_kws={"size": 10, "va": "center_baseline", "color": "black"}, )
    figname = f'figure/heatmap_fossil_use.pdf'
    #plt.savefig(figname, bbox_inches='tight')
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    fig.show()

    # radar
    #df3 = df[df.ele_impact == 0.07]
    #df3 = df3[df3.demand_ratio.isin([0.2, 0.5, 0.8, 1, 1.2])]
    df3 = df.copy()
    df3 = df3[df3.biomass_ratio == 1]
    df3['GHG'] = df3['GHG'] / df3['GHG'].max()
    df3['BDV'] = df3['BDV'] / df3['BDV'].max()
    df3['Health'] = df3['Health'] / df3['Health'].max()
    df4 = df3[df3.biomass_ratio == 1]
    df5 = df3[df3.biomass_ratio != 1]
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(df4['demand_ratio'], df4['GHG'], label='GHG', color=colors_5[0])
    ax.plot(df4['demand_ratio'], df4['BDV'], label='BDV', color=colors_5[1])
    ax.plot(df4['demand_ratio'], df4['Health'], label='Health', color=colors_5[2])
    ax.plot(df5['demand_ratio'], df5['GHG'], label='GHG', color=colors_5[0], linestyle='--')
    ax.plot(df5['demand_ratio'], df5['BDV'], label='BDV', color=colors_5[1], linestyle='--')
    ax.plot(df5['demand_ratio'], df5['Health'], label='Health', color=colors_5[2], linestyle='--')

    ax.set_xlabel('Demand ratio')
    ax.set_ylabel('Normalized impact')
    ax.legend()
    plt.show()

    features = ['GHG', 'BDV', 'Health']
    num_vars = len(features)
    angles = np.linspace(np.pi / 2, 2 * np.pi + np.pi / 2, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    offset = 0.1
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, scenario in enumerate(df3['demand_ratio'].unique()):
        values = df3.loc[df3['demand_ratio'] == scenario, features].values.flatten().tolist()
        values += values[:1]
        line_style = 'solid'
        color = cmp_purple().reversed()(i / 5)
        ax.fill(angles, values, color=color, alpha=0.1)
        ax.plot(angles, values, color=color, linewidth=1, label=scenario, linestyle=line_style)
    # ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_ylim(-0.4, 1)
    ax.set_xticklabels(features)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
    plt.show()

    # heatmap, total health
    table = pd.pivot_table(df, index='ele_impact', columns='demand_ratio',
                           values='Health', aggfunc='mean')
    fig, ax = plt.subplots(1, 1, figsize=(9, 9), squeeze=True, subplot_kw={'aspect': 1})
    sns.heatmap(table, square=True, annot=True, fmt=".0f",
                linewidth=0.5, cbar=False, ax=ax, cmap=cmap, cbar_kws={"shrink": 0.5},
                annot_kws={"size": 10, "va": "center_baseline", "color": "black"}, )
    #figname = f'figures/lcia/lcia_heatmap_{year}_{scenario}_{impact}.pdf'
    #plt.savefig(figname, bbox_inches='tight')
    ax.invert_yaxis()
    fig.show()

    # heatmap, ghg intensity
    df['ghg_intensity'] = df['ghg'] / df['plastic_production']
    table = pd.pivot_table(df, index='ele_impact', columns='demand_ratio',
                           values='ghg_intensity', aggfunc='mean')
    fig, ax = plt.subplots(1, 1, figsize=(9, 9), squeeze=True, subplot_kw={'aspect': 1})
    sns.heatmap(table, square=True, annot=True, fmt=".2f",
                linewidth=0.5, cbar=False, ax=ax, cmap=cmap,
                annot_kws={"size": 10, "va": "center_baseline", "color": "black"}, )
    fig.show()

    # heatmap, health intensity
    df['health_intensity'] = df['health'] / df['plastic_production']
    table = pd.pivot_table(df, index='ele_impact', columns='demand_ratio',
                           values='health_intensity', aggfunc='mean')
    fig, ax = plt.subplots(1, 1, figsize=(9, 9), squeeze=True, subplot_kw={'aspect': 1})
    sns.heatmap(table, square=True, annot=True, fmt=".2f",
                linewidth=0.5, cbar=False, ax=ax, cmap=cmap,
                annot_kws={"size": 10, "va": "center_baseline", "color": "black"}, )
    fig.show()

    # heatmap, methanol from co2 hydrogenation
    columns = [x for x in df.columns if 'methanol, from co2' in x]
    df['methanol_co2'] = df[columns].sum(axis=1)
    table = pd.pivot_table(df, index='ele_impact', columns='demand_ratio',
                           values='methanol_co2', aggfunc='mean')
    fig, ax = plt.subplots(1, 1, figsize=(9, 9), squeeze=True, subplot_kw={'aspect': 1})
    sns.heatmap(table, square=True, annot=True, fmt=".2f",
                linewidth=0.5, cbar=False, ax=ax, cmap=cmap,
                annot_kws={"size": 10, "va": "center_baseline", "color": "black"}, )
    fig.show()

    # heatmap, biomass use (MT biomass)
    df['biomass_use'] = df['agricultural_residue'] + df['forest_residue']
    df['biomass_intensity'] = df['biomass_use'] / df['plastics_bio_virgin']
    table = pd.pivot_table(df, index='ele_impact', columns='demand_ratio',
                           values='biomass_intensity', aggfunc='mean')
    fig, ax = plt.subplots(1, 1, figsize=(9, 9), squeeze=True, subplot_kw={'aspect': 1})
    sns.heatmap(table, square=True, annot=True, fmt=".0f",
                linewidth=0.5, cbar=False, ax=ax, cmap=cmap, cbar_kws={"shrink": 0.5},
                annot_kws={"size": 10, "va": "center_baseline", "color": "black"}, )
    ax.invert_yaxis()
    fig.show()

    # heatmap, mr
    df['plastics_mr'] = df['plastics_bio_mr'] + df['plastics_fossil_mr'] + df['plastics_co2_mr']
    df['plastics_virgin'] = df['plastics_bio_virgin'] + df['plastics_fossil_virgin'] + df['plastics_co2_virgin']
    df['plastics_total'] = df['plastics_mr'] + df['plastics_virgin']
    df['mr_percent'] = df['plastics_mr'] / df['plastics_total']
    # heatmap, electricity use (TWh)
    table = pd.pivot_table(df, index='ele_impact', columns='demand_ratio',
                           values='electricity', aggfunc='mean')
    fig, ax = plt.subplots(1, 1, figsize=(9, 9), squeeze=True, subplot_kw={'aspect': 1})
    sns.heatmap(table, square=True, annot=True, fmt=".0f",
                linewidth=0.5, cbar=False, ax=ax, cmap=cmap,
                annot_kws={"size": 10, "va": "center_baseline", "color": "black"}, )
    fig.show()

    # heatmap, ele_intensity
    df['ele_intensity'] = df['electricity'] / df['plastic_production']
    table = pd.pivot_table(df, index='ele_impact', columns='demand_ratio',
                           values='ele_intensity', aggfunc='mean')
    fig, ax = plt.subplots(1, 1, figsize=(9, 9), squeeze=True, subplot_kw={'aspect': 1})
    sns.heatmap(table, square=True, annot=True, fmt=".2f",
                linewidth=0.5, cbar=False, ax=ax, cmap=cmap,
                annot_kws={"size": 10, "va": "center_baseline", "color": "black"}, )
    fig.show()

    # electricity from biomass
    df['electricity_biomass'] = df['electricity'] - df['electricity_non_biomass']
    table = pd.pivot_table(df, index='ele_impact', columns='demand_ratio',
                           values='electricity_biomass', aggfunc='mean')
    fig, ax = plt.subplots(1, 1, figsize=(9, 9), squeeze=True, subplot_kw={'aspect': 1})
    sns.heatmap(table, square=True, cbar_kws={'shrink': 0.6},
                linewidth=0.5, cbar=True, ax=ax, cmap=cmap,
                annot_kws={"size": 10, "va": "center_baseline", "color": "black"}, )
    fig.show()

    # heatmap, c_fossil (Mt)
    df.loc[df.c_fossil < 0, 'c_fossil'] = 0
    table = pd.pivot_table(df, index='ele_impact', columns='demand_ratio',
                           values='c_fossil', aggfunc='mean')

    fig, ax = plt.subplots(1, 1, figsize=(9, 9), squeeze=True, subplot_kw={'aspect': 1})
    sns.heatmap(table, square=True, annot=True, fmt=".0f",
                linewidth=0.5, cbar=False, ax=ax, cmap=cmap,
                annot_kws={"size": 10, "va": "center_baseline", "color": "black"}, )
    fig.show()

    # heatmap, fossil PJ
    # lhv: ecoinvent 3.10, market for petroleum, market for natural gas, high pressure
    df['fossil_PJ'] = (df['petroleum'] * 43.2 + df['natural_gas'] * 36 / 0.735)
    df.loc[df.fossil_PJ < 0, 'fossil_PJ'] = 0
    table = pd.pivot_table(df, index='ele_impact', columns='demand_ratio',
                           values='fossil_PJ', aggfunc='mean')
    fig, ax = plt.subplots(1, 1, figsize=(9, 9), squeeze=True, subplot_kw={'aspect': 1})
    sns.heatmap(table, square=True, annot=True, fmt=".0f",
                linewidth=0.5, cbar=False, ax=ax, cmap=cmap,
                annot_kws={"size": 10, "va": "center_baseline", "color": "black"}, )
    fig.show()

    # heatmap, new bioplastics PLA, PBS
    df['new_plastics_total'] = df['pla_total'] + df['pbs_total']
    table = pd.pivot_table(df, index='ele_impact', columns='demand_ratio',
                           values='new_plastics_total', aggfunc='mean')
    fig, ax = plt.subplots(1, 1, figsize=(9, 9), squeeze=True, subplot_kw={'aspect': 1})
    sns.heatmap(table, square=True, annot=True, fmt=".0f",
                linewidth=0.5, cbar=False, ax=ax, cmap=cmap,
                annot_kws={"size": 10, "va": "center_baseline", "color": "black"}, )
    fig.show()

    # heatmap, new bioplastics ratio PLA, PBS
    df['new_plastics_ratio'] = df['new_plastics_total'] / df['plastic_production']
    table = pd.pivot_table(df, index='ele_impact', columns='demand_ratio',
                           values='new_plastics_ratio', aggfunc='mean')
    fig, ax = plt.subplots(1, 1, figsize=(9, 9), squeeze=True, subplot_kw={'aspect': 1})
    sns.heatmap(table, square=True, annot=True, fmt=".1%",
                linewidth=0.5, cbar=False, ax=ax, cmap=cmap,
                annot_kws={"size": 10, "va": "center_baseline", "color": "black"}, )
    fig.show()

    # heatmap, forest residue
    table = pd.pivot_table(df, index='ele_impact', columns='demand_ratio',
                           values='forest_residue', aggfunc='mean')
    fig, ax = plt.subplots(1, 1, figsize=(9, 9), squeeze=True, subplot_kw={'aspect': 1})
    sns.heatmap(table, square=True, annot=True, fmt=".0f",
                linewidth=0.5, cbar=False, ax=ax, cmap=cmap,
                annot_kws={"size": 10, "va": "center_baseline", "color": "black"}, )
    fig.show()

    # heatmap, agricultural residue
    table = pd.pivot_table(df, index='ele_impact', columns='demand_ratio',
                           values='c_agri', aggfunc='mean')
    fig, ax = plt.subplots(1, 1, figsize=(9, 9), squeeze=True, subplot_kw={'aspect': 1})
    sns.heatmap(table, square=True, annot=True, fmt=".0f",
                linewidth=0.5, cbar=False, ax=ax, cmap=cmap,
                annot_kws={"size": 10, "va": "center_baseline", "color": "black"}, )
    fig.show()

    # fossil input percentage
    df['c_total'] = df['c_agri'] + df['c_forest'] + df['c_co2'] + df['c_fossil']
    df['fossil_percentage'] = df['c_fossil'] / df['c_total']
    table = pd.pivot_table(df, index='ele_impact', columns='demand_ratio',
                           values='fossil_percentage', aggfunc='mean')
    fig, ax = plt.subplots(1, 1, figsize=(9, 9), squeeze=True, subplot_kw={'aspect': 1})
    sns.heatmap(table, square=True, annot=True, fmt=".0f",
                linewidth=0.5, cbar=False, ax=ax, cmap=cmap,
                annot_kws={"size": 10, "va": "center_baseline", "color": "black"}, )
    fig.show()

    # plot heat vs health
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    sns.scatterplot(data=df, x='total_heat', y='health', ax=ax, hue='demand_ratio')
    plt.show()

    # plot electricity use vs biomass use
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    sns.scatterplot(data=df, x='electricity', y='biomass_use', ax=ax, hue='demand_ratio')
    plt.show()


def plot_ele_biomass_fossil_sensitivity(master_file_path, plastics_file_path):
    ele_impact_list = [round(i, 2) for i in np.arange(0, 0.401, 0.01)]
    biomass_ratio_list = [round(i, 2) for i in np.arange(0.0, 2.001, 0.05)]
    demand_list = [1]
    #demand_list = [0.4638, 1]
    base_path = os.path.join("data", "raw", "user_inputs_ccs_no_ccs")
    files = glob.glob(os.path.join(base_path, "*.xlsx"))
    df = pd.DataFrame()
    for user_input_file in files:
        user_input = MasterFile(user_input_file, master_file_path, plastics_file_path)
        if 'default' in user_input_file:
            scenario = 'with_ccs'
        else:
            scenario = 'without_ccs'
        df0 = user_input.sensitivity_demand_ele_biomass(ele_impact_list, demand_list, biomass_ratio_list)
        df0['scenario'] = scenario
        df = pd.concat([df, df0], ignore_index=True)
    df['fossil'] = df['natural_gas'] + df['petroleum']
    cmap = 'magma_r'
    df1 = df.loc[df.scenario == 'with_ccs']
    df2 = df.loc[df.scenario == 'without_ccs']
    table1 = pd.pivot_table(df1, index='ele_impact', columns='biomass_ratio', values='fossil', aggfunc='mean')
    table2 = pd.pivot_table(df2, index='ele_impact', columns='biomass_ratio', values='fossil', aggfunc='mean')
    table1 = table1.applymap(lambda x: np.nan if x < 1 else x)
    table2 = table2.applymap(lambda x: np.nan if x < 1 else x)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4), squeeze=True, subplot_kw={'aspect': 1})
    sns.heatmap(
        np.where(table1.isna(), 0, np.nan),
        ax=ax1, vmin=0, vmax=0,
        cbar=False, cmap=ListedColormap([colors_4[0]]), linewidth=0)
    sns.heatmap(table1, square=True, annot=False, fmt=".0f", vmin=0, vmax=800,
            linewidth=0.5, cbar=False, ax=ax1, cmap=cmap, cbar_kws={"shrink": 0.5},
            annot_kws={"size": 10, "va": "center_baseline", "color": "black"}, )
    sns.heatmap(
        np.where(table2.isna(), 0, np.nan),
        ax=ax2, vmin=0, vmax=0,
        cbar=False, cmap=ListedColormap(colors_4[0]), linewidth=0)
    sns.heatmap(table2, square=True, annot=False, fmt=".0f", vmin=0, vmax=800,
            linewidth=0.5, cbar=False, ax=ax2, cmap=cmap, cbar_kws={"shrink": 0.5},
            annot_kws={"size": 10, "va": "center_baseline", "color": "black"}, )
    sns.heatmap(table2, square=True, annot=False, fmt=".0f", vmin=0, vmax=3000,
                linewidth=0.5, cbar=True, ax=ax3, cmap=cmap, cbar_kws={"shrink": 0.5},
                annot_kws={"size": 10, "va": "center_baseline", "color": "black"}, )
    figname = f'figure/heatmap_fossil_use.pdf'
    #plt.savefig(figname, bbox_inches='tight')
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    fig.show()

def plot_scenarios(master_file_path, plastics_file_path):
    df0 = different_scenarios(master_file_path, plastics_file_path)
    df = df0[['scenario', 'ghg', 'bdv', 'health']].copy()
    df_fossil_bau = pd.DataFrame({'scenario': ['fossil_bau'], 'ghg': [3048.9], 'bdv': [0], 'health': [0.001338512]})
    df = pd.concat([df, df_fossil_bau], ignore_index=True)
    scenario_list = ['step1_fossil_linear', 'step6_ccs']
    df = df.loc[df.scenario.isin(scenario_list)].copy()
    df['scenario'] = pd.Categorical(df['scenario'], categories=scenario_list, ordered=True)
    df.sort_values('scenario', inplace=True)
    df.reset_index(drop=True, inplace=True)
    # normalize by dividing the highest value
    df['ghg'] = df['ghg'] / df['ghg'].max()
    df['bdv'] = df['bdv'] / df['bdv'].max()
    df['health'] = df['health'] / df['health'].max()

    features = ['ghg', 'bdv', 'health']
    num_vars = len(features)
    angles = np.linspace(np.pi / 2, 2 * np.pi + np.pi / 2, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    offset = 0.1
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, scenario in enumerate(df['scenario'].unique()):
        values = df.loc[df['scenario'] == scenario, features].values.flatten().tolist()
        values += values[:1]
        line_style = 'solid'
        if 'fossil' in scenario:
            color = 'grey'
            if 'bau' in scenario:
                line_style = '--'
        else:
            color = colors_5[4]
        ax.fill(angles, values, color=color, alpha=0.1)
        ax.plot(angles, values, color=color, linewidth=1, label=scenario, linestyle=line_style)
    # ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_ylim(-0.4, 1)
    ax.set_xticklabels(features)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
    plt.savefig(r'figure/scenarios_three_pillar.pdf')
    plt.show()

    # sensitivity, with no low bdv
    df = df0[['scenario', 'ghg', 'bdv', 'health']].copy()
    scenario_list2 = ['step1_fossil_linear', 'step6_ccs', 'sensitivity_no_low_bdv']
    df2 = df.loc[df.scenario.isin(scenario_list2)].copy()
    df1 = df2.loc[df.scenario.isin(scenario_list)].copy()
    df2['ghg'] = df2['ghg'] / df1['ghg'].max()
    df2['bdv'] = df2['bdv'] / df1['bdv'].max()
    df2['health'] = df2['health'] / df1['health'].max()
    df2['scenario'] = pd.Categorical(df2['scenario'], categories=scenario_list2, ordered=True)
    df2.sort_values('scenario', inplace=True)
    df2.reset_index(drop=True, inplace=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, scenario in enumerate(df2['scenario'].unique()):
        values = df2.loc[df2['scenario'] == scenario, features].values.flatten().tolist()
        values += values[:1]
        line_style = 'solid'
        if 'fossil' in scenario:
            color = 'grey'
        elif 'sensitivity' in scenario:
            color = colors_4[0]
        else:
            color = colors_5[4]
        ax.fill(angles, values, color=color, alpha=0.1)
        ax.plot(angles, values, color=color, linewidth=1, label=scenario, linestyle=line_style)
    # ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_ylim(-0.4, 2)
    ax.set_xticklabels(features)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
    plt.savefig(r'figure/scenarios_three_pillar_sensitivity_bdv.pdf')
    plt.show()


    df = df0[['scenario', 'ghg', 'bdv', 'health']].copy()
    df_fossil_bau = pd.DataFrame({'scenario': ['fossil_bau'], 'ghg': [3048.9], 'bdv': [0], 'health': [0.001338512]})
    df = pd.concat([df, df_fossil_bau], ignore_index=True)
    scenario_list = ['default', 'ccs', 'all_biomass', 'no_fossil', 'no_chem_recycling', 'fossil_lockin', 'half_biomass']
    df = df.loc[df.scenario.isin(scenario_list)].copy()
    df['scenario'] = pd.Categorical(df['scenario'], categories=scenario_list, ordered=True)
    df.sort_values('scenario', inplace=True)
    df.reset_index(drop=True, inplace=True)
    # normalize by dividing the highest value
    df['ghg'] = df['ghg'] / df['ghg'].max()
    df['bdv'] = df['bdv'] / df['bdv'].max()
    df['health'] = df['health'] / df['health'].max()
    features = ['ghg', 'bdv', 'health']

    # waterfall ghg
    df = df0[['scenario', 'ghg', 'bdv', 'health']].copy()
    df['ghg'] /= 1000  # Gt
    df = df.loc[df.scenario.str.contains('step')]
    df['ghg_diff'] = df['ghg'] - df['ghg'].shift(1)
    df['ghg_diff'].fillna(df['ghg'], inplace=True)
    df['ghg_bottom'] = df['ghg'].shift(1)
    df['ghg_bottom'].fillna(0, inplace=True)
    ghg = df.loc[df.scenario == 'step6_ccs', 'ghg'].values[0]
    df_final = pd.DataFrame({'scenario': ['after'], 'ghg_diff': [ghg], 'ghg_bottom': [0]})
    df = pd.concat([df, df_final], ignore_index=True)
    #df['scenario'] = ['fossil linear', 'renewable feedstock', 'non-drop-in bioplastics', 'mechanical recycling', 'chemical recycling', 'ccs', 'renewable circular']
    df['scenario'] = ['fossil linear', 'mechanical recycling', 'renewable feedstock', 'non-drop-in bioplastics', 'chemical recycling', 'ccs', 'renewable circular']
    fig, ax = plt.subplots(figsize=(16, 6))
    colors = [colors_5[3] for x in range(len(df))]
    colors[0] = 'grey'
    colors[-1] = colors_5[4]
    ax.bar(df['scenario'], df['ghg_diff'], bottom=df['ghg_bottom'], color=colors)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('GHG emissions (Gt CO2-eq)')
    #plt.savefig(r'figure/ghg_waterfall.pdf')
    #df.to_excel(r'data/figure/ghg_waterfall.xlsx')
    plt.show()

    df1 = df.copy()
    df1['reduction'] = df1['ghg_diff'] / df1.iloc[0, 1]
    ghg1 = df1.iloc[0, 1]
    ghg2 = df1.iloc[-2, 1]
    df1.loc[df1.scenario == 'renewable circular', 'reduction'] = (ghg2 - ghg1) / ghg1
    df1.loc[df1.scenario == 'fossil linear', 'reduction'] = 0
    df1 = df1[['scenario', 'ghg', 'reduction']]
    df1 = df1.iloc[0:6]



def plot_sensitivity_biomass_old(user_input):
    df0 = user_input.sensitivity_biomass_availability()
    biomass_c_list = ['c_biomass_to_plastics', 'c_biomass_to_ccs',
                      'c_biomass_loss', 'c_biomass_to_heat', 'c_biomass_to_electricity']
    df = df0[df0.biomass_ratio <= 2].copy()
    df2 = df.copy()
    for x in biomass_c_list:
        df2[x] = df2[x] / df2['c_biomass_in']
    dfp = df

    color_list = [colors_5[2], colors_5[3], colors_7[-1], colors_5[1], colors_5[0]]
    # stackplot
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.stackplot(dfp['biomass_ratio'], dfp[biomass_c_list[0]], dfp[biomass_c_list[1]],
                 dfp[biomass_c_list[2]], dfp[biomass_c_list[3]], dfp[biomass_c_list[4]],
                 colors=color_list, labels=biomass_c_list)
    max_point = dfp.loc[dfp['forest_residue'].idxmax(), 'biomass_ratio']
    # ax.axvline(x=max_point, color='grey', linestyle='--', linewidth=1)
    zero_point = dfp.loc[dfp.loc[dfp.index != 0, 'forest_residue'].idxmin(), 'biomass_ratio']
    # ax.axvline(x=zero_point, color='grey', linestyle='--', linewidth=1)
    ax.set_xlabel('Biomass availability ratio')
    ax.set_ylabel('Biomass use')
    ax.set_xlim(dfp['biomass_ratio'].min(), dfp['biomass_ratio'].max())
    ax.set_ylim(0, 2200)
    ax.legend()
    plt.savefig(r'figure/biomass_sensitivity_biomass_use_no_ccs.pdf')
    plt.show()

    # ghg impact plot
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.axhline(y=0, color='grey', linestyle='--', linewidth=1)
    sns.lineplot(x=dfp['biomass_ratio'], y=dfp['GHG'], ax=ax, color=colors_7[4], linewidth=1)
    ax.set_ylim(-1500, 2700)
    ax.set_xlim(dfp['biomass_ratio'].min(), dfp['biomass_ratio'].max())
    ax.legend()
    plt.savefig(r'figure/biomass_sensitivity_ghg_no_ccs.pdf')
    plt.show()

    #stacked bar plot
    fig, ax = plt.subplots(figsize=(20, 6))
    for i in range(len(biomass_c_list)):
        y = dfp[biomass_c_list[i]]
        x = dfp['biomass_ratio']
        if i == 0:
            bottom = 0
        else:
            bottom += dfp[biomass_c_list[i - 1]]
        sns.barplot(x=x, y=y, ax=ax, color=color_list[i], bottom=bottom, label=biomass_c_list[i])
    # plot a vertical line at forest residue max use:

    max_point = dfp['forest_residue'].idxmax()
    ax.axvline(x=max_point, color='black', linestyle='--', linewidth=0.5)
    zero_point = dfp.loc[dfp.index!=0, 'forest_residue'].idxmin()
    ax.axvline(x=zero_point, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Biomass availability ratio')
    ax.set_ylabel('Biomass use')
    ax.legend()
    plt.show()


def plot_system_contribution(master_file_path, plastics_file_path):
    df = system_contribution_analysis(master_file_path, plastics_file_path)
    df.loc[df.scenario.str.contains('fossil_linear'), 'scenario'] = 'fossil linear'
    df.loc[df.scenario.str.contains('ccs'), 'scenario'] = 'renewable circular'
    df['bdv'] *= 1e-6  # PDF
    df['health'] *= 1e9  # DALY
    df['ghg'] *= 1e-3  #Gt CO2eq
    dfp = pd.pivot_table(df, index='scenario', columns='contributor', values='health')
    df.to_excel(r'data/figure/system_contribution.xlsx')
    # ghg
    dfp = pd.pivot_table(df, index='scenario', columns='contributor', values='ghg')
    dfp['scope1_emission'] = dfp['onsite_heat'] + dfp['onsite_process']
    dfp['scope3_ccs'] = dfp['ccs']
    dfp['scope2_raw_material'] = dfp['feedstock_fossil_total'] + dfp['feedstock_biomass'] + dfp['feedstock_other']
    dfp['scope2_electricity'] = dfp['electricity_grid']
    dfp['scope3_waste'] = dfp['waste_treatment']
    dfp = dfp[['scope1_emission', 'scope2_raw_material', 'scope2_electricity', 'scope3_waste', 'scope3_ccs']]
    color_list = [colors_7[2], colors_7[1], colors_7[3], colors_7[0], colors_7[4]]
    fig, ax = plt.subplots(figsize=(7, 12))
    dfp.plot(kind='bar', stacked=True, ax=ax, color=color_list, edgecolor='white')
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=0.5)
    ax.scatter(x=[0, 1], y=list(dfp.sum(axis=1)), color='black', s=50, label='sum')
    ax.set_ylabel('climate change impact (Mt CO2eq)')
    plt.savefig(r'figure/system_contribution_ghg.pdf')

    plt.show()

    # bdv
    dfp = pd.pivot_table(df, index='scenario', columns='contributor', values='bdv')
    dfp['scope1_emission'] = dfp['onsite_heat'] + dfp['onsite_process']
    dfp['scope3_ccs'] = dfp['ccs']
    dfp['scope2_raw_material'] = dfp['feedstock_fossil_total'] + dfp['feedstock_biomass'] + dfp['feedstock_other']
    dfp['scope2_electricity'] = dfp['electricity_grid']
    dfp['scope3_waste'] = dfp['waste_treatment']
    dfp = dfp[['scope1_emission', 'scope2_raw_material', 'scope2_electricity', 'scope3_waste', 'scope3_ccs']]
    color_list = [colors_7[2], colors_7[1], colors_7[3], colors_7[0], colors_7[4]]
    fig, ax = plt.subplots(figsize=(7, 12))
    dfp.plot(kind='bar', stacked=True, ax=ax, color=color_list, edgecolor='white')
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=0.5)
    ax.scatter(x=[0, 1], y=list(dfp.sum(axis=1)), color='black', s=50, label='sum')
    ax.set_ylabel('biodiversity loss impact (PDF)')
    plt.savefig(r'figure/system_contribution_bdv.pdf')
    plt.show()

    # health
    dfp = pd.pivot_table(df, index='scenario', columns='contributor', values='health')
    dfp.sum(axis=1)
    dfp['scope1_emission'] = dfp['onsite_heat'] + dfp['onsite_process']
    dfp['scope3_ccs'] = dfp['ccs']
    dfp['scope2_raw_material'] = dfp['feedstock_fossil_total'] + dfp['feedstock_biomass'] + dfp['feedstock_other']
    dfp['scope2_electricity'] = dfp['electricity_grid']
    dfp['scope3_waste'] = dfp['waste_treatment']
    dfp = dfp[['scope1_emission', 'scope2_raw_material', 'scope2_electricity', 'scope3_waste', 'scope3_ccs']]
    color_list = [colors_7[2], colors_7[1], colors_7[3], colors_7[0], colors_7[4]]
    fig, ax = plt.subplots(figsize=(7, 12))
    dfp.plot(kind='bar', stacked=True, ax=ax, color=color_list, edgecolor='white')
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=0.5)
    ax.scatter(x=[0, 1], y=list(dfp.sum(axis=1)), color='black', s=50, label='sum')
    ax.set_ylabel('PM-related health impact (DALY)')
    plt.savefig(r'figure/system_contribution_health.pdf')
    plt.show()
    a=0


def plot_pareto_curves(user_input):
    df_result = user_input.model_results_multi_objective()
    df_result = df_result.loc[df_result.GHG < 5000]
    df1 = df_result[(df_result['GHG weight'] >= 0.999) & (df_result['GHG weight'] <= 1)]
    cmp = cmp_purple()
    cmp = cmp.reversed()
    indices = np.linspace(0, 255, 5, dtype=int)
    colors = cmp(indices)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(x='GHG', y='BDV', data=df_result, hue='Health Epsilon', palette=colors, ax=ax)
    sns.scatterplot(x='GHG', y='BDV', data=df1, color=colors_4[1], ax=ax)
    sns.lineplot(x='GHG', y='BDV', data=df1, color=colors_4[1], ax=ax, linestyle='--')
    ax.axvline(x=0, color='grey', linestyle='-', linewidth=0.25)
    plt.savefig(r'figure/pareto_curves.pdf')
    df_result.to_excel(r'data/figure/pareto_curves.xlsx')
    plt.show()
    a=0


def plot_sensitivity_electricity(master_file_path, plastics_file_path):
    user_input_file1 = r'data/raw/user_inputs_scenarios/user_inputs_step5_cr.xlsx'
    user_input_file2 = r'data/raw/user_inputs_scenarios/user_inputs_step6_ccs.xlsx'
    user_input_list = [user_input_file2, user_input_file1]
    ele_impact_list = [0, 0.001, 0.005] + [round(i, 3) for i in np.arange(0.01, 0.41, 0.01)]
    df = pd.DataFrame()
    for i in user_input_list:
        if i == user_input_file1:
            scenario = 'no_ccs'
        else:
            scenario = 'ccs'
        user_input = MasterFile(i, master_file_path, plastics_file_path)
        df0 = user_input.sensitivity_demand_ele_biomass(ele_impact_list, [1], [1])
        df0['scenario'] = scenario
        df = pd.concat([df, df0], ignore_index=True)
    df['ele_total'] = df['ele_non_biomass'] + df['ele_biomass']
    df.to_excel(r'data/figure/sensitivity_electricity.xlsx')
    df1 = df.loc[(df.scenario == 'no_ccs') & (df.biomass_ratio == 1)].copy()
    df2 = df.loc[(df.scenario == 'ccs') & (df.biomass_ratio == 1)].copy()
    # all
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    ax1.axhline(y=0, color='grey', linestyle='-', linewidth=0.25)
    sns.lineplot(x=df1['ele_impact'], y=df1['GHG'], ax=ax1, color=colors_7[4], linestyle='--', linewidth=1.5, label='without ccs, 100% biomass')
    sns.lineplot(x=df2['ele_impact'], y=df2['GHG'], ax=ax1, color=colors_7[4], linestyle='-', linewidth=1.5, label='with ccs, 100% biomass')
    ax1.set_xlim(0, 0.4)
    ax1.set_ylim(-600, 1800)
    dfp = df1
    ax2.stackplot(dfp['ele_impact'], dfp['ele_use_h2'], dfp['ele_use_base_chemical'], dfp['ele_use_polymerization'],
                 dfp['ele_use_mr'], dfp['ele_use_other'],
                 labels=['hydrogen production', 'base chemicals production', 'polymerization', 'mechanical recycling',
                         'other'],
                 colors=colors_5_with_grey)
    ax2.set_xlabel('electricity impact (kg CO2eq/kWh)')
    ax2.set_ylabel('electricity use (TWh)')
    ax2.set_xlim(0, 0.4)
    ax2.set_ylim(0, 9800)
    ax2.legend(loc='upper left', frameon=False)
    dfp = df2
    ax3.stackplot(dfp['ele_impact'], dfp['ele_use_h2'], dfp['ele_use_base_chemical'], dfp['ele_use_polymerization'],
                 dfp['ele_use_mr'], dfp['ele_use_other'],
                 labels=['hydrogen production', 'base chemicals production', 'polymerization', 'mechanical recycling',
                         'other'],
                 colors=colors_5_with_grey)
    ax3.set_xlabel('electricity impact (kg CO2eq/kWh)')
    ax3.set_ylabel('electricity use (TWh)')
    ax3.set_xlim(0, 0.4)
    ax3.set_ylim(0, 9800)
    plt.savefig(r'figure/sensitivity_electricity_all.pdf')
    plt.show()

    # plot ghg vs ele_impact
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), squeeze=True)
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=0.25)
    sns.lineplot(x=df1['ele_impact'], y=df1['GHG'], ax=ax, color=colors_7[4], linestyle='--', linewidth=1.5, label='without ccs, 100% biomass')
    sns.lineplot(x=df2['ele_impact'], y=df2['GHG'], ax=ax, color=colors_7[4], linestyle='-', linewidth=1.5, label='with ccs, 100% biomass')
    #sns.lineplot(x=df3['ele_impact'], y=df3['GHG'], ax=ax, color=colors_7[0], linestyle='--', linewidth=0.75, label='without ccs, 50% biomass')
    #sns.lineplot(x=df4['ele_impact'], y=df4['GHG'], ax=ax, color=colors_7[0], linestyle='-', linewidth=0.75, label='with ccs, 50% biomass')
    ax.set_xlim(0, 0.4)
    ax.set_ylim(-600, 1800)
    plt.savefig(r'figure/sensitivity_electricity_ghg.pdf')
    plt.show()


    # plot electricity use vs ele_impact
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), squeeze=True)
    dfp = df1
    ax.stackplot(dfp['ele_impact'], dfp['ele_use_h2'], dfp['ele_use_base_chemical'], dfp['ele_use_polymerization'],
                 dfp['ele_use_mr'], dfp['ele_use_other'],
                 labels=['hydrogen production', 'base chemicals production', 'polymerization', 'mechanical recycling',
                         'other'],
                 colors=colors_5_with_grey)
    plt.xlabel('electricity impact (kg CO2eq/kWh)')
    plt.ylabel('electricity use (TWh)')
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0, 9800)
    plt.legend(loc='upper left', frameon=False)
    plt.savefig(r'figure/sensitivity_electricity_use_no_ccs.pdf')
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), squeeze=True)
    dfp = df2
    ax.stackplot(dfp['ele_impact'], dfp['ele_use_h2'], dfp['ele_use_base_chemical'], dfp['ele_use_polymerization'],
                 dfp['ele_use_mr'], dfp['ele_use_other'],
                 labels=['hydrogen production', 'base chemicals production', 'polymerization', 'mechanical recycling',
                         'other'],
                 colors=colors_5_with_grey)
    plt.xlabel('electricity impact (kg CO2eq/kWh)')
    plt.ylabel('electricity use (TWh)')
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0, 9800)
    plt.legend(loc='upper left', frameon=False)
    plt.savefig(r'figure/sensitivity_electricity_use_with_ccs.pdf')
    plt.show()
    return df


def plot_sensitivity_biomass(master_file_path, plastics_file_path):
    user_input_file1 = r'data/raw/user_inputs_scenarios/user_inputs_step5_cr.xlsx'
    user_input_file2 = r'data/raw/user_inputs_scenarios/user_inputs_step6_ccs.xlsx'
    user_input_list = [user_input_file2, user_input_file1]
    biomass_ratio_list = [round(i, 2) for i in np.arange(0, 4.1, 0.1)]
    df = pd.DataFrame()
    for i in user_input_list:
        if i == user_input_file1:
            scenario = 'no_ccs'
        else:
            scenario = 'ccs'
        user_input = MasterFile(i, master_file_path, plastics_file_path)
        df0 = user_input.sensitivity_demand_ele_biomass([-999], [1], biomass_ratio_list)
        df0['scenario'] = scenario
        df = pd.concat([df, df0], ignore_index=True)
    df.to_excel(r'data/figure/sensitivity_biomass.xlsx')
    df1 = df.loc[(df.scenario == 'no_ccs')].copy()
    df2 = df.loc[(df.scenario == 'ccs')].copy()
    biomass_c_list = ['c_biomass_to_plastics', 'c_biomass_to_ccs',
                      'c_biomass_loss', 'c_biomass_to_heat', 'c_biomass_to_electricity']
    color_list = [colors_5[2], colors_5[3], colors_7[-1], colors_5[1], colors_5[0]]
    # all
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    ax1.axhline(y=0, color='grey', linestyle='-', linewidth=1)
    sns.lineplot(x=df1['biomass_ratio'], y=df1['GHG'], ax=ax1, color=colors_7[4],
                 linestyle='--', linewidth=1.5, label='without ccs, 100% biomass')
    sns.lineplot(x=df2['biomass_ratio'], y=df2['GHG'], ax=ax1, color=colors_7[4],
                 linestyle='-', linewidth=1.5, label='without ccs, 100% biomass')
    ax1.set_ylim(-1600, 2700)
    ax1.set_xlim(0, 2)
    ax1.legend()
    dfp = df1
    ax2.stackplot(dfp['biomass_ratio'], dfp[biomass_c_list[0]], dfp[biomass_c_list[1]],
                 dfp[biomass_c_list[2]], dfp[biomass_c_list[3]], dfp[biomass_c_list[4]],
                 colors=color_list, labels=biomass_c_list)
    ax2.set_xlabel('Biomass availability ratio')
    ax2.set_ylabel('Biomass use')
    ax2.set_xlim(0, 2)
    ax2.set_ylim(0, 2200)
    ax2.legend()
    dfp = df2
    ax3.stackplot(dfp['biomass_ratio'], dfp[biomass_c_list[0]], dfp[biomass_c_list[1]],
                 dfp[biomass_c_list[2]], dfp[biomass_c_list[3]], dfp[biomass_c_list[4]],
                 colors=color_list, labels=biomass_c_list)
    ax3.set_xlabel('Biomass availability ratio')
    ax3.set_ylabel('Biomass use')
    ax3.set_xlim(0, 2)
    ax3.set_ylim(0, 2200)
    plt.savefig(r'figure/biomass_sensitivity_all.pdf')
    plt.show()
    df3 = df[df.biomass_ratio==1].copy()
    for x in biomass_c_list:
        df3[x] = df3[x] / df3['c_biomass_in']
    # plot ghg vs biomass_ratio
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=1)
    sns.lineplot(x=df1['biomass_ratio'], y=df1['GHG'], ax=ax, color=colors_7[4],
                 linestyle='--', linewidth=1.5, label='without ccs, 100% biomass')
    sns.lineplot(x=df2['biomass_ratio'], y=df2['GHG'], ax=ax, color=colors_7[4],
                 linestyle='-', linewidth=1.5, label='without ccs, 100% biomass')
    ax.set_ylim(-1600, 2700)
    ax.set_xlim(0, 2)
    ax.legend()
    plt.savefig(r'figure/biomass_sensitivity_ghg.pdf')
    plt.show()

    biomass_c_list = ['c_biomass_to_plastics', 'c_biomass_to_ccs',
                      'c_biomass_loss', 'c_biomass_to_heat', 'c_biomass_to_electricity']
    df3 = df.copy()
    for x in biomass_c_list:
        df3[x] = df3[x] / df3['c_biomass_in']

    color_list = [colors_5[2], colors_5[3], colors_7[-1], colors_5[1], colors_5[0]]
    # stackplot
    dfp = df1
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.stackplot(dfp['biomass_ratio'], dfp[biomass_c_list[0]], dfp[biomass_c_list[1]],
                 dfp[biomass_c_list[2]], dfp[biomass_c_list[3]], dfp[biomass_c_list[4]],
                 colors=color_list, labels=biomass_c_list)
    ax.set_xlabel('Biomass availability ratio')
    ax.set_ylabel('Biomass use')
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2200)
    ax.legend()
    plt.savefig(r'figure/biomass_sensitivity_biomass_use_no_ccs.pdf')
    plt.show()

    dfp = df2
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.stackplot(dfp['biomass_ratio'], dfp[biomass_c_list[0]], dfp[biomass_c_list[1]],
                 dfp[biomass_c_list[2]], dfp[biomass_c_list[3]], dfp[biomass_c_list[4]],
                 colors=color_list, labels=biomass_c_list)
    ax.set_xlabel('Biomass availability ratio')
    ax.set_ylabel('Biomass use')
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2200)
    ax.legend()
    plt.savefig(r'figure/biomass_sensitivity_biomass_use_ccs.pdf')
    plt.show()


def plot_fossil_fuel_impacts(master_file_path, plastics_file_path):
    user_input_file = r'data/raw/user_inputs.xlsx'
    base_path1 = os.path.join("data", "raw", "user_inputs_fossil_impacts")
    base_path2 = os.path.join("data", "raw", "user_inputs_fossil_impacts_no_ccs")
    ele_impact_list = [0, 0.001, 0.005] + [round(i, 3) for i in np.arange(0.01, 0.41, 0.01)]
    demand_list = [1]
    df_all = pd.DataFrame()
    for base_path in [base_path1, base_path2]:
        files = glob.glob(os.path.join(base_path, "*.xlsx"))
        if base_path == base_path1:
            scenario2 = 'ccs'
        else:
            scenario2 = 'no_ccs'
        for user_input_file in files:
            user_input = MasterFile(user_input_file, master_file_path, plastics_file_path)
            scenario = user_input_file.split('_')[-1].split('.')[0]
            print(scenario2, user_input_file)
            df1 = user_input.sensitivity_demand_ele_biomass(ele_impact_list, demand_list, [1])
            df1['scenario'] = scenario
            df1['scenario2'] = scenario2
            df_all = pd.concat([df_all, df1], ignore_index=True)
    df = df_all.copy()
    df.loc[df.scenario == 'in', 'scenario'] = 'fossil_lock_in'
    df.loc[df.scenario == 'fossil', 'scenario'] = 'no_fossil'
    df.to_excel(r'data/figure/fossil_fuel_impacts.xlsx')
    df1 = df.loc[df.scenario2 == 'ccs'].copy()
    df2 = df.loc[df.scenario2 == 'no_ccs'].copy()
    table1 = pd.pivot_table(df1, index='ele_impact', columns='scenario', values='GHG')
    table1['delta'] = table1['fossil_lock_in'] - table1['default']
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [1, 1]}, figsize=(12, 6))
    sns.lineplot(x='ele_impact', y='GHG', data=df2, hue='scenario', ax=ax1, palette=colors_5)
    sns.lineplot(x='ele_impact', y='GHG', data=df1, hue='scenario', ax=ax2, palette=colors_5)
    ax1.set_ylim(-500, 6000)
    ax1.set_xlim(0, 0.4)
    ax2.set_xlim(0, 0.4)
    ax1.axhline(y=0, color='grey')
    ax1.axvline(x=0.07, color='grey')
    ax2.axhline(y=0, color='grey')
    ax2.axvline(x=0.07, color='grey')
    #plt.savefig(r'figure/fossil_fuel_impacts_no_ccs.pdf')
    plt.savefig(r'figure/fossil_fuel_impacts.pdf')
    plt.show()
    return df


def plot_allocation(master_file_path, plastics_file_path):
    base_path = os.path.join("data", "raw", "user_inputs_allocation")
    files = glob.glob(os.path.join(base_path, "*.xlsx"))
    df_list = []
    for f in files:
        scenario_name = f.split('inputs_')[2].split('.')[0]
        print('-----------', scenario_name, '-----------')
        user_input = MasterFile(f, master_file_path, plastics_file_path)

        if scenario_name == 'default':
            df_default1, df_default2 = user_input.carbon_flow_sankey('GHG')
        else:
            df_se1, df_se2 = user_input.carbon_flow_sankey('GHG')

        df1, df2 = user_input.model_results('GHG')
        df1['scenario'] = scenario_name
        df_list.append(df1)
    df0 = pd.concat(df_list, ignore_index=True)
    df0.fillna(0, inplace=True)

    df_default1['impact_flow'] = df_default1['flowxvalue'] * df_default1['ghg']
    df_default11 = df_default1.dropna(subset=['impact_flow'])
    df_default11 = df_default11.loc[df_default11['impact_flow'] != 0]
    df_default11 = df_default11[['process', 'product_name', 'type', 'unit', 'impact_flow']]
    df_se1['impact_flow'] = df_se1['flowxvalue'] * df_se1['ghg']
    df_se11 = df_se1.dropna(subset=['impact_flow'])
    df_se11 = df_se11.loc[df_se11['impact_flow'] != 0]
    df_se11 = df_se11[['process', 'product_name', 'type', 'unit', 'impact_flow']]
    df11 = pd.merge(df_default11, df_se11, on=['process', 'product_name', 'type', 'unit'], how='outer', suffixes=('_default', '_se'))
    df11.fillna(0, inplace=True)
    df11['impact_flow_delta'] = df11['impact_flow_se'] - df11['impact_flow_default']
    df11['group'] = 'others'
    #df11.loc[df11.type == 'EMISSION', 'group'] = 'process emissions'
    #df11.loc[df11.process.str.contains('CCS'), 'group'] = 'CCS'
    #df11.loc[df11.product_name.isin(residue_list_code), 'group'] = 'feedstock_biomass'
    #df11.loc[df11.product_name.isin(['petroleum', 'natural_gas']), 'group'] = 'feedstock_fossil'
    df11.loc[df11.process.str.contains('heat'), 'group'] = 'heat'
    #df11.loc[df11.product_name.str.contains('electricity'), 'group'] = 'electricity'
    #df11.loc[df11.process.str.contains('waste'), 'group'] = 'plastic_waste_treatment'
    df12 = df11.groupby('group').sum()
    df12.rename(columns={'impact_flow_default': 'economic allocation (baseline)',
                         'impact_flow_se': 'substitution'}, inplace=True)
    df12 = df12[['economic allocation (baseline)', 'substitution']].T
    df12 /= 1000
    mpl.rcParams.update({'font.size': 6})
    fig = plt.figure(figsize=(18 / 2.54, 10 / 2.54))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    ax = plt.subplot(gs[0])
    df12.plot(kind='bar', stacked=True, ax=ax, color=colors_4, rot=0, legend=False)
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=0.5)
    ax.scatter(x=[0, 1], y=list(df12.sum(axis=1)), color=colors_7[4], s=10, label='net impact')
    legend_ax = plt.subplot(gs[1])
    legend_ax.axis('off')
    legend_ax.legend(*ax.get_legend_handles_labels(), loc='lower left', frameon=False, title='')
    ax.set_ylabel('climate change impact (Gt CO$_{2}$-eq)')
    ax.xaxis.set_ticks_position('none')
    plt.tight_layout()
    plt.savefig(r'figure/ghg_allocation.png', bbox_inches='tight', dpi=300)
    plt.show()
    return df12


def plot_biogenic_carbon_impact(user_input):
    df_flow, df_result = user_input.sensitivity_biogenic_carbon_impact()
    df = df_flow.copy()
    df.fillna(0, inplace=True)
    df['group'] = 'others'
    #df = df.copy()
    df['delta'] = df['ghg_flow_zero'] - df['ghg_flow_default']
    df.loc[df.process.str.startswith('heat'), 'group'] = 'heat from natural gas'
    df.loc[df.product_name == 'co2_emission_biogenic_long', 'group'] = 'biogenic CO$_{2}$ from forest residues'
    df1 = df.groupby('group').sum(numeric_only=True)
    df1 /= 1000
    df1.rename(columns={'ghg_flow_default': 'GWP100$_{bio}$=0.38 (baseline)', 'ghg_flow_zero': 'GWP100$_{bio}$=0'},
               inplace=True)
    df1 = df1[['GWP100$_{bio}$=0.38 (baseline)', 'GWP100$_{bio}$=0']].copy().T
    mpl.rcParams.update({'font.size': 6})
    fig = plt.figure(figsize=(18 / 2.54, 10 / 2.54))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    ax = plt.subplot(gs[0])
    color_order = [colors_4[0], colors_4[2], colors_4[1]]
    df1.plot(kind='bar', stacked=True, ax=ax, color=color_order, rot=0, legend=False)
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=0.5)
    ax.scatter(x=[0, 1], y=list(df1.sum(axis=1)), color=colors_7[4], s=10, label='net impact')
    legend_ax = plt.subplot(gs[1])
    legend_ax.axis('off')
    legend_ax.legend(*ax.get_legend_handles_labels(), loc='lower left', frameon=False, title='')
    ax.set_ylabel('climate change impact (Gt CO$_{2}$-eq)')
    ax.xaxis.set_ticks_position('none')
    plt.tight_layout()
    plt.savefig(r'figure/ghg_biogenic_carbon.png', bbox_inches='tight', dpi=300)
    plt.show()
    return df