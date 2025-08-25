import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import networkx as nx
import geopandas as gpd
import os
import glob

from src.variable_declaration import (final_product_list,
                                      )
from src.optimization import (regional_results,
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
    vals[:, 0] = np.concatenate((np.linspace(251 / 256, 242 / 256, 100),
                                 np.linspace(242 / 256, 201 / 256, 156)))
    vals[:, 1] = np.concatenate((np.linspace(243 / 256, 225 / 256, 100),
                                 np.linspace(225 / 256, 100 / 256, 156)))
    vals[:, 2] = np.concatenate((np.linspace(207 / 256, 139 / 256, 100),
                                 np.linspace(139 / 256, 32 / 256, 156)))

    newcmp = ListedColormap(vals)
    return newcmp


def cmp_purple():
    vals = np.ones((256, 4))
    vals[:, 0] = (np.linspace(39 / 256, 230 / 256, 256))
    vals[:, 1] = (np.linspace(67 / 256, 231 / 256, 256))
    vals[:, 2] = (np.linspace(146 / 256, 242 / 256, 256))
    newcmp = ListedColormap(vals)
    return newcmp


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
    #df2 = df2.groupby(['product_from', 'product_to']).sum(numeric_only=True).reset_index()
    df1 = pd.concat([df1, df2], ignore_index=True)
    suffix_list = ['_biogenic_short', '_biogenic_long', '_fossil', '_co2']
    for suffix in suffix_list:
        df1.loc[df1['product_from'].str.contains(suffix), 'product_from'] = df1['product_from'].str.replace(suffix, '')
        df1.loc[df1['product_to'].str.contains(suffix), 'product_to'] = df1['product_to'].str.replace(suffix, '')
        df.loc[df['product_name'].str.contains(suffix), 'product_name'] = df['product_name'].str.replace(suffix, '')
    df1['flowxvalue'] = df1['flowxvalue'].astype(float)
    df1 = df1.groupby(['product_from', 'product_to']).sum(numeric_only=True).reset_index()
    df.loc[df['product_type'] == 'intermediate', 'color'] = colors_5[1]
    df.loc[df['product_type'] == 'waste', 'color'] = colors_5[4]
    df1 = df1.loc[df1.product_from != 'water']
    df1 = df1.loc[df1.product_to != 'electricity']
    df1 = df1.loc[df1.product_to != 'heat_high']
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


def plot_region(master_file_path, plastics_file_path):
    df = regional_results(master_file_path, plastics_file_path)

    world_ghg = df.loc[df.country.str.contains('World'), 'ghg'].values[0]
    world_ghg_r = df.loc[~df.country.str.contains('World'), 'ghg'].sum()
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
    df1['plastics_health_intensity'] *= 1e6
    fig, ax = plt.subplots(1, 1, figsize=(11.5, 8), squeeze=True)
    p1 = sns.scatterplot(data=df1, x='plastics_ghg_intensity', y='plastics_bdv_intensity', size='size',
                         sizes=(300, 3000), hue='plastics_health_intensity', ax=ax, palette=cmp_purple().reversed(),
                         alpha=0.8, hue_norm=(0.2, 1.5))
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

    # stacked bar bdv
    dfp = df1[['country', 'bdv']].sort_values(by='bdv', ascending=False).copy()
    bottom = 0
    fig, ax = plt.subplots(1, 1, figsize=(6, 8), squeeze=True)
    for x in dfp.index:
        ax.bar('regional optimization', dfp.loc[x, 'bdv'], bottom=bottom)
        bottom += dfp.loc[x, 'bdv']
    plt.show()

    # stacked bar health
    dfp = df1[['country', 'health']].sort_values(by='health', ascending=False).copy()
    bottom = 0
    fig, ax = plt.subplots(1, 1, figsize=(6, 8), squeeze=True)
    for x in dfp.index:
        ax.bar('regional optimization', dfp.loc[x, 'health'], bottom=bottom)
        bottom += dfp.loc[x, 'health']
    plt.show()

    # stacked bar all impacts， percentage of the world
    dfp = df1[['country', 'plastic_production', 'ghg', 'bdv', 'health']].copy()
    dfp.loc[dfp.country == 'CEU', 'country'] = 'EUR'
    dfp.loc[dfp.country == 'WEU', 'country'] = 'EUR'
    dfp = dfp.groupby('country').sum(numeric_only=True).reset_index()
    dfp['plastic_production'] = dfp['plastic_production'] / dfp['plastic_production'].sum() * 100
    ghg_total_pos = dfp.loc[dfp.ghg > 0, 'ghg'].sum()
    dfp['ghg'] = dfp['ghg'] / dfp['ghg'].sum() * 100
    dfp['bdv'] = dfp['bdv'] / dfp['bdv'].sum() * 100
    dfp['health'] = dfp['health'] / dfp['health'].sum() * 100
    dfp['color'] = colors_7[-1]
    dfp.loc[dfp.country == 'CHN', 'color'] = colors_7[0]
    dfp.loc[dfp.country == 'USA', 'color'] = colors_7[3]
    dfp.loc[dfp.country == 'BRA', 'color'] = "#E2B597"
    dfp.loc[dfp.country == 'SEAS', 'color'] = colors_7[5]
    dfp.loc[dfp.country == 'IND', 'color'] = colors_7[1]
    dfp.loc[dfp.country == 'EUR', 'color'] = colors_7[2]
    dfp.loc[dfp.country == 'ME', 'color'] = colors_7[4]
    dfp.loc[dfp.color == colors_7[-1], 'country'] = 'other'
    dfp_ghg_p = dfp.loc[dfp.ghg > 0].copy()
    dfp_ghg_n = dfp.loc[dfp.ghg < 0].copy()
    dfp_ghg_p = dfp_ghg_p.groupby(['color', 'country']).sum(numeric_only=True).reset_index()
    dfp_ghg_n = dfp_ghg_n.groupby(['color', 'country']).sum(numeric_only=True).reset_index()
    dfp_ghg_p.sort_values(by='ghg', ascending=True, inplace=True)
    dfp_ghg_n.sort_values(by='ghg', ascending=False, inplace=True)
    dfp_ghg_p.loc[dfp_ghg_p.country == 'other', 'country'] = 'other positive'
    dfp_ghg_n.loc[dfp_ghg_n.country == 'other', 'country'] = 'other negative'
    dfp_ghg = pd.concat([dfp_ghg_p, dfp_ghg_n], ignore_index=True)
    #dfp_ghg.sort_values(by='ghg', ascending=False, inplace=True)
    condition = dfp_ghg['country'] == 'CHN'
    dfp_ghg = pd.concat([dfp_ghg[condition], dfp_ghg[~condition]]).reset_index(drop=True)
    country_order = dfp_ghg['country'].unique()
    country_order = [x for x in country_order if 'other' not in x][::-1]
    country_order = country_order + ['other']
    dfp = dfp.groupby(['color', 'country']).sum(numeric_only=True).reset_index()
    dfp['country'] = pd.Categorical(dfp['country'],
                                    categories=country_order,
                                    ordered=True)
    dfp = dfp.sort_values('country')

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), squeeze=True)
    bottom = 0
    for x in dfp.index:
        ax.bar('plastic_production', dfp.loc[x, 'plastic_production'], bottom=bottom, color=dfp.loc[x, 'color'])
        bottom += dfp.loc[x, 'plastic_production']
    bottom = 0
    for x in dfp.index:
        ax.bar('health', dfp.loc[x, 'health'], bottom=bottom, color=dfp.loc[x, 'color'])
        bottom += dfp.loc[x, 'health']
    bottom = 0
    for x in dfp.index:
        ax.bar('bdv', dfp.loc[x, 'bdv'], bottom=bottom, color=dfp.loc[x, 'color'])
        bottom += dfp.loc[x, 'bdv']
    ax.set_ylim(0, 100)
    plt.savefig(r'figure/regional_contribution_impacts.pdf')
    plt.show()

    # stacked bar climate change impacts
    fig, ax = plt.subplots(1, 1, figsize=(6, 8), squeeze=True)
    neg_b = 0
    pos_b = 0
    for x in dfp_ghg.index:
        value = dfp_ghg.loc[x, 'ghg']
        color = dfp_ghg.loc[x, 'color']
        if value < 0:
            ax.bar('regional optimization', value, bottom=neg_b, color=color)
            neg_b += value
        else:
            ax.bar('regional optimization', value, bottom=pos_b, color=color)
            pos_b += value
    ax.bar('global optimization', world_ghg, color=colors_7[6])
    ax.scatter('regional optimization', world_ghg_r, color=colors_7[6], edgecolor='white', s=80)
    ax.scatter('global optimization', world_ghg, color=colors_7[6], edgecolor='white', s=80)
    #plt.savefig(r'figure/ghg_global_vs_regional_sum.pdf')
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
    df3.to_csv(r'data/figure/carbon_input_source_key_countries.csv')
    plt.show()
    mpl.rcParams['hatch.linewidth'] = 1.0
    mpl.rcParams['hatch.color'] = 'black'


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
    # for paper
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
    # for thesis
    fig, ax = plt.subplots(1, 1, figsize=(18 / 2.54, 10 / 2.54))
    df_image.plot(column='IMAGE_region',
                  ax=ax,
                  legend=True,
                  linewidth=.5,
                  edgecolor='white',
                  cmap=cmap,
                  missing_kwds={'color': 'white', 'label': 'Missing values'},
                  legend_kwds={'loc': 'lower center',
                               'bbox_to_anchor': (0.5, -0.25),
                               'ncol': 7,  # Adjust this value as needed
                               'frameon': False,
                               'markerscale': 0.7,
                               'fontsize': 8})
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(r'figure/image_region_map.pdf', bbox_inches='tight')
    plt.show()
    return df


def plot_demand_sensitivity(master_file_path, plastics_file_path):
    ele_impact_list = [-999]
    biomass_ratio_list = [1]
    demand_list = [round(i, 2) for i in np.arange(0.1, 1.501, 0.025)]
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


def plot_ele_biomass_demand_sensitivity(user_input_file, master_file_path, plastics_file_path):
    ele_impact_list = [round(i, 2) for i in np.arange(0, 0.401, 0.01)]
    biomass_ratio_list = [round(i, 2) for i in np.arange(0.0, 2.001, 0.05)]
    base_path = os.path.join("data", "raw", "user_inputs_fossil_impacts_no_ccs")
    files = glob.glob(os.path.join(base_path, "*.xlsx"))
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
    cmap = 'magma_r'
    df1 = df.loc[df.scenario == 'default']
    df2 = df.loc[df.scenario == 'no_fossil']
    df3 = df.loc[df.scenario == 'fossil_lock_in']
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


def plot_scenarios(master_file_path, plastics_file_path):
    df0 = different_scenarios(master_file_path, plastics_file_path)
    df = df0[['scenario', 'ghg', 'bdv', 'health']].copy()
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

    # radar plot, three pilar
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, scenario in enumerate(df['scenario'].unique()):
        values = df.loc[df['scenario'] == scenario, features].values.flatten().tolist()
        values += values[:1]
        line_style = 'solid'
        if 'fossil' in scenario:
            color = 'grey'
        else:
            color = colors_5[4]
        ax.fill(angles, values, color=color, alpha=0.1)
        ax.plot(angles, values, color=color, linewidth=1, label=scenario, linestyle=line_style)
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
    ax.set_xticks(angles[:-1])
    ax.set_ylim(-0.4, 2)
    ax.set_xticklabels(features)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
    plt.savefig(r'figure/scenarios_three_pillar_sensitivity_bdv.pdf')
    plt.show()

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
    plt.savefig(r'figure/ghg_waterfall.pdf')
    df.to_excel(r'data/figure/ghg_waterfall.xlsx')
    plt.show()


def plot_system_contribution(master_file_path, plastics_file_path, country):
    df = system_contribution_analysis(master_file_path, plastics_file_path, country)
    df.to_excel(r'data/figure/system_contribution.xlsx')
    mpl.rcParams['hatch.linewidth'] = 3
    hatch_patterns = ['', '//', '', '', '', '']
    color_list = [colors_7[2], colors_7[1], colors_7[1], colors_7[3], colors_7[0], colors_7[4]]
    # ghg
    dfp = pd.pivot_table(df, index='scenario', columns='contributor', values='ghg')
    dfp['scope1_emission'] = dfp['onsite_heat'] + dfp['onsite_process']
    dfp['scope3_ccs'] = dfp['ccs']
    dfp['scope2_raw_material_b'] = dfp['feedstock_biomass']
    dfp['scope2_raw_material_o'] = dfp['feedstock_fossil_total'] + dfp['feedstock_other']
    dfp['scope2_electricity'] = dfp['electricity_grid']
    dfp['scope3_waste'] = dfp['waste_treatment']
    dfp = dfp[['scope1_emission', 'scope2_raw_material_b', 'scope2_raw_material_o','scope2_electricity', 'scope3_waste', 'scope3_ccs']]
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = dfp.plot(kind='bar', stacked=True, ax=ax, color=color_list, edgecolor='white')
    for i, (col_name, hatch) in enumerate(zip(dfp.columns, hatch_patterns)):
        if hatch:  # Only apply if hatch pattern is not empty
            for j, bar in enumerate(ax.containers[i]):
                bar.set_hatch(hatch)
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=0.5)
    ax.scatter(x=[0, 1], y=list(dfp.sum(axis=1)), color='black', s=50, label='sum')
    ax.set_ylabel('climate change impact (Mt CO2eq)')
    plt.savefig(r'figure/system_contribution_ghg.pdf')
    plt.show()

    # bdv
    dfp = pd.pivot_table(df, index='scenario', columns='contributor', values='bdv')
    dfp['scope1_emission'] = dfp['onsite_heat'] + dfp['onsite_process']
    dfp['scope3_ccs'] = dfp['ccs']
    dfp['scope2_raw_material_b'] = dfp['feedstock_biomass']
    dfp['scope2_raw_material_o'] = dfp['feedstock_fossil_total'] + dfp['feedstock_other']
    dfp['scope2_electricity'] = dfp['electricity_grid']
    dfp['scope3_waste'] = dfp['waste_treatment']
    dfp = dfp[['scope1_emission', 'scope2_raw_material_b', 'scope2_raw_material_o','scope2_electricity', 'scope3_waste', 'scope3_ccs']]
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = dfp.plot(kind='bar', stacked=True, ax=ax, color=color_list, edgecolor='white')
    for i, (col_name, hatch) in enumerate(zip(dfp.columns, hatch_patterns)):
        if hatch:  # Only apply if hatch pattern is not empty
            for j, bar in enumerate(ax.containers[i]):
                bar.set_hatch(hatch)
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
    dfp['scope2_raw_material_b'] = dfp['feedstock_biomass']
    dfp['scope2_raw_material_o'] = dfp['feedstock_fossil_total'] + dfp['feedstock_other']
    dfp['scope2_electricity'] = dfp['electricity_grid']
    dfp['scope3_waste'] = dfp['waste_treatment']
    dfp = dfp[['scope1_emission', 'scope2_raw_material_b', 'scope2_raw_material_o','scope2_electricity', 'scope3_waste', 'scope3_ccs']]
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = dfp.plot(kind='bar', stacked=True, ax=ax, color=color_list, edgecolor='white')
    for i, (col_name, hatch) in enumerate(zip(dfp.columns, hatch_patterns)):
        if hatch:  # Only apply if hatch pattern is not empty
            for j, bar in enumerate(ax.containers[i]):
                bar.set_hatch(hatch)
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=0.5)
    ax.scatter(x=[0, 1], y=list(dfp.sum(axis=1)), color='black', s=50, label='sum')
    ax.set_ylabel('PM-related health impact (DALY)')
    plt.savefig(r'figure/system_contribution_health.pdf')
    plt.show()


def plot_pareto_curves(user_input):
    df_result = user_input.model_results_multi_objective()
    df_result = df_result.loc[df_result.GHG < 5000]
    df1 = df_result[(df_result['GHG weight'] >= 0.999) & (df_result['GHG weight'] <= 1)]
    df2 = df_result
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(x='GHG', y='BDV', data=df2, color=colors_7[0], ax=ax)
    ax.axvline(x=0, color='grey', linestyle='-', linewidth=0.25)
    plt.savefig(r'figure/pareto_curves.pdf')
    df_result.to_excel(r'data/figure/pareto_curves.xlsx')
    plt.show()


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
    sns.lineplot(x=df1['ele_impact'], y=df1['GHG'], ax=ax1, color=colors_4[0], linestyle='-', linewidth=1.5, label='without ccs, 100% biomass')
    sns.lineplot(x=df2['ele_impact'], y=df2['GHG'], ax=ax1, color=colors_4[1], linestyle='-', linewidth=1.5, label='with ccs, 100% biomass')
    ax1.set_xlim(0, 0.4)
    ax1.set_ylim(-600, 1800)
    dfp = df1
    dfp['ele_use_other'] += dfp['ele_use_base_chemical'] + dfp['ele_use_polymerization'] + dfp['ele_use_mr']
    ax2.stackplot(dfp['ele_impact'], dfp['ele_use_h2'], dfp['ele_use_other'],
                 labels=['hydrogen production', 'other'],
                 colors=colors_5_with_grey)
    ax2.set_xlabel('electricity impact (kg CO2eq/kWh)')
    ax2.set_ylabel('electricity use (TWh)')
    ax2.set_xlim(0, 0.4)
    ax2.set_ylim(0, 9800)
    ax2.legend(loc='upper left', frameon=False)
    dfp = df2
    dfp['ele_use_other'] += dfp['ele_use_base_chemical'] + dfp['ele_use_polymerization'] + dfp['ele_use_mr']
    ax3.stackplot(dfp['ele_impact'], dfp['ele_use_h2'], dfp['ele_use_other'],
                 labels=['hydrogen production', 'other'],
                 colors=colors_5_with_grey)
    ax3.set_xlabel('electricity impact (kg CO2eq/kWh)')
    ax3.set_ylabel('electricity use (TWh)')
    ax3.set_xlim(0, 0.4)
    ax3.set_ylim(0, 9800)
    plt.savefig(r'figure/sensitivity_electricity_all.pdf')
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
    sns.lineplot(x=df1['biomass_ratio'], y=df1['GHG'], ax=ax1, color=colors_4[0],
                 linestyle='-', linewidth=1.5, label='without ccs, 100% biomass')
    sns.lineplot(x=df2['biomass_ratio'], y=df2['GHG'], ax=ax1, color=colors_4[1],
                 linestyle='-', linewidth=1.5, label='with ccs, 100% biomass')
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
    return df


def plot_fossil_fuel_impacts(master_file_path, plastics_file_path):
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

        df1, df2, df3 = user_input.model_results('GHG')
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
    df11.loc[df11.process.str.contains('heat'), 'group'] = 'heat'
    df12 = df11.groupby('group').sum(numeric_only=True)
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
    plt.savefig(r'figure/ghg_allocation.pdf', bbox_inches='tight', dpi=300)
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


def plot_ammonia_emission_heatmap():
    df0 = pd.read_csv('data/raw/raw_material_impact_2050_scenRCP1p9.csv', index_col=0)
    ag_list = ['barley_straw', 'maize_stover', 'rapeseed_straw', 'rice_straw', 'sorghum_straw', 'soybean_straw',
               'sugarcane_tops_and_leaves', 'wheat_straw']
    loc_list = ['BR', 'CN', 'IN', 'US', 'WAF', 'WEU', 'SEAS', 'World']
    df = df0.loc[df0.Product.isin(ag_list)].copy()
    df = df.loc[df.Location.isin(loc_list)].copy()
    df = df[['Product', 'Location', 'Ammonia']].copy()
    df['Ammonia'] *= 1000  # 1e-3 kg/kg straw
    table = pd.pivot_table(df, index='Product', columns='Location', values='Ammonia', aggfunc='mean')
    table = table[loc_list]
    mpl.rcParams['svg.fonttype'] = 'none'  # Don't convert fonts to paths
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    fig, ax = plt.subplots(figsize=(10, 9))

    sns.heatmap(
        np.where(table.isna(), 0, np.nan),
        ax=ax, vmin=0, vmax=0,
        cbar=False,
        annot=np.full_like(table, "NA", dtype=object),
        fmt="",
        annot_kws={"size": 10, "va": "center_baseline", "color": "black"},
        cmap=ListedColormap(['#d8d8d8']),
        linewidth=0)
    sns.heatmap(table, cmap=cmp_yellow_orange(), square=True, annot=True, fmt=".2f",
                linewidth=0.5, cbar=False, ax=ax, vmax=2, vmin=0,
                annot_kws={"size": 10, "va": "center_baseline", "color": "black"}, )
    figname = f'figures/lcia/lcia_heatmap_ammonia.pdf'
    #plt.savefig(figname, bbox_inches='tight')
    plt.tight_layout()
    fig.show()

