import glob
import pandas as pd
import os

from src.others.variable_declaration import (gas_density_dict, products_purity_dict,
                                             cut_off_raw_material_list, fuel_lhv_dict,
                                             rename_dict_ihs, fuel_list, dilute_product_list,
                                             fuel_co2_dict, ihs_to_master_name_alignment_dict, df_pm_emission)


def read_ihs_data(product_name, process_name, allocation_choice):
    # Get all .csv files in the data folder
    base_path = os.path.join("data", "external", "ihs", product_name, process_name)
    files = glob.glob(os.path.join(base_path, "*.csv"))
    df_all = pd.DataFrame()
    i = 1
    for file in files:
        normalized_file_path = os.path.normpath(file)
        process_name_full = f"{product_name}, {process_name}_nr{i}"
        df = pd.read_csv(normalized_file_path, encoding='unicode_escape')
        unit = df.iloc[6, 3]
        try:
            price_info = df.iloc[2, 0].split("Price: ")[1]
            price, price_unit = price_info.split(" ")[:2]
            price = float(price)
            price_unit = price_unit.split(",")[0]
        except ValueError:
            if product_name == "pet":
                price = 199.08
            else:
                price = -999
            price_unit = "¢/KG"
        df = df.iloc[:, [0, 4, 1, 2, 3]]
        df.columns = ['product', 'unit', 'price', 'price_unit', process_name_full]
        df1 = pd.DataFrame()
        for x in ["RAW MATERIALS", "BY-PRODUCT CREDITS", "UTILITIES"]:
            if x in df['product'].values:
                index1 = df.index[df['product'] == x].tolist()[0]  # find row index of "RAW MATERIALS"
                df_temp = df.iloc[index1:, :]
                index2 = df_temp.index[df_temp['product'].isnull()].tolist()[0]
                df_temp = df.iloc[index1 + 1:index2, :].copy()
                df_temp[process_name_full] = df_temp[process_name_full].astype(float)
                df_temp['type'] = x
                if x == "RAW MATERIALS":
                    cut_off = df_temp.loc[(df_temp['unit'] == "TONNE") & (df_temp[process_name_full] < 0.025)].index
                    df_temp.drop(cut_off, inplace=True)
                    cut_off = df_temp.loc[(df_temp['unit'] == "G") & (df_temp[process_name_full] < 25000)].index
                    df_temp.drop(cut_off, inplace=True)
                elif x == "BY-PRODUCT CREDITS":
                    cut_off = df_temp.loc[(df_temp['unit'] == "TONNE") & (df_temp[process_name_full] > -0.025)].index
                    df_temp.drop(cut_off, inplace=True)
                df_temp[process_name_full] *= -1
                df1 = pd.concat([df1, df_temp])

        df1['price'] = pd.to_numeric(df1['price'], errors='coerce')

        # convert volume base to mass base for CO
        if unit == "per MNM3":
            df1[process_name_full] /= 1.165
            price_unit = "¢/KG"

        # for PEF, convert DMF to FDCA
        if product_name == "pef":
            condition = df1['product'] == "DIMETHYL FURANOATE"
            df1.loc[condition, process_name_full] *= 156 / 184
            df1.loc[condition, 'product'] = "FDCA"
            df1.drop(df1[df1['product'] == "METHANOL"].index, inplace=True)

        # for PBAT, convert DMT to TPA
        elif product_name == "pbat":
            condition = df1['product'] == "DIMETHYL TEREPHTHALATE"
            df1.loc[condition, process_name_full] *= 166 / 194
            df1.loc[condition, 'product'] = "TEREPHTHALIC ACID"
            df1.drop(df1[df1['product'] == "METHANOL"].index, inplace=True)
            a = 0

        # convert fuel unit from tonne to GJ
        for product, factor in fuel_lhv_dict.items():
            condition = df1['product'] == product
            df1.loc[condition, process_name_full] *= factor
            df1.loc[condition, 'unit'] = "GJ"
            df1.loc[condition, 'price'] /= factor
            df1.loc[condition, 'price_unit'] = "¢/MJ"

        # convert gas unit from NM3 to tonne
        for product, factor in gas_density_dict.items():
            condition = (df1['product'] == product) & (df1['unit'] == "NM3")
            df1.loc[condition, process_name_full] *= factor / 1000
            df1.loc[condition, 'unit'] = "TONNE"
            condition_price = (df1['product'] == product) & (df1['price_unit'] == "¢/NM3")
            df1.loc[condition_price, 'price'] /= factor
            df1.loc[condition_price, 'price_unit'] = "¢/KG"

        # convert unit from G to tonne
        df1.loc[df1['unit'] == "G", process_name_full] /= 1000000
        df1.loc[df1['unit'] == "G", 'unit'] = "TONNE"

        # natural gas: 1 Mcal = 4.184 MJ, change unit to GJ
        df1.loc[df1['unit'] == "MMCAL", process_name_full] *= 4.184 / 1000
        df1.loc[df1['unit'] == "MMCAL", 'unit'] = "GJ"
        df1.loc[df1['price_unit'] == "¢/MMCAL", 'price'] /= 4.184
        df1.loc[df1['price_unit'] == "¢/MMCAL", 'price_unit'] = "¢/MJ"

        # electricity: change unit to MWh
        df1.loc[df1['product'] == "ELECTRICITY", process_name_full] /= 1000
        df1.loc[df1['product'] == "ELECTRICITY", 'unit'] = "MWh"

        # steam: change unit to MJ, assume 1 kg steam = 3.2 MJ, ihs methodology, 600psig, 400 degreeC
        df1.loc[df1['product'] == "STEAM", process_name_full] *= 3.2
        df1.loc[df1['product'] == "STEAM", 'unit'] = "GJ"
        df1.loc[df1['product'] == "STEAM", 'price'] /= 3200
        df1.loc[df1['product'] == "STEAM", 'price_unit'] = "¢/MJ"
        df1.drop(df1[df1['unit'] == "EA"].index, inplace=True)

        # cooling water and process water unit
        df1.loc[df1.unit == 'M3', 'price'] /= 1000
        df1.loc[df1.unit == 'M3', 'price_unit'] = '¢/KG'
        df1.loc[df1.unit == 'M3', 'unit'] = 'TONNE'

        # fuel credits, consider as heat_high, efficiency = 90%, add co2 emission
        co2 = 0
        co = 0
        nmvoc = 0
        nox = 0
        pm25 = 0
        sox = 0
        df_pm = df_pm_emission(r'data/raw/master_file_20240327.xlsx')
        for fuel in fuel_list:
            df_pm1 = df_pm.loc[df_pm['product_name'] == fuel].copy()
            if fuel in df1.loc[df1['type'] == 'BY-PRODUCT CREDITS', 'product'].values:
                # Fixed condition by properly closing the bracket after 'type'
                fuel_amount = df1.loc[(df1['product'] == fuel) & (df1['type'] == 'BY-PRODUCT CREDITS'),
                                      process_name_full].values[0]
                co2 += fuel_co2_dict[fuel] * fuel_amount
                co += df_pm1.loc[df_pm1['pollutant'] == 'co_emission', 'value'].values[0] * fuel_amount
                nmvoc += df_pm1.loc[df_pm1['pollutant'] == 'nmvoc_emission', 'value'].values[0] * fuel_amount
                nox += df_pm1.loc[df_pm1['pollutant'] == 'nox_emission', 'value'].values[0] * fuel_amount
                pm25 += df_pm1.loc[df_pm1['pollutant'] == 'pm25_emission', 'value'].values[0] * fuel_amount
                sox += df_pm1.loc[df_pm1['pollutant'] == 'sox_emission', 'value'].values[0] * fuel_amount
            df1.loc[df1['product'] == fuel, process_name_full] *= 0.9
            df1.loc[df1['product'] == fuel, 'product'] = 'heat_high'
        if allocation_choice == 'system_expansion':
            if co2 > 0:
                df1 = pd.concat([df1, pd.DataFrame([{'type': 'EMISSION', 'product': 'co2_emission', 'unit': 'TONNE',
                                                     'price': 0, 'price_unit': '¢/KG', process_name_full: co2}])],
                                ignore_index=True)
            if co > 0:
                df1 = pd.concat([df1, pd.DataFrame([{'type': 'EMISSION', 'product': 'co_emission', 'unit': 'TONNE',
                                                     'price': 0, 'price_unit': '¢/KG', process_name_full: co}])],
                                ignore_index=True)
            if nmvoc > 0:
                df1 = pd.concat([df1, pd.DataFrame([{'type': 'EMISSION', 'product': 'nmvoc_emission', 'unit': 'TONNE',
                                                     'price': 0, 'price_unit': '¢/KG', process_name_full: nmvoc}])],
                                ignore_index=True)
            if nox > 0:
                df1 = pd.concat([df1, pd.DataFrame([{'type': 'EMISSION', 'product': 'nox_emission', 'unit': 'TONNE',
                                                     'price': 0, 'price_unit': '¢/KG', process_name_full: nox}])],
                                ignore_index=True)
            if pm25 > 0:
                df1 = pd.concat([df1, pd.DataFrame([{'type': 'EMISSION', 'product': 'pm25_emission', 'unit': 'TONNE',
                                                     'price': 0, 'price_unit': '¢/KG', process_name_full: pm25}])],
                                ignore_index=True)
            if sox > 0:
                df1 = pd.concat([df1, pd.DataFrame([{'type': 'EMISSION', 'product': 'sox_emission', 'unit': 'TONNE',
                                                     'price': 0, 'price_unit': '¢/KG', process_name_full: sox}])],
                                ignore_index=True)

        # add n2o emissions for adipic acid
        if product_name == "adipic_acid":
            df_temp = pd.DataFrame({'type': 'EMISSION', 'product': 'n2o_emission', 'unit': 'TONNE',
                                    'price': 0, 'price_unit': '¢/KG', process_name_full: 0.01191}, index=[0])
            df1 = pd.concat([df1, df_temp], ignore_index=True)

        # create a new row for type=product, product=product_name, unit=tonne, process_name=1
        df_product = pd.DataFrame({'type': 'PRODUCT', 'product': product_name,
                                   'price': price, 'price_unit': price_unit,
                                   'unit': 'TONNE', process_name_full: 1},
                                  index=[0])
        df1 = pd.concat([df1, df_product], ignore_index=True)
        df1.set_index(['type', 'product', 'unit', 'price', 'price_unit'], inplace=True)
        df_all = pd.concat([df_all, df1], axis=1)
        df_all.dropna(how='all', inplace=True)
        i += 1
    df_all.reset_index(inplace=True)
    df_all = df_all.copy()
    df_all['product'] = df_all['product'].str.lower()
    df_all['product'].replace(' ', '_', regex=True, inplace=True)
    df_all['product'].replace('\\,', '', regex=True, inplace=True)
    df_all['product'].replace('\\.', 'p', regex=True, inplace=True)
    df_all['product'].replace('\\(', '', regex=True, inplace=True)
    df_all['product'].replace('\\)', '', regex=True, inplace=True)
    df_all['product'].replace('\\:', '_to_', regex=True, inplace=True)
    df_all['product'] = df_all['product'].str.replace("'", "")
    return df_all


def read_ihs_data_all(file_path, allocation_choice):
    df_ihs = pd.read_excel(file_path, engine='openpyxl', sheet_name='process_ihs')
    df_ihs = df_ihs[df_ihs.include == "yes"].copy()
    dfs = []  # List to store individual DataFrames
    for i in df_ihs.index:
        product_name = df_ihs.loc[i, 'product']
        process_name = df_ihs.loc[i, 'process']
        print(f"Reading {product_name}, {process_name}")
        df_temp = read_ihs_data(product_name, process_name, allocation_choice)
        df_temp.set_index(['type', 'product', 'unit', 'price', 'price_unit'], inplace=True)
        dfs.append(df_temp)
    df = pd.concat(dfs, axis=1)
    df.reset_index(inplace=True)
    # df.to_csv("data/intermediate/ihs_matrix.csv", index=False)
    return df


def ihs_data_harmonization(df):
    # df = read_ihs_data_all(file_path)
    process_columns = df.iloc[:, 5:].columns
    df_price = df.groupby(by=['product'])["price"].mean()
    df_price_std = df.groupby(by=['product'])["price"].std()

    # convert purity to 100%
    df1 = df.copy()
    df1['price'] = df1['product'].map(df_price)
    # remove additional cut-off raw materials
    df1.drop(df1[df1['product'].isin(cut_off_raw_material_list)].index, inplace=True)
    df1.reset_index(drop=True, inplace=True)

    # adjust purity if specified
    for product, purity in products_purity_dict.items():
        condition = df1['product'] == product
        df1.loc[condition, process_columns] *= purity
        new_product_name = rename_dict_ihs[product]
        if new_product_name in df_price.index:
            df1.loc[condition, 'price'] = df_price.loc[new_product_name]
        else:
            df1.loc[condition, 'price'] /= purity
        df1.loc[condition, 'product'] = new_product_name
    df_price = df1.groupby(by=['product'])["price"].mean()
    df_price_std = df1.groupby(by=['product'])["price"].std()
    df1['price'] = df1['product'].map(df_price)

    # "dilute", "crude", "conc", "credit", update purity
    for product in dilute_product_list:
        if product in df_price.index:
            new_product_name = rename_dict_ihs[product]
            price_dilute = df_price[product]
            price_pure = df_price[new_product_name]
            purity_proxy = price_dilute / price_pure
            condition = df1['product'] == product
            df1.loc[condition, process_columns] *= purity_proxy
            if product in ['propylene_dilute', "benzene-rich_stream", "purge_ethylene", 'sulfuric_acid_dilute']:
                df_temp = df1.loc[(df1['product'] == product)].copy().dropna(how='all', axis=1)
                df_temp2 = df1[df_temp.columns].copy().dropna(axis=0, thresh=6)
                column_to_update = df_temp.columns[5:]
                feed_amount = df_temp2.loc[(df_temp2["product"] == new_product_name) &
                                           (df_temp2["type"] == "RAW MATERIALS"), column_to_update].values
                by_product_amount = df_temp2.loc[(df_temp2["product"] == product) &
                                                 (df_temp2["type"] == "BY-PRODUCT CREDITS"), column_to_update].values
                df_temp2.loc[(df_temp2["product"] == new_product_name) &
                             (df_temp2["type"] == "RAW MATERIALS"), column_to_update] = feed_amount + by_product_amount
                df1.loc[df_temp2.index, df_temp2.columns] = df_temp2
            if product in ['propylene_dilute', "benzene-rich_stream", "purge_ethylene", 'sulfuric_acid_dilute']:
                df1.drop(df1[(df1["product"] == product) & (df1["type"] == "BY-PRODUCT CREDITS")].index, inplace=True)
            df1.loc[condition, 'price'] = df_price.loc[new_product_name]
            df1.loc[condition, 'product'] = new_product_name

    # update C4
    temp_dict = {'c4_fraction5': 'butadiene_raffinate', 'c4_fraction8': 'c4_by-products_mixed',
                 'c4_feed': 'c4_by-products_mixed', 'hydrogen_chloride': 'hydrochloric_acid'}
    for product in ['hydrogen_chloride', 'c4_fraction5', 'c4_fraction8', 'c4_feed']:
        by_product_name = temp_dict[product]
        df_temp = df1.loc[(df1['product'] == product)].copy().dropna(how='all', axis=1)
        df_temp2 = df1[df_temp.columns].copy().dropna(axis=0, thresh=6)
        if df_temp2.shape[0] > 0:
            df_temp = df_temp2.loc[(df_temp2["product"] == by_product_name)].copy().dropna(how='all', axis=1)
            df_temp2 = df_temp2[df_temp.columns].copy().dropna(axis=0, thresh=6)
            if df_temp2.shape[0] > 0:
                column_to_update = df_temp.columns[5:]
                feed_amount = df_temp2.loc[(df_temp2["product"] == product) &
                                           (df_temp2["type"] == "RAW MATERIALS"), column_to_update].values
                by_product_amount = df_temp2.loc[(df_temp2["product"] == by_product_name) &
                                                 (df_temp2["type"] == "BY-PRODUCT CREDITS"), column_to_update].values
                df_temp2.loc[(df_temp2["product"] == product) &
                             (df_temp2["type"] == "RAW MATERIALS"), column_to_update] = feed_amount + by_product_amount
                df1.loc[df_temp2.index, df_temp2.columns] = df_temp2
            if product != 'hydrogen_chloride':
                df1.loc[df1['product'] == product, 'price'] = df_price.loc['butadiene']
                df1.loc[df1['product'] == product, 'product'] = 'butadiene'
    for by_product_name in ['c4_by-products_mixed', 'butadiene_raffinate']:
        df1.drop(df1[(df1["product"] == by_product_name) & (df1["type"] == "BY-PRODUCT CREDITS")].index, inplace=True)
    # update names
    for product, new_product_name in rename_dict_ihs.items():
        condition = df1['product'] == product
        df1.loc[condition, 'product'] = new_product_name
    df_price = df1.groupby(by=['product'])["price"].mean()
    df_price_std = df1.groupby(by=['product'])["price"].std()
    df1['price'] = df1['product'].map(df_price)

    df2 = df1.groupby(by=['type', 'product', 'unit', 'price', 'price_unit']).sum().reset_index()
    df2.to_csv("data/intermediate/ihs_harmonized_matrix.csv", index=False)
    return df2


def process_by_product(df, df_product, product_name, by_product_name, product_process_name):
    df_by = df[df.type == 'BY-PRODUCT CREDITS'].copy()
    if by_product_name in df_by['product'].values and product_name in df['product'].unique():
        by_product_amount = df.loc[(df["product"] == by_product_name) &
                                   (df["type"] == "BY-PRODUCT CREDITS"), 0].values[0]
        replace_consumption_amount = df_product.loc[(df_product["product"] == by_product_name), 0].values[0]
        replace_ratio = abs(by_product_amount / replace_consumption_amount)
        df_temp = df_product.copy()
        df_temp.loc[df_temp.type == 'PRODUCT', 'type'] = 'RAW MATERIALS'
        df_temp.iloc[:, 5:] *= replace_ratio
        df = pd.concat([df, df_temp], ignore_index=True)
        df['process'] = product_process_name
        df = df.loc[df['product'] != by_product_name].copy()
        df = df.groupby(by=['type', 'product', 'unit', 'price', 'price_unit', 'process']).sum().reset_index()
    return df


def allocation_standard(df, process_name):
    allocation_list = ['acetone', 'diethylene_glycol', '1-decene', 'butenes_and_decenes', 'c8+_aromatics',
                       'cyclohexanone', 'dibasic_acids_mixed', 'methyl_acetate', 'methyl_valerate',
                       'mixed_methyl_ester', 'o-dichlorobenzene', 'p-dichlorobenzene', 'propylene_dichloride',
                       'toluenediamine_crude']
    product_in_system_list = ['diethylene_glycol']
    by_product_list = list(df[(df.type == 'BY-PRODUCT CREDITS') &
                              (df['product'].isin(allocation_list))]['product'].unique())
    total_production = df.loc[(df.type == 'PRODUCT') |
                              (df['product'].isin(allocation_list)), 0].sum()
    df = df[~df['product'].isin(by_product_list)].copy()
    df[0] /= total_production
    df.loc[df.type == 'PRODUCT', 0] = 1
    for product in product_in_system_list:
        if product in by_product_list:
            df1 = df.copy()
            df1.loc[df1.type == 'PRODUCT', 'product'] = product
            product_process_name2 = f"{product}, {process_name}"
            df1['process'] = product_process_name2
            df = pd.concat([df, df1], ignore_index=True)
    return df


def allocation_economic(df, process_name):
    df1 = df[(df['type'] == 'BY-PRODUCT CREDITS') |
             (df['type'] == 'PRODUCT')].copy()
    df1['revenue'] = df1[0] * df1['price']
    df1['allocation'] = df1['revenue'] / df1['revenue'].sum()
    if -999 in df1['price'].values:
        print(f"Warning: {process_name} has no price information")
    allocation_main_product = df1.loc[df1['type'] == 'PRODUCT', 'allocation'].values[0]
    df.loc[df['type'] != 'PRODUCT', 0] *= allocation_main_product
    product_in_system_list = ['acetone', 'diethylene_glycol']
    for product in product_in_system_list:
        if product in df['product'].values:
            allocation_by_product = df1.loc[df1['product'] == product, 'allocation'].values[0]
            by_product_amount = df.loc[(df["product"] == product), 0].values[0]
            df2 = df.copy()
            df2[0] *= (allocation_by_product / by_product_amount / allocation_main_product)
            df2.loc[df2.type == 'PRODUCT', 0] = 1
            df2.loc[df2.type == 'PRODUCT', 'product'] = product
            product_process_name2 = f"{product}, {process_name}"
            df2['process'] = product_process_name2
            df = pd.concat([df, df2], ignore_index=True)
    df.drop(df[df['type'] == 'BY-PRODUCT CREDITS'].index, inplace=True)
    return df


def ihs_data_inventory(file_path, allocation_choice='economic'):
    if os.path.exists("data/intermediate/ihs_matrix.csv"):
        df_all = pd.read_csv("data/intermediate/ihs_matrix.csv")
    else:
        df_all = read_ihs_data_all(file_path, allocation_choice)
    df = ihs_data_harmonization(df_all)
    df_price = df.groupby(by=['product'])["price"].mean()
    df_cl2 = read_ihs_data('chlorine', 'from hcl', allocation_choice)
    df_cl2 = ihs_data_harmonization(df_cl2)
    df_cl2['mean'] = df_cl2.iloc[:, 5:].mean(axis=1)
    df_cl2.rename(columns={'mean': 0}, inplace=True)
    df_cl2['price'] = df_cl2['product'].map(df_price)
    df_cl2.drop(df_cl2[(df_cl2["product"] == 'hydrochloric_acid') &
                       (df_cl2["type"] == "BY-PRODUCT CREDITS")].index, inplace=True)
    df_cl2 = df_cl2[['type', 'product', 'unit', 'price', 'price_unit', 0]].copy()
    df_aa = read_ihs_data('acetic_anhydride', 'from acetic acid via ketene', allocation_choice)
    df_aa = ihs_data_harmonization(df_aa)
    df_aa['mean'] = df_aa.iloc[:, 5:].mean(axis=1)
    df_aa.rename(columns={'mean': 0}, inplace=True)
    df_aa['price'] = df_aa['product'].map(df_price)
    df_aa = df_aa[['type', 'product', 'unit', 'price', 'price_unit', 0]].copy()
    df.set_index(['type', 'product', 'unit', 'price', 'price_unit'], inplace=True)
    df_ihs = pd.read_excel(file_path, engine='openpyxl', sheet_name='process_ihs')
    df_ihs = df_ihs[df_ihs.include == "yes"].copy()
    df2 = pd.DataFrame()
    for i in df_ihs.index:
        product_name = df_ihs.loc[i, 'product']
        process_name = df_ihs.loc[i, 'process']
        product_process_name = f"{product_name}, {process_name}"
        df_temp = df[df.columns[df.columns.str.contains(product_process_name)]]
        df_temp2 = df_temp.mean(axis=1)
        df_temp3 = df_temp2[df_temp2 != 0].copy()
        df_temp3 = df_temp3.reset_index()
        df_temp3['process'] = product_process_name

        if 'BY-PRODUCT CREDITS' in df_temp3.type.values:
            df_temp3 = process_by_product(df_temp3, df_aa, 'ca', 'acetic_acid', product_process_name)
            df_temp3 = process_by_product(df_temp3, df_cl2, 'chlorine', 'hydrochloric_acid', product_process_name)
        if 'BY-PRODUCT CREDITS' in df_temp3.type.values:
            if allocation_choice == 'system_expansion':
                df_temp3 = allocation_standard(df_temp3, process_name)
            elif allocation_choice == 'economic':
                df_temp3 = allocation_economic(df_temp3, process_name)
        if product_process_name == 'ethylene_oxide, from ethylene by oxygen oxidation':
            ethylene_amount = -df_temp3.loc[(df_temp3["product"] == "ethylene"), 0].values[0]
            ethylene_left = ethylene_amount - 28 / 44
            co2_amount = ethylene_left * 44 / 28
            df_temp3 = pd.concat([df_temp3, pd.DataFrame({'type': 'BY-PRODUCT CREDITS',
                                                          'product': 'co2_feedstock',
                                                          'unit': 'TONNE',
                                                          'price': -999,
                                                          'price_unit': '¢/KG',
                                                          0: co2_amount,
                                                          'process': product_process_name}, index=[0])],
                                 ignore_index=True)
        df2 = pd.concat([df2, df_temp3], axis=0)
    df2.rename(columns={0: 'value'}, inplace=True)
    utility_list = list(df2[df2['type'] == 'UTILITIES']['product'].unique())
    df3 = df2.groupby(by=['product', 'unit', 'price', 'price_unit', 'process']).sum(numeric_only=True).reset_index()
    df3.loc[df3.value < 0, 'type'] = 'RAW MATERIALS'
    df3.loc[df3['product'].isin(utility_list), 'type'] = 'UTILITIES'
    df3.loc[df3.value > 0, 'type'] = 'BY-PRODUCT CREDITS'
    df3.loc[df3['product'].str.contains('co2_emission'), 'type'] = 'EMISSION'
    df3.loc[df3.value == 1, 'type'] = 'PRODUCT'
    for ihs_name, master_name in ihs_to_master_name_alignment_dict.items():
        df3.loc[df3['product'] == ihs_name, 'product'] = master_name
    df3.loc[df3.unit == 'TONNE', 'unit'] = 'kg'
    df3.loc[df3.unit == 'MWh', 'unit'] = 'kWh'
    df3.loc[df3.unit == 'GJ', 'unit'] = 'MJ'
    df3.rename(columns={'product': 'product_name'}, inplace=True)
    df3.loc[df3.product_name == 'nitrogen_liquid', 'type'] = 'UTILITIES'
    df3.loc[df3.product_name == 'n2o_emission', 'type'] = 'EMISSION'
    df3 = df3.groupby(by=['product_name', 'process', 'unit', 'type']).sum(
        numeric_only=True).reset_index()
    if allocation_choice == 'economic':
        df3.rename(columns={'value': 0, 'product_name': 'product'}, inplace=True)
        process_with_byproduct_list = list(df3.loc[(df3.type == 'BY-PRODUCT CREDITS') & (df3.price != -999), 'process'].unique())
        df4 = df3.copy()
        df3 = df3.loc[~df3['process'].isin(process_with_byproduct_list)]
        for p in process_with_byproduct_list:
            df_temp = df4[df4['process'] == p].copy()
            df_temp = allocation_economic(df_temp, p)
            df3 = pd.concat([df3, df_temp], axis=0)
        df3.rename(columns={0: 'value', 'product': 'product_name'}, inplace=True)
    df3 = df3[~df3.process.str.contains('chlorobenzene')].copy()
    df3.to_csv(f"data/intermediate/ihs_inventory_{allocation_choice}.csv", index=False)
    return df3
