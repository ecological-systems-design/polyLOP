import pandas as pd

from src.others.variable_declaration import plastics_sector_match_dict, sector_subsector_dict


def get_plastic_types_dict():
    df = pd.read_excel(r'data/external/klotz_plastics_recycling.xlsx', sheet_name='S2.1 plastic types')
    df.loc[df.name_plastic_types == 'PS', 'name_plastic_types'] = 'GPPS'
    id_list = df['id_plastic_types'].tolist()
    name_list = df['name_plastic_types'].str.lower().tolist()
    plastic_types_dict = dict(zip(id_list, name_list))
    return plastic_types_dict


def waste_to_secondary_plastics_ratio():
    df0 = pd.read_excel(r'data/external/klotz_plastics_recycling.xlsx', sheet_name='mass_flow_2')
    df = df0.copy()
    df['waste_to_secondary_plastics_ratio'] = df['uptaken secondary material from mechanical recycling'] / df['waste']
    df.loc[df['plastic_type'] == 'PS', 'plastic_type'] = 'GPPS'
    df['plastic_type'] = df['plastic_type'].str.lower()
    df = df.loc[~df['plastic_type'].isin(['abs', 'pa', 'pc']), :].copy()
    df['waste_to_secondary_plastics_ratio'].fillna(0, inplace=True)
    df.loc[df['plastic_type'] == 'pur', 'plastic_type'] = 'pur_rigid'
    df_temp = df.loc[df['plastic_type'] == 'pur_rigid'].copy()
    df_temp['plastic_type'] = 'pur_flexible'
    df = pd.concat([df, df_temp], ignore_index=True)
    ratio_dict_mr = df.set_index('plastic_type')['waste_to_secondary_plastics_ratio'].to_dict()
    df_temp2 = df.loc[df.chemical_recycling > 0].copy()
    df_temp2['waste_to_chemical_recycling_ratio'] = df_temp2['chemical_recycling'] / df_temp2['waste']
    ratio_dict_cr = df_temp2.set_index('plastic_type')['waste_to_chemical_recycling_ratio'].to_dict()
    return df, ratio_dict_mr, ratio_dict_cr


def consumption_to_secondary_plastics_ratio():
    df0 = pd.read_excel(r'data/external/klotz_plastics_recycling.xlsx', sheet_name='mass_flow_2')
    df = df0.copy()
    df['consumption_to_secondary_plastics_ratio'] = df['uptaken secondary material from mechanical recycling'] / df['consumption']
    df.loc[df['plastic_type'] == 'PS', 'plastic_type'] = 'GPPS'
    df['plastic_type'] = df['plastic_type'].str.lower()
    df = df.loc[~df['plastic_type'].isin(['abs', 'pa', 'pc']), :].copy()
    df['consumption_to_secondary_plastics_ratio'].fillna(0, inplace=True)
    df.loc[df['plastic_type'] == 'pur', 'plastic_type'] = 'pur_rigid'
    df_temp = df.loc[df['plastic_type'] == 'pur_rigid'].copy()
    df_temp['plastic_type'] = 'pur_flexible'
    df = pd.concat([df, df_temp], ignore_index=True)
    ratio_dict_mr = df.set_index('plastic_type')['consumption_to_secondary_plastics_ratio'].to_dict()
    df_temp2 = df.loc[df.chemical_recycling > 0].copy()
    df_temp2['consumption_to_chemical_recycling_ratio'] = df_temp2['chemical_recycling'] / df_temp2['consumption']
    ratio_dict_cr = df_temp2.set_index('plastic_type')['consumption_to_chemical_recycling_ratio'].to_dict()
    return df, ratio_dict_mr, ratio_dict_cr


def mechanical_recycling_flows():
    df0 = pd.read_excel(r'data/external/klotz_plastics_recycling.xlsx', sheet_name='S4.8 results_rec_inventories')
    df = df0[['future']].copy()
    df = df[~df['future'].str.contains('transport')].copy()
    df = df[~df['future'].str.contains('0.730460')].copy()
    df = df[~df['future'].str.contains('chemical, inorganic')].copy()
    extracted_data = df['future'].str.extract(
        r'^(pl\d+),.*Exchange: ([\d.]+) (megajoule|kilowatt hour).*?\'market (for|group for) (heat|electricity)')
    extracted_data.columns = ['id_plastic_types', 'value', 'unit', 'market_prefix', 'product_name']
    extracted_data['product_name'] = extracted_data['product_name']
    extracted_data.drop('market_prefix', axis=1, inplace=True)
    if not extracted_data.empty:
        df[['id_plastic_types', 'value', 'unit', 'product_name']] = extracted_data
    df['value'] = pd.to_numeric(df['value'])
    df = df.groupby(['id_plastic_types', 'product_name', 'unit']).sum(numeric_only=True).reset_index()
    df['type'] = 'UTILITIES'
    df['value'] *= -1
    df.loc[df.product_name == 'heat', 'product_name'] = 'heat_high'
    df.loc[df.unit == 'megajoule', 'unit'] = 'MJ'
    df.loc[df.unit == 'kilowatt hour', 'unit'] = 'kWh'
    plastic_types_dict = get_plastic_types_dict()
    df_ratio = waste_to_secondary_plastics_ratio()[0]
    df_output = pd.DataFrame()
    for code in df['id_plastic_types'].unique():
        plastic1 = f'{plastic_types_dict[code]}'
        if plastic1 in df_ratio['plastic_type'].values:
            plastic = f'{plastic_types_dict[code]}'
            if plastic != 'hips_co2':
                plastic_mr = f'{plastic}_mr'
                plastic_waste = f'{plastic}_waste'
                process_name = f'{plastic_mr}, from mechanical recycling'
                df_temp = df.loc[df['id_plastic_types'] == code].copy()
                df_temp['process'] = process_name
                df_temp2 = pd.DataFrame(
                    {'id_plastic_types': [code, code], 'product_name': [plastic_waste, plastic_mr],
                     'unit': ['kg', 'kg'], 'value': [-1, 1],
                     'type': ['RAW MATERIALS', 'PRODUCT'],
                     'process': [process_name, process_name]})
                df_temp = pd.concat([df_temp, df_temp2], ignore_index=True)
                '''
                process_name = f'{plastic}, from {plastic_mr}'
                df_temp3 = pd.DataFrame({'id_plastic_types': [code, code],
                                         'product_name': [plastic_mr, f'{plastic1}_mr'],
                                         'unit': ['kg', 'kg'], 'value': [-1, 1],
                                         'type': ['RAW MATERIALS', 'PRODUCT'],
                                         'process': [process_name, process_name]})
                df_temp = pd.concat([df_temp, df_temp3], ignore_index=True)
                '''
                df_output = pd.concat([df_output, df_temp], ignore_index=True)
    df_output = df_output[~df_output['process'].str.contains('pc|pa|abs')].copy()
    df_output.drop('id_plastic_types', axis=1, inplace=True)
    return df_output


def gasification_flows():
    # raw data: https://pubs.acs.org/doi/10.1021/acs.iecr.2c03929; Table 1, 20% PS situation
    methanol = 3076
    plastic_waste = -2500 / methanol
    o2 = -1965 / methanol
    ng = -300.8 * 50 / methanol  # kg-> MJ
    water = -2464 / methanol
    ele = -1345 / methanol  # kWh
    co2e = (2092 + 187.8 * 2.6 / 100) / methanol
    co2f = 2475 / methanol
    # ng-> heat
    heat = ng * 0.9  # MJ
    df = pd.DataFrame()
    for p in ['pp', 'ldpe', 'hdpe', 'gpps', 'hips']:
        plastic = f'{p}_waste'
        product_name = f'methanol'
        process_name = f'{product_name}, from {plastic} gasification'
        df_temp = pd.DataFrame({'product_name': [product_name, plastic, 'oxygen_liquid', 'water', 'electricity',
                                                 f'co2_emission', f'co2_feedstock', 'heat_high'],
                                'unit': ['kg', 'kg', 'kg', 'kg', 'kWh', 'kg', 'kg', 'MJ'],
                                'value': [1, plastic_waste, o2, water, ele, co2e, co2f, heat],
                                'type': ['PRODUCT', 'RAW MATERIALS', 'RAW MATERIALS', 'RAW MATERIALS',
                                         'UTILITIES', 'EMISSION', 'BY-PRODUCT CREDITS', 'UTILITIES'],
                                'process': [process_name, process_name, process_name, process_name, process_name,
                                            process_name, process_name, process_name]})
        df = pd.concat([df, df_temp], ignore_index=True)
    return df


def pyrolysis_flows():
    # raw data: IHS
    naphtha = 1
    sodium_hydroxide = -0.28811 / 2
    plastic_waste = -110.2 / 70.4
    ele = -1.221361  # kWh
    lime = -0.08482
    steam = 3.2 * 0.399  # MJ
    co2_1 = (-plastic_waste * 0.857 - 0.84826) / 12 * 44
    co2_2 = (-plastic_waste * 0.923 - 0.84826) / 12 * 44
    df = pd.DataFrame()
    for p in ['pp', 'ldpe', 'hdpe', 'gpps', 'hips']:
        if p == 'gpps' or p == 'hips':
            co2 = co2_2
        else:
            co2 = co2_1
        plastic = f'{p}_waste'
        product_name = f'naphtha_mix'
        process_name = f'{product_name}, from {plastic} pyrolysis'
        df_temp = pd.DataFrame({'product_name': [product_name, plastic, 'electricity', 'co2_emission',
                                                 'sodium_hydroxide', 'lime', 'steam_high'],
                                'unit': ['kg', 'kg', 'kWh', 'kg', 'kg', 'kg', 'MJ'],
                                'value': [1, plastic_waste, ele, co2, sodium_hydroxide, lime, steam],
                                'type': ['PRODUCT', 'RAW MATERIALS', 'UTILITIES', 'EMISSION', 'RAW MATERIALS',
                                         'RAW MATERIALS', 'BY-PRODUCT CREDITS'],
                                'process': [process_name] * 7})
        df = pd.concat([df, df_temp], ignore_index=True)


    return df
