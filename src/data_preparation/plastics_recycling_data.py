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


def waste_to_production_ratio_by_subsector(demand_scenario='2050', country='World'):
    file_path_2 = r'data/raw/bioplastics_vs_fossil.xlsx'
    from src.data_preparation.master_file_back_up import read_plastics_demand
    df = read_plastics_demand(file_path_2)[0]
    df = df[df['Region'] == country].copy()
    df['sector'] = df['Variable'].map(plastics_sector_match_dict)
    lifetime_mapping = {
        'Building and Construction': 33,
        'Automotive': 12,
        'Electrical and Electronic Equipment': 8,
        'Agriculture': 4,
        'Household items, furniture, leisure and others': 5,
        'Textiles': 5,
        'Packaging': 0
    }
    df = df[['sector', 'Unit', '2020', str(demand_scenario), f'cagr_{demand_scenario}']].copy()
    # df = df[df['sector'] != 'Packaging'].copy()
    df['lifetime'] = df['sector'].map(lifetime_mapping)
    df['year_difference_to_2020'] = 30 - df['lifetime']
    df['waste_availability'] = df['2020'] * (1 + df[f'cagr_{demand_scenario}']) ** (df['year_difference_to_2020'])
    df['waste_to_production_ratio'] = df['waste_availability'] / df[str(demand_scenario)]
    df.loc[df.sector == 'Household items, furniture, leisure and others', 'sector'] = 'Consumer products'
    df.loc[df.sector == 'Electrical and Electronic Equipment', 'sector'] = 'Electrical and Electronics'
    df.loc[df.sector == 'Automotive', 'sector'] = 'Transport'
    df.loc[df.sector == 'Agriculture', 'sector'] = 'Other'
    ratio_dict = {}
    for sector in df['sector'].unique():
        ratio = df.loc[df['sector'] == sector, 'waste_to_production_ratio'].values[0]
        for subsector in sector_subsector_dict[sector]:
            ratio_dict[subsector] = ratio
    return ratio_dict


def waste_availability_non_packaging(demand_scenario='2050', country='World'):
    file_path_2 = r'data/raw/bioplastics_vs_fossil.xlsx'
    from src.data_preparation.master_file_back_up import read_plastics_demand
    df = read_plastics_demand(file_path_2)[0]
    df = df[df['Region'] == country].copy()
    df['sector'] = df['Variable'].map(plastics_sector_match_dict)
    lifetime_mapping = {
        'Building and Construction': 33,
        'Automotive': 12,
        'Electrical and Electronic Equipment': 8,
        'Agriculture': 4,
        'Household items, furniture, leisure and others': 5,
        'Textiles': 5,
        'Packaging': 0
    }
    df = df[['sector', 'Unit', '2020', str(demand_scenario), f'cagr_{demand_scenario}']].copy()
    # df = df[df['sector'] != 'Packaging'].copy()
    df['lifetime'] = df['sector'].map(lifetime_mapping)
    df['year_difference_to_2020'] = 30 - df['lifetime']
    df['waste_availability'] = df['2020'] * (1 + df[f'cagr_{demand_scenario}']) ** (df['year_difference_to_2020'])
    df['waste_to_production_ratio'] = df['waste_availability'] / df[str(demand_scenario)]
    df_share = read_plastics_demand(file_path_2)[1]
    df_share['sector'] = df_share['Sector'].map(plastics_sector_match_dict)
    df_share = df_share[df_share['sector'] != 'Packaging'].copy()
    df_share = df_share.groupby('sector').sum(numeric_only=True).reset_index()
    df = pd.merge(df, df_share, on='sector', how='left')
    availability_by_plastics_dict = {}
    for plastics in df_share.columns[1:-1]:
        availability = df['waste_availability'] * df[plastics]
        availability_by_plastics_dict[plastics.lower()] = availability.sum()
    return availability_by_plastics_dict


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
    # raw data: MK3
    naphtha = 0.77
    fuel_gas = 0.115 / naphtha
    char = 0.115 / naphtha
    plastic_waste = -1 / naphtha
    ele = -0.18 / naphtha  # kWh
    heat = -1.79 * 3.6 / naphtha  # MJ
    heat_fuel_gas = fuel_gas * 47.1 * 0.9  # MJ, 47.1 MJ/kg, natural gas, US market
    co2_fuel_gas = fuel_gas * 47.1 * 0.0561
    co_fuel_gas = fuel_gas * 47.1 * 0.000039
    nmvoc_fuel_gas = fuel_gas * 47.1 * 2.6e-6
    nox_fuel_gas = fuel_gas * 47.1 * 0.000089
    pm_fuel_gas = fuel_gas * 47.1 * 1.4e-7
    sox_fuel_gas = fuel_gas * 47.1 * 2.44e-7
    heat_char = char * 28.4 * 0.9  # MJ, charcoal
    co2_char = char * 28.4 * 0.112
    co_char = char * 28.4 * 0.00009
    nmvoc_char = char * 28.4 * 0.00000731
    nox_char = char * 28.4 * 0.000081
    pm_char = char * 28.4 * 0.000133
    sox_char = char * 28.4 * 0.000133
    heat_net = heat + heat_fuel_gas + heat_char
    co2 = co2_fuel_gas + co2_char
    co = co_fuel_gas + co_char
    nmvoc = nmvoc_fuel_gas + nmvoc_char
    nox = nox_fuel_gas + nox_char
    pm = pm_fuel_gas + pm_char
    sox = sox_fuel_gas + sox_char
    df = pd.DataFrame()
    for p in ['pp', 'ldpe', 'hdpe', 'gpps', 'hips']:
        suffix_list = ['_co2', '_biogenic_short', '_biogenic_long', '_fossil']
        if p == 'hips':
            suffix_list = ['_biogenic_short', '_biogenic_long', '_fossil']
        for suffix in suffix_list:
            plastic = f'{p}{suffix}_waste'
            product_name = f'naphtha{suffix}'
            if suffix == '_co2':
                s = '_fossil'
            else:
                s = suffix
            process_name = f'{product_name}, from {plastic} pyrolysis'
            df_temp = pd.DataFrame({'product_name': [product_name, plastic, 'electricity', f'co2{s}',
                                                     'co', 'nmvoc', 'nox', 'pm', 'sox', 'heat_high'],
                                    'unit': ['kg', 'kg', 'kWh', 'kg', 'kg', 'kg', 'kg', 'kg', 'kg', 'MJ'],
                                    'value': [1, plastic_waste, ele, co2, co, nmvoc, nox, pm, sox, heat_net],
                                    'type': ['PRODUCT', 'RAW MATERIALS', 'UTILITIES', 'EMISSION', 'EMISSION',
                                             'EMISSION', 'EMISSION', 'EMISSION', 'EMISSION', 'BY-PRODUCT CREDITS'],
                                    'process': [process_name] * 10})
            df = pd.concat([df, df_temp], ignore_index=True)
    return df
