import pandas as pd
import os
import pycountry

from src.others.variable_declaration import (residue_dict, product_demand_dict, product_demand_dict2,
                                             emissions_ghg_dict, residue_list_code,
                                             ihs_to_master_name_alignment_dict,
                                             carbon_content_dict)
from src.data_preparation.ihs_data import ihs_data_inventory
from src.data_preparation.iam_data import iam_data_preparation
from src.data_preparation.plastics_recycling_data import mechanical_recycling_flows, waste_availability_non_packaging, \
    waste_to_production_ratio_by_subsector, gasification_flows


def get_df_impact(year, scenario, country, low_biodiversity):
    df_impact = pd.read_csv(f'data/raw/raw_material_impact_{year}_{scenario}.csv', index_col=0)
    country_dict = {'CAN': 'CA', 'CHN': 'CN', 'INDIA': 'IN', 'JAP': 'JP', 'USA': 'US', 'MEX': 'MX', 'KOR': 'KR',
                    'INDO': 'ID', 'TUR': 'TR', 'BRA': 'BR', 'UKR': 'UA', 'RUS': 'RU', 'SAF': 'ZA'}
    for x in country_dict.keys():
        df_impact.loc[df_impact.Location == x, 'Location'] = country_dict[x]
    if low_biodiversity:
        biomass_list = residue_list_code + ['agricultural_residue']
        biomass_list_lbdv = [f'{x}_lbdv' for x in biomass_list]
        biomass_list_all = biomass_list + biomass_list_lbdv
        df_impact_rest = df_impact[~df_impact.Product.isin(biomass_list_all)].copy()
        if country == 'World':
            df_impact_biomass = df_impact[df_impact.Product.isin(biomass_list_lbdv)].copy()
            df_impact = pd.concat([df_impact_rest, df_impact_biomass], ignore_index=True)
        else:
            biomass_list2 = residue_list_code + ['agricultural_residue_lbdv', 'forest_residue_lbdv']
            biomass_list2.remove('forest_residue')
            df_impact_biomass = df_impact[df_impact.Product.isin(biomass_list2)].copy()
            df_impact = pd.concat([df_impact_rest, df_impact_biomass], ignore_index=True)
        df_impact['Product'] = df_impact['Product'].str.replace('_lbdv', '', regex=False)
    if country == "World":
        df_impact = df_impact[(df_impact.Location == country) | (df_impact.Location == "Global average")].copy()
    elif pycountry.countries.get(alpha_3=country):
        country_iso2 = pycountry.countries.get(alpha_3=country).alpha_2
        df_impact = df_impact[(df_impact.Location == country_iso2) | (df_impact.Location == "Global average")].copy()
    else:
        print('regional impact data to be developed')
    df_impact[['GHG', 'Biodiversity']] *= -1
    df_impact.rename(columns={'Product': 'product_name'}, inplace=True)
    return df_impact


def master_file_preparation(year, scenario, country, file_path, allocation_choice='standard',
                            iam_scenario='SSP2_SPA2_19I_D', ele_share=0.02, low_biodiversity=True, fossil_routes=True):
    if os.path.exists(f"data/intermediate/ihs_inventory_{allocation_choice}.csv"):
        df_flow_ihs = pd.read_csv(f"data/intermediate/ihs_inventory_{allocation_choice}.csv")
    else:
        df_flow_ihs = ihs_data_inventory(file_path, allocation_choice)
    for ihs_name, master_name in ihs_to_master_name_alignment_dict.items():
        df_flow_ihs.loc[df_flow_ihs['product'] == ihs_name, 'product'] = master_name
    # df_product
    df_product = pd.read_excel(file_path, engine='openpyxl', sheet_name='product')
    df_product = df_product[df_product.include == "yes"].copy()
    ihs_product_list = list(df_flow_ihs['product_name'].unique())
    ihs_intermediate_list = [x for x in ihs_product_list if x not in list(df_product['product_name'].unique())]
    df_product_all = pd.concat([df_product, pd.DataFrame({'product_name': ihs_intermediate_list,
                                                          'unit': ['kg'] * len(ihs_intermediate_list),
                                                          'product_type': ['intermediate'] * len(ihs_intermediate_list),
                                                          'include': ['yes'] * len(ihs_intermediate_list)
                                                          })], ignore_index=True)
    for p, c in carbon_content_dict.items():
        if p in df_product_all['product_name'].unique():
            df_product_all.loc[df_product_all.product_name == p, 'carbon_content'] = c
    # df_process
    df_process = pd.read_excel(file_path, engine='openpyxl', sheet_name='process')
    df_process = df_process[df_process.include == "yes"].copy()
    df_process_ihs = df_flow_ihs[df_flow_ihs['type'] == 'PRODUCT'].copy()
    df_process_ihs.rename(columns={'process': 'product_process'}, inplace=True)
    df_process_ihs['include'] = 'yes'
    df_process_ihs['Data source'] = 'IHS'
    df_process_ihs = df_process_ihs[['product_name', 'product_process', 'include', 'Data source']].copy()
    df_process_ihs2 = pd.read_excel(file_path, engine='openpyxl', sheet_name='process_ihs')
    df_process_ihs2 = df_process_ihs2.loc[df_process_ihs2['keep_in_model'] == 'yes'].copy()
    df_process_ihs2 = df_process_ihs2.iloc[:, [0, 1, -5, -4, -3, -2, -1]].copy()
    df_process_ihs2['process'] = df_process_ihs2['product'] + ', ' + df_process_ihs2['process']
    df_process_ihs2.rename(columns={'product': 'product_name', 'process': 'product_process'}, inplace=True)
    df_process_ihs = pd.merge(df_process_ihs, df_process_ihs2, how='left', on=['product_name', 'product_process'])
    df_process_ihs.loc[df_process_ihs.product_name == 'diethylene_glycol', ['co2_route', 'agricultural_residue_route',
                                                                            'forest_residue_route',
                                                                            'fossil_route']] = 'yes'
    df_process_all = pd.concat([df_process, df_process_ihs], ignore_index=True)
    # df_flow
    df_flow = pd.read_excel(file_path, engine='openpyxl', sheet_name='flows')
    df_flow = df_flow[df_flow.process.isin(list(df_process.product_process.unique()))].copy()
    df_flow_ihs = df_flow_ihs[['product_name', 'process', 'unit', 'value', 'type']].copy()
    df_flow_all = pd.concat([df_flow, df_flow_ihs], ignore_index=True)
    # cooling water unit conversion: # https://www.fao.org/3/bc822e/bc822e.pdf, table 1, once-through cooling
    df_flow_all.loc[df_flow_all.product_name == 'cooling_water_kg', 'value'] /= 36.944
    df_flow_all.loc[df_flow_all.product_name == 'cooling_water_kg', 'unit'] = 'MJ'
    df_flow_all.loc[df_flow_all.product_name == 'cooling_water_kg', 'product_name'] = 'cooling_water'

    # separate for agricultural residue, forest residue, and fossil
    df_flow_new = pd.DataFrame()
    df_process_new = pd.DataFrame()
    product_list_temp = ['agricultural_residue', 'corn_steep_liquor', 'enzyme', 'co2_emission_biogenic_short',
                         'co2_emission_biogenic_long', 'co2_emission_fossil']
    if fossil_routes:
        routes_list = ['co2_route', 'agricultural_residue_route', 'forest_residue_route', 'fossil_route']
    else:
        routes_list = ['co2_route', 'agricultural_residue_route', 'forest_residue_route']
    for x in routes_list:
        suffix_dict = {'co2_route': '_co2', 'agricultural_residue_route': '_biogenic_short',
                       'forest_residue_route': '_biogenic_long', 'fossil_route': '_fossil'}
        df_process_temp = df_process_all[df_process_all[x] == 'yes'].copy()
        product_list = list(df_product_all.loc[df_product_all.product_type == 'product', 'product_name'].unique())
        df_flow_temp = df_flow_all[df_flow_all.process.isin(list(df_process_temp.product_process.unique()))].copy()
        df_process_temp['carbon_content'] = df_process_temp['product_name'].map(df_product_all.set_index('product_name')
                                                                                ['carbon_content'])
        df_flow_temp['carbon_content'] = df_flow_temp['product_name'].map(df_product_all.set_index('product_name')
                                                                          ['carbon_content'])
        df_flow_temp['product_type'] = df_flow_temp['product_name'].map(df_product_all.set_index('product_name')
                                                                        ['product_type'])
        df_process_temp['product_name'] += suffix_dict[x]
        df_process_temp['product_process'] += suffix_dict[x]
        df_flow_temp['process'] += suffix_dict[x]
        product_list = list(df_flow_temp.loc[df_flow_temp.product_type == 'product', 'product_name'].unique())
        df_flow_temp.loc[df_flow_temp.product_type == 'product', 'product_type'] = 'intermediate'
        df_flow_temp.loc[(df_flow_temp.carbon_content > 0) &
                         (df_flow_temp.product_type.isin(['intermediate', 'emission'])) &
                         (~df_flow_temp.product_name.isin(product_list_temp)), 'product_name'] += suffix_dict[x]
        df_flow_temp.loc[df_flow_temp.product_name == 'co2_emission_co2', 'product_name'] = 'co2_emission_fossil'

        for p in product_list:
            carbon_content = df_product_all.loc[df_product_all.product_name == p, 'carbon_content'].iloc[0]
            new_rows = pd.DataFrame({
                'product_name': [p, f'{p}{suffix_dict[x]}'],
                'process': [f'{p} from {p}{suffix_dict[x]}', f'{p} from {p}{suffix_dict[x]}'],
                'unit': ['kg', 'kg'],
                'value': [1, -1],
                'type': ['PRODUCT', 'RAW MATERIAL'],
                'product_type': ['product', 'intermediate'],
                'carbon_content': [carbon_content, carbon_content]
            })
            df_flow_temp = pd.concat([df_flow_temp, new_rows], ignore_index=True)
            new_rows = pd.DataFrame({
                'product_name': [f'{p}{suffix_dict[x]}'],
                'product_process': [f'{p} from {p}{suffix_dict[x]}'],
                'include': ['yes'],
                'Data source': ['name change only']
            })
            df_process_temp = pd.concat([df_process_temp, new_rows], ignore_index=True)
        df_flow_temp.loc[df_flow_temp.product_name == 'co2_feedstock', 'product_name'] += suffix_dict[x]
        df_flow_temp.loc[df_flow_temp.product_name == 'co2_feedstock_co2', 'product_name'] = 'co2_feedstock_fossil'
        df_flow_new = pd.concat([df_flow_new, df_flow_temp], ignore_index=True)
        df_process_new = pd.concat([df_process_new, df_process_temp], ignore_index=True)
        for y in df_flow_temp.product_name.unique():
            carbon_content = df_flow_temp.loc[df_flow_temp.product_name == y, 'carbon_content'].iloc[0]
            if y not in df_product_all['product_name'].unique():
                df_product_all = pd.concat([df_product_all, pd.DataFrame([{'product_name': y, 'unit': 'kg',
                                                                           'product_type': 'intermediate',
                                                                           'include': 'yes',
                                                                           'carbon_content': carbon_content}])],
                                           ignore_index=True)
    df_process_temp = df_process_all[df_process_all['all'] == 'yes'].copy()
    df_flow_temp = df_flow_all[df_flow_all.process.isin(list(df_process_temp.product_process.unique()))].copy()
    df_process_new = pd.concat([df_process_new, df_process_temp], ignore_index=True)
    df_flow_new = pd.concat([df_flow_new, df_flow_temp], ignore_index=True)
    for x in df_product_all['product_name'].unique():
        if x not in df_flow_new['product_name'].unique():
            df_product_all.drop(df_product_all[df_product_all.product_name == x].index, inplace=True)
    df_product_all.loc[df_product_all.product_name.str.contains('co2_feedstock'), 'product_type'] = 'raw_material'
    df_process_all = df_process_new.copy()
    df_flow_all = df_flow_new.copy()


    # df_impact.loc[df_impact.Biodiversity == 0, 'Biodiversity'] = 1e-10
    df_avai_biomass = pd.read_csv(r'data/raw/lignocellulose_feedstock_combined_potential_impacts_all_scenarios.csv',
                                  index_col=0)
    df_avai_biomass['Country'] = df_avai_biomass['Country'].fillna('NA')
    df_avai_biomass = df_avai_biomass[(df_avai_biomass.YEAR == year) &
                                      (df_avai_biomass.SCENARIO == scenario) &
                                      (df_avai_biomass.Price == 'normal')].copy()
    if low_biodiversity:
        df_avai_biomass = df_avai_biomass[df_avai_biomass.BDV < 1e-14].copy()
    df_avai_biomass.loc[df_avai_biomass.Product.str.contains('conifer'), 'Product'] = 'Forest residue'
    df_avai_biomass["product_name"] = df_avai_biomass["Product"].map(residue_dict)
    df_avai_biomass = pd.pivot_table(df_avai_biomass, index=['Country', 'product_name'],
                                     values=['AVAI_MIN', 'AVAI_MAX'],
                                     aggfunc='sum').reset_index()
    if country == "World":
        df_avai_biomass = pd.pivot_table(df_avai_biomass, index=['product_name'],
                                         values=['AVAI_MIN', 'AVAI_MAX'],
                                         aggfunc='sum').reset_index()
    elif pycountry.countries.get(alpha_3=country):
        country_iso2 = pycountry.countries.get(alpha_3=country).alpha_2
        df_avai_biomass = df_avai_biomass[df_avai_biomass.Country == country_iso2].copy()
        new_rows = []
        for residue in residue_dict.values():
            if residue not in df_avai_biomass['product_name'].unique():
                new_rows.append({'Country': country_iso2, 'product_name': residue, 'AVAI_MIN': 0, 'AVAI_MAX': 0})
        if new_rows:
            df_avai_biomass = pd.concat([df_avai_biomass, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        print('regional availability data to be developed')
    df_impact = get_df_impact(year, scenario, country, low_biodiversity)
    df_impact.drop_duplicates(inplace=True)
    df = pd.merge(df_product_all, df_impact, how='left', on='product_name')
    df = pd.merge(df, df_avai_biomass, how='left', on='product_name')
    df['GHG'] = df['GHG'].fillna(0)
    df.loc[df.product_type == 'emission', 'GHG'] = df.loc[df.product_type == 'emission',
    'product_name'].map(emissions_ghg_dict)
    df['Biodiversity'] = df['Biodiversity'].fillna(0)
    df['supply_demand'] = df['AVAI_MIN'].copy() / 1000  # Mt
    df['supply_demand'].fillna(10e15, inplace=True)
    df.loc[(df.product_type == 'raw_material') & (df['GHG'] == 0), 'GHG'] = 0
    df.loc[(df.product_type == 'raw_material') & (df['Biodiversity'] == 0), 'Biodiversity'] = 0
    df.loc[df.product_type != 'raw_material', 'supply_demand'] = 0
    df.loc[df.product_type == 'product', 'supply_demand'] = df.loc[df.product_type == 'product',
    'product_name'].map(product_demand_dict2)
    df.loc[(df.product_type == 'intermediate'), 'GHG'] = 0
    df.loc[(df.product_type == 'intermediate'), 'Biodiversity'] = 0
    # add iam constraints
    if os.path.exists('data/intermediate/iam_scenarios.csv'):
        df_iam = pd.read_csv('data/intermediate/iam_scenarios.csv')
    else:
        df_iam = iam_data_preparation()
    df_iam_1 = df_iam[df_iam.Scenario == iam_scenario].copy()
    if country in df_iam_1.Region.unique():
        df_iam_1 = df_iam_1[df_iam_1.Region == country].copy()
        df_electricity = df_iam_1[df_iam_1.Variable == 'Secondary Energy|Electricity'].copy().reset_index(drop=True)
        df_temp = df_iam_1[df_iam_1.Variable == 'Secondary Energy|Electricity|Biomass'].copy().reset_index(drop=True)
        df_electricity.iloc[:, 7:] -= df_temp.iloc[:, 7:]
        df_electricity['Variable'] = 'Secondary Energy|Electricity|Non-Biomass'
        assert df_electricity.shape[0] == 1
        electricity_constraint = df_electricity[str(year)].values[0] * 277.778 * ele_share  # EJ to TWh
        df_co2_fossil = df_iam_1[df_iam_1.Variable == 'Carbon Sequestration|CCS|Fossil'].copy()
        assert df_co2_fossil.shape[0] == 1
        co2_fossil_constraint = df_co2_fossil[str(year)].values[0]  # Mt
        df_co2_bio = df_iam_1[df_iam_1.Variable == 'Carbon Sequestration|CCS|Biomass'].copy()
        assert df_co2_bio.shape[0] == 1
        co2_bio_constraint = df_co2_bio[str(year)].values[0]  # Mt
        df_temp = df[(df.product_type == 'raw_material') & (df.AVAI_MAX > 0)].copy()
        df.loc[df.product_name == 'electricity_non_biomass', 'supply_demand'] = electricity_constraint
        df.loc[df.product_name == 'co2_feedstock_fossil', 'supply_demand'] = co2_fossil_constraint + co2_bio_constraint
        df.loc[df.product_name == 'co2_feedstock_biogenic_short', 'supply_demand'] = 0
        df.loc[df.product_name == 'co2_feedstock_biogenic_long', 'supply_demand'] = 0
    else:
        print('country does not exist in the IAM data, or if it is a region, to be developed')
    df_flow_all['product_type'] = df_flow_all['product_name'].map(df.set_index('product_name')['product_type'])
    return df, df_process_all, df_flow_all


def add_fossil_routes(year, scenario, country, file_path, allocation_choice, iam_scenario, ele_share,
                      low_biodiversity, fossil_routes):
    df, df_process_all, df_flow_all = master_file_preparation(year, scenario, country, file_path, allocation_choice,
                                                              iam_scenario, ele_share, low_biodiversity, fossil_routes)
    dff = pd.read_csv('data/raw/ecoinvent_hvc_processes.csv')
    dff = dff.groupby(by=['product_name', 'process', 'unit', 'type']).sum(numeric_only=True).reset_index()
    dff.loc[dff.value == 1, 'type'] = 'PRODUCT'
    for product in dff.product_name.unique():
        if product in df.product_name.unique():
            carbon_content = df.loc[df.product_name == product, 'carbon_content'].iloc[0]
        elif f'{product}_biogenic_short' in df.product_name.unique():
            carbon_content = df.loc[df.product_name == f'{product}_biogenic_short', 'carbon_content'].iloc[0]
        else:
            carbon_content_dict = {'pyrolysis_gas': 0.9075, 'ch4': 0.75, 'reformate': 0.92, 'butane': 0.8276,
                                   'diesel': 0.86, 'ethane': 0.8, 'naphtha': 0.84826, 'natural_gas_liquids': 0.82,
                                   'propane': 0.828, 'butadiene_crude': 0.8882, 'petroleum': 0.845,
                                   'isobutane': 0.8276, 'pentane': 0.833, 'natural_gas': 0.75,
                                   'heavy_fuel_oil': 0.86}
            if product in carbon_content_dict.keys():
                carbon_content = carbon_content_dict[product]
            else:
                carbon_content = 0
        dff.loc[dff.product_name == product, 'carbon_content'] = carbon_content

    if not fossil_routes:
        return df, df_process_all, df_flow_all
    else:
        dff['process'] += '_fossil'
        dff.loc[(dff.carbon_content > 0) & (dff.type != 'EMISSION'), 'product_name'] += '_fossil'
        dff.loc[dff.product_name == 'ammonia', 'product_name'] = 'ammonia_fossil'
        dff.loc[dff.product_name == 'hydrogen', 'product_name'] = 'hydrogen_fossil'
        dff.loc[dff.product_name == 'natural_gas_fossil', 'product_name'] = 'natural_gas'
        dff.loc[dff.product_name == 'petroleum_fossil', 'product_name'] = 'petroleum'
        dff_process = dff[dff['type'] == 'PRODUCT'].copy()
        dff_process['Data source'] = 'ecoinvent 3.10'
        dff_process['include'] = 'yes'
        dff_process = dff_process.rename(columns={'process': 'product_process'})
        dff_process = dff_process[
            ['product_name', 'product_process', 'unit', 'include', 'Data source', 'carbon_content']].copy()
        dff_product = dff[~dff.product_name.isin(list(df['product_name'].unique()))].copy()
        dff_product = dff_product[['product_name', 'unit', 'carbon_content']].copy()
        dff_product.drop_duplicates(inplace=True)
        dff_product['product_type'] = 'intermediate'
        dff_product.loc[dff_product.product_name.str.contains('emission'), 'product_type'] = 'emission'
        dff_product.loc[dff_product.product_name.isin(['petroleum', 'natural_gas']), 'product_type'] = 'raw_material'
        dff_product['supply_demand'] = 0
        dff_product.loc[dff_product.product_type == 'raw_material', 'supply_demand'] = 1e16
        dff_product['include'] = 'yes'
        df_impact = get_df_impact(year, scenario, country, low_biodiversity)
        dff_product = pd.merge(dff_product, df_impact, how='left', on='product_name')
        dff_product.loc[dff_product.product_type == 'emission',
        'GHG'] = dff_product.loc[dff_product.product_type == 'emission',
        'product_name'].map(emissions_ghg_dict)
        dff_product['GHG'] = dff_product['GHG'].fillna(0)
        dff_product['Biodiversity'] = dff_product['Biodiversity'].fillna(0)
        df = pd.concat([df, dff_product], ignore_index=True)
        df_process_all = pd.concat([df_process_all, dff_process], ignore_index=True)
        df_flow_all = pd.concat([df_flow_all, dff], ignore_index=True)

        return df, df_process_all, df_flow_all


def read_plastics_demand(file_path):
    df_plastics_demand = pd.read_excel(file_path, engine='openpyxl', sheet_name='production_sector')
    df_plastics_demand.loc[df_plastics_demand['Variable'] == 'Plastics|Production|Sector|Industrial Machinery',
    'Variable'] = 'Plastics|Production|Sector|Other'
    df_plastics_demand = df_plastics_demand.groupby(by=['Region', 'Variable',
                                                        'Unit']).sum(numeric_only=True).reset_index()
    for year in [2005, 2010, 2015, 2020, 2025, 2030, 2035, 2040, 2045, 2050]:
        df_plastics_demand[f'cagr_{year}'] = (df_plastics_demand[str(year)] / df_plastics_demand[str(2020)]) ** (
                1 / 30) - 1
    df_plastics_demand[f'cagr_1'] = (1 / df_plastics_demand[str(2020)]) ** (1 / 30) - 1
    df_plastics_share = pd.read_excel(file_path, engine='openpyxl', sheet_name='share_subsector')
    df_plastics_share['GPPS'] += df_plastics_share['EPS']
    df_plastics_share.drop('EPS', axis=1, inplace=True)
    return df_plastics_demand, df_plastics_share


def polymer_subsector_demand_dict(country, demand_scenario):
    file_path_2 = r'data/raw/bioplastics_vs_fossil.xlsx'
    df_plastics_share = read_plastics_demand(file_path_2)[1].copy()
    df_plastics_demand = read_plastics_demand(file_path_2)[0].copy()
    plastics_demand_dict = {}
    if country in df_plastics_demand['Region'].unique():
        df_plastics_demand = df_plastics_demand[df_plastics_demand['Region'] == country].copy()
    else:
        print('country does not exist in the plastics demand data, or if it is a region, to be developed')
    for i in df_plastics_share.index:
        sector = df_plastics_share.loc[i, 'Sector']
        if demand_scenario == 1:
            sector_demand = 1
        else:
            sector_demand = df_plastics_demand.loc[df_plastics_demand['Variable'] == sector,
            str(demand_scenario)].iloc[0]
        subsector = df_plastics_share.loc[i, 'Subsector']
        for polymer in df_plastics_share.columns[2:-1]:
            polymer_demand = df_plastics_share.loc[i, polymer] * sector_demand
            if polymer_demand > 0:
                product_name_temp = f"{subsector}_{polymer}".lower().replace(' ', '_')
                plastics_demand_dict[product_name_temp] = polymer_demand
    return plastics_demand_dict


def electricity_impact_scenarios(year, scenario, country, file_path, allocation_choice='standard',
                                 iam_scenario='SSP2_SPA2_19I_D',
                                 ele_share=0.02,
                                 ele_impact=-999,
                                 low_biodiversity=True, fossil_routes=True,
                                 ):
    df, df_process_all, df_flow_all = add_fossil_routes(year, scenario, country, file_path, allocation_choice,
                                                        iam_scenario, ele_share,
                                                        low_biodiversity, fossil_routes)
    if ele_impact == -999:
        return df, df_process_all, df_flow_all
    else:
        df.loc[df.product_name == 'electricity_non_biomass', 'GHG'] = -ele_impact
        return df, df_process_all, df_flow_all


def add_mechanical_recycling(year, scenario, country, file_path, allocation_choice='standard',
                             iam_scenario='SSP2_SPA2_19I_D',
                             ele_share=0.02,
                             ele_impact=-999,
                             low_biodiversity=True, fossil_routes=True,
                             mechanical_recycling=True):
    df, df_process_all, df_flow_all = electricity_impact_scenarios(year, scenario, country, file_path,
                                                                   allocation_choice,
                                                                   iam_scenario, ele_share, ele_impact,
                                                                   low_biodiversity, fossil_routes)
    if mechanical_recycling:
        df_flow_recycling = mechanical_recycling_flows()
        df_flow_all = pd.concat([df_flow_all, df_flow_recycling], ignore_index=True)
        df_process_recycling = df_flow_recycling[df_flow_recycling.type == 'PRODUCT'].copy()
        df_process_recycling.rename(columns={'process': 'product_process'}, inplace=True)
        df_process_recycling = df_process_recycling[['product_name', 'product_process']].copy()
        df_process_recycling['include'] = 'yes'
        df_process_recycling['Data source'] = 'MK2'
        df_process_all = pd.concat([df_process_all, df_process_recycling], ignore_index=True)
        df_product_recycling = df_flow_recycling[['product_name', 'unit']].copy()
        df_int = df_product_recycling[df_product_recycling.product_name.str.contains('_mr')].copy()
        df_int['product_type'] = 'intermediate'
        df_int['include'] = 'yes'
        df_int['GHG'] = 0
        df_int['Biodiversity'] = 0
        df_int['supply_demand'] = 0
        df_int = df_int.drop_duplicates()
        df = pd.concat([df, df_int], ignore_index=True)
        df_flow_all['product_type'] = df_flow_all['product_name'].map(df.set_index('product_name')['product_type'])
        return df, df_process_all, df_flow_all
    else:
        return df, df_process_all, df_flow_all


def add_chemical_recycling_gasification(year, scenario, country, file_path, allocation_choice='standard',
                                        iam_scenario='SSP2_SPA2_19I_D',
                                        ele_share=0.02,
                                        ele_impact=-999,
                                        low_biodiversity=True, fossil_routes=True,
                                        mechanical_recycling=True, chemical_recycling_gasi=True):
    df, df_process_all, df_flow_all = add_mechanical_recycling(year, scenario, country, file_path, allocation_choice,
                                                               iam_scenario, ele_share, ele_impact, low_biodiversity,
                                                               fossil_routes, mechanical_recycling)
    if chemical_recycling_gasi:
        df_gasi = gasification_flows()
        df_flow_all = pd.concat([df_flow_all, df_gasi], ignore_index=True)
        df_process_gasi = df_gasi[df_gasi.type == 'PRODUCT'].copy()
        df_process_gasi.rename(columns={'process': 'product_process'}, inplace=True)
        df_process_gasi['include'] = 'yes'
        df_process_gasi['Data source'] = 'Prifti2023'
        df_process_all = pd.concat([df_process_all, df_process_gasi], ignore_index=True)
        df_flow_all['product_type'] = df_flow_all['product_name'].map(df.set_index('product_name')['product_type'])
        return df, df_process_all, df_flow_all
    else:
        return df, df_process_all, df_flow_all


def refine_plastics_demand(year, scenario, country, file_path, allocation_choice='standard',
                           demand_scenario='2050',
                           iam_scenario='SSP2_SPA2_19I_D',
                           ele_share=0.02, ele_impact=-999,
                           low_biodiversity=True, fossil_routes=True,
                           bio_plastics=True, mechanical_recycling=True, chemical_recycling_gasi=True):
    df, df_process_all, df_flow_all = add_chemical_recycling_gasification(year, scenario, country, file_path,
                                                                          allocation_choice, iam_scenario,
                                                                          ele_share, ele_impact, low_biodiversity,
                                                                          fossil_routes, mechanical_recycling,
                                                                          chemical_recycling_gasi)
    file_path_2 = r'data/raw/bioplastics_vs_fossil.xlsx'
    df_plastics_share = read_plastics_demand(file_path_2)[1].copy()
    df_plastics_demand = read_plastics_demand(file_path_2)[0].copy()
    if country in df_plastics_demand['Region'].unique():
        df_plastics_demand = df_plastics_demand[df_plastics_demand['Region'] == country].copy()
    else:
        print('country does not exist in the plastics demand data, or if it is a region, to be developed')
    df_plastics_substitute = pd.read_excel(file_path_2, engine='openpyxl', sheet_name='substitution_factor_subsector')
    df_plastics_substitute = df_plastics_substitute[df_plastics_substitute['include'] == 'yes'].copy()
    df_plastics_substitute.loc[df_plastics_substitute['traditional_plastics'] == 'EPS', 'traditional_plastics'] = 'GPPS'
    df_plastics_substitute = df_plastics_substitute.groupby(by=['sector', 'subsector', 'bio-based_plastics',
                                                                'traditional_plastics']).mean(
        numeric_only=True).reset_index()
    df.loc[df.product_type == 'product', 'product_type'] = 'intermediate'
    polymer_source_list1 = ['_co2', '_biogenic_short', '_biogenic_long', '_fossil']
    polymer_source_list2 = ['_co2', '_biogenic_short', '_biogenic_long', '_fossil',
                            '_co2_mr', '_biogenic_short_mr', '_biogenic_long_mr', '_fossil_mr']
    waste_to_production_dict = waste_to_production_ratio_by_subsector(demand_scenario, country)
    for i in df_plastics_share.index:
        sector = df_plastics_share.loc[i, 'Sector']
        if demand_scenario == 1:
            sector_demand = 1
        else:
            sector_demand = df_plastics_demand.loc[df_plastics_demand['Variable'] == sector,
            str(demand_scenario)].iloc[0]
        subsector = df_plastics_share.loc[i, 'Subsector']
        subsector2 = subsector.lower().replace(' ', '_')
        for polymer in df_plastics_share.columns[2:-1]:
            if 'PUR' in polymer:
                polymer_source_list = polymer_source_list1
            else:
                polymer_source_list = polymer_source_list2
            polymer_demand = df_plastics_share.loc[i, polymer] * sector_demand
            carbon_content = df.loc[df.product_name == polymer.lower(), 'carbon_content'].iloc[0]
            if polymer_demand > 0:
                for suffix in polymer_source_list:
                    polymer2 = f"{polymer}{suffix}".lower()
                    if polymer2 in df['product_name'].unique():
                        product_name = f"{subsector}_{polymer2}".lower().replace(' ', '_')
                        waste_name = f'{polymer2}_waste'.replace('_mr', '')

                        waste_to_production_ratio = waste_to_production_dict[subsector2]
                        new_process = pd.DataFrame([{'product_name': product_name,
                                                     'product_process': f'{subsector}, from {polymer2}',
                                                     'include': 'yes'}])
                        df_process_all = pd.concat([df_process_all, new_process], ignore_index=True)
                        new_flows_df = pd.DataFrame({
                            'product_name': [product_name, polymer2, waste_name],
                            'process': [f'{subsector}, from {polymer2}'] * 3,
                            'unit': ['kg'] * 3,
                            'value': [1, -1, waste_to_production_ratio],
                            'type': ['PRODUCT', 'RAW MATERIAL', 'WASTE'],
                            'carbon_content': [carbon_content] * 3
                        })
                        df_flow_all = pd.concat([df_flow_all, new_flows_df], ignore_index=True)
                        new_product = pd.DataFrame([{'product_name': product_name,
                                                     'unit': 'kg',
                                                     'product_type': 'product',
                                                     'carbon_content': carbon_content,
                                                     'include': 'yes',
                                                     'GHG': 0,
                                                     'Biodiversity': 0,
                                                     'supply_demand': 0}])
                        df = pd.concat([df, new_product], ignore_index=True)
                        if waste_name not in df['product_name'].unique():
                            new_product = pd.DataFrame([{'product_name': waste_name,
                                                         'unit': 'kg',
                                                         'product_type': 'waste',
                                                         'carbon_content': carbon_content,
                                                         'include': 'yes',
                                                         'GHG': 0,
                                                         'Biodiversity': 0,
                                                         'supply_demand': 0}])
                            df = pd.concat([df, new_product], ignore_index=True)
    if bio_plastics:
        for j in df_plastics_substitute.index:
            subsector = df_plastics_substitute.loc[j, 'subsector']
            subsector2 = subsector.lower().replace(' ', '_')
            polymer_0 = df_plastics_substitute.loc[j, 'traditional_plastics']
            waste_to_production_ratio = waste_to_production_dict[subsector2]
            polymer_1 = f"{polymer_0}".lower()
            # product_name = f"{subsector}_{polymer_1}".lower().replace(' ', '_')
            '''
            for suffix in polymer_source_list1:
                polymer_1 = f"{polymer_0}{suffix}".lower()
                product_name = f"{subsector}_{polymer_1}".lower().replace(' ', '_')
                if polymer_1 in df['product_name'].unique():
            '''
            polymer_2 = df_plastics_substitute.loc[j, 'bio-based_plastics'].lower()
            factor = df_plastics_substitute.loc[j, 'substitution_factor']
            for suffix in ['_biogenic_short', '_biogenic_long']:
                polymer_2s = f"{polymer_2}{suffix}"
                product_name = f"{subsector}_{polymer_1}_{polymer_2s}".lower().replace(' ', '_')
                if '+' in polymer_2:
                    polymer_21 = polymer_2.split('+')[0].split('%')[1].lower()
                    polymer_21s = f'{polymer_21}{suffix}'
                    carbon_content_21 = df.loc[df.product_name == polymer_21, 'carbon_content'].iloc[0]
                    waste_21 = f'{polymer_21s}_waste'
                    polymer_21_share = float(polymer_2.split('+')[0].split('%')[0]) / 100
                    polymer_22 = polymer_2.split('+')[1].split('%')[1].lower()
                    polymer_22s = f'{polymer_22}{suffix}'
                    carbon_content_22 = df.loc[df.product_name == polymer_22, 'carbon_content'].iloc[0]
                    waste_22 = f'{polymer_22s}_waste'
                    polymer_22_share = float(polymer_2.split('+')[1].split('%')[0]) / 100
                    polymer_21_flow = factor * polymer_21_share
                    polymer_22_flow = factor * polymer_22_share
                    waste_21_flow = waste_to_production_ratio * polymer_21_flow
                    waste_22_flow = waste_to_production_ratio * polymer_22_flow
                    carbon_content_product = carbon_content_21 * polymer_21_share + carbon_content_22 * polymer_22_share
                    new_flow = pd.DataFrame(
                        {'product_name': [f'{product_name}', polymer_21s, polymer_22s, waste_21, waste_22],
                         'process': [f'{subsector}, from {polymer_2s} replacing {polymer_1}'] * 5,
                         'unit': ['kg'] * 5,
                         'value': [1, -polymer_21_flow, -polymer_22_flow, waste_21_flow, waste_22_flow],
                         'carbon_content': [carbon_content_product, carbon_content_21, carbon_content_22,
                                            carbon_content_21, carbon_content_22],
                         'type': ['PRODUCT', 'RAW MATERIALS', 'RAW MATERIALS', 'WASTE', 'WASTE']})
                    df_flow_all = pd.concat([df_flow_all, new_flow], ignore_index=True)
                    new_process = pd.DataFrame({'product_name': [f'{product_name}'],
                                                'product_process': [
                                                    f'{subsector}, from {polymer_2s} replacing {polymer_1}'],
                                                'include': ['yes']})
                    df_process_all = pd.concat([df_process_all, new_process], ignore_index=True)
                    if waste_21 not in df['product_name'].unique():
                        new_product = pd.DataFrame([{'product_name': waste_21,
                                                     'unit': 'kg',
                                                     'product_type': 'waste',
                                                     'carbon_content': carbon_content_21,
                                                     'include': 'yes',
                                                     'GHG': 0,
                                                     'Biodiversity': 0,
                                                     'supply_demand': 0}])
                        df = pd.concat([df, new_product], ignore_index=True)
                    if waste_22 not in df['product_name'].unique():
                        new_product = pd.DataFrame([{'product_name': waste_22,
                                                     'unit': 'kg',
                                                     'product_type': 'waste',
                                                     'carbon_content': carbon_content_22,
                                                     'include': 'yes',
                                                     'GHG': 0,
                                                     'Biodiversity': 0,
                                                     'supply_demand': 0}])
                        df = pd.concat([df, new_product], ignore_index=True)
                    if product_name not in df['product_name'].unique():
                        new_product = pd.DataFrame([{'product_name': product_name,
                                                        'unit': 'kg',
                                                        'product_type': 'product',
                                                        'carbon_content': carbon_content_product,
                                                        'include': 'yes',
                                                        'GHG': 0,
                                                        'Biodiversity': 0,
                                                        'supply_demand': 0}])
                        df = pd.concat([df, new_product], ignore_index=True)
                else:
                    waste_2 = f'{polymer_2s}_waste'
                    carbon_content_2 = df.loc[df.product_name == polymer_2, 'carbon_content'].iloc[0]
                    polymer_2_flow = factor
                    waste_2_flow = waste_to_production_ratio * polymer_2_flow
                    new_flow = pd.DataFrame({'product_name': [f'{product_name}', polymer_2s, waste_2],
                                             'process': [
                                                            f'{subsector}, from {polymer_2s} replacing {polymer_1}'] * 3,
                                             'carbon_content': [carbon_content_2] * 3,
                                             'unit': ['kg'] * 3,
                                             'value': [1, -polymer_2_flow, waste_2_flow],
                                             'type': ['PRODUCT', 'RAW MATERIALS', 'WASTE']})
                    df_flow_all = pd.concat([df_flow_all, new_flow], ignore_index=True)
                    new_process = pd.DataFrame({'product_name': [f'{product_name}_replaced'],
                                                'product_process': [
                                                    f'{subsector}, from {polymer_2s} replacing {polymer_1}'],
                                                'include': ['yes']})
                    df_process_all = pd.concat([df_process_all, new_process], ignore_index=True)
                    if waste_2 not in df['product_name'].unique():
                        new_product = pd.DataFrame([{'product_name': waste_2,
                                                     'unit': 'kg',
                                                     'product_type': 'waste',
                                                     'carbon_content': carbon_content_2,
                                                     'include': 'yes',
                                                     'GHG': 0,
                                                     'Biodiversity': 0,
                                                     'supply_demand': 0}])
                        df = pd.concat([df, new_product], ignore_index=True)
                    if product_name not in df['product_name'].unique():
                        new_product = pd.DataFrame([{'product_name': product_name,
                                                        'unit': 'kg',
                                                        'product_type': 'product',
                                                        'carbon_content': carbon_content_2,
                                                        'include': 'yes',
                                                        'GHG': 0,
                                                        'Biodiversity': 0,
                                                        'supply_demand': 0}])
                        df = pd.concat([df, new_product], ignore_index=True)
    '''
    int_list = list(df[df.product_type == 'intermediate']['product_name'].unique())
    for x in int_list:
        df_temp = df_flow_all[df_flow_all.product_name == x].copy()
        has_positive = (df_temp['value'] > 0).any()
        has_negative = (df_temp['value'] < 0).any()
        if not has_negative:
            print(f"{x} is not used downstream")
        if not has_positive:
            print(f"{x} has no production information!!! check!!!")
    '''
    df_flow_all['product_type'] = df_flow_all['product_name'].map(df.set_index('product_name')['product_type'])
    return df, df_process_all, df_flow_all


def add_eol_incineration(year, scenario, country, file_path, allocation_choice,
                         demand_scenario, iam_scenario, ele_share, ele_impact,
                         low_biodiversity, fossil_routes, bio_plastics, mechanical_recycling, chemical_recycling_gasi,
                         eol_incineration):
    df, df_process_all, df_flow_all = refine_plastics_demand(year, scenario, country, file_path, allocation_choice,
                                                             demand_scenario, iam_scenario, ele_share, ele_impact,
                                                             low_biodiversity, fossil_routes, bio_plastics,
                                                             mechanical_recycling, chemical_recycling_gasi)
    if not eol_incineration:
        return df, df_process_all, df_flow_all
    else:
        waste_list = list(df[df.product_type == 'waste']['product_name'].unique())
        for waste in waste_list:
            if 'pur' in waste:
                plastics = f"{waste.split('_')[0]}_{waste.split('_')[1]}"
            else:
                plastics = waste.split('_')[0]
            carbon_content = df.loc[df.product_name == plastics, 'carbon_content'].iloc[0]
            co2 = carbon_content * 44 / 12
            if 'biogenic_short' in waste:
                co2_name = 'co2_emission_biogenic_short'
            elif 'biogenic_long' in waste:
                co2_name = 'co2_emission_biogenic_long'
            else:
                co2_name = 'co2_emission_fossil'
            new_process = pd.DataFrame([{'product_name': co2_name,
                                         'product_process': f'waste incineration from {waste}',
                                         'include': 'yes'}])
            df_process_all = pd.concat([df_process_all, new_process], ignore_index=True)
            new_flows_df = pd.DataFrame({
                'product_name': [co2_name, waste],
                'process': [f'waste incineration from {waste}'] * 2,
                'unit': ['kg', 'kg'],
                'value': [co2, -1],
                'type': ['EMISSION', 'RAW MATERIAL'],
                'product_type': ['emission', 'waste']
            })
            df_flow_all = pd.concat([df_flow_all, new_flows_df], ignore_index=True)
            df.loc[df.product_name.str.contains(plastics), 'carbon_content'] = carbon_content
        return df, df_process_all, df_flow_all


def add_ccs(year, scenario, country, file_path, allocation_choice,
            demand_scenario, iam_scenario, ele_share, ele_impact,
            low_biodiversity, fossil_routes, bio_plastics, mechanical_recycling, chemical_recycling_gasi,
            eol_incineration, ccs):
    df, df_process_all, df_flow_all = add_eol_incineration(year, scenario, country, file_path, allocation_choice,
                                                           demand_scenario, iam_scenario, ele_share, ele_impact,
                                                           low_biodiversity, fossil_routes, bio_plastics,
                                                           mechanical_recycling, chemical_recycling_gasi,
                                                           eol_incineration)
    suffix_list = ['_fossil', '_biogenic_short', '_biogenic_long']
    df.loc[df.product_name == 'electricity_non_biomass', 'Biodiversity'] = 0
    # differentiate coefficients for agricultural residue and forest residue
    df_flow_all.loc[(df_flow_all.process.str.contains('biogenic_long'))
                    & (df_flow_all['type'] != 'PRODUCT'), 'value'] *= 1.01
    df_flow_all.loc[(df_flow_all.process.str.endswith('_co2'))
                    & (df_flow_all['type'] != 'PRODUCT'), 'value'] *= 1.01
    if not ccs:
        for suffix in suffix_list:
            co2e_name = f'co2_emission{suffix}'
            co2f_name = f'co2_feedstock{suffix}'
            df_flow = pd.DataFrame()
            new_flows_df = pd.DataFrame({
                'product_name': [co2e_name, co2f_name],
                'process': [f'{co2f_name } release'] * 2,
                'unit': ['kg', 'kg'],
                'value': [1, -1],
                'type': ['EMISSION', 'RAW MATERIAL'],
                'product_type': ['emission', 'raw_material']
            })
            df_flow = pd.concat([df_flow, new_flows_df], ignore_index=True)
            df_flow_all = pd.concat([df_flow_all, df_flow], ignore_index=True)
            df_process = df_flow[df_flow.type == 'EMISSION'].copy()
            df_process.rename(columns={'process': 'product_process'}, inplace=True)
            df_process = df_process[['product_name', 'product_process']].copy()
            df_process['include'] = 'yes'
            df_process['Data source'] = 'mass balacne'
            df_process_all = pd.concat([df_process_all, df_process], ignore_index=True)
        return df, df_process_all, df_flow_all
    else:
        pipeline = -0.00152  # tkm
        electricity = -0.00948  # kWh
        co2_emission = 0.00026  # kg
        co2_feedstock = -1 - co2_emission  # kg
        df_flow = pd.DataFrame()
        for suffix in suffix_list:
            co2f_name = f'co2_feedstock{suffix}'
            co2s_name = f'co2_storage{suffix}'
            co2e_name = f'co2_emission{suffix}'
            new_flows_df = pd.DataFrame({
                'product_name': [co2s_name, co2f_name, co2e_name, 'electricity', 'pipeline'],
                'process': [f'CCS{suffix}', f'CCS{suffix}', f'CCS{suffix}', f'CCS{suffix}', f'CCS{suffix}'],
                'unit': ['kg', 'kg', 'kg', 'kWh', 'tkm'],
                'value': [1, co2_feedstock, co2_emission, electricity, pipeline],
                'type': ['PRODUCT', 'RAW MATERIAL', 'EMISSION', 'UTILITIES', 'RAW MATERIAL'],
                'product_type': ['emission', 'raw_material', 'emission', 'intermediate', 'raw_material']
            })
            df_flow = pd.concat([df_flow, new_flows_df], ignore_index=True)
        df_flow_all = pd.concat([df_flow_all, df_flow], ignore_index=True)
        df_process = df_flow[df_flow.type == 'PRODUCT'].copy()
        df_process.rename(columns={'process': 'product_process'}, inplace=True)
        df_process = df_process[['product_name', 'product_process']].copy()
        df_process['include'] = 'yes'
        df_process['Data source'] = 'premise'
        df_process_all = pd.concat([df_process_all, df_process], ignore_index=True)
        df_product = df_flow.loc[~df_flow.product_name.isin(list(df['product_name'].unique()))].copy()
        df_product = df_product[['product_name', 'unit']].copy()
        df_product.drop_duplicates(inplace=True)
        df_product['product_type'] = 'emission'
        df_product.loc[df_product.product_name.str.contains('pipeline'), 'product_type'] = 'raw_material'
        df_product.loc[df_product.product_name.str.contains('pipeline'), 'supply_demand'] = 1e16
        df_product['include'] = 'yes'
        df_product['GHG'] = 0
        df_product.loc[df_product.product_name.str.contains('biogenic'), 'GHG'] = -1
        df_product.loc[df_product.product_name.str.contains('pipeline'), 'GHG'] = -0.0024789
        df_product['Biodiversity'] = 0
        df_product['carbon_content'] = 0
        df_product.loc[df_product.product_name.str.contains('co2_storage'), 'carbon_content'] = 12/44
        df = pd.concat([df, df_product], ignore_index=True)

        return df, df_process_all, df_flow_all
