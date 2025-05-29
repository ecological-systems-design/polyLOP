import pandas as pd
import pycountry
import os
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import glob
import time

from src.others.variable_declaration import (residue_list_code, image_region_list, ihs_to_master_name_alignment_dict,
                                             carbon_content_dict, plastics_sector_match_dict, sector_subsector_dict,
                                             emissions_ghg_dict, residue_dict, max_replacing_rate_dict,
                                             co2_feedstock_list, final_product_list, df_pm_emission)
from src.data_preparation.ihs_data import ihs_data_inventory
from src.data_preparation.plastics_recycling_data import (mechanical_recycling_flows, gasification_flows,
                                                          pyrolysis_flows,
                                                          waste_to_secondary_plastics_ratio,
                                                          consumption_to_secondary_plastics_ratio)


class MasterFile:
    def __init__(self, user_input_file_path, master_file_path, plastics_file_path):
        self.df_input = pd.read_excel(user_input_file_path)
        self.master_file_path = master_file_path
        self.plastics_file_path = plastics_file_path
        self.year = self.df_input.loc[self.df_input['parameter'] == 'year', 'value'].values[0]
        self.scenario = self.df_input.loc[self.df_input['parameter'] == 'scenario', 'value'].values[0]
        self.country = self.df_input.loc[self.df_input['parameter'] == 'country', 'value'].values[0]
        self.allocation_choice = self.df_input.loc[self.df_input['parameter'] == 'allocation_choice', 'value'].values[0]
        self.system_boundary = self.df_input.loc[self.df_input['parameter'] == 'system_boundary', 'value'].values[0]
        self.mechanical_recycling = self.df_input.loc[self.df_input['parameter'] == 'mechanical_recycling',
                                                      'value'].values[0]
        self.gasi = self.df_input.loc[self.df_input['parameter'] == 'chemical_recycling_gasification',
                                      'value'].values[0]
        if 'chemical_recycling_pyrolysis' in self.df_input['parameter'].values:
            self.pyrolysis = self.df_input.loc[self.df_input['parameter'] == 'chemical_recycling_pyrolysis',
                                               'value'].values[0]
        else:
            self.pyrolysis = False
        self.fossil_routes = self.df_input.loc[self.df_input['parameter'] == 'fossil_routes', 'value'].values[0]
        self.bs_routes = self.df_input.loc[self.df_input['parameter'] == 'agricultural_residue_routes',
                                           'value'].values[0]
        self.bl_routes = self.df_input.loc[self.df_input['parameter'] == 'forest_residue_routes', 'value'].values[0]
        self.co2_routes = self.df_input.loc[self.df_input['parameter'] == 'co2_routes', 'value'].values[0]
        self.new_bio_plastics = self.df_input.loc[self.df_input['parameter'] == 'new_bio_plastics', 'value'].values[0]
        self.ccs_process_co2 = self.df_input.loc[self.df_input['parameter'] == 'ccs_process_co2', 'value'].values[0]
        self.ele_availability = self.df_input.loc[self.df_input['parameter'] == 'electricity_availability',
                                                  'value'].values[0]
        self.ele_ratio = self.df_input.loc[self.df_input['parameter'] == 'electricity_ratio', 'value'].values[0]
        self.iam_scenario = self.df_input.loc[self.df_input['parameter'] == 'iam_scenario', 'value'].values[0]
        self.biomass_availability = self.df_input.loc[self.df_input['parameter'] == 'biomass_availability',
                                                      'value'].values[0]
        self.biomass_ratio = self.df_input.loc[self.df_input['parameter'] == 'biomass_ratio', 'value'].values[0]
        self.plastics_demand = self.df_input.loc[self.df_input['parameter'] == 'plastics_demand', 'value'].values[0]
        self.plastics_demand_ratio = self.df_input.loc[self.df_input['parameter'] == 'plastics_demand_ratio',
                                                       'value'].values[0]
        self.ele_impact = self.df_input.loc[self.df_input['parameter'] == 'electricity_impact', 'value'].values[0]
        self.low_biodiversity = self.df_input.loc[self.df_input['parameter'] == 'low_biodiversity', 'value'].values[0]
        self.fossil_lock_in = self.df_input.loc[self.df_input['parameter'] == 'fossil_lock_in', 'value'].values[0]
        if self.ele_availability == 'default':
            ele_file_path = r'data/external/iam/image_rcp19_electricity_production.xlsx'
            self.df_ele_avai_default = pd.read_excel(ele_file_path, engine='openpyxl', sheet_name='main.ElecProdTot')
            self.df_region_ele = pd.read_excel(ele_file_path, engine='openpyxl', sheet_name='region')

        self.get_rm_impact()
        self.read_plastics_demand()
        self.rm_supply_dict()
        self.polymer_subsector_demand_dict()
        self.waste_to_production_ratio_by_subsector()

    def get_rm_impact(self):
        df0 = pd.read_csv(f'data/raw/raw_material_impact_{self.year}_{self.scenario}.csv', index_col=0)
        df_pm = pd.read_csv(f'data/raw/cf_pm_health.csv')
        #general_column = ['Region'] + [x for x in df_pm.columns if 'general' in x]
        #df_pm = df_pm[general_column].copy()
        column_long = list(df_pm.columns)
        column_short = [x.split('CF_')[1].split('_DALY')[0] for x in column_long if '_' in x]
        column_short = [x.replace('.', '') for x in column_short]
        column_short = [x.replace('agricultural_soil', 'agriculture') for x in column_short]
        column_short = [x.replace('SO2', 'sox') for x in column_short]
        column_short = [f'{x.split("_")[0]}_emission_{x.split("_")[1]}' for x in column_short]
        column_short = ['Region'] + [x.lower() for x in column_short]
        df_pm.columns = column_short
        df_country = pd.read_excel(r'data/raw/Country.xlsx', engine='openpyxl', sheet_name='Sheet1')
        df_country.loc[df_country.Country == "Namibia", "ISO2"] = "NA"
        country_dict = {'CAN': 'CA', 'CHN': 'CN', 'INDIA': 'IN', 'JAP': 'JP', 'USA': 'US', 'MEX': 'MX', 'KOR': 'KR',
                        'INDO': 'ID', 'TUR': 'TR', 'BRA': 'BR', 'UKR': 'UA', 'RUS': 'RU', 'SAF': 'ZA'}
        df = df0.copy()
        df['Location'] = df['Location'].fillna('NA')
        for x in country_dict.keys():
            df.loc[df.Location == x, 'Location'] = country_dict[x]
        df_pm.loc[df_pm.Region == 'INDIA', 'Region'] = 'IND'
        df_pm.loc[df_pm.Region == 'INDO', 'Region'] = 'IDN'
        df_pm.loc[df_pm.Region == 'JAP', 'Region'] = 'JPN'
        df_pm.loc[df_pm.Region == 'SAF', 'Region'] = 'ZAF'
        df[['GHG', 'Biodiversity']] *= -1
        df.rename(columns={'Product': 'product_name'}, inplace=True)
        df = df[~df.product_name.str.contains('agricultural_residue')]
        df = df.drop_duplicates()

        if self.low_biodiversity:
            biomass_list = residue_list_code
            biomass_list_lbdv = [f'{x}_lbdv' for x in biomass_list]
            biomass_list_all = biomass_list + biomass_list_lbdv
            df_impact_rest = df[~df.product_name.isin(biomass_list_all)].copy()
            if self.country == 'World' or self.country in image_region_list:
                biomass_list2 = biomass_list_lbdv
            else:
                biomass_list2 = residue_list_code + ['forest_residue_lbdv']
                biomass_list2.remove('forest_residue')
            df_impact_biomass = df[df.product_name.isin(biomass_list2)].copy()
            df = pd.concat([df_impact_rest, df_impact_biomass], ignore_index=True)
            df['product_name'] = df['product_name'].str.replace('_lbdv', '', regex=False)
        if self.country == "World":
            df = df[(df.Location == self.country) | (df.Location == "Global average")].copy()
        elif pycountry.countries.get(alpha_3=self.country):
            country_iso2 = pycountry.countries.get(alpha_3=self.country).alpha_2
            df = df[(df.Location == country_iso2) | (df.Location == "Global average")].copy()
        elif self.country in image_region_list:
            df = df[(df.Location == self.country) | (df.Location == "Global average")].copy()
        else:
            print('country / region not found')
            df = pd.DataFrame()
        if self.ele_impact != 'default':
            df.loc[df.product_name == 'electricity_non_biomass', 'GHG'] = -self.ele_impact
        # add_health_impact
        df['Sector'] = 'agriculture'
        df.loc[df.product_name.str.contains('electricity'), 'Sector'] = 'energy'
        df.loc[df.Location == 'Global average', 'Sector'] = 'general'
        df.loc[df.product_name.isin(['natural_gas', 'petroleum']), 'Sector'] = 'energy'
        if self.country in df_pm['Region'].unique():
            df_pm = df_pm.loc[df_pm.Region == self.country].copy()
            df_pm.fillna(0, inplace=True)
            self.pm_dict = df_pm.to_dict(orient='records')[0]
            df['Health'] = df['Particulates'] * df['Sector'].apply(lambda x: self.pm_dict['pm25_emission_' + x]) + \
                           df['Sulfur dioxide'] * df['Sector'].apply(lambda x: self.pm_dict['sox_emission_' + x]) + \
                           df['Nitrogen oxides'] * df['Sector'].apply(lambda x: self.pm_dict['nox_emission_' + x]) + \
                           df['Ammonia'] * df['Sector'].apply(lambda x: self.pm_dict['nh3_emission_' + x])
            df['Health'] *= -1
            df1 = df.copy()
            df1['Health_pm25'] = df['Particulates'] * df['Sector'].apply(lambda x: self.pm_dict['pm25_emission_' + x]) * 1e6
            df1['Health_sox'] = df['Sulfur dioxide'] * df['Sector'].apply(lambda x: self.pm_dict['sox_emission_' + x]) * 1e6
            df1['Health_nox'] = df['Nitrogen oxides'] * df['Sector'].apply(lambda x: self.pm_dict['nox_emission_' + x]) * 1e6
            df1['Health_nh3'] = df['Ammonia'] * df['Sector'].apply(lambda x: self.pm_dict['nh3_emission_' + x]) * 1e6
            df2 = df1.loc[(df1.product_name.isin(residue_dict.values())) | (df1.product_name.isin(['natural_gas', 'petroleum']))].copy()
            df2 = df2[['product_name', 'Health_pm25', 'Health_sox', 'Health_nox', 'Health_nh3']].copy()
            df2['Health'] = df2.sum(axis=1, numeric_only=True)
            df2['pm25_contribution'] = df2['Health_pm25'] / df2['Health']
            df2['sox_contribution'] = df2['Health_sox'] / df2['Health']
            df2['nox_contribution'] = df2['Health_nox'] / df2['Health']
            df2['nh3_contribution'] = df2['Health_nh3'] / df2['Health']
            df2 = df2[['product_name', 'Health', 'pm25_contribution', 'sox_contribution', 'nox_contribution', 'nh3_contribution']].copy()
        else:
            print('country does not exist in the health impact data')
        self.df_rm_impact = df.copy()
        return df

    def read_plastics_demand(self):
        if self.plastics_demand == 'default':
            df_demand = pd.read_excel(self.plastics_file_path, engine='openpyxl',
                                               sheet_name='production_sector')
            df_demand.loc[df_demand['Variable'] == 'Plastics|Production|Sector|Industrial Machinery',
                          'Variable'] = 'Plastics|Production|Sector|Other'
            df_demand = df_demand.groupby(by=['Region', 'Variable', 'Unit']).sum(numeric_only=True).reset_index()
            df_demand = df_demand[['Region', 'Variable', 'Unit', '2020', '2050']].copy()
            demand_ratio = self.plastics_demand_ratio
            df_demand['2050'] *= demand_ratio
            df_demand[f'cagr_2050'] = (df_demand[str(2050)] / df_demand[str(2020)]) ** (1 / 30) - 1
            df_share = pd.read_excel(self.plastics_file_path, engine='openpyxl', sheet_name='share_subsector')
            df_share['GPPS'] += df_share['EPS']
            df_share.drop('EPS', axis=1, inplace=True)
            if self.country in df_demand['Region'].unique():
                df_demand = df_demand[df_demand['Region'] == self.country].copy()
            else:
                print('country does not exist in the plastics demand data')
                df_demand = pd.DataFrame()
        else:
            df_demand = pd.DataFrame()
            df_share = pd.DataFrame()
            print('user-defined scenarios to be developed')
        self.df_plastics_demand = df_demand
        self.df_plastics_share = df_share
        return df_demand, df_share

    def polymer_subsector_demand_dict(self):
        df_plastics_share = self.df_plastics_share.copy()
        df_plastics_demand = self.df_plastics_demand.copy()
        plastics_demand_dict = {}
        for i in df_plastics_share.index:
            sector = df_plastics_share.loc[i, 'Sector']
            sector_demand = df_plastics_demand.loc[df_plastics_demand['Variable'] == sector, '2050'].iloc[0]
            subsector = df_plastics_share.loc[i, 'Subsector']
            for polymer in df_plastics_share.columns[2:-1]:
                polymer_demand = df_plastics_share.loc[i, polymer] * sector_demand
                if polymer_demand > 0:
                    product_name_temp = f"{subsector}_{polymer}".lower().replace(' ', '_')
                    plastics_demand_dict[product_name_temp] = polymer_demand
        self.demand_dict = plastics_demand_dict
        df = pd.merge(df_plastics_share, df_plastics_demand, left_on='Sector', right_on='Variable', how='left')
        df1 = df.copy()
        for polymer in df1.columns[2:12]:
            df1[polymer] = df1[polymer] * df1['2050']
        df1 = df1.iloc[:, 1:12]
        df1.to_excel('data/processed/plastics_production_2050.xlsx', index=False)
        total_demand = df1.iloc[:, 2:12].sum().sum()
        return plastics_demand_dict

    def rm_supply_dict(self):
        df_avai_biomass = pd.read_csv(r'data/raw/lignocellulose_feedstock_combined_potential_impacts_all_scenarios.csv',
                                      index_col=0)
        df_avai_biomass['Country'] = df_avai_biomass['Country'].fillna('NA')
        df_avai_biomass = df_avai_biomass[(df_avai_biomass.YEAR == self.year) &
                                          (df_avai_biomass.SCENARIO == self.scenario) &
                                          (df_avai_biomass.Price == 'normal')].copy()
        if self.biomass_availability == 'default':
            if self.low_biodiversity:
                df_avai_biomass = df_avai_biomass[df_avai_biomass.BDV < 1e-14].copy()
            df_avai_biomass.loc[df_avai_biomass.Product.str.contains('conifer'), 'Product'] = 'Forest residue'
            df_avai_biomass["product_name"] = df_avai_biomass["Product"].map(residue_dict)
            df_avai_biomass = pd.pivot_table(df_avai_biomass, index=['Country', 'product_name'],
                                             values=['AVAI_MIN', 'AVAI_MAX'],
                                             aggfunc='sum').reset_index()
            if self.country == "World":
                df_avai_biomass = pd.pivot_table(df_avai_biomass, index=['product_name'],
                                                 values=['AVAI_MIN', 'AVAI_MAX'],
                                                 aggfunc='sum').reset_index()
            elif pycountry.countries.get(alpha_3=self.country):
                country_iso2 = pycountry.countries.get(alpha_3=self.country).alpha_2
                df_avai_biomass = df_avai_biomass[df_avai_biomass.Country == country_iso2].copy()
                new_rows = []
                for residue in residue_dict.values():
                    if residue not in df_avai_biomass['product_name'].unique():
                        new_rows.append({'Country': country_iso2, 'product_name': residue, 'AVAI_MIN': 0, 'AVAI_MAX': 0})
                if new_rows:
                    df_avai_biomass = pd.concat([df_avai_biomass, pd.DataFrame(new_rows)], ignore_index=True)
            elif self.country in image_region_list:
                df_country = pd.read_excel(r'data/raw/Country.xlsx', engine='openpyxl', sheet_name='Sheet1')
                df_country.loc[df_country.Country == "Namibia", "ISO2"] = "NA"
                df_country = df_country.loc[df_country.IMAGE_region == self.country].copy()
                country_list = list(df_country.ISO2.unique())
                df_avai_biomass = df_avai_biomass[df_avai_biomass.Country.isin(country_list)].copy()
                df_avai_biomass = pd.pivot_table(df_avai_biomass, index=['product_name'],
                                                 values=['AVAI_MIN', 'AVAI_MAX'],
                                                 aggfunc='sum').reset_index()
                new_rows = []
                for residue in residue_dict.values():
                    if residue not in df_avai_biomass['product_name'].unique():
                        new_rows.append({'Country': self.country, 'product_name': residue,
                                         'AVAI_MIN': 0, 'AVAI_MAX': 0})
                if new_rows:
                    df_avai_biomass = pd.concat([df_avai_biomass, pd.DataFrame(new_rows)], ignore_index=True)
            else:
                print('country does not exist in the biomass availability data')
                df_avai_biomass = pd.DataFrame()
            df_avai_biomass['supply'] = df_avai_biomass['AVAI_MIN'] / 1000 * self.biomass_ratio  # convert to Mt
            df_avai_biomass = df_avai_biomass[['product_name', 'supply']].copy()
            supply_dict = df_avai_biomass.set_index('product_name').to_dict()['supply']
        else:
            supply_dict = {}
            print('user-defined scenarios to be developed')
        if self.ele_availability == 'default':
            df_ele = self.df_ele_avai_default.copy()
            df_region = self.df_region_ele.copy()
            df_ele = df_ele.loc[df_ele['t'] == self.year].copy()
            df_ele = pd.merge(df_ele, df_region, how='left', on='Regions')
            df_ele.dropna(subset=['Region'], inplace=True)
            df_ele['Electricity Production~per Technology'] /= (1000000000 / 277.778)  # TWh
            if self.country in df_ele.Region.unique():
                df_ele_1 = df_ele[df_ele.Region == self.country].copy()
                ele_tot = df_ele_1.loc[df_ele_1.Technology == 'total', 'Electricity Production~per Technology'].sum()
                ele_bio = df_ele_1.loc[df_ele_1.Technology.str.contains('bio'),
                                       'Electricity Production~per Technology'].sum()
                ele_avai = (ele_tot - ele_bio) * self.ele_ratio
            else:
                print('region does not exist in the electricity production data')
                ele_avai = 0
        else:
            ele_avai = self.ele_availability
        supply_dict['electricity_non_biomass'] = ele_avai
        if not self.bl_routes:
            supply_dict['forest_residue'] = 0
        self.supply_dict = supply_dict
        return supply_dict

    def waste_to_production_ratio_by_subsector(self):
        df = self.df_plastics_demand.copy()
        df['sector'] = df['Variable'].map(plastics_sector_match_dict)
        lifetime_mapping = {
            'Building and Construction': 33,
            'Automotive': 12,
            'Electrical and Electronic Equipment': 8,
            'Agriculture': 4,
            'Household items, furniture, leisure and others': 3,
            'Textiles': 5,
            'Packaging': 0
        }
        # df = df[['sector', 'Unit', '2020', str(demand_scenario), f'cagr_{demand_scenario}']].copy()
        # df = df[df['sector'] != 'Packaging'].copy()
        df['lifetime'] = df['sector'].map(lifetime_mapping)
        df['year_difference_to_2020'] = 30 - df['lifetime']
        df['waste_availability'] = df['2020'] * (1 + df[f'cagr_2050']) ** (df['year_difference_to_2020'])
        df['waste_to_production_ratio'] = df['waste_availability'] / df['2050']
        df.loc[df.sector == 'Household items, furniture, leisure and others', 'sector'] = 'Consumer products'
        df.loc[df.sector == 'Electrical and Electronic Equipment', 'sector'] = 'Electrical and Electronics'
        df.loc[df.sector == 'Automotive', 'sector'] = 'Transport'
        df.loc[df.sector == 'Agriculture', 'sector'] = 'Other'
        ratio_dict = {}
        for sector in df['sector'].unique():
            ratio = df.loc[df['sector'] == sector, 'waste_to_production_ratio'].values[0]
            for subsector in sector_subsector_dict[sector]:
                ratio_dict[subsector] = ratio
        self.waste_to_production_ratio_dict = ratio_dict
        return ratio_dict

    def prepare_master_file_1_integrate_ihs(self):

        # step 1: combine ihs inventory and master inventory

        if os.path.exists(f"data/intermediate/ihs_inventory_{self.allocation_choice}.csv"):
            df_flow_ihs = pd.read_csv(f"data/intermediate/ihs_inventory_{self.allocation_choice}.csv")
        else:
            df_flow_ihs = ihs_data_inventory(self.master_file_path, self.allocation_choice)
        for ihs_name, master_name in ihs_to_master_name_alignment_dict.items():
            df_flow_ihs.loc[df_flow_ihs['product_name'] == ihs_name, 'product_name'] = master_name
        # df_product
        df_product = pd.read_excel(self.master_file_path, engine='openpyxl', sheet_name='product')
        df_product = df_product[df_product.include == "yes"].copy()
        # df_process_ihs
        df_process_ihs = pd.read_excel(self.master_file_path, engine='openpyxl', sheet_name='process_ihs')
        df_process_ihs = df_process_ihs.loc[df_process_ihs['include'] == 'yes'].copy()
        df_process_ihs['process'] = df_process_ihs['product'] + ', ' + df_process_ihs['process']
        df_process_ihs_new = pd.DataFrame({'product': 'diethylene_glycol',
                                           'process': 'diethylene_glycol, from ethylene via eo',
                                           'include': 'yes',
                                           'co2_route': 'yes', 'agricultural_residue_route': 'yes',
                                           'forest_residue_route': 'yes', 'fossil_route': 'yes'}, index=[0])
        df_process_ihs = pd.concat([df_process_ihs, df_process_ihs_new], ignore_index=True)
        df_process_ihs.rename(columns={'product': 'product_name', 'process': 'product_process'}, inplace=True)
        df_process_ihs['Data source'] = 'ihsmarkit'
        df_flow_ihs = df_flow_ihs.loc[df_flow_ihs.process.isin(list(df_process_ihs['product_process'].unique()))].copy()
        # df_process
        df_process = pd.read_excel(self.master_file_path, engine='openpyxl', sheet_name='process')
        df_process = df_process[df_process.include == "yes"].copy()
        df_process = pd.concat([df_process, df_process_ihs], ignore_index=True)
        # df_flow
        df_flow = pd.read_excel(self.master_file_path, engine='openpyxl', sheet_name='flows')
        df_flow = df_flow[df_flow.process.isin(list(df_process.product_process.unique()))].copy()
        df_flow_ihs = df_flow_ihs[['product_name', 'process', 'unit', 'value', 'type']].copy()
        df_flow = pd.concat([df_flow, df_flow_ihs], ignore_index=True)
        # cooling water unit conversion: # https://www.fao.org/3/bc822e/bc822e.pdf, table 1, once-through cooling
        df_flow.loc[df_flow.product_name == 'cooling_water_kg', 'value'] /= 36.944
        df_flow.loc[df_flow.product_name == 'cooling_water_kg', 'unit'] = 'MJ'
        df_flow.loc[df_flow.product_name == 'cooling_water_kg', 'product_name'] = 'cooling_water'
        ihs_product_list = list(df_flow_ihs['product_name'].unique())
        ihs_intermediate_list = [x for x in ihs_product_list if x not in list(df_product['product_name'].unique())]
        df = pd.concat([df_product, pd.DataFrame({'product_name': ihs_intermediate_list,
                                                              'unit': ['kg'] * len(ihs_intermediate_list),
                                                              'product_type': ['intermediate'] * len(
                                                                  ihs_intermediate_list),
                                                              'include': ['yes'] * len(ihs_intermediate_list)
                                                              })], ignore_index=True)
        for p, c in carbon_content_dict.items():
            if p in df['product_name'].unique():
                df.loc[df.product_name == p, 'carbon_content'] = c
        df.dropna(subset=['carbon_content'], inplace=True)
        return df, df_process, df_flow

    def prepare_master_file_2_integrate_plastics_product_waste(self):
        df, df_process, df_flow = self.prepare_master_file_1_integrate_ihs()
        df_plastics_share = self.df_plastics_share.copy()
        df_plastics_demand = self.df_plastics_demand.copy()
        df_plastics_demand.set_index('Variable', inplace=True)
        sector_demands = df_plastics_demand.loc[df_plastics_share['Sector'], '2050'].values
        process_pl_list = []
        flow_pl_list = []
        product_pl_list = []
        for i, row in df_plastics_share.iterrows():
            sector_demand = sector_demands[i]
            subsector = row['Subsector']
            for polymer in df_plastics_share.columns[2:-1]:
                polymer_demand = row[polymer] * sector_demand
                if polymer_demand > 0:
                    polymer2 = polymer.lower()
                    carbon_content = df.loc[df.product_name == polymer2, 'carbon_content'].iloc[0]
                    product_name = f"{subsector}_{polymer2}".replace(' ', '_').lower()
                    waste_name = f'{polymer2}_waste'
                    waste_to_production_ratio = self.waste_to_production_ratio_dict[subsector]
                    process_pl_list.append({
                        'product_name': product_name,
                        'product_process': f'{subsector}, from {polymer2}',
                        'Data source': 'MK123',
                        'include': 'yes',
                        'co2_route': 'yes',
                        'agricultural_residue_route': 'yes',
                        'forest_residue_route': 'yes',
                        'fossil_route': 'yes'
                    })
                    flow_pl_list.extend([
                        {'product_name': product_name, 'process': f'{subsector}, from {polymer2}', 'unit': 'kg',
                         'value': 1, 'type': 'PRODUCT'},
                        {'product_name': polymer2, 'process': f'{subsector}, from {polymer2}', 'unit': 'kg',
                         'value': -1, 'type': 'RAW MATERIALS'},
                        {'product_name': waste_name, 'process': f'{subsector}, from {polymer2}', 'unit': 'kg',
                         'value': waste_to_production_ratio, 'type': 'WASTE'}
                    ])
                    product_pl_list.append({
                        'product_name': product_name,
                        'unit': 'kg',
                        'product_type': 'product',
                        'include': 'yes',
                        'carbon_content': carbon_content
                    })
                    product_pl_list.append({
                        'product_name': waste_name,
                        'unit': 'kg',
                        'product_type': 'waste',
                        'include': 'yes',
                        'carbon_content': carbon_content
                    })

        # Convert lists to DataFrames once outside the loop
        df_process_pl = pd.DataFrame(process_pl_list)
        df_flow_pl = pd.DataFrame(flow_pl_list)
        df_pl = pd.DataFrame(product_pl_list).drop_duplicates('product_name')
        df = pd.concat([df, df_pl], ignore_index=True)
        df_process = pd.concat([df_process, df_process_pl], ignore_index=True)
        df_flow = pd.concat([df_flow, df_flow_pl], ignore_index=True)
        return df, df_process, df_flow

    def prepare_master_file_3_add_mechanical_recycling(self):
        df, df_process_all, df_flow_all = self.prepare_master_file_2_integrate_plastics_product_waste()
        if self.mechanical_recycling:
            df_flow_recycling = mechanical_recycling_flows()
            df_process = df_process_all.loc[(df_process_all['Data source'] == 'MK123') &
                                            (~df_process_all['product_name'].str.contains('pur'))].copy()
            df_flow = df_flow_all.loc[df_flow_all.process.isin(list(df_process.product_process.unique()))].copy()
            df_process['product_process'] += '_mr'
            df_process['product_name'] += '_mr'
            df_flow['process'] += '_mr'
            df_flow.loc[~df_flow.product_name.str.contains('waste'), 'product_name'] += '_mr'
            df_flow_recycling = pd.concat([df_flow_recycling, df_flow], ignore_index=True)
            df_flow_all = pd.concat([df_flow_all, df_flow_recycling], ignore_index=True)
            df_process_recycling = df_flow_recycling[df_flow_recycling.type == 'PRODUCT'].copy()
            df_process_recycling.rename(columns={'process': 'product_process'}, inplace=True)
            df_process_recycling = df_process_recycling[['product_name', 'product_process']].copy()
            df_process_recycling['include'] = 'yes'
            df_process_recycling['Data source'] = 'MK2'
            df_process_recycling['co2_route'] = 'yes'
            df_process_recycling['agricultural_residue_route'] = 'yes'
            df_process_recycling['forest_residue_route'] = 'yes'
            df_process_recycling['fossil_route'] = 'yes'
            # df_process_recycling = pd.concat([df_process_recycling, df_process], ignore_index=True)
            df_process_all = pd.concat([df_process_all, df_process_recycling], ignore_index=True)
            df_product_recycling = df_flow_recycling[['product_name', 'unit']].copy()
            df_int = df_product_recycling[df_product_recycling.product_name.str.contains('_mr')].copy()
            df_int['product_type'] = 'intermediate'
            df_int['include'] = 'yes'
            df_int = df_int.drop_duplicates()
            product_list = df.loc[df.product_type == 'product', 'product_name'].to_list()
            for p in product_list:
                carbon_content = df.loc[df.product_name == p, 'carbon_content'].iloc[0]
                df_int.loc[df_int.product_name.str.contains(p), 'product_type'] = 'product'
                df_int.loc[df_int.product_name.str.contains(p), 'carbon_content'] = carbon_content
            for p in df_int.loc[df_int.product_type == 'intermediate', 'product_name'].to_list():
                p1 = p.replace('_mr', '')
                carbon_content = df.loc[df.product_name == p1, 'carbon_content'].iloc[0]
                df_int.loc[df_int.product_name == p, 'carbon_content'] = carbon_content
            df = pd.concat([df, df_int], ignore_index=True)
            df_flow_all['product_type'] = df_flow_all['product_name'].map(df.set_index('product_name')['product_type'])
            return df, df_process_all, df_flow_all
        else:
            return df, df_process_all, df_flow_all

    def prepare_master_file_4_add_chemical_recycling(self):
        df, df_process_all, df_flow_all = self.prepare_master_file_3_add_mechanical_recycling()
        if self.gasi:
            df_gasi = gasification_flows()
            df_flow_all = pd.concat([df_flow_all, df_gasi], ignore_index=True)
            df_process_gasi = df_gasi[df_gasi.type == 'PRODUCT'].copy()
            df_process_gasi.rename(columns={'process': 'product_process'}, inplace=True)
            df_process_gasi['include'] = 'yes'
            df_process_gasi['Data source'] = 'Prifti2023'
            df_process_gasi['co2_route'] = 'yes'
            df_process_gasi['agricultural_residue_route'] = 'yes'
            df_process_gasi['forest_residue_route'] = 'yes'
            df_process_gasi['fossil_route'] = 'yes'
            df_process_all = pd.concat([df_process_all, df_process_gasi], ignore_index=True)
            df_flow_all['product_type'] = df_flow_all['product_name'].map(df.set_index('product_name')['product_type'])
        if self.pyrolysis:
            df_pyrolysis = pyrolysis_flows()
            df_flow_all = pd.concat([df_flow_all, df_pyrolysis], ignore_index=True)
            df_process_pyrolysis = df_pyrolysis[df_pyrolysis.type == 'PRODUCT'].copy()
            df_process_pyrolysis.rename(columns={'process': 'product_process'}, inplace=True)
            df_process_pyrolysis['include'] = 'yes'
            df_process_pyrolysis['Data source'] = 'IHSmarkit'
            df_process_pyrolysis['co2_route'] = 'yes'
            df_process_pyrolysis['agricultural_residue_route'] = 'yes'
            df_process_pyrolysis['forest_residue_route'] = 'yes'
            df_process_pyrolysis['fossil_route'] = 'yes'
            df_process_all = pd.concat([df_process_all, df_process_pyrolysis], ignore_index=True)
            df_flow_all['product_type'] = df_flow_all['product_name'].map(df.set_index('product_name')['product_type'])
            df_temp = pd.DataFrame({'product_name': 'naphtha_mix', 'unit': 'kg', 'product_type': 'intermediate',
                                    'include': 'yes', 'carbon_content': 0.84826}, index=[0])
            df = pd.concat([df, df_temp], ignore_index=True)
        return df, df_process_all, df_flow_all


    def prepare_master_file_5_separate_routes(self):
        df_product_all, df_process_all, df_flow_all = self.prepare_master_file_4_add_chemical_recycling()
        df_process_all.loc[df_process_all.product_process.str.contains('hips'), 'co2_route'] = ''
        route_list = ['co2_route', 'forest_residue_route', 'agricultural_residue_route', 'fossil_route']
        active_routes = [self.co2_routes, self.bl_routes, self.bs_routes, self.fossil_routes]
        route_suffixes = {
            'co2_route': '_co2',
            'agricultural_residue_route': '_biogenic_short',
            'forest_residue_route': '_biogenic_long',
            'fossil_route': '_fossil'
        }
        product_list_temp = ['agricultural_residue', 'corn_steep_liquor', 'enzyme', 'co2_emission_biogenic_short',
                             'co2_emission_biogenic_long', 'co2_emission_fossil', 'potato_starch']
        active_route_list = [route for route, active in zip(route_list, active_routes) if active]
        process_frames = []
        flow_frames = []
        unique_product_names = []
        product_name_index = df_product_all.set_index('product_name')
        df_product_new = df_product_all.loc[(df_product_all.carbon_content == 0) |
                             (~df_product_all.product_type.isin(['intermediate', 'waste', 'product'])) |
                             (df_product_all.product_name.isin(product_list_temp))].copy()
        df_product_modify = df_product_all.loc[~df_product_all.product_name.isin(df_product_new.product_name.unique())].copy()
        for route in active_route_list:
            route_processes = df_process_all[df_process_all[route] == 'yes'].copy()
            flow_data = df_flow_all[df_flow_all['process'].isin(route_processes['product_process'])].copy()
            process_suffix = route_suffixes[route]
            route_processes['product_name'] += process_suffix
            route_processes['product_process'] += process_suffix

            flow_data['process'] += process_suffix
            flow_data['carbon_content'] = flow_data['product_name'].map(product_name_index['carbon_content'])
            flow_data['product_type'] = flow_data['product_name'].map(product_name_index['product_type'])
            flow_data.loc[(flow_data.carbon_content > 0) &
                             (flow_data.product_type.isin(['intermediate', 'emission', 'waste', 'product'])) &
                             (~flow_data.product_name.isin(product_list_temp)), 'product_name'] += process_suffix
            flow_data.loc[flow_data.product_name == 'co2_emission_co2', 'product_name'] = 'co2_emission_fossil'
            flow_data.loc[flow_data.product_name == 'co2_feedstock', 'product_name'] += process_suffix
            flow_data.loc[flow_data.product_name == 'co2_feedstock_co2', 'product_name'] = 'co2_feedstock_fossil'
            process_frames.append(route_processes)
            flow_frames.append(flow_data)
            unique_product_names.extend(route_processes['product_name'].unique())
            df_product_temp = df_product_modify.copy()
            df_product_temp['product_name'] += process_suffix
            df_product_new = pd.concat([df_product_new, df_product_temp], ignore_index=True)
        df_process_new = pd.concat(process_frames, ignore_index=True)
        df_flow_new = pd.concat(flow_frames, ignore_index=True)
        df_process_temp = df_process_all[df_process_all['all'] == 'yes'].copy()
        df_flow_temp = df_flow_all[df_flow_all.process.isin(list(df_process_temp.product_process.unique()))].copy()
        df_process_new = pd.concat([df_process_new, df_process_temp], ignore_index=True)
        df_flow_new = pd.concat([df_flow_new, df_flow_temp], ignore_index=True)
        valid_products = set(df_flow_new['product_name'].unique())
        df_product_new = df_product_new[df_product_new['product_name'].isin(valid_products)]
        df_flow_new['carbon_content'] = df_flow_new['product_name'].map(df_product_new.set_index('product_name')['carbon_content'])
        return df_product_new, df_process_new, df_flow_new

    def prepare_master_file_6_add_fossil_routes(self):
        df, df_process_all, df_flow_all = self.prepare_master_file_5_separate_routes()
        dff = pd.read_csv('data/raw/ecoinvent_hvc_processes.csv')
        dff = dff.groupby(by=['product_name', 'process', 'unit', 'type']).sum(numeric_only=True).reset_index()
        dff.loc[dff.value == 1, 'type'] = 'PRODUCT'
        for product in dff.product_name.unique():
            if product in df.product_name.unique():
                carbon_content = df.loc[df.product_name == product, 'carbon_content'].iloc[0]
            elif f'{product}_biogenic_short' in df.product_name.unique():
                carbon_content = df.loc[df.product_name == f'{product}_biogenic_short', 'carbon_content'].iloc[0]
            elif f'{product}_fossil' in df.product_name.unique():
                carbon_content = df.loc[df.product_name == f'{product}_fossil', 'carbon_content'].iloc[0]
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

        if not self.fossil_routes:
            return df, df_process_all, df_flow_all
        else:
            dff['process'] += '_fossil'
            dff.loc[(dff.carbon_content > 0) & (dff.type != 'EMISSION'), 'product_name'] += '_fossil'
            #dff.loc[dff.product_name == 'ammonia', 'product_name'] = 'ammonia_fossil'
            #dff.loc[dff.product_name == 'hydrogen', 'product_name'] = 'hydrogen_fossil'
            dff.loc[dff.product_name == 'natural_gas_fossil', 'product_name'] = 'natural_gas'
            dff.loc[dff.product_name == 'petroleum_fossil', 'product_name'] = 'petroleum'
            dff.loc[dff.product_name == 'natural_gas', 'value'] *= 0.735    # density in kg/m3
            dff.loc[dff.product_name == 'natural_gas', 'unit'] = 'kg'
            dff.loc[dff.process.str.startswith('C3 hydrocarbon mixture'),
                    'process'] = 'propylene, from potroleum refinery openration_fossil'
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
            dff_product.loc[
                dff_product.product_name.isin(['petroleum', 'natural_gas']), 'product_type'] = 'raw_material'
            dff_product['include'] = 'yes'
            df = pd.concat([df, dff_product], ignore_index=True)
            df_process_all = pd.concat([df_process_all, dff_process], ignore_index=True)
            df_flow_all = pd.concat([df_flow_all, dff], ignore_index=True)

            df_sc = dff.loc[dff.process.str.contains('steam cracking')]
            sc_process_list = list(df_sc.process.unique())
            feedstock_list = ['butane_fossil', 'diesel_fossil', 'ethane_fossil', 'naphtha_fossil',
                              'natural_gas_liquids_fossil', 'propane_fossil']
            suffix_list = ['_co2', '_biogenic_short', '_biogenic_long', '_fossil']
            df_sc2 = pd.DataFrame()
            for sc_process in sc_process_list:
                df_temp = df_sc[df_sc.process == sc_process].copy()
                df_temp['process'] = df_temp['process'].str.replace('_fossil', '_pyrolysis')
                process_name = df_temp['process'].iloc[0]
                df_temp1 = df_temp[df_temp.product_name.isin(feedstock_list)].copy()
                feedstock_amount = df_temp1.value.sum()
                df_temp = df_temp[~df_temp.product_name.isin(feedstock_list)].copy()
                df_temp['product_name'] = df_temp['product_name'].str.replace('_fossil', '')
                df_temp2 = pd.DataFrame({'product_name': 'naphtha_mix', 'process': process_name, 'unit': 'kg',
                                         'type': 'RAW MATERIALS', 'value': feedstock_amount, 'carbon_content': 0.84826},
                                        index=[0])
                df_temp = pd.concat([df_temp, df_temp2], ignore_index=True)
                for suffix in suffix_list:
                    df_temp3 = df_temp.copy()
                    df_temp3['process'] += suffix
                    df_temp3.loc[df_temp3.product_name == 'naphtha_mix', 'product_name'] += suffix
                    df_temp3.loc[df_temp3.product_name == 'co2_emission', 'product_name'] += suffix
                    df_temp3.loc[df_temp3['type'] == 'PRODUCT', 'product_name'] += suffix
                    df_sc2 = pd.concat([df_sc2, df_temp3], ignore_index=True)
            df_sc2['product_name'] = df_sc2['product_name'].str.replace('co2_emission_co2', 'co2_emission_fossil')
            df_sc_process = df_sc2[df_sc2['type'] == 'PRODUCT'].copy()
            df_sc_process['Data source'] = 'ecoinvent 3.10'
            df_sc_process['include'] = 'yes'
            df_sc_process = df_sc_process.rename(columns={'process': 'product_process'})
            df_sc_process = df_sc_process[
                ['product_name', 'product_process', 'unit', 'include', 'Data source', 'carbon_content']].copy()
            df_sc_product = df_sc2[~df_sc2.product_name.isin(list(df['product_name'].unique()))].copy()
            df_sc_product = df_sc_product[['product_name', 'unit', 'carbon_content']].copy()
            df_sc_product.drop_duplicates(inplace=True)
            df_sc_product['product_type'] = 'intermediate'
            df_sc_product['include'] = 'yes'
            df = pd.concat([df, df_sc_product], ignore_index=True)
            df_process_all = pd.concat([df_process_all, df_sc_process], ignore_index=True)
            df_flow_all = pd.concat([df_flow_all, df_sc2], ignore_index=True)
            return df, df_process_all, df_flow_all

    def prepare_master_file_7_new_bio_plastics(self):
        df, df_process_all, df_flow_all = self.prepare_master_file_6_add_fossil_routes()
        if not self.new_bio_plastics:
            return df, df_process_all, df_flow_all
        else:
            df_plastics_substitute = pd.read_excel(self.plastics_file_path, engine='openpyxl',
                                                   sheet_name='substitution_factor_subsector')
            df_plastics_substitute = df_plastics_substitute[df_plastics_substitute['include'] == 'yes'].copy()
            df_plastics_substitute.loc[
                df_plastics_substitute['traditional_plastics'] == 'EPS', 'traditional_plastics'] = 'GPPS'
            df_plastics_substitute = df_plastics_substitute.groupby(by=['sector', 'subsector', 'bio-based_plastics',
                                                                        'traditional_plastics']).mean(
                numeric_only=True).reset_index()
            for j in df_plastics_substitute.index:
                subsector = df_plastics_substitute.loc[j, 'subsector']
                polymer_0 = df_plastics_substitute.loc[j, 'traditional_plastics']
                waste_to_production_ratio = self.waste_to_production_ratio_dict[subsector]
                polymer_1 = f"{polymer_0}".lower()
                polymer_2 = df_plastics_substitute.loc[j, 'bio-based_plastics'].lower()
                factor = df_plastics_substitute.loc[j, 'substitution_factor']
                suffix_list = []
                if self.bs_routes:
                    suffix_list.append('_biogenic_short')
                if self.bl_routes:
                    suffix_list.append('_biogenic_long')
                for suffix in suffix_list:
                    polymer_2s = f"{polymer_2}{suffix}"
                    product_name = f"{subsector}_{polymer_1}_{polymer_2s}".lower().replace(' ', '_')
                    if '+' in polymer_2:
                        polymer_21 = polymer_2.split('+')[0].split('%')[1].lower()
                        polymer_21s = f'{polymer_21}{suffix}'
                        carbon_content_21 = df.loc[df.product_name == polymer_21s, 'carbon_content'].iloc[0]
                        waste_21 = f'{polymer_21s}_waste'
                        polymer_21_share = float(polymer_2.split('+')[0].split('%')[0]) / 100
                        polymer_22 = polymer_2.split('+')[1].split('%')[1].lower()
                        polymer_22s = f'{polymer_22}{suffix}'
                        carbon_content_22 = df.loc[df.product_name == polymer_22s, 'carbon_content'].iloc[0]
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
                                                         'include': 'yes'}])
                            df = pd.concat([df, new_product], ignore_index=True)
                        if waste_22 not in df['product_name'].unique():
                            new_product = pd.DataFrame([{'product_name': waste_22,
                                                         'unit': 'kg',
                                                         'product_type': 'waste',
                                                         'carbon_content': carbon_content_22,
                                                         'include': 'yes'}])
                            df = pd.concat([df, new_product], ignore_index=True)
                        if product_name not in df['product_name'].unique():
                            new_product = pd.DataFrame([{'product_name': product_name,
                                                         'unit': 'kg',
                                                         'product_type': 'product',
                                                         'carbon_content': carbon_content_product,
                                                         'include': 'yes'}])
                            df = pd.concat([df, new_product], ignore_index=True)
                    else:
                        waste_2 = f'{polymer_2s}_waste'
                        carbon_content_2 = df.loc[df.product_name == polymer_2s, 'carbon_content'].iloc[0]
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
                                                         'include': 'yes'}])
                            df = pd.concat([df, new_product], ignore_index=True)
                        if product_name not in df['product_name'].unique():
                            new_product = pd.DataFrame([{'product_name': product_name,
                                                         'unit': 'kg',
                                                         'product_type': 'product',
                                                         'carbon_content': carbon_content_2,
                                                         'include': 'yes'}])
                            df = pd.concat([df, new_product], ignore_index=True)
            df_flow_all['product_type'] = df_flow_all['product_name'].map(df.set_index('product_name')['product_type'])
            return df, df_process_all, df_flow_all

    def prepare_master_file_8_add_eol_incineration(self):
        df, df_process_all, df_flow_all = self.prepare_master_file_7_new_bio_plastics()
        waste_list = list(df[df.product_type == 'waste']['product_name'].unique())
        for waste in waste_list:
            if 'pur' in waste:
                plastics = f"{waste.split('_')[0]}_{waste.split('_')[1]}"
            else:
                plastics = waste.split('_')[0]
            carbon_content = df.loc[df.product_name == waste, 'carbon_content'].iloc[0]
            co2 = carbon_content * 44 / 12
            if 'biogenic_short' in waste:
                co2_name = 'co2_emission_biogenic_short'
            elif 'biogenic_long' in waste:
                co2_name = 'co2_emission_biogenic_long'
            else:
                co2_name = 'co2_emission_fossil'
            # pm emissions, ecoivnent 3.10, treatment of waste plastic, mixture, municipal incineration, GLO
            co = 2.5353e-5
            #nh3 = 1.7204e-6
            nox = 0.00054479
            pm = 1.7807e-6
            so2 = 1.496e-5
            nmvoc = 6.7875e-7
            # pm emissions, from Christopher
            #nh3 = 30.79 * 0.29 * 1e-6 #35 MJ/kg mixed plastics, pollution unit in 1e-6 kg pollutant/MJ
            pm = 30.79 * 8.55 * 1e-6  # 35 MJ/kg mixed plastics, pollution unit in 1e-6 kg pollutant/MJ
            nox = 30.79 * 81.68 * 1e-6  # 35 MJ/kg mixed plastics, pollution unit in 1e-6 kg pollutant/MJ
            so2 = 30.79 * 68.32 * 1e-6  # 35 MJ/kg mixed plastics, pollution unit in 1e-6 kg pollutant/MJ
            # pm emissions, EEA guideline, per kg basis
            pm = 3e-6
            nox = 1071e-6
            so2 = 87e-6
            #nh3 = 3e-6
            # pm emissions, based on regulations
            pm = 1.4e-4
            nox = 2.84e-3
            so2 = 7e-4
            nh3 = 0
            new_process = pd.DataFrame([{'product_name': co2_name,
                                         'product_process': f'waste incineration from {waste}',
                                         'include': 'yes', 'Data source': 'mass balance'}])
            df_process_all = pd.concat([df_process_all, new_process], ignore_index=True)
            new_flows_df = pd.DataFrame({
                'product_name': ['co_emission', 'nh3_emission', 'nox_emission', 'pm25_emission', 'sox_emission',
                                 'nmvoc_emission', co2_name, waste],
                'process': [f'waste incineration from {waste}'] * 8,
                'unit': ['kg'] * 8,
                'value': [co, nh3, nox, pm, so2, nmvoc, co2, -1],
                'type': ['EMISSION', 'EMISSION', 'EMISSION', 'EMISSION', 'EMISSION', 'EMISSION', 'EMISSION',
                         'RAW MATERIALS'],
                'product_type': ['emission', 'emission', 'emission', 'emission', 'emission', 'emission', 'emission',
                                 'waste']
            })
            df_flow_all = pd.concat([df_flow_all, new_flows_df], ignore_index=True)
            df.loc[df.product_name.str.contains(plastics), 'carbon_content'] = carbon_content
        return df, df_process_all, df_flow_all

    def prepare_master_file_9_add_ccs_process_co2(self):
        df, df_process_all, df_flow_all = self.prepare_master_file_8_add_eol_incineration()
        suffix_list = ['_fossil', '_biogenic_short', '_biogenic_long']
        if not self.ccs_process_co2:
            for suffix in suffix_list:
                co2e_name = f'co2_emission{suffix}'
                co2f_name = f'co2_feedstock{suffix}'
                df_flow = pd.DataFrame()
                new_flows_df = pd.DataFrame({
                    'product_name': [co2e_name, co2f_name],
                    'process': [f'{co2f_name} release'] * 2,
                    'unit': ['kg', 'kg'],
                    'value': [1, -1],
                    'type': ['EMISSION', 'RAW MATERIALS'],
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
                    'type': ['PRODUCT', 'RAW MATERIALS', 'EMISSION', 'UTILITIES', 'RAW MATERIALS'],
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
            df_product['include'] = 'yes'
            df_product['carbon_content'] = 0
            df_product.loc[df_product.product_name.str.contains('co2_storage'), 'carbon_content'] = 12 / 44
            df = pd.concat([df, df_product], ignore_index=True)
        return df, df_process_all, df_flow_all

    def prepare_master_file_10_add_rm_impact(self):
        df, df_process_all, df_flow_all = self.prepare_master_file_9_add_ccs_process_co2()
        df_impact = self.df_rm_impact.copy()
        df_impact = df_impact[['product_name', 'GHG', 'Biodiversity', 'Health']].copy()
        df = pd.merge(df, df_impact, on='product_name', how='left')
        df.loc[df.product_name.str.contains('pipeline'), 'GHG'] = -0.0024789
        df.loc[df.product_name.str.contains('pipeline'), 'Biodiversity'] = 0
        df.loc[df.product_type == 'emission', 'GHG'] = df.loc[df.product_type == 'emission',
                                                              'product_name'].map(emissions_ghg_dict)
        df.GHG.fillna(0, inplace=True)
        df.loc[df.product_type == 'intermediate', 'GHG'] = 0
        df.Biodiversity.fillna(0, inplace=True)
        pm_list = ['nh3_emission', 'nox_emission', 'pm25_emission', 'sox_emission']
        sector_list = ['energy', 'chemical', 'agriculture', 'general']
        df_temp = df.loc[df.product_name.isin(pm_list)].copy()
        df = df.loc[~df.product_name.isin(pm_list)].copy()
        for sector in sector_list:
            df_temp2 = df_temp.copy()
            df_temp2['product_name'] = df_temp2['product_name'] + '_' + sector
            df_temp2['Health'] = df_temp2['product_name'].map(self.pm_dict)
            df = pd.concat([df, df_temp2], ignore_index=True)
        df.Health.fillna(0, inplace=True)
        return df, df_process_all, df_flow_all

    def prepare_master_file_11_economic_allocation(self):
        df, df_process_all, df_flow_all = self.prepare_master_file_10_add_rm_impact()
        if self.allocation_choice == 'economic':
            if os.path.exists(f"data/intermediate/ihs_inventory_system_expansion.csv"):
                df_price = pd.read_csv(f"data/intermediate/ihs_inventory_system_expansion.csv")
            else:
                df_price = ihs_data_inventory(self.master_file_path, 'system_expansion')
            df_price = df_price.groupby(by=['product_name']).mean(numeric_only=True).reset_index()
            process_with_byproducts = df_flow_all.loc[(df_flow_all['type'] == 'BY-PRODUCT CREDITS') &
                                                      (~df_flow_all['product_name'].str.contains('co2_feedstock')), 'process'].unique()
            df_flow_temp = df_flow_all.loc[df_flow_all['process'].isin(process_with_byproducts)].copy()
            df_flow_temp['product_name2'] = df_flow_temp['product_name']
            for suffix in ['_biogenic_short', '_biogenic_long', '_fossil', '_co2']:
                df_flow_temp.loc[:, 'product_name2'] = df_flow_temp['product_name2'].str.replace(suffix, '')
            df_flow_temp['price'] = df_flow_temp['product_name2'].map(df_price.set_index('product_name')['price'])
            price_dict = {'tetrahydrofuran': 159.95, 'glucose': 100.87, 'succinic_acid': 273.37,
                          'syngas_2_to_1': 11.79, 'ammonia': 42.55}  # price from ihsmarkit
            df_flow_temp['price'].fillna(df_flow_temp['product_name2'].map(price_dict), inplace=True)
            df_flow_temp.loc[df_flow_temp.product_name.str.contains('naphtha_mix'), 'price'] = 58.26  # price from ihsmarkit
            for p in process_with_byproducts:
                df_temp = df_flow_temp.loc[df_flow_temp['process'] == p].copy()
                df_temp2 = df_temp.loc[(df_temp.type.isin(['PRODUCT', 'BY-PRODUCT CREDITS'])) &
                                       (~df_temp.product_name.str.contains('co2_feedstock'))].copy()
                df_temp2['revenue'] = df_temp2['value'] * df_temp2['price']
                df_temp2['allocation'] = df_temp2['revenue'] / df_temp2['revenue'].sum()
                alloc = df_temp2.loc[df_temp2['type'] == 'PRODUCT', 'allocation'].values[0]
                df_temp.loc[df_temp['type'] != 'PRODUCT', 'value'] *= alloc
                df_temp.drop(df_temp[(df_temp['type'] == 'BY-PRODUCT CREDITS') &
                                     (~df_temp.product_name.str.contains('co2_feedstock'))].index, inplace=True)
                df_temp.drop(columns=['product_name2', 'price'], inplace=True)
                df_flow_all.drop(df_flow_all[df_flow_all['process'] == p].index, inplace=True)
                df_flow_all = pd.concat([df_flow_all, df_temp], ignore_index=True)

        return df, df_process_all, df_flow_all

    def prepare_master_file_12_add_pm_emissions(self):
        df, df_process_all, df_flow_all = self.prepare_master_file_11_economic_allocation()
        df_pm = df_pm_emission(self.master_file_path)
        df_pm = df_pm.loc[df_pm.include == 'yes'].copy()
        for p in df_pm['product_name'].unique():
            df_process = df_process_all.loc[(df_process_all['product_name'].str.startswith(p)) &
                                            (~df_process_all['product_name'].str.contains('_mr')) &
                                            (~df_process_all['product_name'].str.contains('oxide')) &
                                            (~df_process_all['product_name'].str.contains('glycol'))].copy()
            df_process = df_process.loc[df_process['Data source'] != 'ecoinvent 3.10'].copy()
            if df_process.shape[0] > 0:
                df_temp = df_pm.loc[df_pm['product_name'] == p].copy()
                df_temp = df_temp[['pollutant', 'value']].copy()
                df_temp.rename(columns={'pollutant': 'product_name'}, inplace=True)
                for process in df_process['product_process'].unique():
                    df_temp1 = df_temp.copy()
                    df_temp1['process'] = process
                    df_temp1['unit'] = 'kg'
                    df_temp1['type'] = 'EMISSION'
                    df_temp1['product_type'] = 'emission'
                    df_temp1['carbon_content'] = 0
                    df_temp1.loc[df_temp1.product_name == 'co_emission', 'carbon_content'] = 0.4286
                    df_flow_all = pd.concat([df_flow_all, df_temp1], ignore_index=True)
        #'''
        emission_list = ['pm25_emission', 'sox_emission', 'nox_emission', 'nh3_emission']
        df_flow_all.loc[(df_flow_all['product_name'].isin(emission_list)) &
                        (df_flow_all['process'].str.contains('electricity')), 'product_name'] += '_energy'
        df_flow_all.loc[(df_flow_all['product_name'].isin(emission_list)) &
                        (df_flow_all['process'].str.contains('waste incineration')), 'product_name'] += '_energy'
        df_flow_all.loc[(df_flow_all['product_name'].isin(emission_list)), 'product_name'] += '_general'

        #'''
        return df, df_process_all, df_flow_all

    def prepare_master_file_final_adjustment(self):
        df, df_process_all, df_flow_all = self.prepare_master_file_12_add_pm_emissions()
        # differentiate coefficients for agricultural residue and forest residue

        df_flow_all.loc[(df_flow_all.process.str.contains('biogenic_long'))
                        & (df_flow_all['type'].isin(['UTILITIES', 'RAW MATERIALS']))
                        & (df_flow_all['value'] != 1) & (df_flow_all['value'] != -1), 'value'] *= 1
        df_flow_all.loc[(df_flow_all.process.str.endswith('_co2'))
                        & (df_flow_all['type'].isin(['UTILITIES', 'RAW MATERIALS']))
                        & (df_flow_all['value'] != 1) & (df_flow_all['value'] != -1), 'value'] *= 1

        df_flow_all['carbon_content'] = df_flow_all['product_name'].map(df.set_index('product_name')['carbon_content'])
        df_flow_all['product_type'] = df_flow_all['product_name'].map(df.set_index('product_name')['product_type'])
        df_flow_all['data_source'] = df_flow_all['process'].map(df_process_all.set_index('product_process')['Data source'])
        #'''
        if not self.bs_routes:
            df_process_all = df_process_all.loc[df_process_all.product_process !=
                                                'heat, high temperature, from agricultural residue'].copy()
            df_flow_all = df_flow_all.loc[df_flow_all.process !=
                                          'heat, high temperature, from agricultural residue'].copy()
            df_process_all = df_process_all.loc[df_process_all.product_process !=
                                                'electricity_biogenic, from agricultural residue'].copy()
            df_flow_all = df_flow_all.loc[df_flow_all.process !=
                                          'electricity_biogenic, from agricultural residue'].copy()
        if not self.bl_routes:
            df_process_all = df_process_all.loc[df_process_all.product_process !=
                                                'heat, high temperature, from forest residue'].copy()
            df_flow_all = df_flow_all.loc[df_flow_all.process != 'heat, high temperature, from forest residue'].copy()
            df_process_all = df_process_all.loc[df_process_all.product_process !=
                                                'electricity_biogenic, from forest residue'].copy()
            df_flow_all = df_flow_all.loc[df_flow_all.process !=
                                          'electricity_biogenic, from forest residue'].copy()
        if not self.bs_routes and not self.bl_routes and not self.co2_routes:
            df_process_all = df_process_all.loc[df_process_all.product_process !=
                                                'hydrogen, from PEM electrolysis'].copy()
            df_flow_all = df_flow_all.loc[df_flow_all.process !=
                                          'hydrogen, from PEM electrolysis'].copy()
            df_process_all = df_process_all.loc[df_process_all.product_process !=
                                                'ammonia, from Haber-Bosch process'].copy()
            df_flow_all = df_flow_all.loc[df_flow_all.process !=
                                          'ammonia, from Haber-Bosch process'].copy()
        #'''
        return df, df_process_all, df_flow_all

    def export_process_list(self):
        df, df_process_all, df_flow_all = self.prepare_master_file_final_adjustment()
        df1 = df_flow_all.copy()
        suffix_list = ['_fossil', '_biogenic_short', '_biogenic_long', '_co2']
        df4 = pd.DataFrame()
        for suffix in suffix_list:
            suffix_list2 = suffix_list.copy()
            suffix_list2.remove(suffix)
            df2 = df1.loc[~df1['process'].apply(lambda x: any(s in x for s in suffix_list2))].copy()
            df3 = df2.loc[df2.type == 'PRODUCT'].copy()
            product_list = df3.loc[df3.product_type == 'product', 'process'].unique()
            df2 = df2.loc[~df2.process.isin(product_list)].copy()
            df2 = df2.loc[~df2.process.str.startswith('agricultural residue, from')]
            df2 = df2.loc[~df2.process.str.contains('mechanical recycling')]
            df2 = df2.loc[~df2.process.str.contains('waste gasification')]
            df2 = df2.loc[~df2.process.str.contains('waste incineration')]
            df2 = df2.loc[~df2.process.str.contains('CCS')]
            df2 = df2.loc[~df2.process.str.contains('electricity, from')]
            df2['process'] = df2['process'].str.replace(suffix, '')
            df2['product_name'] = df2['product_name'].str.replace(suffix, '')
            check_list = ['ammonium_sulfate', 'corn_steep_liquor', 'enzyme', 'hydrogen_peroxide',
                          'isobutane', 'pentane']
            for x in check_list:
                df_temp = df2[df2.product_name == x].copy()
                has_positive = (df_temp['value'] > 0).any()
                has_negative = (df_temp['value'] < 0).any()
                if not has_negative:
                    process_list = df_temp['process'].unique()
                    df2 = df2.loc[~df2.process.isin(process_list)].copy()
            if suffix == '_fossil':
                df2 = df2.loc[~df2.process.str.contains('electricity_biogenic, from')]
                df2 = df2.loc[~df2.process.str.contains('heat, high temperature')]
                df2 = df2.loc[~df2.process.str.contains('hydrogen, from PEM')]
                df2 = df2.loc[~df2.process.str.contains('ammonia, from Haber-Bosch process')]
                df2 = df2.loc[~df2.process.str.contains('potato')]
                df2.to_csv(f"data/processed/fossil_process_list.csv", index=False)
                dff = df2.copy()
            else:
                df2 = df2.loc[~df2.process.isin(dff.process)].copy()
                df4 = pd.concat([df4, df2], ignore_index=True)
        df4 = df4.drop_duplicates()
        df4.to_csv(f"data/processed/alternative_process_list.csv", index=False)
        # waste treatment
        df5 = df1.loc[((df1.process.str.contains('waste')) | (df1.process.str.contains('recycl'))) &
                      (df1.process.str.contains('biogenic_short'))].copy()
        df5['process'] = df5['process'].str.replace('_biogenic_short', '')
        df5['product_name'] = df5['product_name'].str.replace('_biogenic_short', '')
        df5['process'] = df5['process'].str.replace('_mr', '')
        df5['product_name'] = df5['product_name'].str.replace('_mr', '')
        df5.to_csv(f"data/processed/plastic_waste_process_list.csv", index=False)
        # utility
        df6 = df1.loc[(df1.process.str.contains('electricity_biogenic, from agricultural residue')) |
                      (df1.process.str.startswith('heat, district')) |
                      (df1.process.str.contains('heat, high temperature, from agricultural residue')) |
                      (df1.process.str.contains('steam,')) | (df1.process.str.contains('cooling'))].copy()
        # ecoinvent
        df7 = df_process_all.loc[df_process_all['Data source'].str.contains('ecoinvent', case=False, na=False)].copy()
        df7['product_process'] = df7['product_process'].str.replace('_fossil', '')
        df7['product_process'] = df7['product_process'].str.replace('_biogenic_short', '')
        df7['product_process'] = df7['product_process'].str.replace('_biogenic_long', '')
        df7['product_name'] = df7['product_name'].str.replace('_fossil', '')
        df7['product_name'] = df7['product_name'].str.replace('_biogenic_short', '')
        df7['product_name'] = df7['product_name'].str.replace('_biogenic_long', '')
        df7 = df7.drop_duplicates()
        #df7.to_csv(f"data/processed/ecoinvent_process_list.csv", index=False)
        suffix_list = ['_fossil', '_biogenic_short', '_biogenic_long', '_co2']
        df8 = df1.copy()
        for suffix in suffix_list:
            df8['process'] = df8['process'].str.replace(suffix, '')
            df8['product_name'] = df8['product_name'].str.replace(suffix, '')
        df8 = df8.drop_duplicates()
        # from literature
        df8 = df8.loc[df8.data_source != 'ihsmarkit']
        df8 = df8.loc[df8.data_source != 'mass balance']
        df8 = df8.loc[df8.data_source != 'ecoinvent 3.10']
        df8 = df8.loc[df8.data_source != 'name change only']
        product_list = list(df8.loc[df8.product_type == 'product', 'process'].unique())
        df8 = df8.loc[~df8.process.isin(product_list)].copy()
        df8 = df8.loc[df8.product_name != 'nmvoc_emission']
        df8 = df8.loc[df8.product_name != 'co_emission']
        df8.to_csv(f"data/processed/flows_from_literature.csv", index=False)
        df_flow_all.to_csv(f"data/processed/flows_all.csv", index=False)
        return df, df_process_all, df1

    def set_up_optimization_model(self):
        self.df_product, self.df_process, self.df_flow = self.prepare_master_file_final_adjustment()
        df_impact_ghg = self.df_product#.loc[self.df_product.GHG != 0].copy()
        df_impact_bdv = self.df_product#.loc[self.df_product.Biodiversity != 0].copy()
        df_impact_bdv.loc[df_impact_bdv.product_name == 'electricity_non_biomass', 'Biodiversity'] = 0
        df_impact_ghg = df_impact_ghg.loc[df_impact_ghg.product_type.isin(['raw_material', 'emission', 'waste'])]
        df_impact_bdv = df_impact_bdv.loc[df_impact_bdv.product_type.isin(['raw_material', 'emission', 'waste'])]
        self.bdv_dict = dict(zip(df_impact_bdv['product_name'], df_impact_bdv['Biodiversity']))
        self.ghg_dict = dict(zip(df_impact_ghg['product_name'], df_impact_ghg['GHG']))
        self.health_dict = dict(zip(df_impact_ghg['product_name'], df_impact_ghg['Health']))
        self.process_name_list = list(self.df_process.product_process.unique())
        self.inflow_name, self.inflow = gp.multidict({(i, j): self.df_flow.loc[(self.df_flow['process'] == i) &
                                                                               (self.df_flow['product_name'] == j),
        'value'].values[0] for i, j in
                                                      zip(self.df_flow['process'], self.df_flow['product_name'])})
        self.m = gp.Model('plastics_optimization')
        # Initialize flow variables with a lower bound of 0
        self.flow = self.m.addVars(self.process_name_list, lb=0, name="flow")

        # constraints
        # 1. within supply limit
        raw_material_list = list(self.df_product.loc[self.df_product.product_type == 'raw_material', 'product_name'])
        for x in raw_material_list:
            if x not in self.supply_dict.keys():
                self.supply_dict[x] = 1e10
        for i in self.supply_dict.keys():
            supply_flow = -gp.quicksum(self.flow[p] * self.inflow[p, i] for p in
                                       self.df_flow.loc[self.df_flow.product_name == i, "process"].unique())
            self.m.addConstr(supply_flow <= self.supply_dict[i], name=f"supply_flow_{i}")
        # 2. co2_feedstock > 0
        for i in ['co2_feedstock_fossil', 'co2_feedstock_biogenic_short', 'co2_feedstock_biogenic_long']:
            co2_flow = -gp.quicksum(self.flow[p] * self.inflow[p, i] for p in
                                    self.df_flow.loc[self.df_flow.product_name == i, "process"].unique())
            self.m.addConstr(co2_flow >= 0, name=f"co2_flow_{i}_1")
            if i != 'co2_feedstock_fossil': # biogenic co2 can only be from by-products, no net supply
                self.m.addConstr(co2_flow <= 0, name=f"co2_flow_{i}_2")
            else:
                df_temp = self.df_flow[(self.df_flow.product_type == 'product') &
                                       (self.df_flow.product_name.str.endswith('_co2')) &
                                       (~self.df_flow.product_name.str.contains('_mr'))]
                product_co2_flow = gp.quicksum(self.flow[p] * df_temp.loc[df_temp.process == p, 'value'].sum()
                                             for p in df_temp['process'].unique())

                z = self.m.addVar(vtype=gp.GRB.BINARY, name=f"z_{i}")
                self.m.addConstr(co2_flow >= 0.01 * z, name=f"binding_production_co2_{i}_1")
                self.m.addConstr(product_co2_flow * 0.8 <= co2_flow * 0.2727 * z, name=f"binding_production_co2_{i}_2")

        # 3. meet plastics demand
        for i in self.demand_dict.keys():
            df_temp = self.df_flow.loc[(self.df_flow.product_name.str.startswith(i)) &
                                       (self.df_flow['type'] == 'PRODUCT')]
            demand_flow = gp.quicksum(self.flow[p] for p in df_temp['process'].unique())
            self.m.addConstr(demand_flow == self.demand_dict[i], name=f"demand_flow_{i}_1")
            '''
            if 'hdpe' in i and 'gpps' not in i:
                self.m.addConstr(demand_flow == self.demand_dict[i]*8, name=f"demand_flow_{i}_1")
            else:
                self.m.addConstr(demand_flow == 0, name=f"demand_flow_{i}_1")
            '''
        # 4. no stocks for intermediates
        for i in self.df_product.loc[self.df_product.product_type == 'intermediate', 'product_name']:
            intermediate_flow = gp.quicksum(self.flow[p] * self.inflow[p, i] for p in
                                     self.df_flow.loc[self.df_flow.product_name == i, "process"].unique())
            self.m.addConstr(intermediate_flow == 0, name=f"intermediate_flow_{i}_1")
            #self.m.addConstr(intermediate_flow <= 0, name=f"intermediate_flow_{i}_2")

        # 5. all waste is treated; no production of virgin product and then no waste
        for i in self.df_product.loc[self.df_product.product_type == 'waste', 'product_name']:
            virgin_product = i.replace('_waste', '')
            process_list = self.df_flow[self.df_flow.product_name == i]["process"].unique()
            waste_flow = gp.quicksum(self.flow[p] * self.inflow[p, i]
                                     for p in process_list)
            self.m.addConstr(waste_flow == 0, name=f"waste_flow_{i}_1")
            '''
            #self.m.addConstr(waste_flow <= 0, name=f"waste_flow_{i}_2")
            df_temp1 = self.df_flow[(self.df_flow['product_name'] == virgin_product) &
                                    (self.df_flow['type'] == 'PRODUCT')].copy()
            df_temp2 = self.df_flow[(self.df_flow['product_name'] == i) &
                                    (self.df_flow['type'] == 'WASTE')].copy()
            z = self.m.addVar(vtype=gp.GRB.BINARY, name=f"z_{i}")
            virgin_amount = gp.quicksum(self.flow[p] for p in df_temp1['process'].unique())
            waste_amount = gp.quicksum(self.flow[p] * df_temp2.loc[df_temp2.process == p, 'value'].sum()
                                       for p in df_temp2['process'].unique())
            self.m.addConstr(virgin_amount >= 0.0001 * z, name=f"binding_production_waste_{i}_1")
            self.m.addConstr(waste_amount <= 1e20 * z, name=f"binding_production_waste_{i}_2")
            '''
        # 6. maximum recycling
        polymer_source_list = ['_co2', '_biogenic_short', '_biogenic_long', '_fossil']
        plastics_production_2020 = self.df_plastics_demand['2020'].sum()
        plastics_production_2050 = self.df_plastics_demand['2050'].sum()
        if self.mechanical_recycling:
            ratio_dict_w = waste_to_secondary_plastics_ratio()[1]
            ratio_dict_c = consumption_to_secondary_plastics_ratio()[1]
            for i in ratio_dict_w.keys():
                ratio_w = ratio_dict_w[i]
                ratio_c = ratio_dict_c[i]
                if ratio_w != 0:
                    for suffix in polymer_source_list:
                        df_flow_temp1 = self.df_flow[(self.df_flow['product_name'] == f'{i}_mr{suffix}') &
                                                     (self.df_flow['type'] == 'PRODUCT')].copy()
                        df_flow_temp2 = self.df_flow[(self.df_flow['product_name'] == f'{i}_waste{suffix}') &
                                                     (self.df_flow['type'] == 'WASTE')].copy()
                        df_flow_temp3 = self.df_flow[(self.df_flow['product_name'].str.startswith(f'{i}')) &
                                                     (self.df_flow['product_name'].str.contains(f'{suffix}')) &
                                                     (self.df_flow['type'] == 'PRODUCT') &
                                                     (self.df_flow['product_type'] == 'intermediate')].copy()
                        mr_plastics = gp.quicksum(self.flow[p] for p in df_flow_temp1['process'].unique())
                        all_production = gp.quicksum(self.flow[p] for p in df_flow_temp3['process'].unique())
                        waste = gp.quicksum(self.flow[p] * df_flow_temp2.loc[df_flow_temp2.process == p, 'value'].sum()
                                            for p in df_flow_temp2['process'].unique())
                        self.m.addConstr(waste * ratio_w - mr_plastics >= 0, name=f"mr_flow_w_positive_{i}{suffix}")
                        self.m.addConstr(all_production * ratio_c - mr_plastics >= 0,
                                             name=f"mr_flow_c_positive_{i}{suffix}")

        ratio_dict_w = waste_to_secondary_plastics_ratio()[2]
        ratio_dict_c = consumption_to_secondary_plastics_ratio()[2]

        if self.gasi and not self.pyrolysis:
            for i in ratio_dict_w.keys():
                ratio_w = ratio_dict_w[i]
                ratio_c = ratio_dict_c[i]
                for suffix in polymer_source_list[0:4]:
                    waste = f'{i}_waste{suffix}'
                    df_flow_temp1 = self.df_flow[(self.df_flow['product_name'].str.contains(waste)) &
                                                 (self.df_flow['process'].str.contains('waste gasification'))].copy()
                    df_flow_temp2 = self.df_flow[(self.df_flow['product_name'] == f'{i}_waste{suffix}') &
                                                 (self.df_flow['type'] == 'WASTE')].copy()
                    gasi_plastics = -gp.quicksum(self.flow[p] * df_flow_temp1.loc[df_flow_temp1.process == p, 'value'].sum()
                                                 for p in df_flow_temp1['process'].unique())
                    waste = gp.quicksum(self.flow[p] * df_flow_temp2.loc[df_flow_temp2.process == p, 'value'].sum()
                                        for p in df_flow_temp2['process'].unique())
                    self.m.addConstr(waste * ratio_w - gasi_plastics >= 0, name=f"gasi_flow_w_{i}{suffix}")

        elif self.pyrolysis and not self.gasi:
            for i in ratio_dict_w.keys():
                ratio_w = ratio_dict_w[i]
                ratio_c = ratio_dict_c[i]
                for suffix in polymer_source_list[0:4]:
                    waste = f'{i}_waste{suffix}'
                    df_flow_temp1 = self.df_flow[(self.df_flow['product_name'].str.contains(waste)) &
                                                 (self.df_flow['process'].str.contains('waste pyrolysis'))].copy()
                    df_flow_temp2 = self.df_flow[(self.df_flow['product_name'] == f'{i}_waste{suffix}') &
                                                 (self.df_flow['type'] == 'WASTE')].copy()
                    pyro_plastics = -gp.quicksum(self.flow[p] * df_flow_temp1.loc[df_flow_temp1.process == p, 'value'].sum()
                                                 for p in df_flow_temp1['process'].unique())
                    waste = gp.quicksum(self.flow[p] * df_flow_temp2.loc[df_flow_temp2.process == p, 'value'].sum()
                                        for p in df_flow_temp2['process'].unique())
                    self.m.addConstr(waste * ratio_w - pyro_plastics >= 0, name=f"pyro_flow_w_{i}{suffix}")
        elif self.pyrolysis and self.gasi:
            for i in ratio_dict_w.keys():
                ratio_w = ratio_dict_w[i]
                ratio_c = ratio_dict_c[i]
                for suffix in polymer_source_list[0:4]:
                    waste = f'{i}_waste{suffix}'
                    df_flow_temp1 = self.df_flow[(self.df_flow['product_name'].str.contains(waste)) &
                                                 ((self.df_flow['process'].str.contains('pyrolysis')) |
                                                 (self.df_flow['process'].str.contains('waste gasification')))].copy()
                    df_flow_temp2 = self.df_flow[(self.df_flow['product_name'] == f'{i}_waste{suffix}') &
                                                 (self.df_flow['type'] == 'WASTE')].copy()
                    gasi_pyro_plastics = -gp.quicksum(self.flow[p] * df_flow_temp1.loc[df_flow_temp1.process == p, 'value'].sum()
                                                 for p in df_flow_temp1['process'].unique())
                    waste = gp.quicksum(self.flow[p] * df_flow_temp2.loc[df_flow_temp2.process == p, 'value'].sum()
                                        for p in df_flow_temp2['process'].unique())
                    self.m.addConstr(waste * ratio_w - gasi_pyro_plastics >= 0, name=f"gasi_pyro_flow_w_{i}{suffix}")

        # 7. maximum new bio plastics
        if self.new_bio_plastics:
            for npl in ['pla', 'phb', 'pbs']:
                for op in ['ldpe', 'hdpe', 'pp', 'gpps', 'pvc', 'pet']:
                    df_temp1 = self.df_flow[(self.df_flow['process'].str.contains(f'{npl}')) &
                                            (self.df_flow['process'].str.contains(f'replacing {op}')) &
                                            (self.df_flow['product_name'].str.contains(f'{npl}')) &
                                            (self.df_flow['product_type'] == 'intermediate')].copy()
                    df_temp2 = self.df_flow[(self.df_flow['product_name'].str.contains(f'{op}')) &
                                            (self.df_flow['product_type'] == 'product')].copy()
                    if df_temp1.shape[0] > 0:
                        np_replaced = -gp.quicksum(self.flow[p] * df_temp1.loc[df_temp1.process == p, 'value'].sum()
                                            for p in df_temp1['process'].unique())
                        op_produced = gp.quicksum(self.flow[p] for p in df_temp2['process'].unique())
                        self.m.addConstr(np_replaced - op_produced * max_replacing_rate_dict[f'{npl}_{op}'] <= 0,
                                         name=f"np_replaced_{npl}_{op}")

        # '''
        # 8. MTO ethylene to propylene ratio (0.6-1.3)
        for suffix in polymer_source_list:
            df_temp = self.df_flow[(self.df_flow['process'].str.contains('methanol-to-olefin')) &
                                   (self.df_flow['process'].str.contains(suffix))].copy()
            if df_temp.shape[0] > 0:
                ethylene_process_name = df_temp.loc[df_temp['product_name'].str.contains('ethylene'), 'process'].iloc[0]
                propylene_process_name = df_temp.loc[df_temp['product_name'].str.contains('propylene'), 'process'].iloc[0]
                ethylene = self.flow[ethylene_process_name]
                propylene = self.flow[propylene_process_name]
                self.m.addConstr(ethylene >= 0.6 * propylene, name=f"mto_propylene_{suffix}_1")
                self.m.addConstr(ethylene <= 1.3 * propylene, name=f"mto_propylene_{suffix}_2")
        # '''

        # 9. fossil lock-in
        if self.fossil_lock_in:
            df_temp = self.df_flow[(self.df_flow['process'].str.contains('steam cracking')) &
                                   (self.df_flow['type'] == 'PRODUCT')].copy()
            fossil_product = gp.quicksum(self.flow[p] for p in df_temp['process'].unique())
            self.m.addConstr(fossil_product >= 265, name=f"fossil_lock_in")

    def model_optimization(self, objective, ele_impact=-999):
        self.set_up_optimization_model()
        impact_dict_ghg = self.ghg_dict
        if ele_impact != -999:
            impact_dict_ghg['electricity_non_biomass'] = -ele_impact
        impact_dict_bdv = self.bdv_dict
        impact_dict_health = self.health_dict
        total_impact_ghg = gp.quicksum(self.flow[p] * self.inflow[p, i] * impact_dict_ghg[i]
                                       for i in impact_dict_ghg.keys()
                                       for p in self.df_flow.loc[self.df_flow.product_name == i, "process"].unique())
        total_impact_bdv = gp.quicksum(self.flow[p] * self.inflow[p, i] * impact_dict_bdv[i]
                                       for i in impact_dict_bdv.keys()
                                       for p in self.df_flow.loc[self.df_flow.product_name == i, "process"].unique())
        total_impact_health = gp.quicksum(self.flow[p] * self.inflow[p, i] * impact_dict_health[i]
                                          for i in impact_dict_health.keys()
                                          for p in self.df_flow.loc[self.df_flow.product_name == i, "process"].unique())
        if objective == "GHG":
            total_impact = total_impact_ghg
        elif objective == "health":
            total_impact = total_impact_health
        elif objective == "BDV":
            total_impact = total_impact_bdv
        else:
            ghg_min = 1222
            ghg_max = 6013.404
            bdv_min = 0
            bdv_max = 4852.442
            health_min = 0.000225
            health_max = 0.005224
            total_impact = (total_impact_ghg - ghg_min) / (ghg_max - ghg_min) + \
                            (total_impact_bdv - bdv_min) / (bdv_max - bdv_min) + \
                            (total_impact_health - health_min) / (health_max - health_min)

        self.m.setObjective(total_impact, GRB.MINIMIZE)
        #self.m.setObjective(total_impact, GRB.MAXIMIZE)
        self.m.setParam('NumericFocus', 3)
        self.m.setParam('FeasibilityTol', 1e-09)
        self.m.setParam('OutputFlag', 1)
        self.m.setParam('InfUnbdInfo', 1)
        self.m.Params.DualReductions = 0
        self.m.optimize()

    def model_results(self, objective, ele_impact=-999):
        self.model_optimization(objective, ele_impact)
        m = self.m
        if m.status == GRB.OPTIMAL:
            print("Optimal solution found.")
        elif m.status == GRB.INFEASIBLE:
            print("Model is infeasible.")
        else:
            print(f"Optimization ended with status {self.m.status}")

        if self.fossil_routes:
            report_list = residue_list_code + co2_feedstock_list + ['electricity_non_biomass', 'electricity',
                                                                    'petroleum', 'natural_gas']
        else:
            report_list = residue_list_code + co2_feedstock_list + ['electricity_non_biomass', 'electricity']
        report_value_list = []
        plastic_production_amount = sum(self.demand_dict.values())
        for i in report_list:
            if m.status == GRB.OPTIMAL:
                solution = m.getAttr("x", self.flow)
                supply_amount = 0
                for p in list(self.df_flow.loc[(self.df_flow.product_name == i) &
                                               (self.df_flow['type'] != 'PRODUCT'), "process"].unique()):
                    supply_amount += solution[p] * self.inflow[p, i]
                if i in self.supply_dict.keys():
                    print(f"{i}, {-supply_amount} out of {self.supply_dict[i]}")
                elif i == 'electricity':
                    print(f"Total {i}, {-supply_amount} TWh")
                else:
                    print(f"{i}, {-supply_amount} Mt")
                report_value_list.append(-supply_amount)
            else:
                report_value_list.append(-999)
        report_list_2 = [f'{x}_availability' for x in self.supply_dict.keys()]
        for i in [x for x in self.supply_dict.keys()]:
            report_value_list.append(self.supply_dict[i])
        report_list = report_list + report_list_2
        report_list.append('plastic_production')
        report_value_list.append(plastic_production_amount)
        if m.status == GRB.OPTIMAL:
            solution = m.getAttr("x", self.flow)
            df_result = pd.DataFrame.from_dict(solution, orient='index').reset_index()
            df_result.columns = ['process', 'flow_amount']
            ghg_total = 0
            bdv_total = 0
            health_total = 0
            for i in self.ghg_dict.keys():
                for p in list(self.df_flow.loc[self.df_flow.product_name == i, "process"].unique()):
                    ghg_total += df_result.loc[df_result.process == p, "flow_amount"].iloc[0] * self.inflow[p, i] * \
                                 self.ghg_dict[i]
                    bdv_total += df_result.loc[df_result.process == p, "flow_amount"].iloc[0] * self.inflow[p, i] * \
                                 self.bdv_dict[i]
                    health_total += df_result.loc[df_result.process == p, "flow_amount"].iloc[0] * self.inflow[p, i] * \
                                    self.health_dict[i]
            print(f"GHG: {ghg_total}")
            print(f"Biodiversity: {bdv_total}")
            print(f"Health: {health_total} *1e9 DALY")
            report_value_list.append(ghg_total)
            report_value_list.append(bdv_total)
            report_value_list.append(health_total)
            report_list.append('ghg')
            report_list.append('bdv')
            report_list.append('health')
            # df_result.loc[df_result.flow_amount < 0.000001, "flow_amount"] = 0.000001
            df_flow_result = pd.merge(self.df_flow, df_result, on='process', how='left')
            df_flow_result['flowxvalue'] = df_flow_result['flow_amount'] * df_flow_result['value']
            df_flow_result['ghg'] = df_flow_result["product_name"].map(self.ghg_dict)
            df_flow_result['bdv'] = df_flow_result["product_name"].map(self.bdv_dict)
            df_flow_result['health'] = df_flow_result["product_name"].map(self.health_dict)
            plastic_product_list = list(self.df_product.loc[self.df_product.product_type == 'product', 'product_name'])
            print(f"Plastic production: {plastic_production_amount} Mt")
            df_product = df_flow_result[df_flow_result['type'] == 'PRODUCT'].copy()
            plastics_mr = 0
            plastics_bs = 0
            plastics_bl = 0
            plastics_co2 = 0
            plastics_fossil = 0
            for x in final_product_list:
                plastics_mr += df_product.loc[df_product.product_name.isin([f'{x}_mr_biogenic_short',
                                                                            f'{x}_mr_biogenic_long',
                                                                            f'{x}_mr_co2',
                                                                            f'{x}_mr_fossil']), 'flow_amount'].sum()
                plastics_bs += df_product.loc[df_product.product_name == f'{x}_biogenic_short', 'flow_amount'].sum()
                plastics_bl += df_product.loc[df_product.product_name == f'{x}_biogenic_long', 'flow_amount'].sum()
                plastics_co2 += df_product.loc[df_product.product_name == f'{x}_co2', 'flow_amount'].sum()
                plastics_fossil += df_product.loc[df_product.product_name == f'{x}_fossil', 'flow_amount'].sum()
            print(f"Plastics mechanical recycling: {plastics_mr} Mt")
            print(f"Plastics biogenic short: {plastics_bs} Mt")
            print(f"Plastics biogenic long: {plastics_bl} Mt")
            print(f"Plastics co2-based: {plastics_co2} Mt")
            print(f"Plastics fossil-based: {plastics_fossil} Mt")
            report_value_list.append(plastics_mr)
            report_value_list.append(plastics_bs)
            report_value_list.append(plastics_bl)
            report_value_list.append(plastics_co2)
            report_value_list.append(plastics_fossil)
            report_list.append('plastics_mr')
            report_list.append('plastics_bs')
            report_list.append('plastics_bl')
            report_list.append('plastics_co2')
            report_list.append('plastics_fossil')
            heat = df_product.loc[df_product.product_name == 'heat_high', 'flow_amount'].sum()
            print(f"Total heat: {heat} PJ")
            report_value_list.append(heat)
            report_list.append('total_heat')
            # plastic types
            for p in ['pla', 'pef', 'pbs', 'phb', 'pbat',
                      'hdpe', 'ldpe', 'pp', 'gpps', 'hips', 'pvc', 'pet', 'pur_flexible', 'pur_rigid']:
                amount_tot = df_product.loc[df_product.product_name.str.startswith(p), 'flow_amount'].sum()
                print(f"{p}: {amount_tot} Mt")
                report_value_list.append(amount_tot)
                report_list.append(f'{p}_total')
                for suffix in ['_co2', '_biogenic_short', '_biogenic_long', '_fossil', '_mr']:
                    product_name = f'{p}{suffix}'
                    if product_name in df_product.product_name.values:
                        amount = df_product.loc[df_product.product_name == product_name, 'flow_amount'].sum()
                        print(f"{product_name}: {amount} Mt")
                        report_value_list.append(amount)
                        report_list.append(product_name)
            # co2 emissions
            for c in ['co2_emission_fossil', 'co2_emission_biogenic_short', 'co2_emission_biogenic_long']:
                amount = df_flow_result.loc[df_flow_result.product_name == c, 'flowxvalue'].sum()
                print(f"{c}: {amount} Mt")
                report_value_list.append(amount)
                report_list.append(c)
            for b in ['agricultural_residue', 'forest_residue']:
                for p in ['ethanol', 'glucose', 'lactic acid', 'methanol', 'syngas', 'heat', 'electricity']:
                    amount = -df_flow_result.loc[(df_flow_result.product_name == b) &
                                                 (df_flow_result.process.str.startswith(p)), 'flowxvalue'].sum()
                    print(f"{b}_to_{p}: {amount} Mt")
                    report_value_list.append(amount)
                    report_list.append(f'{b}_to_{p}')
            # waste
            waste_mr = -df_flow_result.loc[(df_flow_result.product_name.str.contains('waste')) &
                                           (df_flow_result.process.str.contains('mechanical recycling')),
            'flowxvalue'].sum()
            waste_gasi = -df_flow_result.loc[(df_flow_result.product_name.str.contains('waste')) &
                                             (df_flow_result.process.str.contains('waste gasification')),
            'flowxvalue'].sum()
            waste_incineration = -df_flow_result.loc[(df_flow_result.product_name.str.contains('waste')) &
                                                     (df_flow_result.process.str.contains('incineration')),
            'flowxvalue'].sum()
            print(f"Waste mechanical recycling: {waste_mr} Mt")
            print(f"Waste gasification: {waste_gasi} Mt")
            print(f"Waste incineration: {waste_incineration} Mt")
            report_value_list.append(waste_mr)
            report_value_list.append(waste_gasi)
            report_value_list.append(waste_incineration)
            report_list.append('waste_to_mr')
            report_list.append('waste_to_gasi')
            report_list.append('waste_to_incineration')
            # methanol
            df_temp = df_flow_result.loc[(df_flow_result.product_name.str.contains('methanol')) &
                                         (df_flow_result.type == 'PRODUCT')].copy()
            df_temp1 = df_temp.loc[~df_temp.process.str.contains('waste gasification')]
            for i in df_temp1['process'].unique():
                amount = df_temp1.loc[df_temp1.process == i, 'flowxvalue'].sum()
                print(f"{i}: {amount} Mt")
                report_value_list.append(amount)
                report_list.append(i)
            df_temp2 = df_temp.loc[df_temp.process.str.contains('waste gasification')]
            amount = df_temp2['flowxvalue'].sum()
            print(f"methanol, from plastic waste gasification: {amount} Mt")
            report_value_list.append(amount)
            report_list.append('methanol_from_waste_gasi')

        else:
            report_value_list.extend([-999] * 23)
        df = pd.DataFrame({'product': report_list, 'value': report_value_list})
        df.set_index('product', inplace=True)
        df1 = df.T.reset_index(drop=True)
        df1['electricity_ghg'] = - \
            self.df_product.loc[self.df_product.product_name == 'electricity_non_biomass', 'GHG'].values[0]
        df1['electricity_bdv'] = - \
            self.df_product.loc[self.df_product.product_name == 'electricity_non_biomass', 'Biodiversity'].values[0]

        if m.status == GRB.OPTIMAL:
            df2 = df_flow_result[df_flow_result.flow_amount > 0.01]
            df_flow_result.to_csv(f'data/processed/flow_result_ele_{self.ele_impact}.csv')
            # df1 = df_flow_result[df_flow_result.flow_amount > 0.01]
            df_process = self.df_process[['product_process', 'product_name']]
            df_result_0 = df_result.copy()
            df_result = pd.merge(df_result, df_process, left_on='process', right_on='product_process', how='left')
            df_product2 = self.df_product[['product_name', 'unit']].copy()
            df_result = pd.merge(df_result, df_product2, on='product_name', how='left')
            df_result['unit'] = df_result['unit'].fillna('kg')
            df_result.loc[df_result.unit == 'MJ', 'unit'] = 'PJ'
            df_result.loc[df_result.unit == 'kWh', 'unit'] = 'TWh'
            df_result.loc[df_result.unit == 'kg', 'unit'] = 'Mt'
            df_result = df_result[['process', 'product_name', 'flow_amount', 'unit']].copy()
            df_result.rename(columns={'flow_amount': 'value'}, inplace=True)
            mask = df_result.process.str.contains('waste incineration')
            df_result.loc[mask, 'product_name'] = df_result.loc[mask, 'process'].str.split('from ').str[1]
            return df1, df_flow_result, df_result
        else:
            return df1, None

    def model_results_multi_objective(self):
        self.set_up_optimization_model()
        m = self.m
        impact_dict_ghg = self.ghg_dict
        impact_dict_bdv = self.bdv_dict
        impact_dict_health = self.health_dict
        total_impact_ghg = gp.quicksum(self.flow[p] * self.inflow[p, i] * impact_dict_ghg[i]
                                       for i in impact_dict_ghg.keys()
                                       for p in self.df_flow.loc[self.df_flow.product_name == i, "process"].unique())
        total_impact_bdv = gp.quicksum(self.flow[p] * self.inflow[p, i] * impact_dict_bdv[i]
                                       for i in impact_dict_bdv.keys()
                                       for p in self.df_flow.loc[self.df_flow.product_name == i, "process"].unique())
        total_impact_health = gp.quicksum(self.flow[p] * self.inflow[p, i] * impact_dict_health[i]
                                          for i in impact_dict_health.keys()
                                          for p in self.df_flow.loc[self.df_flow.product_name == i, "process"].unique())
        ghg_min = -263.3
        ghg_max = 3683.4
        bdv_min = 0
        bdv_max = 4234
        ghg_norm = (total_impact_ghg - ghg_min) / (ghg_max - ghg_min)
        bdv_norm = (total_impact_bdv - bdv_min) / (bdv_max - bdv_min)
        pareto_solutions = []
        health_epsilons = [0.005, 0.0006, 0.0005, 0.0004, 0.0003]
        health_epsilons = [1]
        weights = np.linspace(0, 1, 100)
        for health_eps in health_epsilons:
            health_constraint = m.addConstr(total_impact_health <= health_eps, f"HealthConstraint_{health_eps}")
            for weight in weights:
                #m.reset()
                #m.remove(health_constraint)
                #health_constraint = m.addConstr(health_norm <= health_eps, "HealthConstraint")
                m.setObjective(weight * ghg_norm + (1 - weight) * bdv_norm, GRB.MINIMIZE)
                m.update()
                m.optimize()
                if m.status == GRB.OPTIMAL:
                    solution = {
                        'GHG': total_impact_ghg.getValue(),
                        'BDV': total_impact_bdv.getValue(),
                        'Health': total_impact_health.getValue(),
                        'Health Epsilon': health_eps,
                        'GHG weight': weight
                    }
                    pareto_solutions.append(solution)
                    solution2 = m.getAttr("x", self.flow)
                    df_result2 = pd.DataFrame.from_dict(solution2, orient='index').reset_index()
        df_result = pd.DataFrame(pareto_solutions)
        return df_result

    def calculate_carbon_flow(self, df_temp):
        ans = gp.quicksum(self.flow[p] * df_temp.loc[df_temp.process == p, 'value'].values[0] *
                                       df_temp.loc[df_temp.process == p, 'carbon_content'].values[0]
                                       for p in df_temp['process'].unique())
        return ans

    def sensitivity_demand_ele_biomass(self, ele_impact_list, demand_list, biomass_availability_list):
        start_time = time.time()
        self.set_up_optimization_model()
        m = self.m
        impact_dict_ghg = self.ghg_dict
        impact_dict_bdv = self.bdv_dict
        impact_dict_health = self.health_dict
        df_list = []
        total_impact_bdv = gp.quicksum(self.flow[p] * self.inflow[p, i] * impact_dict_bdv[i]
                                       for i in impact_dict_bdv.keys()
                                       for p in self.df_flow.loc[self.df_flow.product_name == i, "process"].unique())
        total_impact_health = gp.quicksum(self.flow[p] * self.inflow[p, i] * impact_dict_health[i]
                                          for i in impact_dict_health.keys()
                                          for p in self.df_flow.loc[self.df_flow.product_name == i, "process"].unique())
        df_temp = self.df_flow.loc[(self.df_flow.product_name == 'electricity_non_biomass')]
        total_ele_non_biomass = gp.quicksum(self.flow[p] for p in df_temp['process'].unique())
        df_temp = self.df_flow.loc[(self.df_flow.product_name.str.contains('electricity_biogenic')) &
                                   (self.df_flow.type == 'PRODUCT')]
        total_ele_biomass = gp.quicksum(self.flow[p] for p in df_temp['process'].unique())

        df_ele = self.df_flow.loc[(self.df_flow.product_name == 'electricity') & (self.df_flow.type == 'UTILITIES')]
        ele_use_total = -gp.quicksum(self.flow[p] * df_ele.loc[df_ele.process == p, 'value'].values[0]
                                 for p in df_ele['process'].unique())
        df_temp = df_ele.loc[df_ele.process.str.contains('hydrogen,')]
        ele_use_h2 = -gp.quicksum(self.flow[p] * df_temp.loc[df_temp.process == p, 'value'].values[0]
                                 for p in df_temp['process'].unique())
        df_temp = df_ele.loc[(df_ele.process.str.startswith('benzene,')) |
                                          (df_ele.process.str.startswith('ethylene,')) |
                                          (df_ele.process.str.startswith('propylene,')) |
                                          (df_ele.process.str.startswith('methanol,')) |
                                          (df_ele.process.str.startswith('toluene,')) |
                                          (df_ele.process.str.startswith('p-xylene,'))]
        ele_use_base_chemical = -gp.quicksum(self.flow[p] * df_temp.loc[df_temp.process == p, 'value'].values[0]
                                 for p in df_temp['process'].unique())
        df_temp = df_ele.loc[(df_ele.process.str.startswith('gpps,')) |
                                           (df_ele.process.str.startswith('hdpe,')) |
                                           (df_ele.process.str.startswith('hips,')) |
                                           (df_ele.process.str.startswith('ldpe,')) |
                                           (df_ele.process.str.startswith('pet,')) |
                                           (df_ele.process.str.startswith('pbs,')) |
                                           (df_ele.process.str.startswith('pp,')) |
                                           (df_ele.process.str.startswith('pvc')) |
                                           (df_ele.process.str.startswith('pur_,')) |
                                           (df_ele.process.str.startswith('pla,'))]
        ele_use_polymerization = -gp.quicksum(self.flow[p] * df_temp.loc[df_temp.process == p, 'value'].values[0]
                                 for p in df_temp['process'].unique())
        df_temp = df_ele.loc[df_ele.process.str.contains('mechanical recycling')]
        ele_use_mr = -gp.quicksum(self.flow[p] * df_temp.loc[df_temp.process == p, 'value'].values[0]
                                 for p in df_temp['process'].unique())
        df_temp = self.df_flow.loc[(self.df_flow.product_type == 'product') &
                                   (self.df_flow.product_name.str.contains('biogenic_short')) &
                                   (~self.df_flow.product_name.str.contains('_mr'))]
        plastics_bio_virgin = gp.quicksum(self.flow[p] for p in df_temp['process'].unique())
        c_biomass_to_plastics = gp.quicksum(self.flow[p] * df_temp.loc[df_temp.process == p, 'carbon_content'].values[0]
                                      for p in df_temp['process'].unique())
        df_temp = self.df_flow.loc[(self.df_flow.product_type == 'product') &
                                   (self.df_flow.product_name.str.contains('biogenic')) &
                                   (self.df_flow.product_name.str.contains('_mr'))]
        plastics_bio_mr = gp.quicksum(self.flow[p] for p in df_temp['process'].unique())
        df_temp = self.df_flow.loc[(self.df_flow.product_type == 'product') &
                                   (self.df_flow.product_name.str.contains('_co2')) &
                                   (~self.df_flow.product_name.str.contains('_mr'))]
        plastics_co2_virgin = gp.quicksum(self.flow[p] for p in df_temp['process'].unique())
        df_temp = self.df_flow.loc[(self.df_flow.product_type == 'product') &
                                   (self.df_flow.product_name.str.contains('_co2')) &
                                   (self.df_flow.product_name.str.contains('_mr'))]
        plastics_co2_mr = gp.quicksum(self.flow[p] for p in df_temp['process'].unique())
        df_temp = self.df_flow.loc[(self.df_flow.product_type == 'product') &
                                   (self.df_flow.product_name.str.contains('_fossil')) &
                                   (~self.df_flow.product_name.str.contains('_mr'))]
        plastics_fossil_virgin = gp.quicksum(self.flow[p] for p in df_temp['process'].unique())
        df_temp = self.df_flow.loc[(self.df_flow.product_type == 'product') &
                                   (self.df_flow.product_name.str.contains('_fossil')) &
                                   (self.df_flow.product_name.str.contains('_mr'))]
        plastics_fossil_mr = gp.quicksum(self.flow[p] for p in df_temp['process'].unique())
        df_temp = self.df_flow.loc[(self.df_flow.product_name.str.contains('agricultural_residue')) &
                                   (self.df_flow.type == 'PRODUCT')]
        agricultural_residue = gp.quicksum(self.flow[p] for p in df_temp['process'].unique())
        df_temp = self.df_flow.loc[(self.df_flow.product_name.str.contains('forest_residue'))]
        forest_residue = -gp.quicksum(self.flow[p] * df_temp.loc[df_temp.process == p, 'value'].values[0]
                                      for p in df_temp['process'].unique())
        df_temp = self.df_flow.loc[(self.df_flow.product_name =='natural_gas')]
        natural_gas = -gp.quicksum(self.flow[p] * df_temp.loc[df_temp.process == p, 'value'].values[0]
                                   for p in df_temp['process'].unique())
        df_temp = self.df_flow.loc[(self.df_flow.product_name == 'petroleum')]
        petroleum = -gp.quicksum(self.flow[p] * df_temp.loc[df_temp.process == p, 'value'].values[0]
                                 for p in df_temp['process'].unique())
        df_temp = self.df_flow.loc[(self.df_flow.product_name == 'co2_feedstock_fossil')]
        co2_source_point = -gp.quicksum(self.flow[p] * df_temp.loc[df_temp.process == p, 'value'].values[0]
                                             for p in df_temp['process'].unique())
        df_temp = self.df_flow.loc[(self.df_flow.product_name.str.contains('co2_feedstock')) &
                                   (self.df_flow.process.str.contains('co2 hydrogenation'))]
        co2_destination_methanol = -gp.quicksum(self.flow[p] * df_temp.loc[df_temp.process == p, 'value'].values[0]
                                 for p in df_temp['process'].unique())
        df_temp = self.df_flow.loc[(self.df_flow.product_name.str.contains('co2_feedstock')) &
                                   (self.df_flow.process.str.contains('carbon_monoxide, from co2'))]
        co2_destination_co = -gp.quicksum(self.flow[p] * df_temp.loc[df_temp.process == p, 'value'].values[0]
                                 for p in df_temp['process'].unique())
        df_temp = self.df_flow.loc[(self.df_flow.product_name.str.contains('co2_feedstock')) &
                                               (self.df_flow.process.str.contains('CCS'))]
        co2_destination_ccs = -gp.quicksum(self.flow[p] * df_temp.loc[df_temp.process == p, 'value'].values[0]
                                 for p in df_temp['process'].unique())
        df_temp = self.df_flow.loc[(self.df_flow.product_name.str.contains('co2_feedstock')) &
                                   (self.df_flow.type == 'BY-PRODUCT CREDITS')]
        co2_source_byproduct = gp.quicksum(self.flow[p] * df_temp.loc[df_temp.process == p, 'value'].values[0]
                                 for p in df_temp['process'].unique())
        df_temp = self.df_flow.loc[(self.df_flow.process.str.contains('mechanical recycling')) &
                                   (self.df_flow.product_type == 'waste')]
        waste_mr = gp.quicksum(self.flow[p] for p in df_temp['process'].unique())
        df_temp = self.df_flow.loc[(self.df_flow.process.str.contains('gasification')) &
                                   (self.df_flow.product_type == 'waste')]
        waste_gasi = -gp.quicksum(self.flow[p] * df_temp.loc[df_temp.process == p, 'value'].values[0]
                                 for p in df_temp['process'].unique())
        df_temp = self.df_flow.loc[(self.df_flow.process.str.contains('heat, high')) &
                                   (self.df_flow.type == 'RAW MATERIALS')]
        c_biomass_to_heat = -self.calculate_carbon_flow(df_temp)
        df_temp = self.df_flow.loc[(self.df_flow.process.str.contains('electricity_biogenic')) &
                                   (self.df_flow.product_name.str.contains('residue'))]
        c_biomass_to_ele = -self.calculate_carbon_flow(df_temp)
        df_temp = self.df_flow.loc[(self.df_flow.product_name.str.contains('co2_storage_biogenic'))]
        c_biomass_to_ccs = self.calculate_carbon_flow(df_temp)
        # calculate the percentage of bio-based plastics coming from plastics waste gasification
        df_temp = self.df_flow.loc[(self.df_flow.product_name.str.contains('co2_feedstock_biogenic')) &
                                   (self.df_flow.type.str.contains('BY-PRODUCT'))]
        co2_feedstock = self.calculate_carbon_flow(df_temp)
        df_temp = df_temp.loc[df_temp['process'].str.contains('waste gasification')]
        co2_feedstock_waste = self.calculate_carbon_flow(df_temp)
        df_temp = self.df_flow.loc[(self.df_flow.product_name.str.contains('methanol_biogenic')) &
                                   (self.df_flow.type == 'PRODUCT')]
        methanol = self.calculate_carbon_flow(df_temp)
        df_temp1 = df_temp.loc[df_temp.process.str.contains('waste gasi')]
        methanol_waste_direct = self.calculate_carbon_flow(df_temp1)
        df_temp2 = df_temp.loc[df_temp.process.str.contains('co2 hydrogenation')]
        methanol_co2 = self.calculate_carbon_flow(df_temp2)
        base_chemical_list = ['ethylene', 'propylene', 'benzene', 'toluene', 'p-xylene']
        suffix_list = ['_biogenic_short', '_biogenic_long']
        base_chemical_list = [x + y for x in base_chemical_list for y in suffix_list]
        df_temp = self.df_flow.loc[(self.df_flow.product_name.isin(base_chemical_list)) &
                                   (self.df_flow.type == 'PRODUCT')]
        base_chemical = self.calculate_carbon_flow(df_temp)
        df_temp1 = df_temp.loc[df_temp.process.str.contains('methanol-to')]
        base_chemical_methanol = self.calculate_carbon_flow(df_temp1)
        df_temp = self.df_flow.loc[(self.df_flow.product_name.str.contains('biogenic')) &
                                   (~self.df_flow.product_name.str.contains('mr')) &
                                   (self.df_flow.product_type == 'product')]
        plastic = self.calculate_carbon_flow(df_temp)
        df_temp1 = df_temp.loc[~df_temp.process.str.contains('replacing')]
        plastic_conventional = self.calculate_carbon_flow(df_temp1)
        for ele_impact in ele_impact_list:
            if ele_impact != -999:
                impact_dict_ghg['electricity_non_biomass'] = -ele_impact
            total_impact_ghg = gp.quicksum(self.flow[p] * self.inflow[p, i] * impact_dict_ghg[i]
                                           for i in impact_dict_ghg.keys()
                                           for p in
                                           self.df_flow.loc[self.df_flow.product_name == i, "process"].unique())
            m.setObjective(total_impact_ghg, GRB.MINIMIZE)
            m.update()
            for demand_ratio in demand_list:
                for i in self.demand_dict.keys():
                    df_temp = self.df_flow.loc[(self.df_flow.product_name.str.contains(i)) &
                                               (self.df_flow['type'] == 'PRODUCT')]
                    demand_flow = gp.quicksum(self.flow[p] for p in df_temp['process'].unique())
                    m.remove(m.getConstrByName(f"demand_flow_{i}_1"))
                    m.addConstr(demand_flow == self.demand_dict[i] * demand_ratio, name=f"demand_flow_{i}_1")
                for biomass_ratio in biomass_availability_list:
                    for i in residue_list_code:
                        supply_flow = -gp.quicksum(self.flow[p] * self.inflow[p, i] for p in
                                                   self.df_flow.loc[self.df_flow.product_name == i, "process"].unique())
                        m.remove(m.getConstrByName(f"supply_flow_{i}"))
                        m.addConstr(supply_flow <= self.supply_dict[i] * biomass_ratio, name=f"supply_flow_{i}")
                    m.optimize()
                    if m.status == GRB.OPTIMAL:
                        if biomass_ratio == 0:
                            bio_plastics_from_waste_percentage = 0
                            plastics_bio_cr = 0
                        else:
                            if co2_feedstock.getValue() == 0:
                                co2_feedstock_waste_percentage = 0
                            else:
                                co2_feedstock_waste_percentage = co2_feedstock_waste.getValue() / co2_feedstock.getValue()
                            if methanol.getValue() == 0:
                                methanol_waste_percentage = 0
                                base_chemical_waste_percentage = 0
                            else:
                                if methanol.getValue() == 0:
                                    methanol_waste_percentage = 0
                                else:
                                    methanol_waste_percentage = (methanol_waste_direct.getValue() + methanol_co2.getValue()
                                                             * co2_feedstock_waste_percentage) / methanol.getValue()
                                if base_chemical.getValue() == 0:
                                    base_chemical_waste_percentage = 0
                                else:
                                    base_chemical_waste_percentage = base_chemical_methanol.getValue() / base_chemical.getValue() * \
                                                                 methanol_waste_percentage
                            if plastic.getValue() == 0:
                                bio_plastics_from_waste_percentage = 0
                            else:
                                bio_plastics_from_waste_percentage = plastic_conventional.getValue() / plastic.getValue() * \
                                                                     base_chemical_waste_percentage
                            plastics_bio_cr = plastics_bio_virgin.getValue() * bio_plastics_from_waste_percentage
                        solution = {
                            'GHG': total_impact_ghg.getValue(),
                            'BDV': total_impact_bdv.getValue(),
                            'Health': total_impact_health.getValue(),
                            'ele_non_biomass': total_ele_non_biomass.getValue(),
                            'ele_biomass': total_ele_biomass.getValue(),
                            'ele_use_total': ele_use_total.getValue(),
                            'ele_use_h2': ele_use_h2.getValue(),
                            'ele_use_base_chemical': ele_use_base_chemical.getValue(),
                            'ele_use_polymerization': ele_use_polymerization.getValue(),
                            'ele_use_mr': ele_use_mr.getValue(),
                            'ele_use_other': ele_use_total.getValue() - ele_use_h2.getValue() -
                                             ele_use_base_chemical.getValue() - ele_use_polymerization.getValue() -
                                             ele_use_mr.getValue(),
                            'plastics_bio_virgin': plastics_bio_virgin.getValue(),
                            'plastics_bio_mr': plastics_bio_mr.getValue(),
                            'plastics_co2_virgin': plastics_co2_virgin.getValue(),
                            'plastics_co2_mr': plastics_co2_mr.getValue(),
                            'plastics_fossil_virgin': plastics_fossil_virgin.getValue(),
                            'plastics_fossil_mr': plastics_fossil_mr.getValue(),
                            'agricultural_residue': agricultural_residue.getValue(),
                            'forest_residue': forest_residue.getValue(),
                            'natural_gas': natural_gas.getValue(),
                            'petroleum': petroleum.getValue(),
                            'co2_source_point': co2_source_point.getValue(),
                            'co2_source_byproduct': co2_source_byproduct.getValue(),
                            'co2_destination_methanol': co2_destination_methanol.getValue(),
                            'co2_destination_co': co2_destination_co.getValue(),
                            'co2_destination_ccs': co2_destination_ccs.getValue(),
                            'waste_mr': waste_mr.getValue(),
                            'waste_gasi': waste_gasi.getValue(),
                            'c_biomass_to_plastics': c_biomass_to_plastics.getValue() * (
                                        1 - bio_plastics_from_waste_percentage),
                            'c_biomass_to_ccs': c_biomass_to_ccs.getValue(),
                            'c_biomass_to_heat': c_biomass_to_heat.getValue(),
                            'c_biomass_to_electricity': c_biomass_to_ele.getValue(),
                            'c_biomass_in': agricultural_residue.getValue() * 0.494 + forest_residue.getValue() * 0.521,
                            'demand_ratio': demand_ratio,
                            'ele_impact': ele_impact,
                            'biomass_ratio': biomass_ratio
                        }
                        df_list.append(solution)
                    else:
                        a=0
        df_result = pd.DataFrame(df_list)

        df_result['c_biomass_loss'] = df_result['c_biomass_in'] - df_result['c_biomass_to_plastics'] - \
                                        df_result['c_biomass_to_ccs'] - df_result['c_biomass_to_heat'] - \
                                        df_result['c_biomass_to_electricity']

        print("--- %s seconds ---" % (time.time() - start_time))
        df = pd.pivot_table(df_result, values='GHG', index='ele_impact', columns='biomass_ratio')
        return df_result

    def sensitivity_biogenic_carbon_impact(self):
        df_list = []
        for scenario in ['default', 'zero']:
            self.set_up_optimization_model()
            m = self.m
            impact_dict_ghg = self.ghg_dict
            impact_dict_bdv = self.bdv_dict
            impact_dict_health = self.health_dict
            if scenario == 'zero':
                impact_dict_ghg['co2_emission_biogenic_long'] = 0.0001
            total_impact_ghg = gp.quicksum(self.flow[p] * self.inflow[p, i] * impact_dict_ghg[i]
                                           for i in impact_dict_ghg.keys()
                                           for p in
                                           self.df_flow.loc[self.df_flow.product_name == i, "process"].unique())
            total_impact_bdv = gp.quicksum(self.flow[p] * self.inflow[p, i] * impact_dict_bdv[i]
                                             for i in impact_dict_bdv.keys()
                                             for p in self.df_flow.loc[self.df_flow.product_name == i, "process"].unique())
            total_impact_health = gp.quicksum(self.flow[p] * self.inflow[p, i] * impact_dict_health[i]
                                              for i in impact_dict_health.keys()
                                              for p in self.df_flow.loc[self.df_flow.product_name == i, "process"].unique())
            df_temp = self.df_flow.loc[(self.df_flow.product_name.str.contains('agricultural_residue')) &
                                       (self.df_flow.type == 'PRODUCT')]
            agricultural_residue = gp.quicksum(self.flow[p] for p in df_temp['process'].unique())
            df_temp = self.df_flow.loc[(self.df_flow.product_name.str.contains('forest_residue'))]
            forest_residue = -gp.quicksum(self.flow[p] * df_temp.loc[df_temp.process == p, 'value'].values[0]
                                          for p in df_temp['process'].unique())
            df_temp = self.df_flow.loc[(self.df_flow.product_name == 'natural_gas')]
            natural_gas = -gp.quicksum(self.flow[p] * df_temp.loc[df_temp.process == p, 'value'].values[0]
                                       for p in df_temp['process'].unique())
            df_temp = self.df_flow.loc[(self.df_flow.product_name == 'petroleum')]
            petroleum = -gp.quicksum(self.flow[p] * df_temp.loc[df_temp.process == p, 'value'].values[0]
                                     for p in df_temp['process'].unique())
            df_temp = self.df_flow.loc[(self.df_flow.product_name == 'co2_feedstock_fossil')]
            co2_source_point = -gp.quicksum(self.flow[p] * df_temp.loc[df_temp.process == p, 'value'].values[0]
                                            for p in df_temp['process'].unique())
            m.setObjective(total_impact_ghg, GRB.MINIMIZE)
            m.update()
            m.optimize()
            if m.status == GRB.OPTIMAL:
                solution = m.getAttr("x", self.flow)
                if scenario == 'default':
                    df_result_default = pd.DataFrame.from_dict(solution, orient='index').reset_index()
                    df_result_default.columns = ['process', 'flow_amount']
                    df_flow_default = pd.merge(self.df_flow, df_result_default, on='process', how='left')
                    df_flow_default['ghg_factor'] = df_flow_default['product_name'].map(impact_dict_ghg)
                    df_flow_default['ghg_flow_default'] = df_flow_default['value'] * df_flow_default['flow_amount'] * df_flow_default['ghg_factor']
                else:
                    df_result_zero = pd.DataFrame.from_dict(solution, orient='index').reset_index()
                    df_result_zero.columns = ['process', 'flow_amount']
                    df_flow_zero = pd.merge(self.df_flow, df_result_zero, on='process', how='left')
                    df_flow_zero['ghg_factor_zero'] = df_flow_zero['product_name'].map(impact_dict_ghg)
                    df_flow_zero['ghg_flow_zero'] = df_flow_zero['value'] * df_flow_zero['flow_amount'] * df_flow_zero['ghg_factor_zero']
                    df_flow_zero = df_flow_zero[['process', 'product_name', 'ghg_factor_zero', 'ghg_flow_zero']]
                solution_dict = {'GHG': total_impact_ghg.getValue(),
                                 'BDV': total_impact_bdv.getValue(),
                                 'Health': total_impact_health.getValue(),
                                 'agricultural_residue': agricultural_residue.getValue(),
                                 'forest_residue': forest_residue.getValue(),
                                 'natural_gas': natural_gas.getValue(),
                                 'petroleum': petroleum.getValue(),
                                 'co2_source_point': co2_source_point.getValue(),
                                 'scenario': scenario}
                df_list.append(solution_dict)
        df_result = pd.DataFrame(df_list)
        df_flow = pd.merge(df_flow_default, df_flow_zero, on=['process', 'product_name'], how='left')
        return df_flow, df_result

    def calculate_product_impacts(self, objective, ele_impact=-999):
        df_flow_result = self.model_results(objective, ele_impact)[1]
        df_flow_result.loc[df_flow_result.flow_amount == 0, 'flow_amount'] = 1e-6
        if df_flow_result is None:
            print("No solution found.")
            return None
        else:
            df_flow_result['cc_product'] = abs(df_flow_result["product_name"].map(self.ghg_dict))
            df_flow_result['cc_process'] = abs(df_flow_result["product_name"].map(self.ghg_dict))
            df_flow_result['bdv_product'] = abs(df_flow_result["product_name"].map(self.bdv_dict))
            df_flow_result['bdv_process'] = abs(df_flow_result["product_name"].map(self.bdv_dict))
            df_flow_result['health_product'] = abs(df_flow_result["product_name"].map(self.health_dict))
            df_flow_result['health_process'] = abs(df_flow_result["product_name"].map(self.health_dict))
            df_flow_result['sequence'] = 0
            sequence = 0
            # while df_flow_result[df_flow_result.cc_product.isna()].shape[0] > 0:
            while sequence < 15:
                sequence += 1
                for process in list(df_flow_result.process.unique()):
                    df_temp = df_flow_result[df_flow_result.process == process].copy()
                    df_temp2 = df_temp[df_temp.cc_product.isna()].copy()
                    df_temp3 = df_temp[df_temp.cc_product.notna()].copy()
                    if df_temp2.shape[0] == 1:
                        flow = df_temp2['value'].values[0]
                        df_temp3['flowximpact'] = df_temp3['value'] * df_temp3['cc_product']
                        df_temp3['flowximpact2'] = df_temp3['value'] * df_temp3['bdv_product']
                        df_temp3['flowximpact3'] = df_temp3['value'] * df_temp3['health_product']
                        impact1a = abs(df_temp3.loc[df_temp3['type'] != 'BY-PRODUCT CREDITS',
                        'flowximpact']).sum() / flow
                        impact1b = abs(df_temp3.loc[df_temp3['type'] == 'BY-PRODUCT CREDITS',
                        'flowximpact']).sum() / flow
                        impact1 = impact1a - impact1b
                        impact2a = abs(df_temp3.loc[df_temp3['type'] != 'BY-PRODUCT CREDITS',
                        'flowximpact2']).sum() / flow
                        impact2b = abs(df_temp3.loc[df_temp3['type'] == 'BY-PRODUCT CREDITS',
                        'flowximpact2']).sum() / flow
                        impact2 = impact2a - impact2b
                        impact3a = abs(df_temp3.loc[df_temp3['type'] != 'BY-PRODUCT CREDITS',
                            'flowximpact3']).sum() / flow
                        impact3b = abs(df_temp3.loc[df_temp3['type'] == 'BY-PRODUCT CREDITS',
                            'flowximpact3']).sum() / flow
                        impact3 = impact3a - impact3b
                        df_flow_result.loc[(df_flow_result.product_name == df_temp2.product_name.values[0]) &
                                           (df_flow_result.process == process), 'cc_process'] = impact1
                        df_flow_result.loc[(df_flow_result.product_name == df_temp2.product_name.values[0]) &
                                           (df_flow_result.process == process), 'bdv_process'] = impact2
                        df_flow_result.loc[(df_flow_result.product_name == df_temp2.product_name.values[0]) &
                                             (df_flow_result.process == process), 'health_process'] = impact3
                        df_flow_result.loc[(df_flow_result.product_name == df_temp2.product_name.values[0]),
                        'sequence'] = sequence
                for product_name in list(df_flow_result.product_name.unique()):
                    if product_name not in self.bdv_dict.keys():
                        df_temp4 = df_flow_result[df_flow_result.product_name == product_name].copy()
                        df_temp4 = df_temp4[df_temp4.value == 1].copy()
                        if df_temp4[df_temp4.cc_process.isna()].shape[0] == 0:
                            df_temp4['flowximpact'] = df_temp4['flow_amount'] * df_temp4['cc_process']
                            df_temp4['flowximpact2'] = df_temp4['flow_amount'] * df_temp4['bdv_process']
                            df_temp4['flowximpact3'] = df_temp4['flow_amount'] * df_temp4['health_process']
                            if df_temp4['flow_amount'].sum() > 0.01:
                                product_impact = df_temp4['flowximpact'].sum() / df_temp4['flow_amount'].sum()
                                product_impact2 = df_temp4['flowximpact2'].sum() / df_temp4['flow_amount'].sum()
                                product_impact3 = df_temp4['flowximpact3'].sum() / df_temp4['flow_amount'].sum()
                                # ghg_dict[product_name] = product_impact
                                # bdv_dict[product_name] = product_impact2
                                df_flow_result.loc[(df_flow_result.product_name == product_name),
                                'cc_product'] = product_impact
                                df_flow_result.loc[(df_flow_result.product_name == product_name),
                                'bdv_product'] = product_impact2
                                df_flow_result.loc[(df_flow_result.product_name == product_name),
                                'health_product'] = product_impact3
                            elif df_temp4['flow_amount'].sum() > 0:
                                product_impact = df_temp4['cc_process'].min()
                                product_impact2 = df_temp4['bdv_process'].min()
                                product_impact3 = df_temp4['health_process'].min()
                                # ghg_dict[product_name] = product_impact
                                # bdv_dict[product_name] = product_impact2
                                df_flow_result.loc[(df_flow_result.product_name == product_name),
                                'cc_product'] = product_impact
                                df_flow_result.loc[(df_flow_result.product_name == product_name),
                                'bdv_product'] = product_impact2
                                df_flow_result.loc[(df_flow_result.product_name == product_name),
                                'health_product'] = product_impact3
                            else:
                                a = 0
            df_flow_result["cc_contribution"] = df_flow_result["value"] * df_flow_result["cc_product"]
            df_flow_result["bdv_contribution"] = df_flow_result["value"] * df_flow_result["bdv_product"]
            df_flow_result["health_contribution"] = df_flow_result["value"] * df_flow_result["health_product"]
            df_sankey = df_flow_result[df_flow_result.flow_amount > 0.000001].copy()
            df_sankey["flow_amount"] = df_sankey["flow_amount"] * df_sankey["value"]
            df_sankey = df_sankey[df_sankey.value != 1].copy()
            df_sankey['carbon_content'] = df_sankey['product_name'].map(
                self.df_product.set_index('product_name')['carbon_content'])
            process_product_dict = dict(zip(self.df_process.product_process, self.df_process.product_name))
            df_sankey['product_name_2'] = df_sankey['process'].map(process_product_dict)
            df_sankey.loc[df_sankey.value < 0, "product_from"] = df_sankey.loc[df_sankey.value < 0, "product_name"]
            df_sankey.loc[df_sankey.value < 0, "product_to"] = df_sankey.loc[df_sankey.value < 0, "product_name_2"]
            df_sankey.loc[df_sankey.value > 0, "product_from"] = df_sankey.loc[df_sankey.value > 0, "product_name_2"]
            df_sankey.loc[df_sankey.value > 0, "product_to"] = df_sankey.loc[df_sankey.value > 0, "product_name"]
            df_sankey["flow_amount"] = abs(df_sankey["flow_amount"])
            df1 = df_flow_result[df_flow_result['type'] == 'PRODUCT'].copy()
            df = df_flow_result[df_flow_result.flow_amount > 0.001].copy()
            # df_flow_result.to_csv(f'flow_result_ele_{self.ele_impact}_with_product_impact.csv')
            df1 = df_flow_result.loc[df_flow_result.product_type.isin(['raw_material', 'emission'])]
            df1['health'] = abs(df1['flowxvalue'] * df1['health_product']) * 1000000
            df1 = df1.loc[df1.health > 0].copy()
            health_heat = df1.loc[df1.process.str.contains('heat, high'), 'health'].sum()
            health_biomass = df1.loc[(df1.process.str.contains('agricultural residue, from')) |
                                     (df1.product_name.str.contains('forest')), 'health'].sum()
            health_waste_incineration = df1.loc[df1.process.str.contains('waste incineration'), 'health'].sum()
            health_total = df1['health'].sum()
            print(f'heat: {health_heat / health_total}%, biomass: {health_biomass / health_total}%, '
            f'waste incineration: {health_waste_incineration / health_total}%')
            return df_sankey, df_flow_result

    def carbon_flow_sankey(self, objective, ele_impact=-999, scenario='with_ccs'):
        df_flow_result = self.model_results(objective, ele_impact)[1]
        df1 = self.model_results(objective, ele_impact)[0]
        if df_flow_result is None:
            print("No solution found.")
            return None
        else:
            df = df_flow_result.copy()
            df['carbon_flow'] = df['flowxvalue'] * df['carbon_content']
            new_rows = []
            # 1. from agricultural residue and forest residue to biomass
            for biomass in ['agricultural_residue', 'forest_residue']:
                df_temp = df.loc[(df.product_name == biomass) & (df['type'] == 'RAW MATERIALS')].copy()
                biomass_amount = abs(df_temp['carbon_flow'].sum())
                new_rows.append({'product_from': biomass, 'product_to': 'biomass', 'flow_amount': biomass_amount})
                methanol = abs(df_temp.loc[df_temp['process'].str.startswith('methanol,'), 'carbon_flow'].sum())
                ethanol = abs(df_temp.loc[df_temp['process'].str.startswith('ethanol,'), 'carbon_flow'].sum())
                lactic_acid = abs(df_temp.loc[df_temp['process'].str.startswith('lactic'), 'carbon_flow'].sum())
                heat = abs(df_temp.loc[df_temp['process'].str.startswith('heat,'), 'carbon_flow'].sum())
                electricity = abs(df_temp.loc[df_temp['process'].str.startswith('electricity,'), 'carbon_flow'].sum())
                other = biomass_amount - methanol - ethanol - lactic_acid - heat - electricity
                new_rows.append({'product_from': 'biomass', 'product_to': 'methanol', 'flow_amount': methanol})
                new_rows.append({'product_from': 'biomass', 'product_to': 'ethanol', 'flow_amount': ethanol})
                new_rows.append({'product_from': 'biomass', 'product_to': 'lactic_acid', 'flow_amount': lactic_acid})
                new_rows.append({'product_from': 'biomass', 'product_to': 'heat', 'flow_amount': heat})
                new_rows.append({'product_from': 'biomass', 'product_to': 'electricity', 'flow_amount': electricity})
                new_rows.append({'product_from': 'biomass', 'product_to': 'other_intermediates', 'flow_amount': other})
                new_rows.append({'product_from': 'heat', 'product_to': 'loss', 'flow_amount': heat})
                new_rows.append({'product_from': 'electricity', 'product_to': 'loss', 'flow_amount': electricity})

            # 2. biomethanol production
            df_temp = df[(df['process'].str.startswith('methanol,')) &
                         (df['process'].str.contains('residue gasification'))].copy()
            biomethanol_product = abs(df_temp.loc[df_temp.type == 'PRODUCT', 'carbon_flow']).sum()
            co2_feedstock = abs(df_temp.loc[df_temp.type == 'BY-PRODUCT CREDITS', 'carbon_flow']).sum()
            carbon_in = abs(df_temp.loc[df_temp.type == 'RAW MATERIALS', 'carbon_flow']).sum()
            loss = carbon_in - biomethanol_product - co2_feedstock
            new_rows.append({'product_from': 'methanol', 'product_to': 'co2_feedstock', 'flow_amount': co2_feedstock})
            new_rows.append({'product_from': 'methanol', 'product_to': 'loss', 'flow_amount': loss})

            # 4. co2 hydrogenation to methanol
            df_temp = df[(df['process'].str.startswith('methanol, from co2 hydrogenation'))]
            co2_feedstock_amount = abs(df_temp.loc[df_temp['product_name'].str.contains('co2_feedstock'), 'carbon_flow'].sum())
            methanol_amount = abs(df_temp.loc[df_temp['product_name'].str.contains('methanol'), 'carbon_flow'].sum())
            loss_amount = co2_feedstock_amount - methanol_amount
            new_rows.append({'product_from': 'co2_feedstock', 'product_to': 'methanol',
                             'flow_amount': co2_feedstock_amount})
            new_rows.append({'product_from': 'methanol', 'product_to': 'loss', 'flow_amount': loss_amount})

            # 5. waste gasification to methanol
            df_temp = df[df['process'].str.startswith('methanol') &
                         df['process'].str.contains(f'waste gasification')]
            methanol_amount = abs(df_temp.loc[df_temp['product_name'].str.startswith('methanol'), 'carbon_flow'].sum())
            co2_f_amount = abs(df_temp.loc[df_temp['product_name'].str.contains('co2_feedstock'), 'carbon_flow'].sum())
            waste_amount = abs(df_temp.loc[df_temp['product_name'].str.contains('waste'), 'carbon_flow'].sum())
            loss_amount = waste_amount - methanol_amount - co2_f_amount
            new_rows.append({'product_from': 'plastic_waste', 'product_to': 'methanol', 'flow_amount': waste_amount})
            new_rows.append({'product_from': 'methanol', 'product_to': 'loss', 'flow_amount': loss_amount})
            new_rows.append({'product_from': 'methanol', 'product_to': f'co2_feedstock',
                             'flow_amount': co2_f_amount})

            # 5. waste pyrolysis to naphtha
            df_temp = df[df['process'].str.startswith('naphtha') &
                            df['process'].str.contains(f'waste pyrolysis')]
            naphtha_amount = abs(df_temp.loc[df_temp['product_name'].str.startswith('naphtha'), 'carbon_flow'].sum())
            waste_amount = abs(df_temp.loc[df_temp['product_name'].str.contains('waste'), 'carbon_flow'].sum())
            loss_amount = waste_amount - naphtha_amount
            new_rows.append({'product_from': 'plastic_waste', 'product_to': 'steam_cracker_feedstock', 'flow_amount': waste_amount})
            new_rows.append({'product_from': 'steam_cracker_feedstock', 'product_to': 'loss', 'flow_amount': loss_amount})

            # 6. methanol to ethylene
            df_temp = df[(df['process'].str.contains('ethylene, from methanol'))]
            methanol_amount = abs(df_temp.loc[df_temp['product_name'].str.contains('methanol'), 'carbon_flow'].sum())
            ethylene_amount = abs(df_temp.loc[df_temp['product_name'].str.contains('ethylene'), 'carbon_flow'].sum())
            loss_amount = methanol_amount - ethylene_amount
            new_rows.append({'product_from': 'methanol', 'product_to': 'ethylene', 'flow_amount': methanol_amount})
            new_rows.append({'product_from': 'ethylene', 'product_to': 'loss', 'flow_amount': loss_amount})
            new_rows.append({'product_from': 'ethylene', 'product_to': 'plastic_product', 'flow_amount': ethylene_amount})

            # 7. methanol to propylene
            df_temp = df[(df['process'].str.contains('propylene, from methanol'))]
            methanol_amount = abs(df_temp.loc[df_temp['product_name'].str.contains('methanol'), 'carbon_flow'].sum())
            propylene_amount = abs(df_temp.loc[df_temp['product_name'].str.contains('propylene'), 'carbon_flow'].sum())
            loss_amount = methanol_amount - propylene_amount
            new_rows.append({'product_from': 'methanol', 'product_to': 'propylene', 'flow_amount': methanol_amount})
            new_rows.append({'product_from': 'propylene', 'product_to': 'loss', 'flow_amount': loss_amount})
            new_rows.append({'product_from': 'propylene', 'product_to': 'plastic_product', 'flow_amount': propylene_amount})

            # 8. methanol to BTX
            df_temp = df[(df['process'].str.contains('methanol-to-aromatics'))]
            methanol_amount = abs(df_temp.loc[df_temp['product_name'].str.contains('methanol'), 'carbon_flow'].sum())
            btx_amount = abs(df_temp.loc[df_temp['type'] == 'PRODUCT', 'carbon_flow'].sum())
            loss_amount = methanol_amount - btx_amount
            new_rows.append({'product_from': 'methanol', 'product_to': 'btx', 'flow_amount': methanol_amount})
            new_rows.append({'product_from': 'btx', 'product_to': 'loss', 'flow_amount': loss_amount})
            new_rows.append({'product_from': 'btx', 'product_to': 'plastic_product', 'flow_amount': btx_amount})

            # 9. methanol to other intermediates
            df_temp = df[(df['product_name'].str.contains('methanol')) &
                         (df['type'] == 'RAW MATERIALS') &
                         (~df['process'].str.contains('methanol-to'))]
            process_list = df_temp['process'].unique()
            df_temp = df.loc[df.process.isin(process_list)].copy()
            methanol_amount = abs(df_temp.loc[df_temp['product_name'].str.contains('methanol'), 'carbon_flow'].sum())
            product_amount = abs(df_temp.loc[df_temp['type'] == 'PRODUCT', 'carbon_flow'].sum())
            propylene_amount = abs(df_temp.loc[df_temp['product_name'].str.contains('propylene'), 'carbon_flow'].sum())
            btx_amount = abs(df_temp.loc[df_temp['product_name'].str.contains('p-xcylene'), 'carbon_flow'].sum())
            co_amount = abs(df_temp.loc[df_temp['product_name'].str.contains('carbon_monoxide'), 'carbon_flow'].sum())
            loss_amount = methanol_amount - product_amount + propylene_amount + btx_amount + co_amount
            new_rows.append({'product_from': 'methanol', 'product_to': 'other_intermediates', 'flow_amount': methanol_amount})
            new_rows.append({'product_from': 'other_intermediates', 'product_to': 'loss', 'flow_amount': loss_amount})
            new_rows.append({'product_from': 'propylene', 'product_to': 'other_intermediates', 'flow_amount': propylene_amount})
            new_rows.append({'product_from': 'btx', 'product_to': 'other_intermediates', 'flow_amount': btx_amount})
            new_rows.append({'product_from': 'propylene', 'product_to': 'plastic_product', 'flow_amount': -propylene_amount})
            new_rows.append({'product_from': 'btx', 'product_to': 'plastic_product', 'flow_amount': -btx_amount})
            new_rows.append({'product_from': 'other_intermediates', 'product_to': 'other_intermediates', 'flow_amount': co_amount})

            # 10. ethanol
            df_temp = df[(df['process'].str.startswith('ethanol,'))]
            ethanol_amount = abs(df_temp.loc[df_temp['product_name'].str.contains('ethanol'), 'carbon_flow'].sum())
            co2_feedstock_amount = abs(df_temp.loc[df_temp['product_name'].str.contains('co2_feedstock'), 'carbon_flow'].sum())
            biomass_amount = abs(df_temp.loc[df_temp['product_name'].str.contains('residue'), 'carbon_flow'].sum())
            other_intermediates = abs(df_temp.loc[df_temp.type == 'RAW MATERIALS', 'carbon_flow'].sum()) - biomass_amount
            loss_amount = -ethanol_amount - co2_feedstock_amount + biomass_amount + other_intermediates
            new_rows.append({'product_from': 'ethanol', 'product_to': 'co2_feedstock', 'flow_amount': co2_feedstock_amount})
            new_rows.append({'product_from': 'other_intermediates', 'product_to': 'ethanol', 'flow_amount': other_intermediates})
            new_rows.append({'product_from': 'ethanol', 'product_to': 'loss', 'flow_amount': loss_amount})

            # 11. ethanol to ethylene
            df_temp = df[(df['process'].str.contains('ethylene, from ethanol'))]
            ethanol_amount = abs(df_temp.loc[df_temp['product_name'].str.contains('ethanol'), 'carbon_flow'].sum())
            ethylene_amount = abs(df_temp.loc[df_temp['product_name'].str.contains('ethylene'), 'carbon_flow'].sum())
            loss_amount = ethanol_amount - ethylene_amount
            new_rows.append({'product_from': 'ethanol', 'product_to': 'ethylene', 'flow_amount': ethanol_amount})
            new_rows.append({'product_from': 'ethylene', 'product_to': 'loss', 'flow_amount': loss_amount})
            new_rows.append({'product_from': 'ethylene', 'product_to': 'plastic_product', 'flow_amount': ethylene_amount})

            # 12. ethanol to other intermediates
            df_temp = df[(df['process'].str.contains('butadiene, from ethanol'))]
            ethanol_amount = abs(df_temp.loc[df_temp['product_name'].str.contains('ethanol'), 'carbon_flow'].sum())
            butadiene_amount = abs(df_temp.loc[df_temp['product_name'].str.contains('butadiene'), 'carbon_flow'].sum())
            loss_amount = ethanol_amount - butadiene_amount
            new_rows.append({'product_from': 'ethanol', 'product_to': 'other_intermediates', 'flow_amount': ethanol_amount})
            new_rows.append({'product_from': 'other_intermediates', 'product_to': 'loss', 'flow_amount': loss_amount})

            # 13. from fossil fuels
            for p in ['ethylene', 'propylene']:
                df_temp = df[(df.product_name == f'{p}_fossil') & (df.type == 'PRODUCT') &
                             (~df.process.str.contains('methanol'))]
                p_amount = abs(df_temp['carbon_flow'].sum())
                new_rows.append({'product_from': 'fossil_fuel', 'product_to': 'steam_cracker_feedstock', 'flow_amount': p_amount})
                new_rows.append({'product_from': 'steam_cracker_feedstock', 'product_to': p, 'flow_amount': p_amount})
                new_rows.append({'product_from': p, 'product_to': 'plastic_product', 'flow_amount': p_amount})
            p_amount = 0
            for p in ['benzene', 'toluene', 'p-xylene']:
                df_temp = df[(df.product_name == f'{p}_fossil') & (df.type == 'PRODUCT') &
                             (~df.process.str.contains('methanol'))]
                p_amount += abs(df_temp['carbon_flow'].sum())
            new_rows.append({'product_from': 'fossil_fuel', 'product_to': 'steam_cracker_feedstock', 'flow_amount': p_amount})
            new_rows.append({'product_from': 'steam_cracker_feedstock', 'product_to': 'btx', 'flow_amount': p_amount})
            new_rows.append({'product_from': 'btx', 'product_to': 'plastic_product', 'flow_amount': p_amount})
            df_temp = df[df.process.str.startswith('heat, district')]
            natural_gas_amount = abs(df_temp.loc[df_temp.product_name == 'natural_gas', 'carbon_flow'].sum())
            new_rows.append({'product_from': 'fossil_fuel', 'product_to': 'heat', 'flow_amount': natural_gas_amount})
            new_rows.append({'product_from': 'heat', 'product_to': 'loss', 'flow_amount': natural_gas_amount})

            # 14. to lactic acid
            df_temp = df[(df['process'].str.contains('lactic acid,'))]
            lactic_acid_amount = abs(df_temp.loc[df_temp['product_name'].str.contains('lactic_acid'), 'carbon_flow'].sum())
            biomass_amount = abs(df_temp.loc[df_temp['product_name'].str.contains('residue'), 'carbon_flow'].sum())
            other_intermediates = abs(df_temp.loc[df_temp.type == 'RAW MATERIALS', 'carbon_flow'].sum()) - biomass_amount
            loss_amount = -lactic_acid_amount + biomass_amount + other_intermediates
            new_rows.append({'product_from': 'lactic_acid', 'product_to': 'loss', 'flow_amount': loss_amount})
            new_rows.append({'product_from': 'other_intermediates', 'product_to': 'lactic_acid', 'flow_amount': other_intermediates})
            new_rows.append({'product_from': 'lactic_acid', 'product_to': 'plastic_product', 'flow_amount': lactic_acid_amount})

            # 15. plastics
            df_temp = df[(df.product_type == 'product')]
            plastic_amount = abs(df_temp['carbon_flow'].sum())
            df_temp = df[(df.product_type == 'waste')]
            plastic_waste_amount = df_temp.loc[df_temp.type == 'WASTE', 'carbon_flow'].sum()
            waste_to_mr = abs(
                df_temp.loc[df_temp.process.str.contains('mechanical recycling'), 'carbon_flow'].sum())
            waste_to_incineration = abs(
                df_temp.loc[df_temp.process.str.contains('incineration'), 'carbon_flow'].sum())
            new_rows.append({'product_from': f'plastic_product', 'product_to': f'plastic_waste',
                             'flow_amount': plastic_waste_amount})
            new_rows.append({'product_from': f'plastic_waste', 'product_to': f'plastic_product',
                             'flow_amount': waste_to_mr})
            new_rows.append({'product_from': f'plastic_waste', 'product_to': 'loss',
                             'flow_amount': waste_to_incineration})

            # 12. co2_feedstock to loss
            df_temp = df[(df.product_name.str.contains('co2_feedstock')) & (df.process.str.contains('release'))]
            co2_amount = df_temp['carbon_flow'].sum()
            new_rows.append({'product_from': "co2_feedstock", 'product_to': 'loss', 'flow_amount': co2_amount})

            # 13. co2 point source to co2 feedstock
            df_temp = df[df.product_name == 'co2_feedstock_fossil']
            co2_fossil_amount = df_temp['carbon_flow'].sum()
            if co2_fossil_amount < 0:
                new_rows.append({'product_from': 'co2_point_source', 'product_to': 'co2_feedstock',
                                 'flow_amount': -co2_fossil_amount})
            # 14. ccs
            df_temp = df[df.product_name.str.contains('co2_storage')]
            ccs_amount = df_temp['carbon_flow'].sum()
            new_rows.append({'product_from': 'co2_feedstock', 'product_to': 'co2_storage', 'flow_amount': ccs_amount})

            # 15. other carbon inputs
            df_temp = df[df.product_name.isin(['glycerin', 'maize_grain', 'potato', 'sds', 'yeast'])]
            other_amount = df_temp['carbon_flow'].sum()
            new_rows.append({'product_from': 'other_raw_material', 'product_to': 'other_intermediates', 'flow_amount': other_amount})

            df_sankey = pd.DataFrame(new_rows)
            new_rows = []
            to_intermediates = df_sankey.loc[df_sankey.product_to == 'other_intermediates', 'flow_amount'].sum()
            from_intermediates = df_sankey.loc[df_sankey.product_from == 'other_intermediates', 'flow_amount'].sum()
            intermediates_delta = to_intermediates - from_intermediates
            new_rows.append({'product_from': 'other_intermediates', 'product_to': 'plastic_product', 'flow_amount': intermediates_delta})
            df_sankey = pd.concat([df_sankey, pd.DataFrame(new_rows)], ignore_index=True)
            new_rows = []
            to_plastics = df_sankey.loc[df_sankey.product_to == 'plastic_product', 'flow_amount'].sum()
            plastics_loss = -plastic_amount + to_plastics
            new_rows.append({'product_from': 'plastic_product', 'product_to': 'loss', 'flow_amount': plastics_loss})
            df_sankey = pd.concat([df_sankey, pd.DataFrame(new_rows)], ignore_index=True)
            if df_sankey.loc[df_sankey.product_from == 'ethanol', 'flow_amount'].sum() > 5:
                df_sankey.loc[df_sankey.product_from == 'ethanol', 'product_from'] = 'other_intermediates'
                df_sankey.loc[df_sankey.product_to == 'ethanol', 'product_to'] = 'other_intermediates'

            df_sankey = df_sankey.groupby(['product_from', 'product_to']).sum().reset_index()
            df_sankey = df_sankey[df_sankey.flow_amount > 0.000001].copy()
            product_list = list(df_sankey['product_from'].unique()) + list(df_sankey['product_to'].unique())
            sorter_dict = dict(zip(product_list, range(len(product_list))))
            df_sankey['product_code_from'] = df_sankey['product_from'].map(sorter_dict)
            df_sankey['product_code_to'] = df_sankey['product_to'].map(sorter_dict)
            df_sankey.to_excel(f'data/figure/sankey_renewable_circular_{scenario}.xlsx')
            node = dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=product_list,
                color="blue"
            )
            link = dict(
                source=df_sankey['product_code_from'],
                target=df_sankey['product_code_to'],
                value=df_sankey['flow_amount']
            )
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Sankey(
                node=node,
                link=link,
            )])
            fig.update_layout(title_text=f"Objective: {objective}, climate change impact flows", font_size=10)
            fig.show()

            return df, df_sankey


def regional_results(master_file_path, plastics_file_path):
    user_input_file = r'data/raw/user_inputs.xlsx'
    base_path = os.path.join("data", "raw", "user_inputs_regions")
    df = pd.read_excel(user_input_file)
    country_list = ['CAN', 'CHN', 'KOR', 'USA', 'JPN', 'BRA', 'IND', 'IDN', 'ZAF', 'RUS', 'TUR', 'MEX', 'UKR',
                    'World'] + image_region_list
    for country in country_list:
        df.loc[df.parameter == 'country', 'value'] = country
        file_name = 'user_inputs_' + country + '.xlsx'
        df.to_excel(os.path.join(base_path, file_name), index=False)
    files = glob.glob(os.path.join(base_path, "*.xlsx"))
    df_list = []
    i = 26
    with pd.ExcelWriter('data/processed/route_choice_by_process_all_regions_1.xlsx', engine='openpyxl') as writer:
        for f in files:
            country_name = f.split('.')[0][41:]
            print('-----------', country_name, '-----------')
            user_input = MasterFile(f, master_file_path, plastics_file_path)
            df1, df2, df3 = user_input.model_results('GHG')
            sheet_name = f'Table S{i}_{country_name}'
            df3.to_excel(writer, sheet_name=sheet_name, index=False)
            df1['country'] = country_name
            df_list.append(df1)
            i+=1
    df = pd.concat(df_list, ignore_index=True)
    agri_list = [x for x in residue_list_code if x != 'forest_residue']
    agri_avai_list = [x + '_availability' for x in agri_list]
    df['c_agri'] = df[agri_list].sum(axis=1) * 0.494
    df['c_forest'] = df['forest_residue'] * 0.521
    df['c_co2'] = df['co2_feedstock_fossil'] / 44 * 12
    df['c_fossil'] = df['petroleum'] * 0.845 + df['natural_gas'] * 0.75
    df['c_plastics'] = df['plastics_mr'] * 0.77
    df['c_plastics_gasi'] = df['waste_to_gasi'] * 0.77
    #df['country'] = ['BRA', 'CHN', 'IDN', 'IND', 'JPN', 'MEX', 'RUS', 'TUR', 'USA', 'World', 'ZAF']
    df['agri_usage'] = df[agri_list].sum(axis=1) / df[agri_avai_list].sum(axis=1)
    df['forest_usage'] = df['forest_residue'] / df['forest_residue_availability']
    df['electricity_usage'] = df['electricity_non_biomass'] / df['electricity_non_biomass_availability']
    df['plastics_ghg_intensity'] = df['ghg'] / df['plastic_production']
    df['plastics_bdv_intensity'] = df['bdv'] / df['plastic_production']
    df['plastics_health_intensity'] = df['health'] / df['plastic_production']
    return df


def different_scenarios(master_file_path, plastics_file_path):
    base_path = os.path.join("data", "raw", "user_inputs_scenarios")
    '''
    df0 = pd.read_excel(user_input_file)
    file_name = 'user_inputs_default.xlsx'
    df0.to_excel(os.path.join(base_path, file_name), index=False)
    '''
    '''
    df = df0.copy()
    df.loc[df.parameter == 'agricultural_residue_routes', 'value'] = False
    df.loc[df.parameter == 'forest_residue_routes', 'value'] = False
    df.loc[df.parameter == 'co2_routes', 'value'] = False
    file_name = 'user_inputs_fossil.xlsx'
    df.to_excel(os.path.join(base_path, file_name), index=False)
    df = df0.copy()
    df.loc[df.parameter == 'fossil_routes', 'value'] = False
    file_name = 'user_inputs_no_fossil.xlsx'
    df.to_excel(os.path.join(base_path, file_name), index=False)
    df = df0.copy()
    df.loc[df.parameter == 'ccs_process_co2', 'value'] = True
    file_name = 'user_inputs_ccs.xlsx'
    df.to_excel(os.path.join(base_path, file_name), index=False)
    df = df0.copy()
    df.loc[df.parameter == 'low_biodiversity', 'value'] = False
    file_name = 'user_inputs_all_biomass.xlsx'
    df.to_excel(os.path.join(base_path, file_name), index=False)
    df = df0.copy()
    df.loc[df.parameter == 'new_bio_plastics', 'value'] = False
    file_name = 'user_inputs_no_new_plastics.xlsx'
    df.to_excel(os.path.join(base_path, file_name), index=False)
    df = df0.copy()
    df.loc[df.parameter == 'biomass_ratio', 'value'] = 0.5
    file_name = 'user_inputs_half_biomass.xlsx'
    df.to_excel(os.path.join(base_path, file_name), index=False)
    df = df0.copy()
    df.loc[df.parameter == 'fossil_lock_in', 'value'] = True
    file_name = 'user_inputs_fossil_lockin.xlsx'
    df.to_excel(os.path.join(base_path, file_name), index=False)
    '''
    '''
    df1 = df0.copy()
    df1.loc[df1.parameter == 'agricultural_residue_routes', 'value'] = False
    df1.loc[df1.parameter == 'forest_residue_routes', 'value'] = False
    df1.loc[df1.parameter == 'co2_routes', 'value'] = False
    df1.loc[df1.parameter == 'mechanical_recycling', 'value'] = False
    df1.loc[df1.parameter == 'chemical_recycling_gasification', 'value'] = False
    df1.loc[df1.parameter == 'new_bio_plastics', 'value'] = False
    df1.loc[df1.parameter == 'ccs_process_co2', 'value'] = False
    file_name = 'user_inputs_step1_fossil_linear.xlsx'
    df1.to_excel(os.path.join(base_path, file_name), index=False)
    df1.loc[df1.parameter == 'agricultural_residue_routes', 'value'] = True
    df1.loc[df1.parameter == 'forest_residue_routes', 'value'] = True
    df1.loc[df1.parameter == 'co2_routes', 'value'] = True
    file_name = 'user_inputs_step2_renewable_linear.xlsx'
    df1.to_excel(os.path.join(base_path, file_name), index=False)
    df1.loc[df1.parameter == 'new_bio_plastics', 'value'] = True
    file_name = 'user_inputs_step3_new_bioplastics.xlsx'
    df1.to_excel(os.path.join(base_path, file_name), index=False)
    df1.loc[df1.parameter == 'mechanical_recycling', 'value'] = True
    file_name = 'user_inputs_step4_mr.xlsx'
    df1.to_excel(os.path.join(base_path, file_name), index=False)
    df1 = df1.copy()
    df1.loc[df1.parameter == 'chemical_recycling_gasification', 'value'] = True
    file_name = 'user_inputs_step5_cr.xlsx'
    df1.to_excel(os.path.join(base_path, file_name), index=False)
    df1 = df1.copy()
    df1.loc[df1.parameter == 'ccs_process_co2', 'value'] = True
    file_name = 'user_inputs_step6_ccs.xlsx'
    df1.to_excel(os.path.join(base_path, file_name), index=False)
    '''
    files = glob.glob(os.path.join(base_path, "*.xlsx"))
    df_list = []
    for f in files:
        scenario_name = f.split('inputs_')[2].split('.')[0]
        print('-----------', scenario_name, '-----------')
        user_input = MasterFile(f, master_file_path, plastics_file_path)
        df1, df2, df3 = user_input.model_results('GHG')
        df1['scenario'] = scenario_name
        df_list.append(df1)
    df = pd.concat(df_list, ignore_index=True)
    df.fillna(0, inplace=True)
    agri_list = [x for x in residue_list_code if x != 'forest_residue']
    df['c_agri'] = df[agri_list].sum(axis=1) * 0.494
    df['c_forest'] = df['forest_residue'] * 0.521
    df['c_co2'] = df['co2_feedstock_fossil'] / 44 * 12
    df['c_fossil'] = df['petroleum'] * 0.845 + df['natural_gas'] * 0.75
    df['c_plastics'] = df['plastics_mr'] * 0.77
    df['c_plastics_gasi'] = df['waste_to_gasi'] * 0.77
    return df


def system_contribution_analysis(master_file_path, plastics_file_path, country):
    user_file_path1 = r'data/raw/user_inputs_scenarios/user_inputs_step1_fossil_linear.xlsx'
    user_file_path2 = r'data/raw/user_inputs_scenarios/user_inputs_step6_ccs.xlsx'
    df1 = pd.read_excel(user_file_path1)
    df2 = pd.read_excel(user_file_path2)
    df1.loc[df1.parameter == 'country', 'value'] = country
    df2.loc[df2.parameter == 'country', 'value'] = country
    df1.to_excel(user_file_path1, index=False)
    df2.to_excel(user_file_path2, index=False)
    df = pd.DataFrame()
    for f in [user_file_path1, user_file_path2]:
        user_input = MasterFile(f, master_file_path, plastics_file_path)
        df1, df2, df3 = user_input.model_results('GHG')
        df_fossil = df2.loc[df2.product_name.isin(['petroleum', 'natural_gas'])]
        df_fossil_heat = df2.loc[df2.product_name.isin(['petroleum', 'natural_gas']) &
                                 (df2.process.str.startswith('heat'))]
        df_biomass = df2.loc[df2.product_name.isin(residue_list_code)]
        df_ele = df2.loc[(df2.product_type == 'raw_material') & (df2.product_name.str.contains('electricity'))]
        df_ele2 = df2.loc[(df2.product_type == 'emission') & (df2.process.str.contains('electricity_bio'))]
        df_heat = df2.loc[(df2.product_type == 'emission') & (df2.process.str.startswith('heat'))]
        df_waste = df2.loc[df2.process.str.contains('waste')]
        df_ccs = df2.loc[df2.process.str.contains('CCS')]
        df_raw_material = df2.loc[(df2.product_type == 'raw_material') &
                                  (~df2.product_name.str.contains('electricity')) &
                                  (~df2.product_name.isin(residue_list_code)) &
                                  (~df2.product_name.isin(['petroleum', 'natural_gas']))]
        total = []
        df_list = []
        impact_list = ['ghg', 'bdv', 'health']
        df_list.append({'contributor': 'feedstock_fossil_total', **{i: (df_fossil['flowxvalue'] * df_fossil[i]).sum() for i in impact_list}})
        #df_list.append({'contributor': 'feedstock_fossil_heat', **{i: (df_fossil_heat['flowxvalue'] * df_fossil_heat[i]).sum() for i in impact_list}})
        df_list.append({'contributor': 'feedstock_biomass', **{i: (df_biomass['flowxvalue'] * df_biomass[i]).sum() for i in impact_list}})
        df_list.append({'contributor': 'feedstock_other', **{i: (df_raw_material['flowxvalue'] * df_raw_material[i]).sum() for i in impact_list}})
        df_list.append({'contributor': 'electricity_grid', **{i: (df_ele['flowxvalue'] * df_ele[i]).sum() for i in impact_list}})
        #df_list.append({'contributor': 'electricity_biomass', **{i: (df_ele2['flowxvalue'] * df_ele2[i]).sum() for i in impact_list}})
        df_list.append({'contributor': 'onsite_heat', **{i: (df_heat['flowxvalue'] * df_heat[i]).sum() for i in impact_list}})
        df_list.append({'contributor': 'waste_treatment', **{i: (df_waste['flowxvalue'] * df_waste[i]).sum() for i in impact_list}})
        df_list.append({'contributor': 'ccs', **{i: (df_ccs['flowxvalue'] * df_ccs[i]).sum() for i in impact_list}})
        for i in ['ghg', 'bdv', 'health']:
            total.append((df2['flowxvalue']*df2[i]).sum())
        df_temp = pd.DataFrame(df_list)
        df_other = pd.DataFrame([{'contributor': 'onsite_process', **{i: total[j] - (df_temp[i].sum()) for j, i in enumerate(impact_list)}}])
        df_temp = pd.concat([df_temp, df_other], ignore_index=True)
        df_temp['scenario'] = f.split('inputs_')[-1].split('.')[0]
        df = pd.concat([df, df_temp], ignore_index=True)
    df.loc[df.scenario.str.contains('fossil_linear'), 'scenario'] = 'fossil-linear'
    df.loc[df.scenario.str.contains('ccs'), 'scenario'] = 'net-zero'
    df['bdv'] *= 1e-6  # PDF
    df['health'] *= 1e9  # DALY
    df['ghg'] *= 1e-3  #Gt CO2eq
    return df


def system_contribution_analysis_all(master_file_path, plastics_file_path):
    df_all = pd.DataFrame()
    country_list = ['CAN', 'CHN', 'KOR', 'USA', 'JPN', 'BRA', 'IND', 'IDN', 'ZAF', 'RUS', 'TUR', 'MEX', 'UKR',
                    'World'] + image_region_list
    for country in country_list:
        df = system_contribution_analysis(master_file_path, plastics_file_path, country)
        df['country'] = country
        df_all = pd.concat([df_all, df], ignore_index=True)
    df = df_all.copy()
    df = df.loc[df.scenario == 'net-zero'].copy()
    df = df[['country', 'contributor', 'ghg', 'bdv', 'health']]
    df['bdv'] *= 1e6  # 10-6 PDF
    df['ghg'] *= 1e3  # Mt CO2eq
    df.rename(columns={'ghg': 'GHG (Mt CO2eq)', 'bdv': 'Biodiversity (1e-6 PDF)', 'health': 'Health (DALY)'}, inplace=True)
    df.loc[df.country == 'ME', 'country'] = 'RME'

    df.to_csv('data/figure/contribution_analysis_all_countries.csv', index=False)