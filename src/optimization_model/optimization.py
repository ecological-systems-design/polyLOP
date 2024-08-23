import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.data_preparation.master_file_back_up import polymer_subsector_demand_dict, \
    add_eol_incineration, add_ccs
from src.data_preparation.plastics_recycling_data import waste_to_secondary_plastics_ratio
from src.others.variable_declaration import residue_list_code, co2_feedstock_list, sector_subsector_dict, \
    max_replacing_rate_dict, final_product_list


class OptimizationModel:
    def __init__(self, year, scenario, country, file_path, low_biodiversity=True, fossil_routes=True,
                 bio_plastics=True, mechanical_recycling=True, chemical_recycling_gasi=True, eol_incineration=True,
                 ccs=True, allocation_choice='standard',
                 demand_scenario=2050, iam_scenario='SSP2_SPA2_19I_D', ele_share=0.02, ele_impact=-999):
        self.df_demand = None
        self.df_supply = None
        self.df_impact = None
        self.df_product = None
        self.year = year
        self.scenario = scenario
        self.country = country
        self.low_biodiversity = low_biodiversity
        self.fossil_routes = fossil_routes
        self.bio_plastics = bio_plastics
        self.mechanical_recycling = mechanical_recycling
        self.chemical_recycling_gasi = chemical_recycling_gasi
        self.eol_incineration = eol_incineration
        self.ccs = ccs
        self.allocation_choice = allocation_choice
        self.iam_scenario = iam_scenario
        self.ele_share = ele_share
        self.ele_impact = ele_impact
        self.demand_scenario = demand_scenario
        self.file_path = file_path
        self.load_data()
        self.setup_model()

    def load_data(self):
        self.df_product, self.df_process, self.df_flow = add_ccs(self.year, self.scenario,
                                                                              self.country, self.file_path,
                                                                              self.allocation_choice,
                                                                              self.demand_scenario,
                                                                              self.iam_scenario, self.ele_share,
                                                                              self.ele_impact, self.low_biodiversity,
                                                                              self.fossil_routes,
                                                                              self.bio_plastics,
                                                                              self.mechanical_recycling,
                                                                              self.chemical_recycling_gasi,
                                                                              self.eol_incineration, self.ccs)

        self.df_impact = self.df_product[
            (self.df_product.product_type == 'raw_material') | (self.df_product.product_type == 'emission') |
            (self.df_product.product_type == 'waste')].copy()
        self.df_supply = self.df_product[self.df_product.product_type == 'raw_material'].copy()
        self.df_demand = self.df_product[self.df_product.product_type == 'product'].copy()
        self.df_intermediate = self.df_product[self.df_product.product_type == 'intermediate'].copy()
        self.df_waste = self.df_product[self.df_product.product_type == 'waste'].copy()
        # self.df_flow = pd.read_excel(self.file_path, engine='openpyxl', sheet_name='flows')
        # self.df_process = pd.read_excel(self.file_path, engine='openpyxl', sheet_name='process')
        # self.df_process = self.df_process[self.df_process.include == "yes"].copy()
        self.process_list = list(self.df_process.product_process.unique())
        # self.df_flow = self.df_flow[self.df_flow.process.isin(self.process_list)].copy()
        self.inflow_name, self.inflow = gp.multidict({(i, j): self.df_flow.loc[(self.df_flow['process'] == i) &
                                                                               (self.df_flow['product_name'] == j),
        'value'].values[0] for i, j in
                                                      zip(self.df_flow['process'], self.df_flow['product_name'])})
        self.process_name_list = list(self.df_flow['process'].unique())
        self.supply_dict = dict(zip(self.df_supply['product_name'], self.df_supply['supply_demand']))
        self.demand_dict = dict(zip(self.df_demand['product_name'], self.df_demand['supply_demand']))
        self.waste_dict = dict(zip(self.df_waste['product_name'], self.df_waste['supply_demand']))
        self.intermediate_dict = dict(zip(self.df_intermediate['product_name'], self.df_intermediate['supply_demand']))
        self.bdv_dict = dict(zip(self.df_impact['product_name'], self.df_impact['Biodiversity']))
        self.ghg_dict = dict(zip(self.df_impact['product_name'], self.df_impact['GHG']))

    def setup_model(self):
        self.m = gp.Model('plastics_optimization')

        # Initialize flow variables with a lower bound of 0
        self.flow = self.m.addVars(self.process_name_list, lb=0, name="flow")

        # Define constraints for supply, demand, and intermediate flows
        for i in self.supply_dict.keys():
            if self.supply_dict[i] < 1e10:
                supply_flow = -gp.quicksum(self.flow[p] * self.inflow[p, i]
                                           for p in
                                           self.df_flow.loc[self.df_flow.product_name == i, "process"].unique())
                self.m.addConstr(supply_flow <= self.supply_dict[i], name=f"supply_flow_{i}")
                if 'co2_feedstock' in i:
                    self.m.addConstr(supply_flow >= 0, name=f"supply_flow_{i}")
        self.polymer_demand_dict = polymer_subsector_demand_dict(self.country, self.demand_scenario)
        for i in self.polymer_demand_dict.keys():
            df_temp = self.df_flow.loc[(self.df_flow.product_name.str.contains(i)) &
                                       (self.df_flow['type'] == 'PRODUCT')]
            demand_flow = gp.quicksum(self.flow[p] for p in df_temp['process'].unique())
            if 'hdpe' in i:
                self.m.addConstr(demand_flow >= self.polymer_demand_dict[i]*10, name=f"demand_flow_{i}_1")
                self.m.addConstr(demand_flow <= self.polymer_demand_dict[i]*10, name=f"demand_flow_{i}_2")
            else:
                self.m.addConstr(demand_flow >= 0, name=f"demand_flow_{i}_1")
                self.m.addConstr(demand_flow <= 0, name=f"demand_flow_{i}_2")

        for i in self.intermediate_dict.keys():
            intermediate_flow = gp.quicksum(self.flow[p] * self.inflow[p, i]
                                            for p in self.df_flow[self.df_flow.product_name == i]["process"].unique())
            self.m.addConstr(intermediate_flow >= 0, name=f"intermediate_flow_{i}_1")
            self.m.addConstr(intermediate_flow <= 0, name=f"intermediate_flow_{i}_2")
            ''''
            # Adjust the constraint condition as needed
            if i != "acetic_acid":
                self.m.addConstr(intermediate_flow <= 0.01, name=f"intermediate_flow_{i}_upper_bound")
            '''
        '''
        product_list = list(self.df_product.loc[self.df_product['product_type'] == 'product', 'product_name'].unique())
        for i in product_list:
            product_flow = gp.quicksum(self.flow[p] * self.inflow[p, i]
                                       for p in self.df_flow[self.df_flow.product_name == i]["process"].unique())
            self.m.addConstr(product_flow >= 0, name=f"product_flow_{i}_1")
            self.m.addConstr(product_flow <= 0, name=f"product_flow_{i}_2")
        '''
        # waste
        for i in self.waste_dict.keys():
            # 1. waste left = 0
            virgin_product = i.replace('_waste', '')
            process_list = self.df_flow[self.df_flow.product_name == i]["process"].unique()
            waste_flow = gp.quicksum(self.flow[p] * self.inflow[p, i]
                                     for p in process_list)
            if self.eol_incineration:
                self.m.addConstr(waste_flow >= 0, name=f"waste_flow_{i}_1")
                self.m.addConstr(waste_flow <= 0, name=f"waste_flow_{i}_2")
            else:
                self.m.addConstr(waste_flow >= 0, name=f"waste_flow_{i}")
            # 2. no production of virgin product and then no waste
            df_temp1 = self.df_flow[(self.df_flow['product_name'] == virgin_product) &
                                    (self.df_flow['type'] == 'PRODUCT')].copy()
            df_temp2 = self.df_flow[(self.df_flow['product_name'] == i) &
                                    (self.df_flow['type'] == 'WASTE')].copy()
            z = self.m.addVar(vtype=gp.GRB.BINARY, name=f"z_{i}")
            virgin_amount = gp.quicksum(self.flow[p] for p in df_temp1['process'].unique())
            waste_amount = gp.quicksum(self.flow[p] * df_temp2.loc[df_temp2.process == p, 'value'].sum()
                                       for p in df_temp2['process'].unique())
            self.m.addConstr(virgin_amount >= 0.0001 * z, name=f"binding_production_waste_{i}_1")
            self.m.addConstr(waste_amount <= 1e10 * z, name=f"binding_production_waste_{i}_2")

        if self.mechanical_recycling:
            polymer_source_list = ['_co2', '_biogenic_short', '_biogenic_long', '_fossil',
                                   '_co2_mr', '_biogenic_short_mr', '_biogenic_long_mr', '_fossil_mr']
            ratio_dict = waste_to_secondary_plastics_ratio()[1]
            for i in ratio_dict.keys():
                ratio = ratio_dict[i]
                if ratio != 0:
                    for suffix in polymer_source_list[0:4]:
                        df_flow_temp1 = self.df_flow[(self.df_flow['product_name'].str.contains(f'{i}{suffix}')) &
                                                     (self.df_flow['product_name'].str.contains('_mr')) &
                                                     (self.df_flow['type'] == 'PRODUCT') &
                                                     (self.df_flow['product_type'] == 'intermediate')].copy()
                        df_flow_temp2 = self.df_flow[(self.df_flow['process'].str.contains(f'{i}{suffix}')) &
                                                     (self.df_flow['type'] == 'WASTE')].copy()
                        mr_plastics = gp.quicksum(self.flow[p] for p in df_flow_temp1['process'].unique())
                        waste = gp.quicksum(self.flow[p] * df_flow_temp2.loc[df_flow_temp2.process == p, 'value'].sum()
                                            for p in df_flow_temp2['process'].unique())
                        self.m.addConstr(waste * ratio - mr_plastics >= 0, name=f"mr_flow_{i}_{suffix}")
        if self.chemical_recycling_gasi:
            polymer_source_list = ['_co2', '_biogenic_short', '_biogenic_long', '_fossil',
                                   '_co2_mr', '_biogenic_short_mr', '_biogenic_long_mr', '_fossil_mr']
            ratio_dict = waste_to_secondary_plastics_ratio()[2]
            for i in ratio_dict.keys():
                ratio = ratio_dict[i]
                for suffix in polymer_source_list[0:4]:
                    waste = f'{i}{suffix}'
                    df_flow_temp1 = self.df_flow[(self.df_flow['product_name'].str.contains(waste)) &
                                                 (self.df_flow['process'].str.contains('waste gasification'))].copy()
                    df_flow_temp2 = self.df_flow[(self.df_flow['process'].str.contains(f'{i}{suffix}')) &
                                                 (self.df_flow['type'] == 'WASTE')].copy()
                    gasi_plastics = -gp.quicksum(self.flow[p] * df_flow_temp1.loc[df_flow_temp1.process == p, 'value'].sum()
                                                 for p in df_flow_temp1['process'].unique())
                    waste = gp.quicksum(self.flow[p] * df_flow_temp2.loc[df_flow_temp2.process == p, 'value'].sum()
                                        for p in df_flow_temp2['process'].unique())
                    self.m.addConstr(waste * ratio - gasi_plastics >= 0, name=f"gasi_flow_{i}_{suffix}")
        if self.bio_plastics:
            for npl in ['pla', 'phb']:
                for op in ['ldpe', 'hdpe', 'pp', 'gpps', 'pvc', 'pet']:
                    df_temp1 = self.df_flow[(self.df_flow['process'].str.contains(f'{npl}')) &
                                            (self.df_flow['process'].str.contains(f'replacing {op}')) &
                                            (self.df_flow['product_type'] == 'product')].copy()
                    df_temp2 = self.df_flow[(self.df_flow['product_name'].str.contains(f'{op}')) &
                                            (self.df_flow['product_type'] == 'product')].copy()
                    if df_temp1.shape[0] > 0:
                        np_replaced = gp.quicksum(self.flow[p] for p in df_temp1['process'].unique())
                        op_produced = gp.quicksum(self.flow[p] for p in df_temp2['process'].unique())
                        self.m.addConstr(np_replaced - op_produced * max_replacing_rate_dict[f'{npl}_{op}'] <= 0,
                                         name=f"np_replaced_{npl}_{op}")

    def single_objective_optimization_simple_outputs(self, objective):
        m = self.m
        if objective == "GHG":
            impact_dict = self.ghg_dict
        else:
            impact_dict = self.bdv_dict

        total_impact = gp.quicksum(self.flow[p] * self.inflow[p, i] * impact_dict[i]
                                   for i in impact_dict.keys()
                                   for p in self.df_flow.loc[self.df_flow.product_name == i, "process"].unique())

        m.setObjective(total_impact, GRB.MINIMIZE)

        m.optimize()

        # Check and print the status of the model
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
        plastic_production_amount = sum(self.polymer_demand_dict.values())
        for i in report_list:
            if m.status == GRB.OPTIMAL:
                flow_values = self.m.getAttr('x', self.flow)
                solution = m.getAttr("x", self.flow)
                supply_amount = 0
                for p in list(self.df_flow.loc[(self.df_flow.product_name == i) &
                                               (self.df_flow['type'] != 'PRODUCT'), "process"].unique()):
                    supply_amount += solution[p] * self.inflow[p, i]
                if i != 'electricity':
                    print(f"{i}, {-supply_amount} out of {self.supply_dict[i]}")
                else:
                    print(f"Total {i}, {-supply_amount} TWh")
                report_value_list.append(-supply_amount)
            else:
                report_value_list.append(-999)
        report_list_2 = [f'{x}_availability' for x in report_list if x != 'electricity']
        for i in [x for x in report_list if x != 'electricity']:
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
            for i in impact_dict.keys():
                for p in list(self.df_flow.loc[self.df_flow.product_name == i, "process"].unique()):
                    ghg_total += df_result.loc[df_result.process == p, "flow_amount"].iloc[0] * self.inflow[p, i] * \
                                 self.ghg_dict[i]
                    bdv_total += df_result.loc[df_result.process == p, "flow_amount"].iloc[0] * self.inflow[p, i] * \
                                 self.bdv_dict[i]
            print(f"GHG: {ghg_total}")
            print(f"Biodiversity: {bdv_total}")
            report_value_list.append(ghg_total)
            report_value_list.append(bdv_total)
            report_list.append('ghg')
            report_list.append('bdv')
            #df_result.loc[df_result.flow_amount < 0.000001, "flow_amount"] = 0.000001
            df_flow_result = pd.merge(self.df_flow, df_result, on='process', how='left')
            df_flow_result['flowxvalue'] = df_flow_result['flow_amount'] * df_flow_result['value']
            plastic_product_list = list(self.df_product.loc[self.df_product.product_type == 'product', 'product_name'])
            print(f"Plastic production: {plastic_production_amount} Mt")
            df_product = df_flow_result[df_flow_result['type'] == 'PRODUCT'].copy()
            plastics_mr = 0
            plastics_bs = 0
            plastics_bl = 0
            plastics_co2 = 0
            plastics_fossil = 0
            for x in final_product_list:
                plastics_mr += df_product.loc[df_product.product_name.isin([f'{x}_biogenic_short_mr',
                                                                            f'{x}_biogenic_long_mr',
                                                                            f'{x}_co2_mr',
                                                                            f'{x}_fossil_mr']), 'flow_amount'].sum()
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
            self.df_impact.loc[self.df_impact.product_name == 'electricity_non_biomass', 'GHG'].values[0]
        df1['electricity_bdv'] = - \
            self.df_impact.loc[self.df_impact.product_name == 'electricity_non_biomass', 'Biodiversity'].values[0]

        if m.status == GRB.OPTIMAL:
            df2 = df_flow_result[df_flow_result.flow_amount > 0.01]
            df_flow_result.to_csv(f'flow_result_ele_{self.ele_impact}.csv')
            #df1 = df_flow_result[df_flow_result.flow_amount > 0.01]
            return df1, df_flow_result
        else:
            return df1, None

    def calculate_product_impacts(self, objective):
        df_flow_result = self.single_objective_optimization_simple_outputs(objective)[1]
        df_flow_result.loc[df_flow_result.flow_amount == 0, 'flow_amount'] = 1e-6
        if df_flow_result is None:
            print("No solution found.")
            return None
        else:
            df_flow_result['cc_product'] = abs(df_flow_result["product_name"].map(self.ghg_dict))
            df_flow_result['cc_process'] = abs(df_flow_result["product_name"].map(self.ghg_dict))
            df_flow_result['bdv_product'] = abs(df_flow_result["product_name"].map(self.bdv_dict))
            df_flow_result['bdv_process'] = abs(df_flow_result["product_name"].map(self.bdv_dict))
            df_flow_result['sequence'] = 0
            sequence = 0
            while df_flow_result[df_flow_result.cc_product.isna()].shape[0] > 0:
                sequence += 1
                for process in list(df_flow_result.process.unique()):
                    df_temp = df_flow_result[df_flow_result.process == process].copy()
                    df_temp2 = df_temp[df_temp.cc_product.isna()].copy()
                    df_temp3 = df_temp[df_temp.cc_product.notna()].copy()
                    if df_temp2.shape[0] == 1:
                        flow = df_temp2['value'].values[0]
                        df_temp3['flowximpact'] = df_temp3['value'] * df_temp3['cc_product']
                        df_temp3['flowximpact2'] = df_temp3['value'] * df_temp3['bdv_product']
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
                        df_flow_result.loc[(df_flow_result.product_name == df_temp2.product_name.values[0]) &
                                           (df_flow_result.process == process), 'cc_process'] = impact1
                        df_flow_result.loc[(df_flow_result.product_name == df_temp2.product_name.values[0]) &
                                           (df_flow_result.process == process), 'bdv_process'] = impact2
                        df_flow_result.loc[(df_flow_result.product_name == df_temp2.product_name.values[0]),
                        'sequence'] = sequence
                for product_name in list(df_flow_result.product_name.unique()):
                    if product_name not in self.bdv_dict.keys():
                        df_temp4 = df_flow_result[df_flow_result.product_name == product_name].copy()
                        df_temp4 = df_temp4[df_temp4.value == 1].copy()
                        if df_temp4[df_temp4.cc_process.isna()].shape[0] == 0:
                            df_temp4['flowximpact'] = df_temp4['flow_amount'] * df_temp4['cc_process']
                            df_temp4['flowximpact2'] = df_temp4['flow_amount'] * df_temp4['bdv_process']
                            if df_temp4['flow_amount'].sum() > 0.01:
                                product_impact = df_temp4['flowximpact'].sum() / df_temp4['flow_amount'].sum()
                                product_impact2 = df_temp4['flowximpact2'].sum() / df_temp4['flow_amount'].sum()
                                # ghg_dict[product_name] = product_impact
                                # bdv_dict[product_name] = product_impact2
                                df_flow_result.loc[(df_flow_result.product_name == product_name),
                                'cc_product'] = product_impact
                                df_flow_result.loc[(df_flow_result.product_name == product_name),
                                'bdv_product'] = product_impact2
                            elif df_temp4['flow_amount'].sum() > 0:
                                product_impact = df_temp4['cc_process'].min()
                                product_impact2 = df_temp4['bdv_process'].min()
                                # ghg_dict[product_name] = product_impact
                                # bdv_dict[product_name] = product_impact2
                                df_flow_result.loc[(df_flow_result.product_name == product_name),
                                'cc_product'] = product_impact
                                df_flow_result.loc[(df_flow_result.product_name == product_name),
                                'bdv_product'] = product_impact2
                            else:
                                a = 0
            df_flow_result["cc_contribution"] = df_flow_result["value"] * df_flow_result["cc_product"]
            df_flow_result["bdv_contribution"] = df_flow_result["value"] * df_flow_result["bdv_product"]
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
            df = df_flow_result[df_flow_result['type'] == 'PRODUCT'].copy()
            df = df_flow_result[df_flow_result.flow_amount > 0.001].copy()
            # df_flow_result.to_csv(f'flow_result_ele_{self.ele_impact}_with_product_impact.csv')
            return df_sankey, df_flow_result

    def single_objective_optimization_full_outputs(self, objective):
        m = self.m
        if objective == "GHG":
            impact_dict = self.ghg_dict
        else:
            impact_dict = self.bdv_dict

        total_impact = gp.quicksum(self.flow[p] * self.inflow[p, i] * impact_dict[i]
                                   for i in impact_dict.keys()
                                   for p in self.df_flow.loc[self.df_flow.product_name == i, "process"].unique())

        m.setObjective(total_impact, GRB.MINIMIZE)
        m.optimize()

        # Check and print the status of the model
        if m.status == GRB.OPTIMAL:
            print("Optimal solution found.")
        elif m.status == GRB.INFEASIBLE:
            print("Model is infeasible.")
        else:
            print(f"Optimization ended with status {self.m.status}")

        # Debugging: Print the flow values to check for negative values

        if m.status == GRB.OPTIMAL:
            flow_values = self.m.getAttr('x', self.flow)
            for key, value in flow_values.items():
                if value < 0:
                    print(f"Negative flow detected: {key} = {value}")
            solution = m.getAttr("x", self.flow)
            df_result = pd.DataFrame.from_dict(solution, orient='index').reset_index()
            df_result.columns = ['process', 'flow_amount']
            report_list = residue_list_code + co2_feedstock_list + ['electricity_non_biomass']
            for i in report_list:
                supply_amount = 0
                for p in list(self.df_flow.loc[self.df_flow.product_name == i, "process"].unique()):
                    supply_amount += solution[p] * self.inflow[p, i]
                print(f"{i}, {-supply_amount} out of {self.supply_dict[i]}")
            '''
            for i in self.demand_dict.keys():
                y = 0
                for p in list(self.df_flow.loc[self.df_flow.product_name == i, "process"].unique()):
                    y += solution[p] * self.inflow[p, i]
                print(f"{i}, {y} out of {self.demand_dict[i]}")
            '''
            ghg_total = 0
            bdv_total = 0
            for i in impact_dict.keys():
                for p in list(self.df_flow.loc[self.df_flow.product_name == i, "process"].unique()):
                    ghg_total += df_result.loc[df_result.process == p, "flow_amount"].iloc[0] * self.inflow[p, i] * \
                                 self.ghg_dict[i]
                    bdv_total += df_result.loc[df_result.process == p, "flow_amount"].iloc[0] * self.inflow[p, i] * \
                                 self.bdv_dict[i]
            print(f"GHG: {ghg_total}")
            print(f"Biodiversity: {bdv_total}")
            df_result.loc[df_result.flow_amount < 0.000001, "flow_amount"] = 0.000001
            df_flow_result = pd.merge(self.df_flow, df_result, on='process', how='left')
            df_flow_result['cc_product'] = abs(df_flow_result["product_name"].map(self.ghg_dict))
            df_flow_result['cc_process'] = abs(df_flow_result["product_name"].map(self.ghg_dict))
            df_flow_result['bdv_product'] = abs(df_flow_result["product_name"].map(self.bdv_dict))
            df_flow_result['bdv_process'] = abs(df_flow_result["product_name"].map(self.bdv_dict))
            df_flow_result['sequence'] = 0
            plastic_product_list = list(self.df_product.loc[self.df_product.product_type == 'product', 'product_name'])
            plastic_production_amount = df_flow_result.loc[df_flow_result.product_name.isin(plastic_product_list),
            'flow_amount'].sum()
            print(f"Plastic production: {plastic_production_amount} Mt")
            sequence = 0
            while df_flow_result[df_flow_result.cc_product.isna()].shape[0] > 0:
                sequence += 1
                for process in list(df_flow_result.process.unique()):
                    df_temp = df_flow_result[df_flow_result.process == process].copy()
                    df_temp2 = df_temp[df_temp.cc_product.isna()].copy()
                    df_temp3 = df_temp[df_temp.cc_product.notna()].copy()
                    if df_temp2.shape[0] == 1:
                        flow = df_temp2['value'].values[0]
                        df_temp3['flowximpact'] = df_temp3['value'] * df_temp3['cc_product']
                        df_temp3['flowximpact2'] = df_temp3['value'] * df_temp3['bdv_product']
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
                        df_flow_result.loc[(df_flow_result.product_name == df_temp2.product_name.values[0]) &
                                           (df_flow_result.process == process), 'cc_process'] = impact1
                        df_flow_result.loc[(df_flow_result.product_name == df_temp2.product_name.values[0]) &
                                           (df_flow_result.process == process), 'bdv_process'] = impact2
                        df_flow_result.loc[(df_flow_result.product_name == df_temp2.product_name.values[0]),
                        'sequence'] = sequence
                for product_name in list(df_flow_result.product_name.unique()):
                    if product_name not in self.bdv_dict.keys():
                        df_temp4 = df_flow_result[df_flow_result.product_name == product_name].copy()
                        df_temp4 = df_temp4[df_temp4.value == 1].copy()
                        if df_temp4[df_temp4.cc_process.isna()].shape[0] == 0:
                            df_temp4['flowximpact'] = df_temp4['flow_amount'] * df_temp4['cc_process']
                            df_temp4['flowximpact2'] = df_temp4['flow_amount'] * df_temp4['bdv_process']
                            if df_temp4['flow_amount'].sum() > 0.01:
                                product_impact = df_temp4['flowximpact'].sum() / df_temp4['flow_amount'].sum()
                                product_impact2 = df_temp4['flowximpact2'].sum() / df_temp4['flow_amount'].sum()
                                # ghg_dict[product_name] = product_impact
                                # bdv_dict[product_name] = product_impact2
                                df_flow_result.loc[(df_flow_result.product_name == product_name),
                                'cc_product'] = product_impact
                                df_flow_result.loc[(df_flow_result.product_name == product_name),
                                'bdv_product'] = product_impact2
                            elif df_temp4['flow_amount'].sum() > 0:
                                product_impact = df_temp4['cc_process'].min()
                                product_impact2 = df_temp4['bdv_process'].min()
                                # ghg_dict[product_name] = product_impact
                                # bdv_dict[product_name] = product_impact2
                                df_flow_result.loc[(df_flow_result.product_name == product_name),
                                'cc_product'] = product_impact
                                df_flow_result.loc[(df_flow_result.product_name == product_name),
                                'bdv_product'] = product_impact2
                            else:
                                a = 0
            df_flow_result["cc_contribution"] = df_flow_result["value"] * df_flow_result["cc_product"]
            df_flow_result["bdv_contribution"] = df_flow_result["value"] * df_flow_result["bdv_product"]
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
            df = df_flow_result[df_flow_result['type'] == 'PRODUCT'].copy()
            df = df_flow_result[df_flow_result.flow_amount > 0.001].copy()
            return df_sankey, df_flow_result
        else:
            print("No solution found.")

    def multi_objective_optimization(self):
        m = self.m

        # Initialize the expressions for total_bdv and total_ghg inside the loop
        num_points = 101
        results_obj1 = []
        results_obj2 = []
        biomass_consumption = []
        electricity_consumption = []
        co2_consumption = []
        df = pd.DataFrame()
        i = 0
        while i < num_points + 1:
            weight = i / (num_points - 1)

            # Recalculate total_bdv and total_ghg inside the loop to ensure they are associated with the current model 'm'
            total_bdv = gp.quicksum(self.flow[p] * self.inflow[p, prod] * self.bdv_dict[prod]
                                    for prod in self.bdv_dict
                                    for p in self.df_flow[self.df_flow.product_name == prod]["process"].unique())
            total_ghg = gp.quicksum(self.flow[p] * self.inflow[p, prod] * self.ghg_dict[prod]
                                    for prod in self.ghg_dict
                                    for p in self.df_flow[self.df_flow.product_name == prod]["process"].unique())

            # Set the objective function
            m.setObjective(weight * total_ghg + (1 - weight) * total_bdv, GRB.MINIMIZE)
            m.update()
            # Optimize the model
            m.optimize()

            # Check the model status and store the results
            if m.status == GRB.Status.OPTIMAL:
                solution = m.getAttr("x", self.flow)
                df_temp = pd.DataFrame.from_dict(solution, orient='index').reset_index()
                df_temp.columns = ['process', f'flow_amount_{i}']
                df_flow_result = pd.merge(self.df_flow, df_temp, on='process', how='left')
                df_temp.set_index('process', inplace=True)
                df = pd.concat([df, df_temp], axis=1)
                results_obj1.append(total_ghg.getValue())
                results_obj2.append(total_bdv.getValue())
                supply_amount = 0
                for r in residue_list_code:
                    for p in list(self.df_flow.loc[self.df_flow.product_name == r, "process"].unique()):
                        supply_amount += solution[p] * self.inflow[p, r]
                biomass_consumption.append(supply_amount)
                supply_amount = 0
                for r in co2_feedstock_list:
                    for p in list(self.df_flow.loc[self.df_flow.product_name == r, "process"].unique()):
                        supply_amount += solution[p] * self.inflow[p, r]
                co2_consumption.append(supply_amount)
                supply_amount = 0
                for p in list(self.df_flow.loc[self.df_flow.product_name == "electricity", "process"].unique()):
                    supply_amount += solution[p] * self.inflow[p, "electricity"]
                electricity_consumption.append(supply_amount)
            i += 1
        # Create a dataframe result lists
        df_result = pd.DataFrame({'GHG': results_obj1, 'Biodiversity': results_obj2,
                                  'biomass_consumption': biomass_consumption,
                                  'electricity_consumption': electricity_consumption,
                                  'co2_consumption': co2_consumption})

        # Convert results to numpy array for plotting
        plt.scatter(results_obj1, results_obj2, marker='o')
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title('Pareto Front')
        plt.show()
        return df, df_result
