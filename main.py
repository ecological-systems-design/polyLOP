import pandas as pd
import numpy as np

from src.data_preparation.iam_data import (iam_data_preparation)
from src.figures.figures import (product_impact_plot, plot_network, plot_ele_sensitivity,
                                 plot_demand_sensitivity, plot_region, plot_ele_demand_sensitivity, plot_demand_sensitivity_2,
                                 product_consumption_analysis, feedstock_analysis, plot_image_map,
                                 plot_sensitivity_biomass, plot_system_contribution, plot_pareto_curves,
                                 plot_scenarios, plot_sensitivity_electricity, plot_fossil_fuel_impacts,
                                 plot_ele_biomass_fossil_sensitivity,
                                 plot_allocation, plot_biogenic_carbon_impact
                                 )
from src.data_preparation.master_file import MasterFile, system_contribution_analysis
from src.others.aux_functions import track_rm_usage, calculate_raw_material_requirement

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # df_iam = iam_data_preparation()

    master_file_path = r'data/raw/master_file_min_health_c.xlsx'
    #master_file_path = r'data/raw/master_file_average_health.xlsx'
    user_input_file = r'data/raw/user_inputs.xlsx'
    plastics_file_path = r'data/raw/bioplastics_vs_fossil.xlsx'
    objective = 'GHG'   # 'GHG' or 'Biodiversity'
    ele_impact_list = [0.001] + [round(i, 2) for i in np.arange(0.01, 0.41, 0.01)]
    demand_list = [round(i, 3) for i in np.arange(0.2, 1.201, 0.025)]
    ele_impact_list = [0, 0.001, 0.005] + [round(i, 3) for i in np.arange(0.01, 0.41, 0.01)]
    biomass_availability_list = [0.5, 1]
    user_input = MasterFile(user_input_file, master_file_path, plastics_file_path)
    #user_input.export_process_list()
    #'''
    # Fig 1
    #plot_scenarios(master_file_path, plastics_file_path)
    #plot_pareto_curves(user_input)
    #user_input.carbon_flow_sankey('GHG', scenario='with_ccs')
    plot_system_contribution(master_file_path, plastics_file_path)
    # Fig 2
    #plot_sensitivity_electricity(master_file_path, plastics_file_path)
    #plot_sensitivity_biomass(master_file_path, plastics_file_path)

    # Fig 3
    #plot_fossil_fuel_impacts(master_file_path, plastics_file_path)
    #plot_ele_demand_sensitivity(user_input_file, master_file_path, plastics_file_path)
    #plot_ele_biomass_fossil_sensitivity(master_file_path, plastics_file_path)
    #'''
    # Fig 4
    #plot_ele_demand_sensitivity(user_input_file, master_file_path, plastics_file_path)
    #plot_demand_sensitivity_2(user_input_file, master_file_path, plastics_file_path)
    # Fig 5
    #plot_region(master_file_path, plastics_file_path)
    # -----------------SI-------------------
    #plot_image_map(master_file_path, plastics_file_path)
    plot_allocation(master_file_path, plastics_file_path)
    plot_biogenic_carbon_impact(user_input)
    #plot_network(user_input)
    #plot_sensitivity_electricity(master_file_path, plastics_file_path)
    #plot_system_contribution(master_file_path, plastics_file_path)

    plot_scenarios(master_file_path, plastics_file_path)
    #plot_fossil_fuel_impacts(master_file_path, plastics_file_path)
    #plot_ele_demand_sensitivity(user_input_file, master_file_path, plastics_file_path)
    #plot_region(master_file_path, plastics_file_path)
    #plot_demand_sensitivity(master_file_path, plastics_file_path)
    user_input = MasterFile(user_input_file, master_file_path, plastics_file_path)
    user_input.export_process_list()
    #plot_pareto_curves(user_input)
    #plot_biomass_sensitivity(user_input)
    #user_input.sensitivity_demand_ele_impact()
    #plot_ele_sensitivity(user_input)
    #user_input.model_results_multi_objective()
    user_input.carbon_flow_sankey('GHG')
    #plot_network(user_input)
    #user_input.calculate_product_impacts('GHG')
    '''
    ele_impact_list = [round(i, 3) for i in
                       np.arange(0.001, 0.402, 0.03)]  # + [round(i, 2) for i in np.arange(0.2, 0.5, 0.1)]
    # ele_impact_list = [0.001, 0.05, 0.11, 0.15]
    df_list = []
    for ele_impact in ele_impact_list:
        df_temp = user_input.calculate_product_impacts('GHG', ele_impact)[1]
        df_temp['ele_impact'] = ele_impact
        df_list.append(df_temp)
    df = pd.concat(df_list, ignore_index=True)
    df1 = df.loc[df['type'] == 'PRODUCT']
    df2 = pd.pivot_table(df1, index=['process', 'product_name', 'unit', 'product_type', 'carbon_content'],
                            columns='ele_impact', values='cc_process', aggfunc='sum')
    df2.reset_index(inplace=True)
    '''
    '''
    user_input.calculate_product_impacts('GHG', 0.031)
    df1 = user_input.model_results('GHG', 0.061)[1]
    df2 = user_input.model_results('GHG', 0.031)[1]
    df1.set_index(['process', 'product_name', 'type', 'unit', 'value', 'product_type', 'carbon_content', 'data_source'],
                  inplace=True)
    df2.set_index(['process', 'product_name', 'type', 'unit', 'value', 'product_type', 'carbon_content', 'data_source'],
                  inplace=True)
    df2.rename(columns={'flow_amount': 'flow_amount_0.321'}, inplace=True)
    df = pd.concat([df1, df2], axis=1)
    df = df.loc[df[['flow_amount', 'flow_amount_0.321']].sum(axis=1) != 0]
    df['delta'] = df['flow_amount_0.321'] - df['flow_amount']
    df = df.loc[abs(df['delta']) > 0.1]
    df.reset_index(inplace=True)
    df3 = df[df.type=='PRODUCT']
    '''
    #user_input.carbon_flow_sankey('GHG', 0.15)
    user_input.calculate_product_impacts('BDV', 0.05)
    plot_ele_sensitivity(user_input)
    #user_input.sensitivity_ele_impact('GHG')
    #user_input.carbon_flow_sankey('GHG', 0.001)
    #user_input.carbon_flow_sankey('GHG', 0.05)
    #user_input.carbon_flow_sankey('GHG', 0.3)
    user_input.calculate_product_impacts('GHG', 0.001)
    user_input.calculate_product_impacts('GHG', 0.05)
    user_input.calculate_product_impacts('GHG', 0.3)
    user_input.carbon_flow_sankey('GHG', 0.121)
    user_input.sensitivity_ele_impact('GHG')
    # sankey_flow_carbon(model, objective)# -999 for default value from ecoinvent (0.06917kgCO2eq/kWh global world average 2050)
    # single_country_analysis(year, scenario, file_path, low_biodiversity, fossil_routes, bio_plastics, mechanical_recycling, chemical_recycling_gasi, eol_incineration, ccs, allocation_choice, demand_scenario, iam_scenario, ele_share, ele_impact)
    # plastics_production_impacts(year, scenario, country, file_path, bio_plastics, mechanical_recycling, chemical_recycling_gasi, allocation_choice, iam_scenario)
    '''
    df1 = pd.read_csv('flow_result_ele_0.001.csv', index_col=0)
    df2 = pd.read_csv('flow_result_ele_0.05.csv', index_col=0)
    df3 = pd.read_csv('flow_result_ele_0.11.csv', index_col=0)
    df4 = pd.read_csv('flow_result_ele_0.15.csv', index_col=0)
    df2 = df2['flow_amount']
    df3 = df3['flow_amount']
    df4 = df4['flow_amount']
    df = pd.concat([df1, df2, df3, df4], axis=1)
    df = df.loc[df['flow_amount'].sum(axis=1) != 0]
    '''


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
