from src.figures.figures import (plot_network, plot_region, plot_ele_biomass_demand_sensitivity, plot_demand_sensitivity,
                                 plot_image_map, plot_sensitivity_biomass, plot_system_contribution, plot_pareto_curves,
                                 plot_scenarios, plot_sensitivity_electricity, plot_fossil_fuel_impacts,
                                 plot_allocation, plot_biogenic_carbon_impact
                                 )
from src.optimization_model.master_file import MasterFile

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    master_file_path = r'data/raw/master_file_health_large_biomass.xlsx'
    user_input_file = r'data/raw/user_inputs.xlsx'
    plastics_file_path = r'data/raw/bioplastics_vs_fossil.xlsx'
    user_input = MasterFile(user_input_file, master_file_path, plastics_file_path)
    #user_input.export_process_list()
    # Fig 1
    plot_scenarios(master_file_path, plastics_file_path)  # waterfall and radar
    plot_pareto_curves(user_input)
    user_input.carbon_flow_sankey('GHG', scenario='with_ccs')
    plot_system_contribution(master_file_path, plastics_file_path)
    # Fig 2
    plot_sensitivity_electricity(master_file_path, plastics_file_path)
    plot_sensitivity_biomass(master_file_path, plastics_file_path)
    # Fig 3
    plot_fossil_fuel_impacts(master_file_path, plastics_file_path)
    plot_ele_biomass_demand_sensitivity(user_input_file, master_file_path, plastics_file_path)
    # Fig 4
    plot_demand_sensitivity(master_file_path, plastics_file_path)
    # Fig 5
    plot_region(master_file_path, plastics_file_path)
    # -----------------SI-------------------
    plot_image_map(master_file_path, plastics_file_path)
    plot_allocation(master_file_path, plastics_file_path)
    plot_biogenic_carbon_impact(user_input)
    plot_network(user_input)

