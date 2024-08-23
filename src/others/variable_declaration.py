import pandas as pd

iam_scenario_list = ['EN_NPi2020_900f', 'EN_NPi2020_400f_lowBECCS', 'NGFS2_Current Policies', 'EN_INDCi2030_3000f',
                     'LowEnergyDemand_1.3_IPCC', 'SSP2_openres_lc_50', 'SusDev_SDP-PkBudg1000',
                     'DeepElec_SSP2_ HighRE_Budg900', 'CO_Bridge', 'SSP2-baseline', 'SSP2_SPA2_19I_D',
                     'SSP2_SPA2_19I_LI', 'SSP2_SPA2_19I_LIRE', 'SSP2_SPA2_19I_RE',
                     'R2p1_SSP2-Base', 'R2p1_SSP2-PkBudg900']

image_region_list = ['WEU', 'OCE', 'RSAF', 'RSAM', 'WAF', 'EAF', 'CEU', 'RSAS', 'STAN', 'ME', 'SEAS', 'RCAM', 'NAF']

gas_density_dict = {
    "INERT GAS": 1.165,
    "HYDROGEN": 0.08375,
    "HYDROGEN (99 VOL%)": 0.0949625,
    "NITROGEN": 1.165,
    "CARBON MONOXIDE": 1.165,
    "SYNTHESIS GAS (1.83:1)": 0.465138,
    "SYNTHESIS GAS (1:1)": 0.62385,
    "SYNTHESIS GAS (2:1)": 0.443467,
    "SYNTHESIS GAS (3:1)": 0.353275,
    "AIR": 1.205,
}

max_replacing_rate_dict = {'pla_ldpe': 0.1, 'pla_hdpe': 0.1, 'pla_pp': 0.1, 'pla_pet': 0.2, 'pla_gpps': 0.1,
                           'phb_ldpe': 0.2, 'phb_hdpe': 0.2, 'phb_pp': 0.1, 'phb_pet': 0.1, 'phb_gpps': 0.2,
                           'phb_pur_flexible': 0.05, 'phb_pur_rigid': 0.05, 'phb_pvc': 0.1,
                           'pbs_ldpe': 0.2, 'pbs_hdpe': 0.2, 'pbs_pp': 0.2, 'pbs_pet': 0.2, 'pbs_gpps': 0.2,
                           'pbs_pur_flexible': 0.2, 'pbs_pur_rigid': 0.2, 'pbs_pvc': 0.2,
                           'pbat_ldpe': 0.2, 'pbat_hdpe': 0.2, 'pbat_pp': 0.2, 'pbat_pet': 0.2, 'pbat_gpps': 0.2,
                           'pbat_pur_flexible': 0.2, 'pbat_pur_rigid': 0.2, 'pbat_pvc': 0.2, 'pef_pet': 0.2,
                           }  # based on https://doi.org/10.1016/j.jclepro.2018.03.014

fuel_lhv_dict = {"FUEL, LIQUID (CREDIT)": 39.8, "METHANE.": 50, "LIGHT AND HEAVY ENDS": 39.8,
                 "FUEL OIL, LOW SULFUR": 39.8, "LIGHT ENDS CREDIT": 40.6}
#https://www.engineeringtoolbox.com/fuels-higher-calorific-values-d_169.html

fuel_list1 = ["fuel",
             "fuel_gas",
             "fuel_heavy_liquids",
             "fuel_liquid",
             "fuel_liquid_credit",
             "fuel_oil",
             "fuel_oil_low_sulfur",
             "fuel_oil_residual",
             "light_ends_credit",
             "natural_gas"
             ]

fuel_list = ["FUEL",
             "FUEL GAS",
             "FUEL, HEAVY LIQUIDS",
             "FUEL, LIQUID",
             "FUEL OIL",
             "FUEL OIL, LOW SULFUR",
             "FUEL OIL, RESIDUAL",
             "LIGHT ENDS CREDIT",
             "NATURAL GAS"
             ]

fuel_co2_dict1 = {"fuel": 0.0733, "fuel_gas": 0.0561, "fuel_heavy_liquids": 0.0733, "fuel_liquid": 0.0733,
                    "fuel_liquid_credit": 0.0733, "fuel_oil": 0.0733, "fuel_oil_low_sulfur": 0.0733,
                    "fuel_oil_residual": 0.0733, "light_ends_credit": 0.0733, "natural_gas": 0.0561}  # kgCO2eq/MJ

fuel_co2_dict = {"FUEL": 0.0733, "FUEL GAS": 0.0561, "FUEL, HEAVY LIQUIDS": 0.0733, "FUEL, LIQUID": 0.0733,
                    "FUEL OIL": 0.0733, "FUEL OIL, LOW SULFUR": 0.0733,
                    "FUEL OIL, RESIDUAL": 0.0733, "LIGHT ENDS CREDIT": 0.0733, "NATURAL GAS": 0.0561}  # kgCO2eq/MJ


products_purity_dict = {
    'hydrogen_99_vol%': 0.99,
    'lactic_acid_88%': 0.88,
    'sulfuric_acid_70%': 0.7,
    'sulfuric_acid_98%': 0.98,
    'oxygen_95_mole_%': 0.95,
    'caustic_soda_50%': 0.5,
    'nitric_acid_60%': 0.6,
    'formaldehyde_37%': 0.37,
    'hydrochloric_acid_22_be': 0.33321,
    'hydrochloric_acid_10p9%': 0.109,
}

carbon_content_dict = {'acetaldehyde': 0.545,
                       'acetic_anhydride': 0.471,
                       'acetone': 0.620,
                       'adipic_acid': 0.493,
                       'aniline': 0.774,
                       'chlorobenzene': 0.640,
                       'cumene': 0.899,
                       'cyclohexane': 0.857,
                       'diethylene_glycol': 0.453,
                       'dimethyl_terephthalate': 0.619,
                       'dinitrotoluene': 0.462,
                       'ethylbenzene': 0.905,
                       'ethylene_dichloride': 0,
                       'ethylene_glycol': 0.387,
                       'ethylene_oxide': 0.545,
                       'formaldehyde': 0.4,
                       'butene_1': 0.857,
                       'nitrobenzene': 0.585,
                       'phenol': 0.766,
                       'polybutadiene': 0.888,
                       'polymeric_mdi': 0.720,
                       'polyol_polyester': 0.475,
                       'polyol_polyether': 0.609,
                       'propylene_oxide': 0.620,
                       'styrene': 0.923,
                       'terephthalic_acid': 0.578,
                       'toluene_diisocyanate': 0.621,
                       'vinyl_chloride': 0.384,
                       'hdpe': 0.857,
                       'ldpe': 0.857,
                       'pp': 0.857,
                       'pvc': 0.38,
                       'gpps': 0.923,
                       'hips': 0.923,
                       'pet': 0.63,
                       'pur_flexible': 0.61,
                       'pur_rigid': 0.63,
                       'pef': 0.52747}

dilute_product_list = ['nitric_acid_conc', 'nitric_acid_dilute', 'hydrochloric_acid_dilute', 'propylene_dilute',
                       'diethylene_glycol_crude', 'acetic_acid_credit', 'methanol_crude', "benzene-rich_stream",
                       "purge_ethylene", 'sulfuric_acid_dilute']

dilute_product_list_2 = ['propylene_dilute', "benzene-rich_stream"]

rename_dict_ihs = {'hydrogen_99_vol%': 'hydrogen',
                   'lactic_acid_88%': 'lactic_acid',
                   'sulfuric_acid_70%': 'sulfuric_acid',
                   'sulfuric_acid_98%': 'sulfuric_acid',
                   'sulfuric_acid_dilute': 'sulfuric_acid',
                   'oxygen_95_mole_%': 'oxygen',
                   'oxygen_high_usage': 'oxygen',
                   'oxygen_low_usage': 'oxygen',
                   'oxygen_moderate_usage': 'oxygen',
                   'caustic_soda_50%': 'caustic_soda',
                   'caustic_soda_beads': 'caustic_soda',
                   'nitric_acid_60%': 'nitric_acid',
                   'nitric_acid_conc': 'nitric_acid',
                   'nitric_acid_dilute': 'nitric_acid',
                   'formaldehyde_37%': 'formaldehyde',
                   'hydrochloric_acid_22_be': 'hydrochloric_acid',
                   'hydrochloric_acid_100%': 'hydrochloric_acid',
                   'hydrochloric_acid_10p9%': 'hydrochloric_acid',
                   'hydrochloric_acid_dilute': 'hydrochloric_acid',
                   'propylene_chem_grade': 'propylene',
                   'propylene_polymer_grade': 'propylene',
                   'propylene_dilute': 'propylene_chem_grade',
                   'diethylene_glycol_crude': 'diethylene_glycol',
                   'acetic_acid_credit': 'acetic_acid',
                   'methanol_crude': 'methanol',
                   'benzene-rich_stream': 'benzene',
                   'purge_ethylene': 'ethylene',
                   "butadiene_raffinate": "butadiene",
                   'butene-1': 'butene_1',
                   'hydrogen_chloride': 'hydrochloric_acid',
                   '14-butanediol': 'bdo_14',
                   'deionized_water': 'water',
                   'demineralized_water': 'water',
                   'process_water': 'water',
                   'ethanol.': 'ethanol',
                   'ethanolp': 'ethanol',
                   }

ihs_to_master_name_alignment_dict = {'oxygen': 'oxygen_liquid',
                                     'nitrogen': 'nitrogen_liquid',
                                     'inert_gas': 'nitrogen_liquid',
                                     'synthesis_gas_1_to_1': 'syngas_1_to_1',
                                     'synthesis_gas_1p83_to_1': 'syngas_1p8_to_1',
                                     'synthesis_gas_2_to_1': 'syngas_2_to_1',
                                     'hydrogenp': 'hydrogen',
                                     'caustic_soda': 'sodium_hydroxide',
                                     'steam': 'steam_high',
                                     'cooling_water': 'cooling_water_kg',
                                     'mdi': 'polymeric_mdi',
                                     'polyol_trifunc_polyether': 'polyol_polyether',
                                     }

cut_off_raw_material_list = ['ash_disposal', 'grit_disposal', 'cfc-11',
                             'extender_oil', 'flame_retardant', 'mineral_oil',
                             'n-hexane', "air"]

residue_list = ['Barley straw', 'Maize stover', 'Rapeseed straw', 'Rice straw', 'Sorghum straw',
                'Soybean straw', 'Sugarcane tops and leaves', 'Wheat straw', 'Forest residue']

btx_list = ['benzene', 'toluene', 'p-xylene']

co2_feedstock_list = ['co2_feedstock_fossil', 'co2_feedstock_biogenic_short', 'co2_feedstock_biogenic_long']

co2_emission_list = ['co2_emission_biogenic_short', 'co2_emission_biogenic_long', 'co2_emission_fossil']

residue_list_code = ["barley_straw", "maize_stover", "rapeseed_straw", "rice_straw", "sorghum_straw",
                     "soybean_straw", "sugarcane_tops_and_leaves", "wheat_straw", "forest_residue"]

sector_subsector_dict = {'Building and Construction': ['Buildings and construction'],
                         'Consumer products': ['Medical and hygiene items', 'Rest of consumer products'],
                         'Electrical and Electronics': ['Electrical and electronic'],
                         'Packaging': ['Food films', 'Non-food films', 'Consumer non-food bags',
                                       'Beverage bottles', 'Containers for food', 'Containers for non-food',
                                       'Other food packaging', 'Rest of packaging'],
                         'Transport': ['Transportation'],
                         'Textiles': ['Textiles'],
                         'Other': ['Mulch films', 'Rest of other']}


def subsector_to_sector(subsector):
    for sector, subsectors in sector_subsector_dict.items():
        if subsector in subsectors:
            return sector
    return 'Other'


plastics_sector_match_dict = {
    'Plastics|Production|Sector|Packaging': 'Packaging',
    'Plastics|Production|Sector|Buildings & Construction': 'Building and Construction',
    'Plastics|Production|Sector|Textiles': 'Textiles',
    'Plastics|Production|Sector|Other': 'Agriculture',
    'Plastics|Production|Sector|Consumer Products': 'Household items, furniture, leisure and others',
    'Plastics|Production|Sector|Transportation': 'Automotive',
    'Plastics|Production|Sector|Electrical & Electronic (products)': 'Electrical and Electronic Equipment',
}

agricultural_residue_list = ["barley_straw", "maize_stover", "rapeseed_straw", "rice_straw", "sorghum_straw",
                             "soybean_straw", "sugarcane_tops_and_leaves", "wheat_straw"]

residue_dict = dict(zip(residue_list, residue_list_code))

product_demand_dict = {'hdpe': 25, "ldpe": 25, "lldpe": 25,
                       "pp": 25, "pvc": 25,
                       "gpps": 25, "hips": 25,
                       "pet": 25, "pur_flexible": 25, "pur_rigid": 25,
                       "pla": 25, "phb": 25, "pbs": 25,
                       "ca": 25, "pef": 25, 'pbat': 25}  # Mt

product_demand_dict2 = {'hdpe': 1, "ldpe": 1, "lldpe": 1,
                        "pp": 1, "pvc": 1,
                        "gpps": 1, "hips": 1,
                        "pet": 1, "pur_flexible": 1, "pur_rigid": 1,
                        "pla": 1, "phb": 1, "pbs": 1,
                        "ca": 1, "pef": 1, 'pbat': 1}  # Mt

emissions_ghg_dict = {'co2_emission_biogenic_long': 0.38, #0.38,
                      'co2_emission_biogenic_short': 0.0001,
                      'co2_emission_fossil': 1,
                      'co2_storage_biogenic_long': -1,
                      'co2_storage_biogenic_short': -1,
                      'co2_storage_fossil': 0,
                      'n2o_emission': 273,
                      'ch4_emission': 27.9,
                      'nh3_emission': 0,
                      'co_emission': 0, 'pm25_emission': 0, 'sox_emission': 0, 'nmvoc_emission': 0, 'nox_emission': 0}

inorganic_chemical_list = ['sodium_hydroxide', 'potassium_sulfate', 'magnesium_oxide',
                           'sulfuric_acid', 'sodium_hypochlorite', "lime",
                           'sodium_phosphate', 'magnesium_sulfate', 'hydrochloric_acid', 'chlorine', 'sulfur',
                           'potassium_hydroxide', 'nitric_acid']

other_intermediate_list_up = ['bdo_14', 'acetic_acid', 'ammonia', 'butadiene', 'ammonium_sulfate',
                              'trimethylamine', 'hydrogen_peroxide', 'sodium_acetate',
                              'tetrahydrofuran', 'formaldehyde', 'cooling_water', 'nitrogen_liquid', 'oxygen_liquid',
                              'sulfur_dioxide', 'corn_steep_liquor', 'potato_starch', 'enzyme', 'syngas_1_to_1',
                              ]

other_intermediate_list_down = ['ethylene_glycol', 'ethylene_oxide', 'nitrobenzene',
                                'acetaldehyde', 'butene_1',
                                'polybutadiene', 'propylene_oxide',
                                'diethylene_glycol', 'acetic_anhydride', 'aniline',
                                'adipic_acid', 'phenol', 'dinitrotoluene',
                                'polymeric_mdi', 'toluene_diisocyanate',
                                'polyol_polyether', 'polyol_polyester'
                                ]

other_raw_material_list = ['sds', 'potato', 'maize_grain', 'yeast', 'glycerin', 'other_inorganic_chemicals',
                           ]

final_product_list = ['hdpe', 'ldpe', 'lldpe', 'pp', 'pvc', 'gpps', 'hips', 'pet', 'pur_flexible', 'pur_rigid',
                      'pla', 'phb', 'pbs', 'ca', 'pef', 'pbat']

emission_list = ['co2_emission_biogenic_long', 'co2_emission_biogenic_short', 'co2_emission_fossil', ]

intermediate0_list = ['agricultural_residue', ]

intermediate1_list = ['syngas_2_to_1', 'cooling_water', 'heat']

intermediate2_list = ['hydrogen', 'carbon_monoxide',
                      'steam']

intermediate3_list = ['syngas_1p8_to_1',
                      'methanol', 'glucose', 'cellulose']

intermediate4_list = ['ethanol', 'other_intermediates_upstream']

intermediate5_list = ['ethylene', 'propylene', 'benzene', 'toluene',
                      'p-xylene', 'lactic_acid', 'fdca']

intermediate6_list = ['terephthalic_acid', 'ethylbenzene', 'other_intermediates_downstream']

intermediate7_list = ['vinyl_chloride', 'styrene', ]

intermediate8_list = []

product_list_ordered = residue_list_code + other_raw_material_list + \
                       intermediate0_list + intermediate1_list + intermediate2_list + intermediate3_list + \
                       other_intermediate_list_up + \
                       intermediate4_list + intermediate5_list + intermediate6_list + other_intermediate_list_down + \
                       intermediate7_list + \
                       final_product_list + emission_list


def update_position_dict_single(list_to_process, x_position, x_position_dict, y_position_dict):
    y = 0.05
    y_increment = 0.9 / len(list_to_process)
    for item in list_to_process:
        x_position_dict[item] = x_position
        y_position_dict[item] = y
        y += y_increment


def position_dict_all():
    x_position_dict = {}
    y_position_dict = {}
    raw_material_list = residue_list_code + other_raw_material_list
    product_emission_list = final_product_list + emission_list
    update_position_dict_single(raw_material_list, 0.05, x_position_dict, y_position_dict)
    update_position_dict_single(intermediate0_list, 0.1, x_position_dict, y_position_dict)
    update_position_dict_single(intermediate1_list, 0.2, x_position_dict, y_position_dict)
    update_position_dict_single(intermediate2_list, 0.3, x_position_dict, y_position_dict)
    update_position_dict_single(intermediate3_list, 0.4, x_position_dict, y_position_dict)
    update_position_dict_single(intermediate4_list, 0.5, x_position_dict, y_position_dict)
    update_position_dict_single(intermediate5_list, 0.6, x_position_dict, y_position_dict)
    update_position_dict_single(intermediate6_list, 0.7, x_position_dict, y_position_dict)
    update_position_dict_single(intermediate7_list, 0.8, x_position_dict, y_position_dict)
    update_position_dict_single(product_emission_list, 0.95, x_position_dict, y_position_dict)
    return x_position_dict, y_position_dict


polymer_source_list1 = ['_co2', '_biogenic_short', '_biogenic_long', '_fossil']

polymer_source_list2 = ['_co2', '_biogenic_short', '_biogenic_long', '_fossil',
                        '_co2_mr', '_biogenic_short_mr', '_biogenic_long_mr', '_fossil_mr']


def df_pm_emission(master_file_path):
    df = pd.read_excel(master_file_path, engine='openpyxl', sheet_name='pm_emissions')
    return df