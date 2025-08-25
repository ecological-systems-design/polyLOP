import pandas as pd

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

final_product_list = ['hdpe', 'ldpe', 'lldpe', 'pp', 'pvc', 'gpps', 'hips', 'pet', 'pur_flexible', 'pur_rigid',
                      'pla', 'phb', 'pbs', 'ca', 'pef', 'pbat']


def df_pm_emission(master_file_path):
    df = pd.read_excel(master_file_path, engine='openpyxl', sheet_name='pm_emissions')
    return df