import pandas as pd


def iam_data_preparation():
    scenario_list = ["R2p1_SSP2-PkBudg900", 'R2p1_SSP2-Base', "SSP2_SPA2_19I_D",
                     "SSP2_SPA2_19I_RE", "SSP2_SPA2_19I_LI", "SSP2_SPA2_19I_LIRE", "SSP2-baseline"]
    variable_list = ["Secondary Energy|Electricity", "Secondary Energy|Electricity|Biomass",
                     "Secondary Energy|Electricity|Biomass",
                     'Secondary Energy|Electricity|Geothermal', 'Secondary Energy|Electricity|Hydro',
                     'Secondary Energy|Electricity|Non-Biomass Renewables', 'Secondary Energy|Electricity|Nuclear',
                     'Secondary Energy|Electricity|Solar',
                     'Secondary Energy|Electricity|Wind', 'Secondary Energy|Electricity|Solar|PV',
                     'Secondary Energy|Electricity|Solar|CSP',
                     'Final Energy|Industry|Electricity', 'Final Energy|Industry|Chemicals|Electricity',
                     'Production|Chemicals',
                     "Carbon Sequestration|CCS|Biomass",
                     "Carbon Sequestration|CCS|Fossil"]
    dfg = pd.read_csv('data/external/iam/AR6_Scenarios_Database_World_v1.1.csv')
    dfr = pd.read_csv('data/external/iam/AR6_Scenarios_Database_R10_regions_v1.1.csv')
    dfc = pd.read_csv('data/external/iam/AR6_Scenarios_Database_ISO3_v1.1.csv')
    df_meta = pd.read_excel('data/external/iam/AR6_Scenarios_Database_metadata_indicators_v1.1.xlsx',
                            sheet_name='meta_Ch3vetted_withclimate')
    dfc1 = dfc[dfc.Variable.isin(variable_list)].copy()
    dfr1 = dfr[dfr.Variable.isin(variable_list)].copy()
    dfg1 = dfg[dfg.Variable.isin(variable_list)].copy()
    df_meta_imp = df_meta.loc[(df_meta.IMP_marker != 'non-IMP'), ['Model', 'Scenario', 'Category', 'IMP_marker']].copy()
    df_meta_to_use = df_meta.loc[df_meta.Scenario.isin(scenario_list), ['Model', 'Scenario', 'Category']].copy()
    df_ld = dfg[dfg.Scenario == 'LowEnergyDemand_1.3_IPCC'].copy()
    df_ld_ele1 = df_ld[df_ld['Variable'].str.contains(r'^Secondary Energy\|Electricity')].copy()
    df_ld_ele = df_ld_ele1[(df_ld_ele1['Variable'] == 'Secondary Energy|Electricity|Fossil') |
                           (df_ld_ele1['Variable'] == 'Secondary Energy|Electricity|Renewables (incl. Biomass)')].copy()
    df_ld_ele['Variable'] = 'Secondary Energy|Electricity'
    df_ld_ele = df_ld_ele.groupby(['Model', 'Scenario', 'Region', 'Variable', 'Unit']).sum().reset_index()
    dfg1 = pd.concat([dfg1, df_ld_ele], axis=0)
    df_ld_ccs = df_ld_ele.copy()
    df_ld_ccs['Variable'] = 'Carbon Sequestration|CCS|Fossil'
    df_ld_ccs.iloc[:, 5:] = 0
    dfg1 = pd.concat([dfg1, df_ld_ccs], axis=0)
    df_imp = pd.DataFrame()
    df_to_use = pd.DataFrame()
    for df_temp in [dfg1, dfr1, dfc1]:
        df_temp2 = pd.merge(df_meta_imp, df_temp, on=['Model', 'Scenario'], how='right')
        df_temp2.dropna(subset=['IMP_marker'], inplace=True)
        df_imp = pd.concat([df_imp, df_temp2], axis=0)
        df_temp3 = df_temp[df_temp.Scenario.isin(scenario_list)].copy()
        df_temp3 = pd.merge(df_meta_to_use, df_temp3, on=['Model', 'Scenario'], how='right')
        df_to_use = pd.concat([df_to_use, df_temp3], axis=0)
    df = pd.concat([df_imp, df_to_use], axis=0, ignore_index=True)
    df = df.loc[:, ['Model', 'Scenario', 'Category', 'IMP_marker',
                    'Region', 'Variable', 'Unit',
                    '2010', '2020', '2030', '2040', '2050']].copy()
    df_ele = df[df.Variable == 'Secondary Energy|Electricity'].copy()
    region_dict = {'CN': ['CHN', 'R10CHINA+'],
                   'IN': ['IND', 'R10INDIA+'],
                   'EU': ['EU', 'R10EUROPE'],
                   'RNA': ['USA', 'CAN', 'R10NORTH_AM'],
                   'RU': ['RUS', 'R10REF_ECON'],
                   'JP': ['JPN', 'R10PAC_OECD']}
    for region_list in region_dict.values():
        df_ele_r = df_ele[df_ele.Region.isin(region_list)].copy()
        for scenario in df_ele_r.Scenario.unique():
            df_temp = df_ele_r[df_ele_r.Scenario == scenario].copy()
            if df_temp.shape[0] > 1:
                val_1 = df_temp.loc[df_temp.Region == region_list[-1], '2050'].values[0]
                val_2 = df_temp.loc[df_temp.Region != region_list[-1], '2050'].values.sum()
                if abs(val_1 - val_2) < 1e-3:
                    df.drop(df[(df.Scenario == scenario) & (df.Region == region_list[-1])].index, inplace=True)
                else:
                    print('Check the data for scenario: ', scenario, ' and region: ', region_list)
                    # df.drop(df[(df.Scenario == scenario) & (df.Region == region_list[1])].index, inplace=True)
    # df_ele = df[df.Variable == 'Secondary Energy|Electricity'].copy()
    # df_ccs_bio = df[df.Variable == 'Carbon Sequestration|CCS|Biomass'].copy()
    # df_ccs_fos = df[df.Variable == 'Carbon Sequestration|CCS|Fossil'].copy()
    # df_temp = pd.merge(df_ele, df_ccs_bio, on=['Model', 'Scenario', 'Region', 'Category', 'IMP_marker'], how='right')
    df.to_csv('data/intermediate/iam_scenarios.csv', index=False)
    return df
