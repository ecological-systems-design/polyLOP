import pandas as pd


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

def get_plastics_production():
    file_path_2 = r'data/raw/bioplastics_vs_fossil.xlsx'
    df_plastics_share = read_plastics_demand(file_path_2)[1].copy()
    df_plastics_demand = read_plastics_demand(file_path_2)[0].copy()
    if country in df_plastics_demand['Region'].unique():
        df_plastics_demand = df_plastics_demand[df_plastics_demand['Region'] == country].copy()
    else:
        print('country does not exist in the plastics demand data, or if it is a region, to be developed')