import pandas as pd

pd.options.mode.chained_assignment = None

# 0. COMMON FUNCTIONS ==================================================================================================

def get_table_from_sql(name,cursor):
    df = pd.read_sql_query(f"SELECT * from {name}", cursor)
    return df

def debugg(df, nombre):
    print('\n\n========================================================')
    print(f'DEBUGG DE {nombre}')
    print('\nSHAPE:')
    print(df.shape)
    print('\nIS NULL:')
    print(df.isnull().sum())
    print('\nCOLUMNAS:')
    print(df.columns)
    print('\nHEAD:')
    print(df.head())
    print('\nTIPOS:')
    print(df.info())


# === 1. VENTAS ===================================================================================================

def select_working_columns_ventas(ventas_df):
    print('selecting columns from ventas operadores df')
    ventas = ventas_df[['date', 'cod_carmila', 'grupo_rótulo', 'sub_grupo_rótulo',
                        'actividad', 'código_contrato','superficie_(m²)', 'cv_neta_n']]

    ventas.rename(columns={'código_contrato': 'cod_contrato',
                           'grupo_rótulo': 'grupo_rotulo',
                           'sub_grupo_rótulo': 'sub_grupo_rotulo',
                           'superficie_(m²)': 'metros_local'}, inplace=True)
    ventas = ventas.drop_duplicates(keep='first')
    ventas['date'] = pd.to_datetime(ventas['date'])

    return ventas


# === 2. ARRENDAMIENTO =================================================================================================

def select_working_columns_arredamientos(arrend_df):
    print('selecting and creating columns from estado arrendamiento')
    arrendamientos = arrend_df[['mes', 'cod_carmila', 'código_lote', 'uso_del_lote', 'código_contrato',
                                'cif/nif_inquilino', 'tipo_de_ocupación', 'fecha_ultimo_fin_cont.',
                                'renta_anual_s/iva', 'bonificaciones_mensuales_s/iva', 'gastos_anuales_s/iva']]

    arrendamientos.loc[:, 'renta_mes'] = ((arrendamientos['renta_anual_s/iva'] + arrendamientos['gastos_anuales_s/iva'])
                                          / 12) - arrendamientos['bonificaciones_mensuales_s/iva']

    arrendamientos = arrendamientos[arrendamientos['uso_del_lote'] == 'GALERIA']

    arrendamientos.drop(['renta_anual_s/iva', 'bonificaciones_mensuales_s/iva', 'gastos_anuales_s/iva'], axis=1,
                        inplace=True)

    arrendamientos.rename(columns={'mes': 'date',
                                   'superficie_referencia_lote_(m²)': 'metros_local',
                                   'código_contrato': 'cod_contrato',
                                   'cif/nif_inquilino': 'inquilino',
                                   'fecha_ultimo_fin_cont.': 'fecha_fin',
                                   'código_lote': 'cod_lote',
                                   'uso_del_lote': 'uso_lote',
                                   'tipo_de_ocupación': 'ocupacion'}, inplace=True)

    arrendamientos = arrendamientos.drop_duplicates(keep='first')
    arrendamientos['date'] = pd.to_datetime(arrendamientos['date'])

    return arrendamientos


# === 3 INFO CENTROS CC ================================================================================================

def helper_merge_info_hiper_w_cods_carmila(df_hiper, df_cod_carmila):
    df_hiper = df_hiper[['mes', 'cod_hip', 'operaciones', 'ventas']]
    df_codigos = df_cod_carmila[['cod_carmila', 'cod_hiper']]
    merged_df = df_hiper.merge(df_codigos, left_on='cod_hip', right_on='cod_hiper', how='left')
    merged_df = merged_df.drop('cod_hiper', axis=1)

    return merged_df


def helper_merge_info_hiper_w_info_cc(df_hiper_codigos, df_centros_carmila):
    print('Associating info from centers with info from cc..')
    merged = df_hiper_codigos.merge(df_centros_carmila, on='cod_carmila', how='right')
    merged.rename(columns={'mes': 'date'}, inplace=True)
    return merged


def helper_merge_info_centros_w_afluencias(df_info_hips_cc, df_afluencias):
    print('Adding afluencias info to df...')
    merged = df_info_hips_cc.merge(df_afluencias, on=['date', 'cod_carmila'], how='left')
    merged['afluencia'] = merged['afluencia'].fillna(0)
    merged = merged.drop_duplicates(keep='first')

    return merged


def combining_asset_info(df_hiper, df_cod_carmila, df_centros_carmila, df_afluencias):
    print('Creating info from Centros Comerciales')
    codes_w_hiper = helper_merge_info_hiper_w_cods_carmila(df_hiper, df_cod_carmila)
    hiper_w_centro = helper_merge_info_hiper_w_info_cc(codes_w_hiper, df_centros_carmila)
    centros_w_afluencias = helper_merge_info_centros_w_afluencias(hiper_w_centro, df_afluencias)
    centros_w_afluencias['date'] = pd.to_datetime(centros_w_afluencias['date'])

    return centros_w_afluencias


# === 4. COMBINATED: RAW FILE ==========================================================================================

def created_raw_combined_df(arrend, activos, ventas):
    df_arrend = select_working_columns_arredamientos(arrend)
    first_step_merged_df = pd.merge(left=df_arrend, right=activos, on=['date', 'cod_carmila'], how='left')
    df_ventas = select_working_columns_ventas(ventas)
    second_step_merged_df = pd.merge(left=first_step_merged_df, right=df_ventas,
                                     on=['date', 'cod_carmila', 'cod_contrato'], how='left')

    return second_step_merged_df



# === 5. COMBINATED: RAW FILE ==========================================================================================
def helper_create_evolution_series (df,column,periods):
    for n in range (1,periods+1):
        df[f'{column}-{n}'] = df.sort_values(by='date').groupby(['cod_contrato'])[[column]].transform(lambda x: x.shift(n).rolling(1).mean())
        df[f'evol_{column}_vs_{n}_periods'] = df.sort_values(by='date').groupby(['cod_contrato'])[[column]].pct_change(periods=n)

    return df

def helper_extract_number_contracts_by_month(df):
    count_number_nifs = df[['date','inquilino','cod_carmila']].groupby(['date', 'inquilino'])['cod_carmila'].count().reset_index()
    filter_no_vacios = count_number_nifs['inquilino'] != '-'
    count_number_nifs = count_number_nifs[filter_no_vacios]
    count_number_nifs.rename(columns={'cod_carmila': 'num_contracts'}, inplace=True)
    df = df.merge(count_number_nifs, on=['date', 'inquilino'], how='left')
    return df

def wrangling_on_raw_df (df):
    df = df.drop_duplicates(keep='first')
    df['long'].fillna(-3.285, inplace=True)
    print('cambiamos long')
    df[['grupo_rotulo','sub_grupo_rotulo','actividad']].fillna('-', inplace=True)
    print('cambiamos actividades en vacios por -')

    # Ventas 3 Meses Anteriores y evolución
    df = helper_create_evolution_series(df, 'cv_neta_n', 3)
    print('añadimos evoluion 3 meses')

    # Calcula Tasa Esfuerzo, las tasas en meses anteriores y su evolución
    df.loc[:, 'tasa_esfuerzo_n'] = df['cv_neta_n'] / df['renta_mes']
    print('añadimos tasa esfuerzo')
    df = helper_create_evolution_series(df, 'tasa_esfuerzo_n', 3)
    print('añadimos evolución tasa esfuerzo')

    df = helper_extract_number_contracts_by_month(df)
    print('añadimos num contrato')

    df['sales_m2'] = df['cv_neta_n'] / df['metros_local']
    print('sales m2')
    df['renta_m2'] = df['renta_mes'] / df['metros_local']
    print('renta m2')


    return df









