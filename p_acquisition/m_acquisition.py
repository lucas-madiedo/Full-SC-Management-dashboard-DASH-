# Import required libraries
import pandas as pd
import time
import os
import sqlite3

# Import required info
from p_acquisition import variables

# setting some options
pd.options.mode.chained_assignment = None


# ==== COMMON FUNCTIONS ================================================================================================

def general_date_format(x):
    ''' Cambia un str de formato mm.yyyy a formato dd/mm/yyyy'''
    x = (x.encode('ascii', 'ignore')).decode("utf-8")
    x = x.replace('.', '')
    month, year = x.split(' ')
    months_dict = {'Ene': 1, 'Febr': 2, 'Mar': 3, 'Abr': 4, 'Mayo': 5,
                   'Jun': 6, 'Jul': 7, 'Ago': 8, 'Sep': 9, 'Oct': 10,
                   'Nov': 11, 'Dic': 12}
    month = months_dict[month]
    return f'01/{month}/{year}'


def general_serie2date(serie):
    '''Recibe una serie de pandas en str (dd/mm/yyyy) y la pasa a timestamp'''
    serie = serie.apply(lambda x: general_date_format(x))
    serie = pd.to_datetime(serie, format='%d/%m/%Y')
    return serie


def general_extract_cod_carmila(serie):
    '''recibe una serie y devulve los primeros 5 caracteres de cada columna en str'''
    return serie.apply(lambda x: x[:5]).astype('str')


def general_columns_to_category(df, columns):
    '''recibe un df y una lista de columnas que queremos pasar a formato: Pandas Category'''
    for column in columns:
        try:
            df[column] = df[column].apply(lambda x: round(x)).astype('str')
        except:
            df[column] = df[column].astype('str')
    return df


def general_clean_column_names(df):
    '''recibe un dataframe y pasa todas las columnas a minusculas sin espacios'''
    new_name_columns = [column.replace(' ', '_').lower() for column in df.columns.to_list()]
    df.columns = new_name_columns
    return df


def general_combine_files(path, function_to_apply, extension='xlsx', name_of_df=''):
    '''Recibe una carpeta y aplica una función de limpiado a todos los archivos del tipo especificado.
    Combina el resultado final en un único DF'''

    files = [f'{path}/{filename}' for filename in os.listdir(path) if filename.endswith(extension)]

    # Creamos un df vacío y lo llenamos con cada interación  de documentos
    df = pd.DataFrame()
    for n, file in enumerate(files):
        temp_df = function_to_apply(file)
        df = df.append(temp_df)
        print(f'\t - {n + 1} of {len(files)} files processed. Size of {name_of_df} df:{df.shape}')

    duplicados = df.duplicated(subset=None, keep='first')
    df = df[-duplicados].reset_index(drop=True)
    return df

def general_export_to_parquet(df,path_root,name):
    df.to_parquet(f'{path_root}{name}.parquet', index=False)

def general_export_to_csv(df, path_root, name):
    df.to_csv(f'{path_root}{name}.csv', index=False)


# === 1.  INFO CENTROS  ================================================================================================

class NewDataframe():
    def __init__(self, path):
        self.path = path

    def get_sheets(self):
        excel_file = pd.ExcelFile(self.path)
        return excel_file.sheet_names

    def get_cols(self, sheet):
        df = pd.read_excel(self.path, sheet)
        return list(df.columns)

    def create_df(self, sheet, columns):
        lista_columnas = self.get_cols(sheet)
        col_numbers = [lista_columnas.index(col) for col in columns]
        df = pd.read_excel(self.path, sheet_name=sheet, usecols=col_numbers)
        return df


def create_df_centros_carmila(path):
    excel_carmila = NewDataframe(path)

    # excel_carmila.get_sheets()
    sheet_name = 'INFO AMPLIADA'

    # excel_carmila.get_cols('INFO AMPLIADA')
    colums_to_use = ['COD', 'COD SAP', 'COD ATICA', 'CENTRO', 'Nom CRF', 'DIRECCIÓN',
                     'PROVINCIA', 'COM.AUTÓNOMA', 'REGIÓN', 'LATITUD', 'LONGITUD', 'TIPO', 'AÑO APERTURA',
                     'SBA TOTAL', 'SBA GALERíA', 'SBA HIPERMERCADO', 'Nº Locales',
                     'PLANTAS', 'PLAZAS PARKING', 'ÁREA de INFLUENCIA', 'VISITAS AÑO']

    centros_df = excel_carmila.create_df(sheet_name, colums_to_use)

    # Rellenamos n valor NAN por 0
    centros_df['ÁREA de INFLUENCIA'].fillna(value=0, inplace=True)

    # Formateamos correctamente las coordenadas del centro.
    centros_df['LATITUD'] = centros_df['LATITUD'].str.replace(',', '.').astype('float')
    centros_df['LONGITUD'] = centros_df['LONGITUD'].str.replace(',', '.').astype('float')
    centros_df.rename(columns={'LATITUD': 'lat',
                               'LONGITUD': 'long'}, inplace=True)

    # Pasamos columnas a categoria
    centros_df = general_columns_to_category(centros_df, ['COD', 'COD SAP', 'COD ATICA',
                                                          'CENTRO', 'Nom CRF', 'DIRECCIÓN',
                                                          'PROVINCIA', 'COM.AUTÓNOMA', 'REGIÓN',
                                                          'TIPO'])

    centros_df.rename(columns={'COD': 'cod_carmila'}, inplace=True)
    centros_df = general_clean_column_names(centros_df)
    centros_df = centros_df.drop_duplicates(keep='first')

    # METER LONG DE 39062 a calzador <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # SENDING PROMP INFO
    print('\nCREATING CENTROS DF...')
    print('CENTROS DF SUCCESFULLY CREATED')
    print('------------------------------------')
    print(centros_df.info())
    print(centros_df.shape)
    print('------------------------------------\n\n')

    return centros_df


# === 2.  TABLA CÓDIGOS  ===============================================================================================

def create_df_codigos_centros(path):
    cods_centros = NewDataframe(path)
    sheet_name = 'Códigos Centros'

    columns = cods_centros.get_cols(sheet_name)
    codigos_df = cods_centros.create_df(sheet_name, columns)

    codigos_df = general_columns_to_category(codigos_df, codigos_df.columns.to_list())
    codigos_df.rename(columns={'COD CARMILA': 'cod_carmila'}, inplace=True)
    codigos_df = general_clean_column_names(codigos_df)

    # SENDING PROMP INFO
    print('\nCREATING CODIGOS DF...')
    print('CODIGOS DF SUCCESFULLY CREATED')
    print('------------------------------------')
    print(codigos_df.info())
    print(codigos_df.shape)
    print('------------------------------------\n\n')

    return codigos_df


# === 3.  TABLA FACTURACIÓN  ===========================================================================================


def clean_file_facturacion(path):
    fact_df = pd.read_excel(path)

    # Eliminamos columna Mes y pasamos código contrato a str (usar como key)
    fact_df.drop('Mes', axis=1, inplace=True)
    fact_df['Código contrato'] = fact_df['Código contrato'].astype('str')

    # Melting el df para tener filas por mes en lugar de por contrato.
    var_names = ['Centro', 'Código sub grupo rótulo', 'Sub grupo rótulo',
                 'Código actividad', 'Actividad', 'Código contrato']

    fact_df = fact_df.melt(id_vars=var_names, value_name='Facturación', var_name='Fecha')

    # Aplicamos función a la columna fecha para convertirla en tipo correcto
    fact_df['Fecha'] = general_serie2date(fact_df['Fecha'])

    # generamos Columna con codigo Carmila
    fact_df['cod_carmila'] = general_extract_cod_carmila(fact_df['Código contrato'])

    # Reorden de las columnas y orden
    column_order = ['Fecha', 'cod_carmila', 'Centro', 'Sub grupo rótulo', 'Código sub grupo rótulo',
                    'Actividad', 'Código actividad', 'Código contrato',
                    'Facturación']

    fact_df = fact_df[column_order].sort_values(by='Fecha', ascending=False)
    fact_df.reset_index(drop=True, inplace=True)

    # Sustituimos - por 0  para pasar a categoría.
    fact_df['Facturación'] = fact_df['Facturación'].replace('-', 0)

    # pasamos columnas a categorías:
    columns_to_cats = ['cod_carmila', 'Centro', 'Sub grupo rótulo', 'Código sub grupo rótulo',
                       'Actividad', 'Código actividad', 'Código contrato']

    fact_df = general_columns_to_category(fact_df, columns_to_cats)
    fact_df = general_clean_column_names(fact_df)

    return fact_df


def create_df_facturacion(path):
    print('\nCreating Facturación DF...')
    df = general_combine_files(path, clean_file_facturacion, 'xlsx', 'facturacion')
    print('FACTURACION DF SUCCESFULLY CREATED')
    print('------------------------------------')
    print(df.info())
    print(df.shape)
    print(f'Date Range: FROM: {df["fecha"].min()} TO: {df["fecha"].max()}')

    print('------------------------------------\n\n')

    return df


# === 4. ESTADO ARRENDAMIENTO  =========================================================================================

def helper_renta_variable_cleaning(x):
    # NO ESTÁ EN USO
    '''limpia la cifra y la pasa a float.
    Revisar. Hay celdas sin el número y valores de texto.'''
    if 'del CV' in x:
        cifra, text = x.split(' %')
        cifra = cifra.replace(',', '.')
        cifra = round(float(cifra) / 100, 3)
        return cifra

    else:
        '''De momento paso a 0. Revisar con Breva'''
        return 0


def clean_file_estado_arrend(file):
    temp_df = pd.read_excel(file)
    temp_df = temp_df.loc[1:, :]

    # Aplicamos la función fecha para formatear columna Mes
    temp_df['Mes'] = general_serie2date(temp_df['Mes'])

    # Pasamos Código de Lote y Código de Contrato a objeto.
    temp_df['Código lote'] = temp_df['Código lote'].apply(lambda x: round(x)).astype('str')

    # Extraemos renta Variable según Función --- ESTE PASO NO SE ESTÁ EJECUTANDO. INFORMACIÓN ERRÓNEA
    # temp_df['% renta variable'] = temp_df['% renta variable'].apply(lambda x : helper_renta_variable_cleaning(x))
    temp_df['% renta variable'] = temp_df['% renta variable'].astype('str')

    # generamos Columna con codigo Carmila
    temp_df['cod_carmila'] = general_extract_cod_carmila(temp_df['Código lote'])

    # Filtro Solo Centros Carmila
    filtro_centros_carmila = temp_df['cod_carmila'].isin(variables.CODIGOS_CARMILA)
    temp_df = temp_df[filtro_centros_carmila].reset_index(drop=True)

    return temp_df


def create_df_estado_arrend(path):
    print('\nCreating Estado Arrendamiento DF...')

    arrend_files = [f'{path}/{filename}' for filename in os.listdir(path) if filename.endswith('.xlsx')]

    # Creamos un df vacío y lo llenamos con cada interación  de documentos
    arrend_df = pd.DataFrame()
    for n, file in enumerate(arrend_files):
        temp_df = clean_file_estado_arrend(file)
        arrend_df = arrend_df.append(temp_df)
        print(f'\t- {n + 1} of {len(arrend_files)} files processed. Size Estado Arrendamiento df:{arrend_df.shape}')

    # Eliminamos valores vacíos
    # filter_time = arrend_df['Mes'] < '2020/10/01'
    # arrend_df = arrend_df[filter_time].sort_values(by='Mes', ascending=False)

    arrend_df.drop_duplicates(keep='first', inplace=True)

    # Objet columns to categories
    columns_to_category = ['Código inmueble', 'Centro comercial', 'n° lotes', 'Código lote',
                           'Uso del lote', 'Código contrato', 'Inquilino', 'Rótulo', 'Actividad',
                           'CIF/NIF inquilino', 'Tipo de ocupación', 'Grupo rótulo', 'Sub grupo rótulo',
                           'N° Local', 'cod_carmila']

    arrend_df = general_columns_to_category(arrend_df, columns_to_category)

    # fecha fin contrato a formato fecha
    arrend_df['Fecha ultimo fin cont.'] = arrend_df[
        'Fecha ultimo fin cont.'].apply(lambda x: None if x == '-' else x)
    arrend_df['Fecha ultimo fin cont.'] = pd.to_datetime(
        arrend_df['Fecha ultimo fin cont.'], format='%d/%m/%Y')

    # formato en minusculas y sin espacios
    arrend_df = general_clean_column_names(arrend_df)

    print('ESTADO ARRENDAMIENTO DF SUCCESFULLY CREATED')
    print('------------------------------------')
    print(arrend_df.info())
    print(arrend_df.shape)
    print(f'Date Range: FROM: {arrend_df["mes"].min()} TO: {arrend_df["mes"].max()}')
    print('------------------------------------\n\n')

    return arrend_df


# === 5. INFO HIPER  ===================================================================================================
def helper_month_to_num(x):
    dict_mes = {"enero": 1, "febrero": 2, "marzo": 3,
                "abril": 4, "mayo": 5, "junio": 6,
                "julio": 7, "agosto": 8, "septiembre": 9,
                "octubre": 10, "noviembre": 11, "diciembre": 12}

    mes, year = str(x).split(' ')
    mes = dict_mes[mes.lower()]
    return f'01/{mes}/{year}'


def clean_file_info_hiper(file):
    df = pd.read_excel(file, sheet_name='Hipers')
    fecha_mes_file = helper_month_to_num(df.iloc[9, 9])

    # seleccionamos y renombreamos las columnas que nos interesan
    cols_to_use = ['Unnamed: 0', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 9', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 35',
                   'Unnamed: 36']
    clean_df = df[cols_to_use]

    clean_df.rename(columns={'Unnamed: 0': 'ciudad', 'Unnamed: 4': 'cod_hip', 'Unnamed: 5': 'tipo_activo',
                             'Unnamed: 9': 'hipermercado', 'Unnamed: 13': 'ventas', 'Unnamed: 14': 'ventas_n-1',
                             'Unnamed: 35': 'operaciones', 'Unnamed: 36': 'operaciones_n-1'}, inplace=True)

    # Eliminamos lo que no sean hipermercados
    clean_df = clean_df[clean_df['tipo_activo'] == 'HIPERMERCADOS']

    # añadimos una columna mes y la pasamos a formato fecha
    clean_df['mes'] = fecha_mes_file
    clean_df['mes'] = pd.to_datetime(clean_df['mes'], format='%d/%m/%Y')

    # creamos una columna con la fecha del mes del año pasado
    clean_df['mes-1'] = clean_df['mes'] - pd.DateOffset(years=1)

    # creamos dos df diferentes. Uno para este mes y otro con las cifreas del mes pasado.
    current_year = clean_df[['mes', 'cod_hip', 'hipermercado', 'ventas', 'operaciones']]
    last_year = clean_df[['mes-1', 'cod_hip', 'hipermercado', 'ventas_n-1', 'operaciones_n-1']]

    # cambiamos el nombre de las columnas del mes pasado para que al unir quede un dataframe limpio
    last_year.rename(columns={'mes-1': 'mes', 'ventas_n-1': 'ventas', 'operaciones_n-1': 'operaciones'}, inplace=True)

    # unimos el df del mes actual con el del mismo mes del año pasado
    final_df = current_year.append(last_year)

    final_df['ventas'] = final_df['ventas'].astype('float')*1000
    final_df['operaciones'] = final_df['operaciones'].astype('float')*1000

    final_df = final_df[['mes', 'cod_hip', 'ventas', 'operaciones']]
    final_df = final_df[final_df['cod_hip'].isin(variables.HIPERS_CARMILA)]

    # devolvemos el df resultante.
    return final_df


def create_df_info_hiper(path):
    print('\nCreating Info Hiper DF...')
    df = general_combine_files(path, clean_file_info_hiper, 'xlsx', 'Info Hiper')

    # Quitamos duplicados. (2019 está en 2020 n-1 y en 2019)
    duplicados = df.duplicated(subset=None, keep='first')
    df = df[-duplicados].reset_index(drop=True)
    df = df.sort_values(by='mes').reset_index(drop=True)

    print('INFO HIPER DF SUCCESFULLY CREATED')
    print('------------------------------------')
    print(df.info())
    print(df.shape)
    print(f'Date Range: FROM: {df["mes"].min()} TO: {df["mes"].max()}')

    print('------------------------------------\n\n')

    return df


# === 6. AFLUENCIAS  ===================================================================================================
def clean_file_afluencias(path, delimiter=';'):
    df = pd.read_csv(path, delimiter=delimiter)

    # Agrupamos para eliminar la información de los accesos
    df = df.groupby(['fecha', 'centro']).sum().reset_index()

    # Pasamos la fecha a formato Datetime
    df['fecha'] = pd.to_datetime(df['fecha'])
    # Creamos categorias por mes y año
    df['mes'] = pd.DatetimeIndex(df['fecha']).month
    df['year'] = pd.DatetimeIndex(df['fecha']).year

    # Pasamos codigo de centro a categoría
    df['centro'] = df['centro'].astype('str')

    # Agrupamos la infromación por año,mes y centro
    df = df.groupby(['year', 'mes', 'centro']).agg({'afluencia': 'sum'}).reset_index()

    # Renombramos centro por cod_carmila
    df.rename(columns={'centro': 'cod_carmila'}, inplace=True)

    # Creamos columna con la fecha en str
    df['date'] = '1/' + df['mes'].astype(str) + '/' + df['year'].astype(str)
    # pasamos date a datetime
    df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y")
    # dejamos olo columnas relevantes
    df = df[['date', 'cod_carmila', 'afluencia']]

    return df


def create_df_afluencias(path):
    print('\nCreating AFLUENCIAS DF...')
    df = general_combine_files(path, clean_file_afluencias, 'csv', 'afluencias')
    print('AFLUENCIAS DF SUCCESFULLY CREATED')
    print('------------------------------------')
    print(df.info())
    print(df.shape)
    print(f'Date Range: FROM: {df["date"].min()} TO: {df["date"].max()}')
    print('------------------------------------\n\n')

    return df


# === 7. VENTAS OPS  ===================================================================================================
def helper_extract_last_year(df):
    ''' crea dos df, resta un año a uno y elimina el valor de n dejando solo n-1 : Df un año anterior.
    Se elimina valor n-1 en el df año corriente y se fusionan los dos. Se quitan duplicados '''

    current_df = df.copy()
    df_last_year = df.copy()
    df_last_year['date'] = df_last_year['date'] - pd.DateOffset(years=1)
    df_last_year.drop('cv_neta_n', axis=1, inplace=True)
    df_last_year.rename(columns={'cv_neta_n-1': 'cv_neta_n'}, inplace=True)

    current_df.drop('cv_neta_n-1', axis=1, inplace=True)

    full_ventas = df_last_year.append(current_df)

    full_ventas.drop_duplicates(keep='first', inplace=True)
    full_ventas = full_ventas[full_ventas['cv_neta_n'] != 0]

    return full_ventas


def clean_file_ventas(file):
    df = pd.read_excel(file)
    df = df.loc[1:, :]

    # Incluimos columna mes con formato de fecha
    df.insert(loc=0, column='date', value=general_serie2date(df['Mes']))

    # eliminamos columnas extra
    df.drop(['Mes', 'N° contratos con CV', 'CV Mensual / m²',
             'Evol. CV Neta N/N-1 Total', 'Evol. CV Neta N/N-1 Comp..'], axis=1, inplace=True)

    # Creamos columnas cod_carmila y pasamos codigo de contrato a str.
    df['Código contrato'] = df['Código contrato'].astype(str).str[:-2]
    df.insert(loc=1, column='cod_carmila', value=df['Código contrato'].str[:5])

    # Columns to Categories
    cols_to_cat = ['Centro', 'Código grupo rótulo', 'Grupo rótulo', 'Código sub grupo rótulo',
                   'Sub grupo rótulo', 'Código actividad', 'Actividad', 'Rótulo']
    df = general_columns_to_category(df, cols_to_cat)

    # Aplicamos un filtro (con base la tabla de codigos) para eliminar los registros que no pertenecesn a centos carmila
    filtro_centros_carmila = df['cod_carmila'].isin(variables.CODIGOS_CARMILA)
    df = df[filtro_centros_carmila].reset_index(drop=True)

    # Se eliminan los espacios en blanco y se pasan a minúscula todos los nombres de las columnas.
    df = general_clean_column_names(df)
    df = helper_extract_last_year(df)
    return df


def create_df_ventas(path):
    print('\nCREATING VENTAS DF...')
    df = general_combine_files(path, clean_file_ventas, 'xlsx', 'ventas')
    print('VENTAS DF SUCCESFULLY CREATED')
    print('------------------------------------')
    print(df.info())
    print(df.shape)
    print(f'Date Range: FROM: {df["date"].min()} TO: {df["date"].max()}')
    print('------------------------------------\n\n')

    return df


# ======================================================================================================================
# ========================================= CREATING DATA FRAMES  ======================================================
# ======================================================================================================================

def general_saving_to_sql(cursor,df,name):
    df.to_sql(name,cursor,if_exists='replace',index=False,)




# CODIGOS_CENTROS_CARMILA = CODIGOS_CARMILA

# RUTAS DE LOS ARCHIVOS EN BRUTO
PATH_EXCEL_CENTROS_CARMILA = 'data/raw/info_centros/cc_carmila.xlsx'
PATH_CARPETA_FACTURACION = 'data/raw/facturacion/'
PATH_EST_ARRENDAMIENTO = 'data/raw/estado_arrendamiento/'
PATH_INFO_HIPER = 'data/raw/hiper/'
PATH_AFLUENCIAS = 'data/raw/afluencias/'
PATH_VENTAS_OPERADORES = 'data/raw/cv/'

#Export:
# EJECUCIÓN DE LAS FUNCIONES


def creating_ddbb(conn):
    print('INITIALIZING DATA ACQUISITION PIPELINE')

    df_centros_carmila = create_df_centros_carmila(PATH_EXCEL_CENTROS_CARMILA)
    general_saving_to_sql(conn,df_centros_carmila,'centros_carmila')

    df_codigos_centros = create_df_codigos_centros(PATH_EXCEL_CENTROS_CARMILA)
    general_saving_to_sql(conn,df_codigos_centros,'codigos_centros')

    df_facturacion = create_df_facturacion(PATH_CARPETA_FACTURACION)
    general_saving_to_sql(conn,df_facturacion,'tabla_facturacion')

    # todo: Incluir nuevas fechas de contrato al df
    df_est_arrendamiento = create_df_estado_arrend(PATH_EST_ARRENDAMIENTO)
    general_saving_to_sql(conn,df_est_arrendamiento,'estado_arrend')

    df_info_hiper = create_df_info_hiper(PATH_INFO_HIPER)
    general_saving_to_sql(conn,df_info_hiper,'info_hiper')

    df_afluencias = create_df_afluencias (PATH_AFLUENCIAS)
    general_saving_to_sql(conn,df_afluencias,'df_afluencias')

    df_ventas_operadores = create_df_ventas(PATH_VENTAS_OPERADORES)
    general_saving_to_sql(conn,df_ventas_operadores,'ventas_operadores')
