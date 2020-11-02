# Import required libraries
import pickle
import pathlib
import dash
import dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html
import numpy as np

# IMPORT FILES
from models import m_churn_model

# IMPORTING MODEL
WORKING_FEATURES = m_churn_model.WORKING_FEATURES
MODEL_PATH = 'models/model_churn_risk_indicator.sav'
churn_model = pickle.load(open(MODEL_PATH, 'rb'))

# ===DASH INIT =========================================================================================================

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server

# === IMPORTING DATA TO DASH ===========================================================================================

df = pd.read_parquet('data/final/full_db.parquet')  # full data set
df_model = m_churn_model.make_churn_predictions(df.tail(3))  # Necesaria una carga previa para la tabla.

# === CREATE CONTROL OPTIONS ===========================================================================================


CENTER_OPTIONS = [{'label': i, 'value': i} for i in df['cod_carmila'].unique()]
CATEGORIES_OPTIONS = [{'label': i, 'value': i} for i in df['grupo_rotulo'].unique() if i != None]
ddf_tipos = df[['cod_carmila', 'tipo']].drop_duplicates(keep='first')

LISTADO_GALERIAS = ddf_tipos[ddf_tipos['tipo'] == 'GALERIA']['cod_carmila'].to_list()
LISTADO_GERENCIAS = ddf_tipos[ddf_tipos['tipo'] == 'GERENCIA']['cod_carmila'].to_list()
LISTADO_CENTROS = LISTADO_GERENCIAS + LISTADO_GALERIAS
LISTADO_CATEGORIAS = [i for i in df['grupo_rotulo'].unique() if i != None]

first_date_in_data = pd.DatetimeIndex(df['date']).min().date()
last_date_in_data = pd.DatetimeIndex(df['date']).max().date()

# ======================================================================================================================
# ========================================= DASHBOARD TEMPLATE =========================================================
# ======================================================================================================================


mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A"

layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(l=30, r=30, b=20, t=40),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation="h"),
    title="Satellite Overview",
    mapbox=dict(
        accesstoken=mapbox_access_token,
        style="light",
        center=dict(lon=-78.05, lat=42.54),
        zoom=7,
    ),
)

# Create app layout
app.layout = html.Div(
    [
        dcc.Store(id="aggregate_data"),
        html.Div(id="output-clientside"),
        dcc.Store(id='intermediate-value'),  # div invisible para almacenar la info filtrada.
        dcc.Store(id='model-value'),

        html.Div([html.H2("INFORMACIÓN CENTROS COMERCIALES", style={'background-color': '#1c485d ',
                                                                    'text-align': 'center',
                                                                    'color': 'white'},
                          className="pretty_container twelve columns",
                          )], className="row flex-display"),

        # ROW 1 ********************************************************************************************************
        # CONTROLLERS AND SELECTORS ====================================================================================

        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            "Selecciona el período de los datos",
                            className="control_label",
                        ),
                        dcc.DatePickerRange(
                            id='rango_fechas',
                            month_format='MMMM - YYYY',
                            start_date='2018-01-01',
                            end_date='2019-12-31',
                            min_date_allowed=first_date_in_data,
                            max_date_allowed=last_date_in_data,
                            display_format='DD / MMM / YYYY',
                            className="dcc_control",
                            minimum_nights=30,
                            first_day_of_week=1,
                            with_portal=False,
                            number_of_months_shown=3,
                        ),
                        html.P("Centros por Categoría:", className="control_label"),
                        dcc.RadioItems(
                            id="radio_tipo_centro",
                            options=[
                                {"label": "Todos", "value": "Todos"},
                                {"label": "Gerencias", "value": "GERENCIAS"},
                                {"label": "Galerías", "value": "GALERIAS"},
                            ],
                            value="GERENCIAS",
                            labelStyle={"display": "inline-block"},
                            className="dcc_control",
                        ),
                        dcc.Dropdown(
                            id="selector_centros",
                            options=[],
                            multi=True,
                            value=[],
                            className="dcc_control",
                        ),
                        html.P("Selecciona Actividad:", className="control_label"),
                        dcc.Dropdown(
                            id="selector_categoria",
                            options=CATEGORIES_OPTIONS,
                            multi=True,
                            value=LISTADO_CATEGORIAS,
                            className="dcc_control",
                        ),
                        html.P("Selecciona Operador:", className="control_label"),
                        dcc.RadioItems(
                            id="radio_selector_operador",
                            options=[
                                {"label": "Todos", "value": "Todos"},
                                {"label": "Seleccionar", "value": "Seleccionar"},
                            ],
                            value="Todos",
                            labelStyle={"display": "inline-block"},
                            className="dcc_control"),

                        dcc.Dropdown(
                            id="selector_operador",
                            options=[],
                            multi=True,
                            value=[],
                            className="dcc_control", )

                    ],
                    className="pretty_container four columns",
                    id="cross-filter-options",
                ),

                # PLOTS AND TABLES =====================================================================================

                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [html.H6(id="num_centros"), html.P("CENTROS", style={'color': '#1c485d',
                                                                                         'font-weight': 'bold'})],
                                    id="cifra_centros",
                                    className="mini_container",
                                ),

                                html.Div(
                                    [html.H6(id="num_locales"), html.P("LOCALES", style={'color': '#1c485d',
                                                                                         'font-weight': 'bold'})],
                                    id="cifra_locales",
                                    className="mini_container",
                                ),

                                html.Div(
                                    [html.H6(id="ventas_mini"), html.P("VENTAS TOTALES", style={'color': '#1c485d',
                                                                                                'font-weight': 'bold'})],
                                    id="cifra_ventas",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="tasa_esf_mini"), html.P("μ TASA ESFUERZO", style={'color': '#1c485d',
                                                                                                   'font-weight': 'bold'})],
                                    id="cifra_tasa_esfuerzo",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="venta_m2"), html.P("μ VENTA M²", style={'color': '#1c485d',
                                                                                         'font-weight': 'bold'})],
                                    id="cifra_venta_m2",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="renta_m2"), html.P("μ RENTA M²", style={'color': '#1c485d',
                                                                                         'font-weight': 'bold'})],
                                    id="cifra_renta_m2",
                                    className="mini_container",
                                ),
                            ],
                            id="info-container",
                            className="row container-display", style={"margin-right": "150px"},
                        ),
                        # ------------------------------------------------------------------------------ MAIN SALES PLOT
                        html.Div([
                            html.Div(
                                [dcc.RadioItems(
                                    id="tipo_ventas_radio",
                                    options=[
                                        {"label": "Evolución Agrupada", "value": "agrupado"},
                                        {"label": "Ver Operadores", "value": "operadores"},
                                        {"label": "Por Centro", "value": "centros"},
                                        {"label": "Por Categoría", "value": "categorias"},
                                    ],
                                    value="agrupado",
                                    labelStyle={'display': 'inline-block'},
                                ),
                                ], className='row flex-display'),

                            dcc.Graph(id="operadores_sales_graph")
                        ],
                            id="countGraphContainer",
                            className="pretty_container",
                        ),
                    ],
                    id="right-column",
                    className="eight columns",
                ),
            ],
            className="row flex-display",
        ),

        # ---------------------------------------------------------------------------------------- AFLUENCIAS Y SUNBRIST
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="afl_ops_graph")],
                    className="pretty_container seven columns",
                ),
                html.Div(
                    [dcc.Graph(id="ventas_cat_pie_graph")],
                    className="pretty_container five columns",
                ),
            ],
            className="row flex-display",
        ),
        # ---------------------------------------------------------------------------------------- TABLAS MIX COMERCIAL
        html.Div([html.H4("MIX COMERCIAL", style={'background-color': '#99a8b2',
                                                  'text-align': 'center',
                                                  'color': 'white'},
                          className="pretty_container twelve columns",
                          )], className="row flex-display"),
        html.Div([
            html.Div(
                [dcc.Graph(id="mix_1")],
                className="pretty_container six columns",
            ),
            html.Div([html.Div(
                [dcc.RadioItems(
                    id="selector_tipo_grup",
                    options=[
                        {"label": "Metros", "value": "metros_local"},
                        {"label": "Venta/metro", "value": "sales_m2"},
                        {"label": "Renta/metro", "value": "renta_m2"},
                    ],
                    value="metros_local",
                    labelStyle={'display': 'inline-block'},
                ),
                ], className='row flex-display'),
                dcc.Graph(id="mix_2")],
                className="pretty_container six columns",
            ), ],

            className="row flex-display",
        ),

        # ------------------------------------------------------------------------------------------------  SCATTER PLOT
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="grafico_scatter")],
                    className="pretty_container twelve columns",
                )
            ],
            className="row flex-display",
        ),

        # ----------------------------------------------------------------------------------------------- SCATTER PLOT 2
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="grafico_scatter_2")],
                    className="pretty_container twelve columns",
                )
            ],
            className="row flex-display",
        ),

        # ------------------------------------------------------------------------------------------------ TASA ESFUERZO
        html.Div([html.H4("TASAS DE ESFUERZO", style={'background-color': '#99a8b2',
                                                      'text-align': 'center',
                                                      'color': 'white'},
                          className="pretty_container twelve columns",
                          )], className="row flex-display"),
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="tasa_esf_polar_graph")],
                    className="pretty_container five columns",
                ),
                html.Div(
                    [dcc.Graph(id="tasa_esf_line_graph")],
                    className="pretty_container seven columns",
                ),
            ],

            className="row flex-display",
        ),

        # -------------------------------------------------------------------------------------------------- TABLA CHURN
        html.Div([html.H4("RIESGO DE FUGA", style={'background-color': '#99a8b2',
                                                   'text-align': 'center',
                                                   'color': 'white'},
                          className="pretty_container twelve columns",
                          )], className="row flex-display"),
        html.Div([
            html.Div([
                dash_table.DataTable(

                    id="tabla_riesgo",
                    columns=[{'name': i, 'id': i, 'selectable': True, 'hideable': True} for i in
                             df_model[['date', 'cod_carmila', 'inquilino', 'renta_mes', 'cv_neta_n', 'tasa_esfuerzo_n',
                                       'y_proba']].columns],
                    data=[],
                    editable=True,
                    filter_action="native",
                    sort_action="native",
                    sort_mode="multi",
                    column_selectable=None,
                    row_selectable="multi",
                    selected_columns=[],
                    selected_rows=[],
                    page_action="native",
                    page_current=0,
                    page_size=10,
                    style_cell={'minWidth': 95, 'maxWidth': 150, 'width': 95},
                    style_data_conditional=([
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }]),
                    style_header={
                        'backgroundColor': 'rgb(230, 240, 240)',
                        'fontWeight': 'bold'
                    }
                ),
            ], className="pretty_container nine columns"),

            html.Div(
                [dcc.Graph(id="main_test_3")],
                className="pretty_container three columns",
            ), ], className="row flex-display"),

        # ------------------------------------------------------------------------------------------------ MIX COMERCIAL

        # ------------------------------------------------------------------------------------------------ ZOOM OPERADOR
        html.Div([html.H2("ZOOM OPERADOR", style={'background-color': '#1c485d',
                                                  'text-align': 'center',
                                                  'color': 'white'},
                          className="pretty_container twelve columns",
                          )], className="row flex-display"),

        html.Div([
            html.Div(
                [
                    html.P("Seleccionar Operador:", className="control_label"),

                    dcc.Dropdown(
                        id="selector_zoom_operador",
                        options=[],
                        multi=False,
                        value=[],
                        className="dcc_control",
                    ),

                    dcc.Dropdown(
                        id="centros_inquilino",
                        options=[],
                        multi=True,
                        value=[],
                        className="dcc_control",
                    ),
                    html.P("Seleccionar Métrica a Estudiar:", className="control_label"),

                    dcc.RadioItems(id='selector_grafica',
                                   options=[
                                       {'label': 'Cifra Ventas', 'value': 'cv_neta_n'},
                                       {'label': 'Ventas m2', 'value': 'sales_m2'},
                                       {'label': 'Renta m2', 'value': 'renta_m2'},
                                       {'label': 'Tasa Esfuerzo', 'value': 'tasa_esfuerzo_n'},
                                   ],
                                   value='cv_neta_n'),

                ],
                className="pretty_container four columns",
                id="second_menu",
            ),
            # GRAFICA ==================================================================================================
            html.Div([
                html.Div(
                    [dcc.RadioItems(
                        id="selector_tipo_grafico",
                        options=[
                            {"label": "Ver Plano", "value": "plano"},
                            {"label": "Ver Gráfico", "value": "barplot"},
                        ],
                        value="plano",
                        labelStyle={'display': 'inline-block'}),

                    ], className='row flex-display'),

                dcc.Graph(id="right-column-2")],
                className="pretty_container eight columns",

            )], className="row flex-display"),

        html.Div([
            html.Div(
                [dcc.Graph(id="main_graph_zoom")],
                className="pretty_container twelve columns",
            ), ],

            className="row flex-display",
        ),

        # --------------------------------------------------------------------------------------------- DATA  BACKOFFICE

        # html.Div(id='intermediate-value', style={'display': 'none'}),  # div invisible para almacenar la info filtrada.
        # html.Div(id='model-value', style={'display': 'none'}),  # INFO MODEL.

    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},

)


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- CALLBACKS DE SELECTORES ------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# Devuelve una lista de centros en función de las opciones seleccionadas en el primer radiobutton.
@app.callback(
    [Output('selector_centros', 'options'),
     Output('selector_centros', 'value')],
    [Input('radio_tipo_centro', 'value')])
def set_center_options_by_type(selected_radio_tipo):
    ddf_tipos = df[['cod_carmila', 'tipo']].drop_duplicates(keep='first')

    if selected_radio_tipo == 'Todos':
        todos_value = ddf_tipos['cod_carmila'].to_list()
        return CENTER_OPTIONS, todos_value

    elif selected_radio_tipo == 'GALERIAS':
        ddf_galerias = ddf_tipos[ddf_tipos['tipo'] == 'GALERIA']
        galeria_options = [{'label': i, 'value': i} for i in ddf_galerias['cod_carmila']]
        galeria_value = ddf_galerias['cod_carmila'].to_list()
        return galeria_options, galeria_value

    else:
        ddf_gerencias = ddf_tipos[ddf_tipos['tipo'] == 'GERENCIA']
        gerencia_options = [{'label': i, 'value': i} for i in ddf_gerencias['cod_carmila']]
        gerencia_value = ddf_gerencias['cod_carmila'].to_list()
        return gerencia_options, gerencia_value


@app.callback(
    [Output('selector_operador', 'options'),
     Output('selector_operador', 'value'), ],
    [Input('radio_selector_operador', 'value'),
     Input('selector_centros', 'value')])
def selector_operadores(radio_button, centros):
    ddf = df[df['cod_carmila'].isin(centros)]
    ddf = ddf[['cod_carmila', 'inquilino']].drop_duplicates(keep='first')
    lista_ops = list(set(ddf['inquilino'].to_list()))

    if radio_button != 'Todos':
        ops_options = [{'label': i, 'value': i} for i in lista_ops if i is not None]
        return ops_options, lista_ops
    else:
        return [], lista_ops


# DATA COMPILER
@app.callback(
    Output('intermediate-value', 'children'),
    [Input('selector_centros', 'value'),
     Input('selector_categoria', 'value'),
     Input('selector_operador', 'value'),
     Input('rango_fechas', 'start_date'),
     Input('rango_fechas', 'end_date')])
def create_working_db(centros, categorias, operador, fecha_inicio, fecha_fin):
    ddf = df.copy().fillna(0)
    filter_centro = ddf['cod_carmila'].isin(centros)
    filter_categoria = ddf['grupo_rotulo'].isin(categorias)
    filter_operador = ddf['inquilino'].isin(operador)
    filter_fechas = (ddf['date'] >= fecha_inicio) & (ddf['date'] <= fecha_fin)
    ddf = ddf[filter_centro & filter_categoria & filter_operador & filter_fechas]
    json = ddf.to_json(date_format='iso', orient='split')
    return json


# MODEL DATA_____________________________________________________________________________________
@app.callback(
    Output('model-value', 'children'),
    [Input('selector_centros', 'value'),
     Input('selector_categoria', 'value'),
     Input('selector_operador', 'value'),
     Input('rango_fechas', 'start_date'),
     Input('rango_fechas', 'end_date')])
def create_df_predict(centros, categorias, operador, fecha_inicio, fecha_fin):
    ddf = df.copy().fillna(0)
    filter_centro = ddf['cod_carmila'].isin(centros)
    filter_categoria = ddf['grupo_rotulo'].isin(categorias)
    filter_operador = ddf['inquilino'].isin(operador)
    filter_ventas = ddf['cv_neta_n'] > 1
    filter_renta = ddf['renta_mes'] > 1
    filter_fechas = (ddf['date'] >= fecha_inicio) & (ddf['date'] <= fecha_fin)
    ddf = ddf[filter_centro & filter_categoria & filter_operador & filter_fechas & filter_ventas & filter_renta]

    predictions = m_churn_model.make_churn_predictions(ddf).head(500)
    predictions.to_csv('data/final/test_data_model.csv')
    json = predictions.to_json(date_format='iso', orient='split')
    return json


# ---------------------------------------------- ZONA  PRINCIPAL -------------------------------------------------------
# Num Centros
@app.callback(
    Output('num_centros', 'children'),
    [Input('intermediate-value', 'children')])
def update_table(jsonified_cleaned_data):
    ddf = pd.read_json(jsonified_cleaned_data, orient='split')
    sum = ddf['cod_carmila'].nunique()
    return sum


# Num Locales
@app.callback(
    Output('num_locales', 'children'),
    [Input('intermediate-value', 'children')])
def update_table(jsonified_cleaned_data):
    ddf = pd.read_json(jsonified_cleaned_data, orient='split')
    sum = ddf['cod_lote'].nunique()
    number = "{:,.1f}".format(sum)
    return number


# Mini: Ventas
@app.callback(
    Output('ventas_mini', 'children'),
    [Input('intermediate-value', 'children')])
def update_table(jsonified_cleaned_data):
    ddf = pd.read_json(jsonified_cleaned_data, orient='split')
    sum = ddf['cv_neta_n'].sum()
    number = "{:,.1f}".format(sum)
    return number + ' €'


# Mini: Tasa Esfuerzo:
@app.callback(
    Output('tasa_esf_mini', 'children'),
    [Input('intermediate-value', 'children')])
def update_table(jsonified_cleaned_data):
    ddf = pd.read_json(jsonified_cleaned_data, orient='split')
    if ddf.size > 1:
        ddf.dropna(subset=['tasa_esfuerzo_n'])
        cifra = ddf['tasa_esfuerzo_n'].mean()
        number = "{:,.1f}".format(cifra)
        return number


# Mini: Sales M2:
@app.callback(
    Output('venta_m2', 'children'),
    [Input('intermediate-value', 'children')])
def update_table(jsonified_cleaned_data):
    ddf = pd.read_json(jsonified_cleaned_data, orient='split')
    cifra = ddf['sales_m2'].mean()
    number = "{:,.1f}".format(cifra)
    return number + ' €'


# Mini: RETA M2:
@app.callback(
    Output('renta_m2', 'children'),
    [Input('intermediate-value', 'children')])
def update_table(jsonified_cleaned_data):
    ddf = pd.read_json(jsonified_cleaned_data, orient='split')
    cifra = ddf['renta_m2'].mean()
    number = "{:,.1f}".format(cifra)
    return number + ' €'


# MAIN SALES GRAPH
@app.callback(
    Output('operadores_sales_graph', 'figure'),
    [Input('intermediate-value', 'children'),
     Input('tipo_ventas_radio', 'value')])
def update_table(jsonified_cleaned_data, option_radio_buton):
    ddf = pd.read_json(jsonified_cleaned_data, orient='split')

    if option_radio_buton == 'agrupado':
        title = 'Evolución Venta Neta Operadores'
        info = ddf.groupby('date')['cv_neta_n'].sum().reset_index()
        fig = px.area(info, x='date', y='cv_neta_n', title=title, template='none', height=600,
                      labels=dict(date="meses",
                                  cv_neta_n="Venta Neta (€)"), color_discrete_sequence=px.colors.diverging.balance)

    elif option_radio_buton == 'operadores':
        title = 'Evolución Venta Neta de Cada Operador'
        info = ddf.groupby(['date', 'inquilino'])['cv_neta_n'].sum().reset_index()
        fig = px.line(info, x='date', y='cv_neta_n', title=title, template='none', height=600,
                      color='inquilino', labels=dict(date="meses",
                                                     cv_neta_n="Venta Neta (€) Operador"),
                      color_discrete_sequence=px.colors.diverging.balance)

    elif option_radio_buton == 'centros':
        title = 'Suma Venta Neta por Provincia'
        info = ddf.groupby(['provincia'])['cv_neta_n'].sum().reset_index()
        fig = px.bar(info, x='provincia', y='cv_neta_n', title=title, template='none', height=600,
                     color='provincia', labels=dict(provincia="Provincia",
                                                    cv_neta_n="Suma Venta neta (€)"),
                     color_discrete_sequence=px.colors.diverging.balance)

    else:
        title = 'Suma Venta Neta por Categoría'
        info = ddf.groupby(['grupo_rotulo'])['cv_neta_n'].sum().reset_index()
        fig = px.bar(info, x='grupo_rotulo', y='cv_neta_n', title=title, template='none', height=600,
                     color='grupo_rotulo', labels=dict(grupo_rotulo="Categoría",
                                                       cv_neta_n="Suma Venta neta (€)"),
                     color_discrete_sequence=px.colors.diverging.balance
                     )

    return fig


# -------------------------------------------- SEGUNDA LÍNEA GRÁFICAS --------------------------------------------------
# Grafico Afuencias y Ops Hiper
@app.callback(
    Output('afl_ops_graph', 'figure'),
    [Input('intermediate-value', 'children'), ])
def afluencias_ventas_ops(jsonified_cleaned_data):
    ddf = pd.read_json(jsonified_cleaned_data, orient='split')  # CREAR DF A PARTIR DEL JSON
    ddf = ddf[['date', 'operaciones', 'ventas', 'afluencia']].fillna(0)

    fig_info = ddf.melt(id_vars=['date'],
                        value_vars=['operaciones', 'ventas', 'afluencia'],
                        value_name='dato').groupby(['date', 'variable']).sum().reset_index()
    title = 'Afluencias y Datos Hiper'
    fig = px.line(fig_info, x="date", y="dato", color='variable', title=title,
                  labels=dict(date='Fecha', dato='Dato', variable='Tipo'),
                  template='none', color_discrete_sequence=px.colors.diverging.balance)
    fig.update_layout(margin=dict(t=30, l=50, r=0, b=0))

    return fig


# Grafico Circular ventas por categoria
@app.callback(
    Output('ventas_cat_pie_graph', 'figure'),
    [Input('intermediate-value', 'children'),
     Input('selector_centros', 'value')])
def circular_venta_categoria(jsonified_cleaned_data, centros_seleccionados):
    ddf = pd.read_json(jsonified_cleaned_data, orient='split')  # CREAR DF A PARTIR DEL JSON

    if len(centros_seleccionados) > 1:
        title = 'Reparto ventas por Categoría y Subcategoría'
        info = ddf.groupby(['grupo_rotulo', 'sub_grupo_rotulo'])['cv_neta_n'].sum().reset_index()
        fig = px.sunburst(info, path=['grupo_rotulo', 'sub_grupo_rotulo'], values='cv_neta_n', template='none',
                          title=title, color_discrete_sequence=px.colors.diverging.balance)

    else:
        title = 'Reparto ventas por Categooría, Subcategoría y Operador'
        info = ddf.groupby(['cod_carmila', 'grupo_rotulo', 'sub_grupo_rotulo', 'inquilino'])[
            'cv_neta_n'].sum().reset_index()
        fig = px.sunburst(info, path=['grupo_rotulo', 'sub_grupo_rotulo', 'inquilino'], values='cv_neta_n',
                          template='none', title=title, color_discrete_sequence=px.colors.diverging.balance)

    fig.update_traces(textinfo='label+percent parent')
    fig.update_layout(margin=dict(t=30, l=50, r=0, b=0))

    return fig


# ------------------------------------------ Scatter  -----------------------------------------------
@app.callback(
    Output('grafico_scatter', 'figure'),
    [Input('intermediate-value', 'children')])
def grafico_ventas_por_categoria(jsonified_cleaned_data):
    title = 'Evolución Ventas m2 por Categoría y Datos Hiper'
    ddf = pd.read_json(jsonified_cleaned_data, orient='split')
    ddf = ddf[['date', 'cod_carmila', 'grupo_rotulo', 'sales_m2', 'cv_neta_n']].replace(
        [np.inf, -np.inf], np.nan).dropna(subset=["sales_m2"], how="all").reset_index(
        drop=True)

    ddf = ddf.groupby(['date', 'grupo_rotulo'], as_index=False)['sales_m2'].sum()
    fig = px.line(ddf, x='date', y='sales_m2', color='grupo_rotulo', template='none',
                  title=title, color_discrete_sequence=px.colors.diverging.balance)
    fig.update_layout(margin=dict(t=30, l=40, r=0, b=30))

    return fig


# ------------------------------------------ Scatter 2  -----------------------------------------------
@app.callback(
    Output('grafico_scatter_2', 'figure'),
    [Input('selector_centros', 'value'),
     Input('rango_fechas', 'start_date'),
     Input('rango_fechas', 'end_date')])
def update_table(centros, fecha_inicio, fecha_fin):
    title = 'Comparativa nºOperaciones Hiper, Tasa de Vacancy y Ventas Netas por centro'
    ddf = df[['date', 'cod_carmila', 'grupo_rotulo', 'cod_contrato', 'ocupacion',
              'cv_neta_n', 'operaciones', 'tipo']].copy().fillna(0)
    filter_centro = ddf['cod_carmila'].isin(centros)
    filter_fechas = (ddf['date'] >= fecha_inicio) & (ddf['date'] <= fecha_fin)
    ddf = ddf[filter_centro & filter_fechas]

    initial_df = ddf[['date', 'cod_carmila', 'cod_contrato', 'ocupacion', 'cv_neta_n', 'operaciones', 'tipo']]
    ocup_df = ddf[['date', 'cod_carmila', 'cod_contrato', 'ocupacion', 'cv_neta_n', 'operaciones']]

    ocup_df = ocup_df.groupby(['date', 'cod_carmila', 'ocupacion']).agg({'cod_contrato': 'count'})
    pivot = ocup_df.unstack('ocupacion').reset_index().fillna(0)
    pivot.columns = pivot.columns.get_level_values(0)
    pivot.columns = ['date', 'cod_carmila', 'disponibles', 'ocupados']
    pivot['vacancy'] = pivot['disponibles'] / (pivot['disponibles'] + pivot['ocupados'])

    # #fusionamos los df. el creado con los datos de ocupacion y el priemero condatos de venta y ocpupación.
    merged_df = initial_df.merge(pivot, on=['date', 'cod_carmila'], how='left').fillna(0)
    merged_df = merged_df.groupby(['cod_carmila', 'tipo']).mean().reset_index()
    fig = px.scatter(merged_df, y="vacancy", x="operaciones", color='tipo',
                     size='cv_neta_n', hover_name='cod_carmila', labels=dict(vacancy='% Vacancy',
                                                                             operaciones='nº Operaciones Hiper ',
                                                                             variable='Tipo de Centro'),
                     template='none', title=title)

    # fig.update_layout(margin=dict(t=0, l=40, r=0, b=30))

    return fig


# --------------------------------------------- LÍNEA TASA ESFUERZO ------------------------------------
# GRAF POLAR TASA ESFUERZO
@app.callback(
    Output('tasa_esf_polar_graph', 'figure'),
    [Input('intermediate-value', 'children')])
def update_grafico_polar(jsonified_cleaned_data):
    title = 'Tasas de Esfuerzo Media por Categoría'
    ddf = pd.read_json(jsonified_cleaned_data, orient='split')  # CREAR DF A PARTIR DEL JSON
    ddf = ddf[['tasa_esfuerzo_n', 'grupo_rotulo']].groupby(['grupo_rotulo']).mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=ddf['tasa_esfuerzo_n'], theta=ddf['grupo_rotulo'], fill='toself'))
    return fig


# GRAF LINEA TASA ESF
@app.callback(
    Output('tasa_esf_line_graph', 'figure'),
    [Input('intermediate-value', 'children')])
def update_graph_tasa_esf(jsonified_cleaned_data):
    title = 'Evolución Tasa de Esfuerzo Media'
    ddf = pd.read_json(jsonified_cleaned_data, orient='split')
    ddf = ddf[['date', 'tasa_esfuerzo_n', 'grupo_rotulo']].groupby(['date']).mean().reset_index()
    fig = px.line(ddf, x='date', y='tasa_esfuerzo_n', template='none', title=title,
                  color_discrete_sequence=px.colors.diverging.balance)
    return fig


# ----------------------------------------------- CUARTA  LÍNEA:CHURN MODEL  -------------------------------------------
@app.callback(
    [Output('tabla_riesgo', 'columns'),
     Output('tabla_riesgo', 'data')],
    [Input('model-value', 'children')])
def update_table(json_data_model):
    ddf = pd.read_json(json_data_model, orient='split')
    ddf = ddf.sort_values(by=['date', 'y_proba'], ascending=[False, True])
    ddf['date'] = ddf['date'].dt.strftime("%B-%y")

    columns = [{'name': i, 'id': i, 'selectable': True, 'hideable': True} for i in
               ddf[['date', 'cod_carmila', 'inquilino', 'renta_mes', 'cv_neta_n', 'tasa_esfuerzo_n',
                    'y_proba']].columns]

    data = ddf[['date', 'cod_carmila', 'inquilino', 'renta_mes', 'cv_neta_n', 'tasa_esfuerzo_n', 'y_proba']].to_dict(
        'records')

    return columns, data


@app.callback(
    Output(component_id='main_test_3', component_property='figure'),
    [Input(component_id='tabla_riesgo', component_property="derived_virtual_data"),
     Input(component_id='tabla_riesgo', component_property='derived_virtual_selected_rows'),
     Input('intermediate-value', 'children')])
def sales_selected_in_table(all_rows_data, slctd_row_indices, json_data):
    ddf = pd.read_json(json_data, orient='split')
    ddf = ddf[['date', 'cod_carmila', 'inquilino', 'fecha_fin', 'sales_m2', 'renta_m2', 'cv_neta_n', 'cv_neta_n-1',
               'cv_neta_n-2', 'cv_neta_n-3']]
    inquilinos_seleccionados = []
    centros_seleccionados = []
    for n in slctd_row_indices:
        inqu = all_rows_data[n]['inquilino']
        centro = all_rows_data[n]['cod_carmila']
        inquilinos_seleccionados.append(inqu)
        centros_seleccionados.append(centro)

    filter_centro = ddf['cod_carmila'].isin(centros_seleccionados)
    filter_categoria = ddf['inquilino'].isin(inquilinos_seleccionados)

    ddf = ddf[filter_centro & filter_categoria]
    ddf = ddf.groupby(['cod_carmila', 'inquilino', 'fecha_fin'], as_index=False).mean().sort_values(
        by=['fecha_fin'], ascending=True).reset_index()

    ddf = ddf.melt(id_vars=['cod_carmila', 'inquilino'],
                   value_vars=['cv_neta_n', 'cv_neta_n-1', 'cv_neta_n-2', 'cv_neta_n-3'],
                   value_name='dato').groupby(['cod_carmila', 'inquilino', 'variable']).mean().reset_index()

    fig = px.bar(ddf, x='variable', y='dato', color='inquilino',
                 category_orders={'variable': ["cv_neta_n-3", "cv_neta_n-2", "cv_neta_n-1", "cv_neta_n"]},
                 template='none', barmode='group', color_discrete_sequence=px.colors.diverging.balance)
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))

    return fig


# ------------------------------------------ MIX COMERCIAL  -----------------------------------------------
# LINE GRAPH OCUPACION
@app.callback(
    Output('mix_1', 'figure'),
    [Input('selector_centros', 'value'),
     Input('rango_fechas', 'start_date'),
     Input('rango_fechas', 'end_date')])
def evolucion_ocupacion(selected_centros, fecha_inicio, fecha_fin):
    ddf = df[['date', 'cod_carmila', 'cod_contrato', 'ocupacion']].copy().fillna(0)
    title = 'Evolución Tasa de Vacancy'
    filter_centro = ddf['cod_carmila'].isin(selected_centros)
    filter_fechas = (ddf['date'] >= fecha_inicio) & (ddf['date'] <= fecha_fin)
    ddf = ddf[filter_centro & filter_fechas]

    ocup_df = ddf.groupby(['date', 'cod_carmila', 'ocupacion']).agg({'cod_contrato': 'count'})
    pivot = ocup_df.unstack('ocupacion').reset_index().fillna(0)
    pivot.columns = pivot.columns.get_level_values(0)
    pivot.columns = ['date', 'cod_carmila', 'disponibles', 'ocupados']
    pivot['vacancy'] = pivot['disponibles'] / (pivot['disponibles'] + pivot['ocupados'])

    if len(selected_centros) > 1:
        pivot = pivot.groupby(['date'])['vacancy'].mean().reset_index()
    fig = px.line(pivot, x='date', y='vacancy', template='none', title=title,
                  color_discrete_sequence=px.colors.diverging.balance)
    return fig


#  BARRAS GRUPO ROTULOS
@app.callback(
    Output('mix_2', 'figure'),
    [Input('selector_centros', 'value'),
     Input('rango_fechas', 'start_date'),
     Input('rango_fechas', 'end_date'),
     Input('selector_tipo_grup', 'value')])
def update_table(selected_centros, fecha_inicio, fecha_fin, selector_grafica):
    ddf = df[['date', 'cod_carmila', 'metros_local', 'grupo_rotulo', 'sales_m2', 'renta_m2']].copy().fillna(0)
    filter_centro = ddf['cod_carmila'].isin(selected_centros)
    filter_fechas = (ddf['date'] >= fecha_inicio) & (ddf['date'] <= fecha_fin)
    filtro_ceros = (ddf['grupo_rotulo'] != 0) & (ddf['metros_local'] != 0)
    ddf = ddf[filter_centro & filter_fechas & filtro_ceros]
    categoria = selector_grafica
    info = ddf.groupby(['grupo_rotulo'])[categoria].sum().reset_index()
    fig = px.bar(info, x='grupo_rotulo', y=categoria, template='none',
                 color_discrete_sequence=px.colors.diverging.balance, color='grupo_rotulo')
    return fig


# ------------------------------------------ ZOOM OPERADOR  -----------------------------------------------
@app.callback(
    [Output('selector_zoom_operador', 'options'),
     Output('selector_zoom_operador', 'value')],
    [Input('radio_tipo_centro', 'value')])
def set_center_options_by_type(selected_radio_tipo):
    ddf_inquilino = df[['inquilino']].drop_duplicates(keep='first')

    arround = (selected_radio_tipo == 'GERENCIAS') | (selected_radio_tipo == 'GALERIAS') | (
            selected_radio_tipo == 'Todos')

    if arround:
        gerencia_options = [{'label': i, 'value': i} for i in ddf_inquilino['inquilino']]
        gerencia_value = ddf_tipos['cod_carmila'].to_list()
        return gerencia_options, gerencia_value


@app.callback(
    [Output('centros_inquilino', 'options'),
     Output('centros_inquilino', 'value')],
    [Input('selector_zoom_operador', 'value')])
def set_center_options_by_type(inquilino):
    ddf = df[df['inquilino'] == inquilino]
    ddf = ddf[['cod_carmila', 'inquilino']].drop_duplicates(keep='first')
    center_options = [{'label': i, 'value': i} for i in ddf['cod_carmila']]
    center_value = ddf['cod_carmila'].to_list()
    return center_options, center_value


@app.callback(
    Output('right-column-2', 'figure'),
    [Input('selector_zoom_operador', 'value'),
     Input('centros_inquilino', 'value'),
     Input('selector_grafica', 'value'),
     Input('selector_tipo_grafico', 'value')])
def create_zoom_map_plot(inquilino, centros, columna, tipo_grafico):
    ddf = df[['date', 'lat', 'long', 'provincia', 'cod_carmila', 'inquilino', 'cv_neta_n',
              'sales_m2', 'renta_m2', 'tasa_esfuerzo_n']].replace(
        [np.inf, -np.inf], np.nan).dropna(subset=["sales_m2", 'tasa_esfuerzo_n', "renta_m2"], how="all").reset_index(
        drop=True)
    filter_centro = ddf['cod_carmila'].isin(centros)
    filter_operador = ddf['inquilino'] == inquilino
    filter_ventas = ddf[columna] > 0
    ddf = ddf[filter_centro & filter_operador & filter_ventas]
    coords = ddf.groupby(['cod_carmila', 'lat', 'long']).mean().reset_index()

    if tipo_grafico == 'plano':
        fig = px.scatter_mapbox(coords, lat='lat', lon='long', hover_name='cod_carmila', size=columna,
                                center=dict(lat=39.863667, lon=-3.74922), zoom=4.85,
                                mapbox_style="stamen-terrain", color=columna,
                                color_discrete_sequence=px.colors.sequential.Magma)
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
    else:
        ddf = ddf.groupby(['provincia', 'inquilino'], as_index=False)[
            'cv_neta_n', 'sales_m2', 'tasa_esfuerzo_n', 'renta_m2'].mean()
        ddf = ddf.melt(id_vars=['provincia', 'inquilino'],
                       value_vars=['cv_neta_n', 'sales_m2', 'renta_m2', 'tasa_esfuerzo_n'], var_name='metrica',
                       value_name='dato')
        fig = px.bar(ddf, x='provincia', y='dato', color='metrica', barmode='group',
                     color_discrete_sequence=px.colors.diverging.balance)
        fig.update_layout(xaxis={'categoryorder': 'total descending'})

    fig.update_layout(template='none')
    return fig


@app.callback(
    Output('main_graph_zoom', 'figure'),
    [Input('selector_zoom_operador', 'value'),
     Input('centros_inquilino', 'value'),
     Input('selector_grafica', 'value')])
def create_zoom_bar_plot(inquilino, centros, columna):
    ddf = df[['date', 'cod_carmila', 'inquilino', 'cv_neta_n', 'sales_m2', 'renta_m2', 'tasa_esfuerzo_n']].replace(
        [np.inf, -np.inf], np.nan).dropna(subset=["sales_m2", 'tasa_esfuerzo_n', "renta_m2"], how="all").reset_index(
        drop=True)
    filter_centro = ddf['cod_carmila'].isin(centros)
    filter_operador = ddf['inquilino'] == inquilino
    filter_ventas = ddf[columna] > 0
    ddf = ddf[filter_centro & filter_operador & filter_ventas]
    ddf = ddf.groupby('date', as_index=False).agg({columna: 'sum'})
    fig = px.bar(ddf, x='date', y=columna, color_discrete_sequence=px.colors.sequential.Plasma)
    fig.update_layout(template='none')
    return fig


# Main
if __name__ == "__main__":
    app.run_server(debug=False)
