import os
import argparse
import pandas as pd
import sqlite3
from p_acquisition import m_acquisition
from p_wrangling import m_wrangling
from models import m_churn_model

# SQLITLE CONFIG
conn = sqlite3.connect("data/processed/sqlite_data_base.db")
TABLA_CENTROS = 'centros_carmila'
TABLA_CODS_CENTROS = 'codigos_centros'
TABLA_FACTURACION = 'tabla_facturacion'
TABLA_ESTADO_ARREND = 'estado_arrend'
TABLA_INFO_HIPER = 'info_hiper'
TABLA_AFLUENCIAS = 'df_afluencias'
TABLA_VENTAS_OPS = 'ventas_operadores'


def main():
    # ACQUISITION
    print('======================= || STARTING ACQUISITION PROCESS || =======================\n')
    m_acquisition.creating_ddbb(conn)

    # GETTING GETTING DATABASES
    df_centros_carmila = m_wrangling.get_table_from_sql(TABLA_CENTROS, conn)
    df_codigos_centros = m_wrangling.get_table_from_sql(TABLA_CODS_CENTROS, conn)
    df_est_arrendamiento = m_wrangling.get_table_from_sql(TABLA_ESTADO_ARREND, conn)
    df_info_hiper = m_wrangling.get_table_from_sql(TABLA_INFO_HIPER, conn)
    df_afluencias = m_wrangling.get_table_from_sql(TABLA_AFLUENCIAS, conn)
    df_ventas_operadores = m_wrangling.get_table_from_sql(TABLA_VENTAS_OPS, conn)

    # WRANGLING
    print('\n\n======================= || STARTING WRANGLING PROCESS || =======================')
    df_informacion_activos = m_wrangling.combining_asset_info(df_info_hiper, df_codigos_centros, df_centros_carmila,
                                                              df_afluencias)
    combined_raw_df = m_wrangling.created_raw_combined_df(activos=df_informacion_activos, ventas=df_ventas_operadores,
                                                          arrend=df_est_arrendamiento)
    # persisting
    m_acquisition.general_saving_to_sql(conn, combined_raw_df, 'raw_combined_table')

    # CLEAN RAW DF
    clean_df = m_wrangling.wrangling_on_raw_df(combined_raw_df)

    # persisting
    m_acquisition.general_saving_to_sql(conn, clean_df, 'full_final_table')
    clean_df.to_parquet('data/final/full_db.parquet')

    # MODEL TRAIN
    print('\n\n======================= || TRAINING CHURN RISK MODEL || =======================')
    data_to_train_model = m_churn_model.wrangling_data_to_model(clean_df)
    m_churn_model.train_model_lgbm(data_to_train_model)

    # #MODEL PREDICT
    # print('\n\n======================= || PREDICTING WITH CHURN RISK MODEL || =======================')
    # persisting_clean_df = pd.read_parquet('data/final/full_db.parquet')
    # df_with_predictions = m_churn_model.make_churn_predictions(persisting_clean_df.tail(100))
    # print(df_with_predictions.head)

    print('\n\n======================= || PIPELINE FINISHED || =======================')


if __name__ == "__main__":
    main()
