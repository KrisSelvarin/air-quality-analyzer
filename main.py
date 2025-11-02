# main.py 

from pathlib import Path
from src.logger_config import setup_logger
from src.clean import data_clean
from src.eda import Exploratory
from src.sensor import Sensor
import pandas as pd
import time

# logger
logger = setup_logger(__name__)

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# for EDA
GRAPH_DIR = BASE_DIR / 'output' / 'graphs' / 'eda'
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# for sensor performance evaluation
SENSOR_DIR = BASE_DIR / 'output' / 'graphs' / 'sensor'
SENSOR_DIR.mkdir(parents=True, exist_ok=True)

def main():

    # ----------------------------------------------------------
    # Data cleaning
    # ----------------------------------------------------------

    logger.info('Cleaning Data')
    df = data_clean()
    logger.info('Data Cleaned')

    # ----------------------------------------------------------
    # Exploratory Data Analysis
    # ----------------------------------------------------------

    logger.info('Create object explore with dataframe')
    explore = Exploratory(df)

    # ----------------------------------------------------------
    # 1. Distribution

    plot_distribution = [
        ('Distribution of Carbon Monoxide Concentration (Ground Truth)', 'CO(GT)', 'CO Concentration (GT)', 'histogram_co.png'),
        ('Distribution of Nitrous Oxides Concentration (Ground Truth)', 'NOx(GT)', 'NOx Concentration (GT)', 'histogram_nox.png'),
        ('Distribution of Nitrous Dioxide Concentration (Ground Truth)', 'NO2(GT)', 'NOx Concentration (GT)', 'histogram_no2.png')
    ]

    for title_dis, var_dis, xlabel_dis, filename_dis in plot_distribution:
        logger.info(f'Creating histogram for {var_dis} distribution')
        fig_dis = explore.distribution(title_dis, var_dis, xlabel_dis)
        fig_dis.savefig(GRAPH_DIR / filename_dis)
        logger.info(f'Figure saved to {GRAPH_DIR}')

    # ----------------------------------------------------------
    # 2. Trends over time

    plot_trends = [
        ('CO Conecentration from 2004-03-10 to 2005-04-04', 'Date', 'CO(GT)', 'CO Concentration Daily Avg. Level', 'line_co.png'),
        ('NOx Concentration from 2004-03-10 to 2005-04-04', 'Date', 'NOx(GT)', 'NOx Concentration Daily Avg. Level', 'line_nox.png'),
        ('NO2 Concentration from 2004-03-10 to 2005-04-04', 'Date', 'NO2(GT)', 'NO2 Concentration Daily Avg. Level', 'line_no2.png')
    ]

    for title_line, x_line, y_line, ylabel_line, filename_line in plot_trends:
        logger.info(f'Creating line graph for {y_line} trend')
        fig_line = explore.trends(title_line, x=x_line, y=y_line, ylabel=ylabel_line)
        fig_line.savefig(GRAPH_DIR / filename_line)
        logger.info(f'Figure saved to {GRAPH_DIR}')

    # ----------------------------------------------------------
    # 3. Relationship between variables

    plot_relation = [
        ('Temperature vs CO(GT)', 'T', 'CO(GT)', 'Temperature (°C)', 'CO Concentration (mg/m³)', 'corr_T_vs_CO(GT).png'),
        ('Temperature vs NOx(GT)', 'T', 'NOx(GT)', 'Temperature (°C)', 'NOx Concentration (µg/m³)', 'corr_T_vs_NOx(GT).png'),
        ('Temperature vs NO2(GT)', 'T', 'NO2(GT)', 'Temperature (°C)', 'NO2 Concentration (µg/m³)', 'corr_T_vs_NO2(GT).png'),
        ('CO(GT) vs NOx(GT)', 'CO(GT)', 'NOx(GT)', 'CO Concentration (mg/m³)', 'NOx Concentration (µg/m³)', 'corr_CO(GT)_vs_NOx(GT).png'),
        ('CO(GT) vs NO2(GT)', 'CO(GT)', 'NO2(GT)', 'CO Concentration (mg/m³)', 'NO2 Concentration (µg/m³)', 'corr_CO(GT)_vs_NO2(GT).png'),
        ('NOx(GT) vs NO2(GT)', 'NOx(GT)', 'NO2(GT)', 'NOx Concentration (µg/m³)', 'NO2 Concentration (µg/m³)', 'corr_NOx(GT)_vs_NO2(GT).png')
    ]

    for title_scat, x_scat, y_scat, xlabel_scat, ylabel_scat, filename_scat in plot_relation:
        logger.info(f'Creating scatterplot for {title_scat}')
        fig_scat = explore.relation(title_scat, x=x_scat, y=y_scat, xlabel=xlabel_scat, ylabel=ylabel_scat)
        fig_scat.savefig(GRAPH_DIR / filename_scat)
        logger.info(f'Figure saved to {GRAPH_DIR}')

    # ----------------------------------------------------------
    # Sensor Performance Evaluation
    # ----------------------------------------------------------

    logger.info('Creating object sensor with dataframe')
    sensor = Sensor(df)

    # ----------------------------------------------------------
    # 1. Sensor Correlation

    correlate = [
        ('Correlation Between PT08.S1(CO) Sensor Signal and CO Concentration', 'PT08.S1(CO)', 'CO(GT)', 'correlation_PT08.S1(CO)_CO(GT).png'),
        ('Correlation Between PT08.S3(NOx) Sensor Signal and NOx Concentration', 'PT08.S3(NOx)', 'NOx(GT)', 'correlation_PT08.S3(NOx)_NOx(GT).png'),
        ('Correlation Between PT08.S4(NO2) Sensor Signal and NO2 Concentration', 'PT08.S4(NO2)', 'NO2(GT)', 'correlation_PT08.S4(NO2)_NO2(GT).png')
    ]

    for title_cor, x_cor, y_cor, filename_cor in correlate:
        logger.info(f'Creating scatterplot for {title_cor}')
        fig_cor = sensor.sensor_correlation(title_cor, x=x_cor, y=y_cor)
        fig_cor.savefig(SENSOR_DIR / filename_cor)
        logger.info(f'Figure saved to {SENSOR_DIR}')

    # ----------------------------------------------------------
    # 2. Residual Analysis, Mean Absolute Error, Root Mean Square Error, Split Validation

    res_analysis = [
        ('PT08.S1(CO)', 'CO(GT)', 'Residual Distribution: PT08.S1(CO) vs CO(GGT)', 'residual_PT08.S1(CO)_CO(GT).png'),
        ('PT08.S3(NOx)', 'NOx(GT)', 'Residual Distribution: PT08.S3(NOx) vs NOx(GT)', 'residual_PT08.S3(NOx)_NOx(GT).png'),
        ('PT08.S4(NO2)', 'NO2(GT)', 'Residual Distribution: PT08.S4(NO2) vs NO2(GT)', 'residual_PT08.S4(NO2)_NO2(GT).png')
    ]

    results = []

    for feature, target, title_res, filename_res in res_analysis:

        # residual analysis
        logger.info(f'Creating histplot for {title_res}')
        fig_res, model, X, y, y_pred = sensor.residual_analysis(feature, target, title_res)
        fig_res.savefig(SENSOR_DIR / filename_res)
        logger.info(f'Figure saved to {SENSOR_DIR}')

        # MAE, RMSE
        logger.info('Calculating MAE and RMSE')
        mae, rmse = sensor.accurate(y, y_pred)

        # Split Validation
        logger.info('Splitting, Training, Testiing Data')
        r2_train, r2_test = sensor.split_validation(X, y)

        # saving results to csv file
        results.append({
            'feature': feature,
            'target': target,
            'coef': float(model.coef_[0]),
            'intercept': float(model.intercept_),
            'mae': mae,
            'rmse': rmse,
            'r2_train': r2_train,
            'r2_test': r2_test
        })

        pd.DataFrame(results).to_csv(SENSOR_DIR / 'sensor_results.csv', index=False)

if __name__ == '__main__':
    start = time.time()
    logger.info(f'Program Started. {start}')

    try:
        main()
        elapsed = time.time() - start
        logger.info(f'Program Done in {elapsed:.2f} seconds.')

    except Exception as e:
        elapsed = time.time() - start
        logger.error(f'Program Crashed after {elapsed:.2f} seconds', exc_info=True)