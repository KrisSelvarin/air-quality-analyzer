# main.py 

from pathlib import Path
from src.logger_config import setup_logger
from src.clean import data_clean
from src.eda import Exploratory
import time

# logger
logger = setup_logger(__name__)

# Directory
BASE_DIR = Path(__file__).resolve().parent
GRAPH_DIR = BASE_DIR / 'output' / 'graphs'
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

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
        ('Distribution of Non-Metanic HydroCarbons Concentration (Ground Truth)', 'NMHC(GT)', 'NMHC Concentration (GT)', 'histogram_nmhc.png'),
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
        ('NMHC Concentration from 2004-03-10 to 2005-04-04', 'Date', 'NMHC(GT)', 'NMHC Concentration Daily Avg. Level', 'line_nmhc.png'),
        ('NOx Concentration from 2004-03-10 to 2005-04-04', 'Date', 'NOx(GT)', 'NOx Concentration Daily Avg. Level', 'line_nox.png'),
        ('NO2 Concentration from 2004-03-10 to 2005-04-04', 'Date', 'NO2(GT)', 'NO2 Concentration Daily Avg. Level', 'line_no2.png')
    ]

    for title_line, x_line, y_line, ylabel_line, filename_line in plot_trends:
        logger.info(f'Creating line graph for {y_line} trend')
        fig_line = explore.trends(title_line, x=x_line, y=y_line, ylabel=ylabel_line)
        fig_line.savefig(GRAPH_DIR / filename_line)
        logger.info(f'Figure saved to {GRAPH_DIR}')


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