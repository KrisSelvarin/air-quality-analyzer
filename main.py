# main.py 

from pathlib import Path
from src.logger_config import setup_logger
from src.clean import data_clean
from src.eda import Exploratory

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

    plot = [
        ('Distribution of Carbon Monoxide Concentration (Ground Truth)', 'CO(GT)', 'CO Concentration (GT)', 'histogram_co.png'),
        ('Distribution of Non-Metanic HydroCarbons (Ground Truth)', 'NMHC(GT)', 'NMHC Concentration (GT)', 'histogram_nmhc.png'),
        ('Distribution of Nitrous Oxides (Ground Truth)', 'NOx(GT)', 'NOx Concentration (GT)', 'histogram_nox.png'),
        ('Distribution of Nitrous Dioxide (Ground Truth)', 'NO2(GT)', 'NOx Concentration (GT)', 'histogram_no2.png')
    ]

    for title, var, xlabel, filename in plot:
        logger.info(f'Creating histogram for {var} distribution')
        fig = explore.distribution(title, var, xlabel)
        fig.savefig(GRAPH_DIR / filename)
        logger.info(f'Figure saved to {GRAPH_DIR}')

    # ----------------------------------------------------------
    # 2. Trends over time

    # TODO: function for line plot



if __name__ == '__main__':
    main()