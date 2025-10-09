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

    # Data cleaning
    logger.info('Cleaning Data')
    df = data_clean()
    logger.info('Data Cleaned')

    # Exploratory Data Analysis
    logger.info('Create object explore with dataframe')
    explore = Exploratory(df)

    # for CO distribution
    title_co = 'Distribution of CO Concentration (Ground Truth)'
    var_co = 'CO(GT)'
    xlabel_co = 'CO Concentration (GT)'

    logger.info('Creating histogram for CO distribution')
    fig_co = explore.distribution(title_co, var_co, xlabel_co)
    logger.info('Created.')

    # save figure
    fig_co.savefig(GRAPH_DIR / 'histogram_co.png')
    logger.info(f'Figure saved to {GRAPH_DIR}')

if __name__ == '__main__':
    main()