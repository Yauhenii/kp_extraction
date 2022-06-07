import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from project.data.datasets.dataset_factory import dataset_processor_factory

project_dir = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = f"{project_dir}/data/processed"


@click.command()
@click.option('--dataset_name', type=click.STRING)
@click.option('--input_dir', type=click.Path(exists=True))
@click.option('--subset', type=click.STRING)
@click.option('--with_undecided_pairs', type=click.BOOL)
def main(dataset_name, input_dir, subset, with_undecided_pairs):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    processor = dataset_processor_factory(f"{dataset_name}_processor")
    
    processed_dir = f"{PROCESSED_DATA_PATH}/{dataset_name}"
    
    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)
    
    processor.process_dataset(input_dir, subset, processed_dir, with_undecided_pairs)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    
    main()
