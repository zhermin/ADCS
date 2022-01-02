import time
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np

import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from src.kla_reader import Wafer

def load_trained_model(
        subdir: str,
        model_name: str = '',
    ) -> 'tf.keras.Model':
    """Finds and loads Keras model

    If given a model_name, tries to find the model in the current working
    directory. But if invalid or can't be found, it'll use the latest model. 

    Args:
        subdir (str): 
            Subdirectory, eg. backside or edgenormal since the
            correct models will be stored in the correct subdir. 
        model_name (str), default = Empty String:
            Specific model name to use, if desired. You may leave
            this argument empty to use the latest model. 

    Returns:
        tf.keras.Model:
            Loaded Keras model using the model_name or the latest model found. 
    """

    adcs_path = Path.cwd()

    found_models = Path.joinpath(adcs_path, 'models', subdir).glob('*.h5')
    found_models = sorted(found_models, key=lambda p: p.stat().st_ctime)

    if len(found_models) == 0:
        raise FileNotFoundError('No models found! Check if the folder structure is correct and if there are .h5 files inside')

    if model_name:
        if not model_name.endswith('.h5'): model_name = f'{model_name}.h5'
        selected_model = Path.joinpath(adcs_path, 'models', subdir, model_name)

    if not model_name or selected_model not in found_models:
        logging.info(f'Using the latest model as no custom model name specified or found')
        selected_model = found_models[-1]

    logging.info(f'[{subdir.upper()}] Model Loaded:\n{selected_model.name}')
    return load_model(selected_model, compile=False)


class Predictor:
    """Prepares data and model variables for prediction and saves results to CSV

    This helper class takes in some model variables such as image size and also
    the loaded Keras model to perform model inference. It also loads the data
    from a dataframe using Keras' ImageDataGenerator. 

    flow_from_dataframe() was used because it is easier to load the relevant
    filenames from a given list of wafers instead of setting up the correct
    folder structure to use flow_from_directory(). 

    The prediction results are also saved at the end into a CSV and can be found
    in .../ADCS/results/production. 

    Attributes (Init):
        img_size (int): 
            Image size must be same as size used during model training. 
            Usually pre-trained models would use size 224, which was what I used. 
        batch_size (int): 
            Number of images to predict on per batch, higher will require 
            more memory and RAM and might crash the system if too high. 
        conf_threshold (float): 
            Confidence % threshold to meet for a prediction to be considered a
            confident prediction. Set from settings.yaml file. 
        wafers (list of Wafer): 
            List of wafers to be predicted on, eg. bs_wafers or en_wafers. 
        defect_mapping (dict{str: int}): 
            The correct defect_mapping, either bs_ or en_. 
            Set from settings.yaml file. 
        adc_drive_new (Path): 
            Path where all new images from AVI are transferred to
        subdir (str): 
            Subdirectory, eg. backside or edgenormal since the
            correct images will be stored in the correct subdir. 
        model (tf.keras.Model): 
            Loaded Keras model after performing load_trained_model()

    Attributes (Post-Init):
        defect_enum (dict{enum: str}):
            Dictionary of enumerated key-value pairs for each
            defect_mapping key, which are the defect names. 
        class_enum (dict{enum: int}):
            Dictionary of enumerated key-value pairs for each
            defect_mapping value, which are the defect classnumbers. 
        filenames (list of str):
            List of filename strings extracted from the given list
            of wafers. Only include if wafer.exists is True. 
        df (pd.DataFrame):
            Dataframe to store relevant filenames for loading. It will 
            also be populated with other useful information such as 
            datetime of prediction and the predictions themselves. 
        generator (keras ImageDataGenerator):
            Loaded image generator from self.df
    """

    def __init__(
            self,
            img_size: int,
            batch_size: int,
            conf_threshold: float,
            wafers: 'list[Wafer]',
            defect_mapping: 'dict[str, int]',
            adc_drive_new: Path,
            subdir: str,
            model: 'tf.keras.Model',
        ) -> None:

        self.img_size = img_size
        self.batch_size = batch_size
        self.conf_threshold = conf_threshold
        self.wafers = wafers
        self.defect_mapping = defect_mapping
        self.adc_drive_new = adc_drive_new
        self.subdir = subdir
        self.model = model

        self.defect_enum = dict(enumerate(self.defect_mapping.keys()))
        self.class_enum = dict(enumerate(self.defect_mapping.values()))

        self.filenames = [
            str(Path.joinpath(adc_drive_new, wafer.filename)).replace(';','')
            for wafer in self.wafers if wafer.exists
        ]
        self.df = pd.DataFrame({
            'datetime': time.strftime('%d-%m-%Y %H:%M:%S'), 
            'filenames': self.filenames,
        })

    def load_generator(self, **kwargs) -> None:
        """Loads image generator from dataframe"""
        self.generator = ImageDataGenerator(**kwargs).flow_from_dataframe(
            dataframe=self.df,
            x_col='filenames',
            target_size=(self.img_size, self.img_size),
            class_mode=None,
            shuffle=False,
            batch_size=self.batch_size,
        )

    def predict_imgs(self, overwrite_results: bool = False) -> None:
        """Predicts on the image generator using CPU and saves results to df"""
        with tf.device('/cpu:0'):
            pred_raw = self.model.predict(self.generator, verbose=1)

        self.organise_results(pred_raw)
        self.save_results(overwrite_results)

    def organise_results(self, pred_raw: np.array) -> None:
        """Stores other metrics to df such as confidence and predicted classnumbers"""
        self.df = pd.concat([
                self.df, 
                pd.DataFrame(pred_raw, columns=self.defect_mapping.keys())
            ], 
            axis=1,
        )
        self.df['confidence'] = np.max(pred_raw, axis=1)
        self.df['unconfident'] = np.where(self.df['confidence'] < self.conf_threshold, True, False)
        self.df['prediction'] = np.argmax(pred_raw, axis=1)
        self.df['prediction_name'] = self.df['prediction'].map(self.defect_enum.get)
        self.df['new_defect_code'] = self.df['prediction'].map(self.class_enum.get)

    def save_results(self, overwrite_results: bool = False) -> None:
        """Saves or overwrites df into CSV format for future reference"""
        csv_path = Path.joinpath(Path.cwd(), 'results', 'production', self.subdir, f'{self.subdir}_pred.csv')
        if overwrite_results or not csv_path.is_file():
            self.df.to_csv(csv_path, index=False)
            logging.info(f'Results overwritten in .../{self.subdir}_pred.csv')
        else:
            df_full_history = pd.read_csv(csv_path)
            df_full_history = df_full_history.append(self.df)
            df_full_history.to_csv(csv_path, index=False)
            logging.info(f'Results added to .../{self.subdir}_pred.csv')