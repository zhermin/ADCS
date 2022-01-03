import os, time
from pathlib import Path
from typing import Union

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logger = logging.getLogger(__name__)

def duration(start: float) -> float:
    """Returns runtime using current time minus input start time"""
    return round((time.perf_counter() - start)/60, 2)

import threading
import ctypes
import tkinter as tk

#------------------------------ Production Mode -------------------------------#

class Production(threading.Thread):
    """Production Mode: Performs image classification on wafer scans continuously

    # Description
    This threaded class starts if adcs_mode is set to PRODUCTION in the GUI. 

    This will first load the necessary back and edge models to avoid reloading
    each cycle in the continuous loop. If the model names are changed in the
    settings.yaml file, the new models will be used in the next cycle. 

    The loop is used to continuously check the adc_drive_new folder for any new
    KLA files. Take note, the system won't start classifying images if there are 
    no KLA files found even if there are wafer scans inside. This is because the 
    KLA files are needed to find the relevant images for the models and to allow 
    them to be edited before being sent to the K drive. 

    Even though FS AVI scans are very slow, if we were to run a full inspection,
    meaning Frontside, Backside and Edge recipes in one go, we cannot get
    around this FS-scan bottleneck. Hence, we just run everything sequentially 
    similar to how the AVI recipes are done. 

    Because FS logic is not implemented yet, this system will only predict on
    back and edge scans while frontside scans will be untouched and moved into its
    own folder inside the "old" directory. However, whoever takes over this project
    in the future can just roughly follow the way I implemented for BE models for FS. 

    # System Flow
    Refer to the "System Flow" section in the README.pdf document. 

    Attributes:
        settings (dict):
            Dictionary of user-specified key-value
            pairs from the settings.yaml file.
        adcs_started (tk.BooleanVar):
            Tkinter boolean variable to indicate if PRODUCTION loop
            is running or not. Updates to False if the loop is stopped. 
        start_stop_btn_text (tk.StringVar):
            Tkinter string variable for the START/STOP button text.
            Updates from STOP to START if the loop is stopped. 
        is_running (tk.Label):
            Tkinter red Label text to indicate "Running" or not. 
        user_inputs (list of tk.Radiobutton, tk.Button, tk.Entry):
            List of Tkinter user inputs such as the radio buttons, all 
            buttons and entry boxes to re-enable them if the loop is stopped.
    """

    def __init__(
            self,
            settings: dict,
            adcs_started: tk.BooleanVar,
            start_stop_btn_text: tk.StringVar,
            is_running: tk.Label,
            user_inputs: 'list[Union[tk.Radiobutton, tk.Button, tk.Entry]]',
        ) -> None:

        threading.Thread.__init__(self)
        self.daemon = True
        self.settings = settings
        self.adcs_started = adcs_started
        self.start_stop_btn_text = start_stop_btn_text
        self.is_running = is_running
        self.user_inputs = user_inputs
        self.e = threading.Event()
        logging.warning('Production Started')
        logging.info('Loading Models (this might take a while)...')

    def run(self) -> None:
        try:

            from src.kla_reader import KlaReader
            from src.predictor import Predictor
            from src.predictor import load_trained_model
            from tensorflow.keras.applications import mobilenet_v2

            #--------------------------- Initial Model Loading ----------------------------#

            bs_subdir = 'backside'
            current_bs_model = self.settings['bs_model']
            bs_model = load_trained_model(
                subdir=bs_subdir,
                model_name=current_bs_model,
            )

            en_subdir = 'edgenormal'
            current_en_model = self.settings['en_model']
            en_model = load_trained_model(
                subdir=en_subdir,
                model_name=current_en_model,
            )

            fs_subdir = 'frontside'

            no_kla_alert = True
            logging.info('All Models Ready')

            #--------------------------------- Main Loop ----------------------------------#

            while True:

                #---------------------------------- Settings ----------------------------------#

                adc_drive_new = Path(self.settings['adc_drive_new'])
                adc_drive_old = Path(self.settings['adc_drive_old'])
                k_drive = Path(self.settings['k_drive'])

                for path in (adc_drive_new, adc_drive_old, k_drive):
                    if not path.is_dir():
                        raise NotADirectoryError(f'{path} cannot be found, please check folder location settings again')

                IMG_SIZE = 224
                BATCH_SIZE = self.settings['BATCH_SIZE']
                CONF_THRESHOLD = self.settings['CONF_THRESHOLD']/100

                #------------------------- Generate Folder Structure --------------------------#

                for submode in (bs_subdir, en_subdir, fs_subdir):
                    submode_path = Path.joinpath(adc_drive_old, submode)
                    if submode == 'frontside':
                        submode_path.mkdir(parents=True, exist_ok=True)
                        break
                    for subdir in ('test', 'trainval', 'unsorted'):
                        subdir_path = Path.joinpath(submode_path, subdir)
                        subdir_path.mkdir(parents=True, exist_ok=True)
                        if submode == 'backside':
                            for defect in self.settings['bs_defect_mapping'].keys():
                                defect_path = Path.joinpath(subdir_path, defect)
                                defect_path.mkdir(parents=True, exist_ok=True)
                        elif submode == 'edgenormal':
                            for defect in self.settings['en_defect_mapping'].keys():
                                defect_path = Path.joinpath(subdir_path, defect)
                                defect_path.mkdir(parents=True, exist_ok=True)

                unclassified_path = Path.joinpath(adc_drive_old, 'unclassified')
                unclassified_path.mkdir(parents=True, exist_ok=True)

                #------------------------------ Find KLA Files --------------------------------#

                kla_files = []
                extensions = ['*.00*', '*.kla']
                for ext in extensions:
                    kla_files.extend(adc_drive_new.glob(ext))
                kla_files = sorted(kla_files, key=lambda p: p.stat().st_ctime)

                # If no KLA files found, loop to beginning
                if len(kla_files) == 0:
                    if no_kla_alert: 
                        logging.info('-'*50)
                        logging.info(f'No KLA files found, checking every {self.settings["pause_if_no_kla"]} seconds...')
                        no_kla_alert = False

                    self.e.wait(self.settings['pause_if_no_kla'])
                    continue

                # Otherwise, start reading the oldest KLA file and classify its wafer lot
                logging.info('-'*50)
                no_kla_alert = True
                start = time.perf_counter()

                #---------------------------- Read Oldest KLA File ----------------------------#

                kla_file = kla_files[0]
                logging.info(f'KLA File Loaded: {kla_file.name}')

                reader = KlaReader(kla_file)
                reader.check_imgs_in_kla(
                    directory=adc_drive_new,
                    retry_num=self.settings['times_to_find_imgs'],
                    timeout=self.settings['pause_to_find_imgs'],
                )

                #----------------------------- Backside Pipeline ------------------------------#

                if any(wafer.exists for wafer in reader.bs_wafers):

                    logging.info(f'{bs_subdir.title()} Classification: {len(reader.bs_wafers)} images')

                    bs_predictor = Predictor(
                        img_size=IMG_SIZE,
                        batch_size=BATCH_SIZE,
                        conf_threshold=CONF_THRESHOLD,
                        wafers=reader.bs_wafers,
                        defect_mapping=self.settings['bs_defect_mapping'],
                        adc_drive_new=adc_drive_new,
                        subdir=bs_subdir,
                        model=bs_model,
                    )
                    bs_predictor.load_generator(preprocessing_function=mobilenet_v2.preprocess_input)
                    bs_predictor.predict_imgs(overwrite_results=False)

                    reader.bs_wafers = reader.edit_kla(reader.bs_wafers, bs_predictor.df['new_defect_code'])
                    reader.bs_wafers = reader.move_predicted_lots(
                        wafers=reader.bs_wafers,
                        source=adc_drive_new,
                        copy_destination=k_drive,
                        move_destination=Path.joinpath(adc_drive_old, bs_subdir, 'unsorted'),
                        defect_mapping=self.settings['bs_defect_mapping'],
                    )

                #---------------------------- Edgenormal Pipeline -----------------------------#

                if any(wafer.exists for wafer in reader.en_wafers):

                    logging.info(f'{en_subdir.title()} Classification: {len(reader.en_wafers)} images')

                    en_predictor = Predictor(
                        img_size=IMG_SIZE,
                        batch_size=BATCH_SIZE,
                        conf_threshold=CONF_THRESHOLD,
                        wafers=reader.en_wafers,
                        defect_mapping=self.settings['en_defect_mapping'],
                        adc_drive_new=adc_drive_new,
                        subdir=en_subdir,
                        model=en_model,
                    )
                    en_predictor.load_generator(preprocessing_function=mobilenet_v2.preprocess_input)
                    en_predictor.predict_imgs(overwrite_results=False)

                    reader.en_wafers = reader.edit_kla(reader.en_wafers, en_predictor.df['new_defect_code'])
                    reader.en_wafers = reader.move_predicted_lots(
                        wafers=reader.en_wafers,
                        source=adc_drive_new,
                        copy_destination=k_drive,
                        move_destination=Path.joinpath(adc_drive_old, en_subdir, 'unsorted'),
                        defect_mapping=self.settings['en_defect_mapping'],
                    )

                #------------------------------------------------------------------------------#
                #------------------- [TODO] Frontside Pipeline (Incomplete) -------------------#
                #------------------------------------------------------------------------------#

                if any(wafer.exists for wafer in reader.fs_wafers):

                    logging.info(f'{fs_subdir.title()} IGNORED: {len(reader.fs_wafers)} images')

                    # fs_predictor = Predictor(
                    #     ...
                    # )
                    # fs_predictor.load_generator(...)
                    # fs_predictor.predict_imgs(overwrite_results=False)

                    # reader.fs_wafers = reader.edit_kla(reader.fs_wafers, fs_predictor.df['new_defect_code'])
                    reader.fs_wafers = reader.move_predicted_lots(
                        wafers=reader.fs_wafers,
                        source=adc_drive_new,
                        copy_destination=k_drive,
                        move_destination=Path.joinpath(adc_drive_old, fs_subdir),
                    )

                #-------------------------- Move Unclassified Wafers --------------------------#

                reader.move_predicted_lots(
                    wafers=reader.wafers,
                    source=adc_drive_new,
                    copy_destination=k_drive,
                    move_destination=Path.joinpath(adc_drive_old, 'unclassified'),
                )
                reader.remove_kla_file()

                #------------------------------ End of Main Loop ------------------------------#

                logging.info(f'{kla_file.name} completed!')
                logging.info(f'Duration: {duration(start)} mins')
                self.e.wait(self.settings['pause_if_kla'])

        except Exception as e:
            logging.exception('Error Occurred; Run Cancelled; Please Fix')

        finally:
            self.adcs_started.set(False)
            self.start_stop_btn_text.set('START')
            self.is_running.pack_forget()
            for child in self.user_inputs:
                child.configure(state='normal')
            logging.warning('Production Stopped')

    def get_id(self) -> int:
        """Returns the ID of the current thread"""
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def stop(self) -> None:
        """Forcefully stops the thread by raising an exception

        Reference: https://www.geeksforgeeks.org/python-different-ways-to-kill-a-thread/
        """

        self.is_running.pack_forget()
        self.e.set()
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)

#------------------------------- Training Mode --------------------------------#

class Training(threading.Thread):
    """Training Mode: Trains .h5 machine learning models using wafer scans

    # Description
    This threaded class starts if adcs_mode is set to TRAINING in the GUI and
    calls the train_once function from src/be_trainer.py to start training a model. 

    This will only work for backside and edgenormal models since frontside is not 
    yet implemented. The code in be_trainer.py are the code used to train some of 
    the models saved in the ADCS path. 

    You should run this mode after the unsorted folder in the backside/edgenormal
    data folders have been manually sorted and moved into the trainval folder. 
    This is because we don't want future trained models to learn the wrong 
    classifications. 

    The settings here and in the settings.yaml file for the trainer are not very
    extensive as otherwise, it will be too overwhelming. If you want to tweak the 
    training code and other hyperparameters, do edit the be_trainer.py file instead. 

    # Usage Notes
    Based on the number of training_runs, this function will run that number of 
    times and try to create models. If the models pass the training_saving_threshold,
    then it will be saved. However, all details of the runs will be saved in
    CSV format under the /ADCS/results/training folder for future reference. 

    Attributes:
        settings (dict):
            Dictionary of user-specified key-value
            pairs from the settings.yaml file.
        adcs_started (tk.BooleanVar):
            Tkinter boolean variable to indicate if TRAINING run
            is running or not. Updates to False if the run is cancelled. 
        start_stop_btn_text (tk.StringVar):
            Tkinter string variable for the START/STOP button text.
            Updates from STOP to START if the current run is cancelled. 
        is_running (tk.Label):
            Tkinter red Label text to indicate "Running" or not. 
        user_inputs (list of tk.Radiobutton, tk.Button, tk.Entry):
            List of Tkinter user inputs such as the radio buttons, all buttons
            and entry boxes to re-enable them if the current run is cancelled. 
    """

    def __init__(
            self,
            settings: dict,
            adcs_started: tk.BooleanVar,
            start_stop_btn_text: tk.StringVar,
            is_running: tk.Label,
            user_inputs: 'list[Union[tk.Radiobutton, tk.Button, tk.Entry]]',
        ) -> None:

        threading.Thread.__init__(self)
        self.daemon = True
        self.settings = settings
        self.adcs_started = adcs_started
        self.start_stop_btn_text = start_stop_btn_text
        self.is_running = is_running
        self.user_inputs = user_inputs
        logging.warning('Training Started')

    def run(self) -> None:
        try:
            from src.be_trainer import train_once
            from statistics import mean

            start = time.perf_counter()

            if self.settings["training_mode"]:
                mode_name = 'Training'
                training_runs = self.settings['training_runs']
            else:
                mode_name = 'Testing'
                training_runs = 1

            logging.info(f'{mode_name} Mode: {self.settings["training_subdir"]}')

            if self.settings['training_subdir'] == 'BACKSIDE':
                defect_mapping = self.settings['bs_defect_mapping']
            elif self.settings['training_subdir'] == 'EDGENORMAL':
                defect_mapping = self.settings['en_defect_mapping']

            runs = [train_once(
                run_idx=run_idx+1,
                data_path=Path(self.settings['adc_drive_old']),
                subdir=self.settings['training_subdir'].lower(),
                defect_mapping=defect_mapping,
                n=self.settings['training_n'],
                saving_threshold=self.settings['training_saving_threshold']/100,
                batch_size=self.settings['BATCH_SIZE'],
                dense_layers=self.settings['dense_layers'],
                dense_layer_size=self.settings['dense_layer_size'],
                dropout=self.settings['dropout'],
                patience=self.settings['patience'],
                training_mode=self.settings['training_mode'],
                test_model=self.settings['test_model'],
            ) for run_idx in range(training_runs)]

            logging.info('--- OVERALL ---')
            [logging.info(f'{i}) {run:.2%}') for i, run in enumerate(runs, 1)]
            logging.info('--- AVERAGE ---')
            logging.info(f'{mean(runs):.2%}')
            logging.info('--- TOTAL DURATION ---')
            logging.info(f'{duration(start)} mins')

        except Exception as e:
            logging.exception('Error Occurred; Run Cancelled; Please Fix')

        finally:
            self.adcs_started.set(False)
            self.start_stop_btn_text.set('START')
            for child in self.user_inputs:
                child.configure(state='normal')
            logging.warning('Training Stopped')
            self.stop()

    def get_id(self) -> int:
        """Returns the ID of the current thread"""
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def stop(self) -> None:
        """Forcefully stops the thread by raising an exception

        Reference: https://www.geeksforgeeks.org/python-different-ways-to-kill-a-thread/
        """

        self.is_running.pack_forget()
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)