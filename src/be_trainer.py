import time, random
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
from PIL import Image

from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.utils import to_categorical

def duration(start: float) -> float:
    """Returns runtime using current time minus input start time"""
    return round((time.perf_counter() - start)/60, 2)

#----------------------------- Dataset Balancing ------------------------------#

def balance_dataset(
        trainval_path: str,
        defect: str,
        n: int,
        img_size: int,
        val_split: float,
    ) -> tuple:
    """Oversamples dataset so that all classes are balanced

    Each class will either oversample (if less than n) or undersample (if more than n).
    If oversampling, it will simply duplicate the images n/num_images times to reach n. 
    If undersampling, it will randomly sample n images. At the end, each class will 
    have n number of images and are stored in the 4 arrays, x/y train and test lists, 
    split according to the val_split percentage. 

    Args:
        trainval_path (str): 
            Path where all trainval images are found, should be stored in ADCS' "old" folder
        defect (str):
            The defect name, eg. aok or chipping, so that the defect images can be found
            inside these defect folders. 
        n (int): 
            Number of images each class should have at the end. This number should be
            chosen based on the number of images in each class in the trainval folder. 
            It should not be lower than the most minority class or higher than the most
            majority class. 
        img_size (int): Image size for the model. Most pre-trained models use size 224. 
        val_split (float): Percentage of images to be used for validation

    Returns:
        Tuple of 4 Lists of np.arrays (x_train, y_train, x_test, y_test):
            They contain the images loaded using PILLOW and resized to img_size. 
            Thus, they will be in the form of np.array tensors (or matrices), basically
            a grid of decimals, which is the format computers can understand. 
    """

    dataset = list(Path.joinpath(trainval_path, defect).glob('*.jpg'))
    dataset_len = len(dataset)
    if len(dataset) == 0:
        raise ValueError(f'No {defect} images found. Please follow the folder structures in README.md for /old/.../trainval/{defect} and .../test/{defect} and see if there are any {defect} images in these folders')

    if dataset_len > n: 
        dataset = random.sample(dataset, n)
    else: 
        dataset = random.sample(dataset, dataset_len)

    logging.info(f'{defect} ({len(dataset)})')
    train_len, test_len = round((1-val_split)*len(dataset)), round(val_split*len(dataset))

    x_train, y_train, x_test, y_test = [], [], [], []
    for _ in range(test_len): # Validation Split
        data = dataset.pop()
        img = Image.open(data)
        img = img.resize((img_size, img_size))
        img = np.asarray(img)[:, :, :3]

        x_test.append(img)
        y_test.append(defect)

    for data in dataset: # Training Split
        img = Image.open(data)
        img = img.resize((img_size, img_size))
        img = np.asarray(img)[:, :, :3]
        if dataset_len < n: # Oversampling by duplication if less than n
            multiplier = int(np.ceil(n*(1-val_split)/train_len))
            for _ in range(multiplier):
                x_train.append(img)
                y_train.append(defect)
        else:
            x_train.append(img)
            y_train.append(defect)

    x_train, y_train = x_train[:round(n*(1-val_split))], y_train[:round(n*(1-val_split))]
    return x_train, y_train, x_test, y_test

#--------------------------------- Main Loop ----------------------------------#

def train_once(
        run_idx: int,
        data_path: Path,
        subdir: str,
        defect_mapping: 'dict[str, int]',
        n: int = 300,
        saving_threshold: float = 95,
        img_size: int = 224,
        val_split: float = 0.2,
        batch_size: int = 16,
        dense_layers: int = 1,
        dense_layer_size: int = 16,
        dropout: float = 0.2,
        patience: int = 10,
        training_mode: bool = True,
        test_model: str = '',
    ) -> float:
    """Main loop to run the training code once

    Depending on max run_idx, it will repeat for that many times. The point of
    repeating is that because the images are sampled randomly, the accuracy 
    will be different each time. Taking the average like this is called k-fold 
    validation but this is not too useful in getting the best model. 

    There are certain hyperparamets to tune if you want to. However, a bigger 
    dense_layer_size leads to a bigger .h5 model that takes up more memory. 

    Args:
        run_idx (int): Current iteration value of the run
        data_path (Path): 
            Data path where all train and test images are stored, 
            should be in the ADCS old folder. 
        subdir (str): 
            Subdirectory, eg. backside or edgenormal since the
            correct images will be stored in the correct subdir.
        defect_mapping (dict{str: int}): 
            The correct defect_mapping, either bs_ or en_.
            Set from settings.yaml file. 
        n (int), default = 300: 
            Number of images each class should have at the end. 
            More details in above balance_defect() function. 
        saving_threshold (float), default = 95: 
            % accuracy threshold to be considered good enough
            to save the model. 
        img_size (int), default = 224: 
            Image size for the model. Most pre-trained models use size 224. 
        val_split (float), default = 0.2: Percentage of images to be used for validation
        batch_size (int), default = 16: 
            Number of images to predict on per batch, higher will require 
            more memory and RAM and might crash the system if too high. 
        training_mode (bool), default = True: 
            Either training mode (True) or testing mode (False). 
            Used to test a specific trained model on the test set. 
        test_model (str), default = '': 
            If testing mode (training_mode = False), specify a model_name here to test. 
    """

    logging.info(f'|-------- RUN {run_idx} --------|')

    adcs_path = Path.cwd()
    data_path = Path.joinpath(data_path, subdir)
    trainval_path = Path.joinpath(data_path, 'trainval')
    test_path = Path.joinpath(data_path, 'test')

    defect_keys = defect_mapping.keys()
    NUM_CLASSES = len(defect_keys)
    defect_enum = dict(enumerate(defect_keys))

    PREPROCESSING_FUNCTION = mobilenet_v2.preprocess_input

    #------------------------------------------------------------------------------#
    #----------------------------- Training Pipeline ------------------------------#
    #------------------------------------------------------------------------------#

    if training_mode:

        #------------------------------ Dataset Creation ------------------------------#

        logging.info('--- COLLECTING IMAGES ---')

        x_train, y_train, x_test, y_test = [], [], [], []
        for defect in defect_keys:
            defect_x_train, defect_y_train, defect_x_test, defect_y_test = balance_dataset(
                trainval_path=trainval_path,
                defect=defect,
                n=n,
                img_size=img_size,
                val_split=val_split,
            )
            x_train.extend(defect_x_train), y_train.extend(defect_y_train), x_test.extend(defect_x_test), y_test.extend(defect_y_test)

        x_train, y_train, x_test, y_test = np.array(x_train, dtype=object), np.asarray(y_train), np.array(x_test, dtype=object), np.asarray(y_test)
        logging.info(f'Train: {x_train.shape}')
        logging.info(f'Test: {x_test.shape}')

        #------------------------------ Dataset Encoding ------------------------------#

        defect_enum_inv = {v: k for k, v in defect_enum.items()}
        encode = np.vectorize(defect_enum_inv.get)
        y_train_encoded = encode(y_train)
        y_test_encoded = encode(y_test)

        y_train_one_hot = to_categorical(y_train_encoded)
        y_test_one_hot = to_categorical(y_test_encoded)

        #------------------------------ Dataset Loading -------------------------------#

        train_datagen = ImageDataGenerator(
            # brightness_range=[0.8,1.2],
            # shear_range=0.2,
            # zoom_range=[0.8,1.2],
            # width_shift_range=0.2,
            # height_shift_range=0.2,
            # fill_mode='constant',
            # cval=128, #255,
            vertical_flip=True,
            # horizontal_flip=True,
            preprocessing_function=PREPROCESSING_FUNCTION,
        )

        train_generator = train_datagen.flow(
            x=x_train,
            y=y_train_one_hot,
            batch_size=batch_size,
            shuffle=True,
        )

        val_generator = ImageDataGenerator(preprocessing_function=PREPROCESSING_FUNCTION).flow(
            x=x_test,
            y=y_test_one_hot,
            batch_size=batch_size,
            shuffle=False,
        )

        #-------------------------------- Model Setup ---------------------------------#

        logging.info('--- START TRAINING ---')

        pretrained_model = mobilenet_v2.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(img_size, img_size, 3),
        )
        PRETRAINED_NAME = pretrained_model.name.split('_')[0]

        for layer in pretrained_model.layers: layer.trainable = False

        #------------------------- Model Layer Customisations -------------------------#

        x = pretrained_model.output

        x = Flatten()(x)
        for _ in range(dense_layers):
            x = Dense(dense_layer_size, activation='relu', kernel_initializer='he_uniform')(x)
        x = Dropout(dropout)(x)
        x = Dense(NUM_CLASSES, activation='softmax')(x)

        #------------------------------------------------------------------------------#

        model = Model(inputs=pretrained_model.input, outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model_path = Path.joinpath(adcs_path, 'models', subdir, 'model_loss.h5')

        DATETIME = time.strftime('%#d%b%Y-%H%M')
        start = time.perf_counter()

        #------------------------------ Model Callbacks -------------------------------#

        earlystop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        lr_stagnate = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=0,
            mode='auto',
        )
        checkpoint_loss = ModelCheckpoint(
            model_path,
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            mode='auto',
        )
        class LossHistoryCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                logging.info(f'[Epoch {epoch:2}] Train: {logs["loss"]:.2f}/{logs["accuracy"]:.2%} | Val: {logs["val_loss"]:.2f}/{logs["val_accuracy"]:.2%}')

        CALLBACKS = [lr_stagnate, checkpoint_loss, earlystop, LossHistoryCallback()]
        logging.info(f'[Epoch  x] Train: Loss/Acc    | Val: Loss/Acc')

        #------------------------------- Model Training -------------------------------#

        hist = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=100,
            callbacks=CALLBACKS,
            verbose=0,
        )

        EPOCHS = len(hist.history['val_loss']) - patience
        logging.info(f'Training Duration: {duration(start)} mins')

        #------------------------------- Model Testing --------------------------------#

        logging.info('--- ACCURACIES (INCORRECT/CORRECT//TOTAL) ---')

        #------------------------------- Validation Set -------------------------------#

        val_preds = model.predict(val_generator, verbose=0)
        val_preds = np.argmax(val_preds, axis=1).tolist()
        val_corrects = (y_test_encoded == val_preds).sum()
        val_acc = val_corrects / len(val_preds)
        logging.info(f'val_acc = {val_acc:.2%} ({len(val_preds)-val_corrects}/{val_corrects}//{len(val_preds)})')

        #-------------------------------- Training Set --------------------------------#

        xtrain_generator = ImageDataGenerator(preprocessing_function=PREPROCESSING_FUNCTION).flow(
            x=x_train,
            batch_size=batch_size,
            shuffle=False,
        )

        train_preds = model.predict(xtrain_generator, verbose=0)
        train_preds = np.argmax(train_preds, axis=1).tolist()
        train_corrects = (y_train_encoded == train_preds).sum()
        train_acc = train_corrects / len(train_preds)
        logging.info(f'train_acc = {train_acc:.2%} ({len(train_preds)-train_corrects}/{train_corrects}//{len(train_preds)})')

    #------------------------------------------------------------------------------#
    #------------------------------ Testing Pipeline ------------------------------#
    #------------------------------------------------------------------------------#

    #---------------------------------- Test Set ----------------------------------#

    if not training_mode: 
        try:
            if not test_model.endswith('.h5'): test_model = f'{test_model}.h5'
            logging.info(f'Testing {test_model}')
            model = load_model(Path.joinpath(adcs_path, 'models', subdir, test_model))
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'],
            )
        except Exception as e:
            raise ValueError('Invalid or no test model name specified')

    test_generator = ImageDataGenerator(preprocessing_function=PREPROCESSING_FUNCTION).flow_from_directory(
        directory=test_path,
        classes=defect_keys,
        target_size=(img_size,img_size),
        batch_size=batch_size,
        shuffle=False,
    )

    start = time.perf_counter()
    test_loss, test_acc = model.evaluate(test_generator, verbose=0)
    test_incorrects = round((1-test_acc)*test_generator.samples)
    test_corrects = round(test_acc*test_generator.samples)
    logging.info(f'test_acc = {test_acc:.2%} ({test_incorrects}/{test_corrects}//{test_generator.samples})')
    logging.info(f'Testing Duration: {duration(start)} mins')

    if not training_mode: return test_acc

    NAME = f'{subdir}_{DATETIME}_{test_acc:.2%}.h5'

    #-------------------------------- Model Saving --------------------------------#

    if test_acc > saving_threshold:
        model.save(Path.joinpath(adcs_path, 'models', subdir, NAME))
        logging.info(f'{NAME} saved')
    else:
        logging.info(f'Model not saved; test_acc below threshold of {saving_threshold:.0%}')

    #------------------------------- Results Saving -------------------------------#

    try:
        results_csv = Path.joinpath(adcs_path, 'results', 'training', subdir, f'{subdir}_training.csv')
        df_results = pd.read_csv(results_csv)
    except FileNotFoundError:
        df_results = pd.DataFrame(columns=[
            "name",
            "pretrained_model",
            "datetime",
            "val_loss",
            "test_acc",
            "test_corrects",
            "test_total",
            "train_acc",
            "train_corrects",
            "train_total",
            "val_acc",
            "val_corrects",
            "val_total",
            "n_sampling",
            "img_size",
            "batch_size",
            "epochs",
            "patience",
            "dense_layers",
            "dense_layer_size",
            "dropout",
        ])

    df_results = df_results.append({
        "name": NAME,
        "pretrained_model": PRETRAINED_NAME,
        "datetime": DATETIME,
        "val_loss": min(hist.history['val_loss']),
        "test_acc": test_acc,
        "test_corrects": test_corrects,
        "test_total": test_generator.samples,
        "train_acc": train_acc,
        "train_corrects": train_corrects,
        "train_total": len(train_preds),
        "val_acc": val_acc,
        "val_corrects": val_corrects,
        "val_total": len(val_preds),
        "n_sampling": n,
        "img_size": img_size,
        "batch_size": batch_size,
        "epochs": EPOCHS,
        "patience": patience,
        "dense_layers": dense_layers,
        "dense_layer_size": dense_layer_size,
        "dropout": dropout,
    }, ignore_index=True)

    df_results.to_csv(results_csv, index=None)
    logging.info(f'Results added to .../{subdir}_training.csv')

    #------------------------------------------------------------------------------#

    return test_acc