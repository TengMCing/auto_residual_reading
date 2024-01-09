import sys
import subprocess
import os
import tensorflow as tf
from tensorflow import keras
import keras_tuner
import pandas as pd
import numpy as np

# Global setttings
DATA_CLASS = 'phn_v2'
INPUT_TYPE = 'es'
RES = 32
COLOR_MODE = 'rgb'
CHANNELS = 3
CLASS_NAMES = ('0')
BATCH_SIZE = 32

project_dir = subprocess.run(['Rscript', '-e', 'cat(here::here())'],
                             check=True,
                             capture_output=True,
                             text=True).stdout

train_dir = os.path.join(project_dir, 
                         'data', 
                         f'{DATA_CLASS}',
                         'residual_plots',
                         f'{RES}',
                         'train')
                         
meta_dir = os.path.join(project_dir, 
                        'data', 
                        f'{DATA_CLASS}',
                        'residual_plots',
                        'meta.csv')
                         
data_gen = keras.preprocessing.image.ImageDataGenerator(validation_split=0.2)
train_dat = data_gen.flow_from_directory(directory=train_dir,
                                         target_size=(RES, RES),
                                         color_mode=COLOR_MODE,
                                         classes=CLASS_NAMES,
                                         class_mode='categorical',
                                         batch_size=BATCH_SIZE,
                                         shuffle=True,
                                         seed=10086,
                                         subset="training")
val_dat = data_gen.flow_from_directory(directory=train_dir,
                                       target_size=(RES, RES),
                                       color_mode=COLOR_MODE,
                                       classes=CLASS_NAMES,
                                       class_mode='categorical',
                                       batch_size=BATCH_SIZE,
                                       shuffle=True,
                                       seed=10086,
                                       subset="validation")
                                         
num_train_dat = len(train_dat.classes)
num_val_dat = len(val_dat.classes)

train_x = list()

for i in range(num_train_dat // BATCH_SIZE + (num_train_dat % BATCH_SIZE > 0)):
  this_batch = next(train_dat)
  train_x.append(this_batch[0])

train_x = np.concatenate(train_x)

val_x = list()

for i in range(num_val_dat // BATCH_SIZE + (num_val_dat % BATCH_SIZE > 0)):
  this_batch = next(val_dat)
  val_x.append(this_batch[0])

val_x = np.concatenate(val_x)

train_plot_uid = [int(train_dat.filenames[index].split('/')[1].split('.')[0]) for index in train_dat.index_array]
val_plot_uid = [int(val_dat.filenames[index].split('/')[1].split('.')[0]) for index in val_dat.index_array]

meta = pd.read_csv(meta_dir, keep_default_na = False)

train_y = pd.merge(pd.DataFrame({'plot_uid': train_plot_uid}), meta, on='plot_uid', how='left')['effect_size'].values
val_y = pd.merge(pd.DataFrame({'plot_uid': val_plot_uid}), meta, on='plot_uid', how='left')['effect_size'].values

train_monotonic = pd.merge(pd.DataFrame({'plot_uid': train_plot_uid}), meta, on='plot_uid', how='left')['measure_monotonic'].values
val_monotonic = pd.merge(pd.DataFrame({'plot_uid': val_plot_uid}), meta, on='plot_uid', how='left')['measure_monotonic'].values
train_sparse = pd.merge(pd.DataFrame({'plot_uid': train_plot_uid}), meta, on='plot_uid', how='left')['measure_sparse'].values
val_sparse = pd.merge(pd.DataFrame({'plot_uid': val_plot_uid}), meta, on='plot_uid', how='left')['measure_sparse'].values
train_splines = pd.merge(pd.DataFrame({'plot_uid': train_plot_uid}), meta, on='plot_uid', how='left')['measure_splines'].values
val_splines = pd.merge(pd.DataFrame({'plot_uid': val_plot_uid}), meta, on='plot_uid', how='left')['measure_splines'].values
train_striped = pd.merge(pd.DataFrame({'plot_uid': train_plot_uid}), meta, on='plot_uid', how='left')['measure_striped'].values
val_striped = pd.merge(pd.DataFrame({'plot_uid': val_plot_uid}), meta, on='plot_uid', how='left')['measure_striped'].values
train_n = pd.merge(pd.DataFrame({'plot_uid': train_plot_uid}), meta, on='plot_uid', how='left')['n'].values
val_n = pd.merge(pd.DataFrame({'plot_uid': val_plot_uid}), meta, on='plot_uid', how='left')['n'].values

train_x_additional = np.column_stack((train_monotonic, train_sparse, train_splines, train_striped, train_n))
val_x_additional = np.column_stack((val_monotonic, val_sparse, val_splines, val_striped, val_n))

train_x = train_x.astype(np.float32)
train_x_additional = train_x_additional.astype(np.float32)
val_x = val_x.astype(np.float32)
val_x_additional = val_x_additional.astype(np.float32)
train_y = train_y.astype(np.float32).reshape((-1, 1))
val_y = val_y.astype(np.float32).reshape((-1, 1))

model_dir = os.path.join(project_dir,
                         "keras_tuner",
                         "best_models",
                         f"{DATA_CLASS}",
                         f"{INPUT_TYPE}",
                         f'final_0.keras')
                         
test_mod = keras.models.load_model(model_dir)
print(np.mean((test_mod.predict([train_x, train_x_additional]) - train_y) ** 2))
print(test_mod.evaluate([train_x, train_x_additional], train_y))
print(np.mean((test_mod.predict([val_x, val_x_additional]) - val_y) ** 2))
print(test_mod.evaluate([val_x, val_x_additional], val_y))
