import sys
import subprocess
import os
import tensorflow as tf
from tensorflow import keras
import keras_tuner
import pandas as pd
import numpy as np

# Global setttings
DATA_CLASS = 'phn'
INPUT_TYPE = 'es'
RES = 64
COLOR_MODE = 'rgb'
CHANNELS = 3
CLASS_NAMES = ('not_null', "null")
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
train_y = np.log(train_y + 1)
val_y = np.log(val_y + 1)

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

def build_model(hp):
    
    # Preprocess the input image
    model_input = keras.layers.Input(shape=(RES, RES, CHANNELS))
    processed_input = keras.applications.vgg16.preprocess_input(model_input)
    grey_scale_input = keras.layers.Lambda(
      lambda image: tf.image.rgb_to_grayscale(image), name='grey_scale'
    )(processed_input)
    
    # Define the convoluational layers
    # From VGG16
    num_blocks = hp.Int("cnn_blocks", min_value=1, max_value=5)
    num_filters = hp.Int("base_filters", min_value=4, max_value=64, step=2, sampling='log')
    
    # Block 1
    # conv 1.1
    x = keras.layers.Conv2D(
        num_filters, (3, 3), padding="same", name="block1_conv1",
    )(grey_scale_input)
    if hp.Boolean('cnn_batch_normalization'):
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    
    # conv 1.2
    x = keras.layers.Conv2D(
        num_filters, (3, 3), padding="same", name="block1_conv2"
    )(x)
    if hp.Boolean('cnn_batch_normalization'):
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    
    # pool 1    
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)
    x = keras.layers.Dropout(hp.Float('cnn_dropout', min_value=0.1, max_value=0.8, step=0.1))(x)
    
    if num_blocks >= 2:
        # Block 2
        # conv 2.1
        x = keras.layers.Conv2D(
            num_filters * 2, (3, 3), padding="same", name="block2_conv1"
        )(x)
        if hp.Boolean('cnn_batch_normalization'):
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        # conv 2.2    
        x = keras.layers.Conv2D(
            num_filters * 2, (3, 3), padding="same", name="block2_conv2"
        )(x)
        if hp.Boolean('cnn_batch_normalization'):
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        # pool 2    
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)
        x = keras.layers.Dropout(hp.Float('cnn_dropout', min_value=0.1, max_value=0.8, step=0.1))(x)
    
    if num_blocks >= 3:
        # Block 3
        # conv 3.1
        x = keras.layers.Conv2D(
            num_filters * 4, (3, 3), padding="same", name="block3_conv1"
        )(x)
        if hp.Boolean('cnn_batch_normalization'):
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        # conv 3.2    
        x = keras.layers.Conv2D(
            num_filters * 4, (3, 3), padding="same", name="block3_conv2"
        )(x)
        if hp.Boolean('cnn_batch_normalization'):
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        # conv 3.3
        x = keras.layers.Conv2D(
            num_filters * 4, (3, 3), padding="same", name="block3_conv3"
        )(x)
        if hp.Boolean('cnn_batch_normalization'):
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        # pool 3
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)
        x = keras.layers.Dropout(hp.Float('cnn_dropout', min_value=0.1, max_value=0.8, step=0.1))(x)
    
    if num_blocks >= 4:
        # Block 4
        # conv 4.1
        x = keras.layers.Conv2D(
            num_filters * 8, (3, 3), padding="same", name="block4_conv1"
        )(x)
        if hp.Boolean('cnn_batch_normalization'):
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        # conv 4.2
        x = keras.layers.Conv2D(
            num_filters * 8, (3, 3), padding="same", name="block4_conv2"
        )(x)
        if hp.Boolean('cnn_batch_normalization'):
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        # conv 4.3
        x = keras.layers.Conv2D(
            num_filters * 8, (3, 3), padding="same", name="block4_conv3"
        )(x)
        if hp.Boolean('cnn_batch_normalization'):
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        # pool 4
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)
        x = keras.layers.Dropout(hp.Float('cnn_dropout', min_value=0.1, max_value=0.8, step=0.1))(x)
    
    if num_blocks >= 5:
        # Block 5
        # conv 5.1
        x = keras.layers.Conv2D(
            num_filters * 8, (3, 3), padding="same", name="block5_conv1"
        )(x)
        if hp.Boolean('cnn_batch_normalization'):
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        # conv 5.2
        x = keras.layers.Conv2D(
            num_filters * 8, (3, 3), padding="same", name="block5_conv2"
        )(x)
        if hp.Boolean('cnn_batch_normalization'):
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        # conv 5.3
        x = keras.layers.Conv2D(
            num_filters * 8, (3, 3), padding="same", name="block5_conv3"
        )(x)
        if hp.Boolean('cnn_batch_normalization'):
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        # pool 5
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)
        x = keras.layers.Dropout(hp.Float('cnn_dropout', min_value=0.1, max_value=0.8, step=0.1))(x)
    
    # Get 1D output
    if hp.Boolean('global_max_or_ave_pooling'):
        x = keras.layers.GlobalMaxPooling2D()(x)
    else:
        x = keras.layers.GlobalAveragePooling2D()(x)
    
    # Get additional information
    additional_x = keras.layers.Input(shape=(5))
    
    # Merge inputs
    x = keras.layers.concatenate([x, additional_x])
    
    # Dense layers    
    x = keras.layers.Dense(
        hp.Int('dense_units', min_value=128, max_value=4096, step=2, sampling='log'),
        kernel_regularizer='l1_l2')(x)
    if hp.Boolean('dense_batch_normalization'):
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(hp.Float('dense_dropout', min_value=0.1, max_value=0.8, step=0.1))(x)
    x = keras.layers.Activation("relu")(x)
    
    model_output = keras.layers.Dense(1, activation="relu")(x)
    this_model = keras.Model([model_input, additional_x], model_output)
    
    # Compile the model
    this_model.compile(
      keras.optimizers.legacy.Adam(
        learning_rate=hp.Float('learning_rate', min_value=1e-6, max_value=1e-1, step=2, sampling='log')
      ),
      loss="mean_squared_error",
      metrics=[keras.metrics.RootMeanSquaredError()]
    )
                       
    this_model.summary()
    return this_model
  

tuner = keras_tuner.BayesianOptimization(hypermodel=build_model,
                                         objective=keras_tuner.Objective("val_root_mean_squared_error", direction="min"),
                                         max_trials=100,
                                         executions_per_trial=1,
                                         overwrite=False,
                                         directory=f"keras_tuner/tuner/{DATA_CLASS}/{INPUT_TYPE}",
                                         project_name=f'{RES}',
                                         max_consecutive_failed_trials=10)

log_dir = os.path.join(project_dir,
                       "keras_tuner",
                       "logs",
                       f"{DATA_CLASS}",
                       f"{INPUT_TYPE}",
                       f'{RES}')
                       
callbacks = []
callbacks.append(keras.callbacks.EarlyStopping(
                 patience=20,
                 restore_best_weights=False,
                 verbose=1))
                 
if os.path.exists(os.path.dirname(log_dir)) is False:
    os.makedirs(os.path.dirname(log_dir))
callbacks.append(keras.callbacks.TensorBoard(
                 log_dir=log_dir,
                 histogram_freq=0,
                 write_graph=False,
                 write_grads=False,
                 write_images=False,
                 update_freq=20))  
callbacks.append(keras.callbacks.ReduceLROnPlateau(
                 factor=0.5,
                 patience=10,
                 verbose=1))
                       
tuner.search_space_summary()

tuner.search(x=[train_x, train_x_additional],
             y=train_y,
             validation_data=([val_x, val_x_additional], val_y),
             batch_size=BATCH_SIZE,
             epochs=2000, 
             callbacks=callbacks)

tuner.results_summary()

best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()
model_dir = os.path.join(project_dir,
                         "keras_tuner",
                         "best_models",
                         f"{DATA_CLASS}",
                         f"{INPUT_TYPE}",
                         f'{RES}')
best_model.save(model_dir)
