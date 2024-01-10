import sys
import subprocess
import os
import tensorflow as tf
from tensorflow import keras
import keras_tuner
import pandas as pd
import numpy as np
import pickle

# from keras import backend as K
# 
# K.set_learning_phase(0)

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
                         'test')
                         
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


# def build_model(hp):
#     
#     # Preprocess the input image
#     model_input = keras.layers.Input(shape=(RES, RES, CHANNELS))
#     processed_input = keras.applications.vgg16.preprocess_input(model_input)
#     grey_scale_input = keras.layers.Lambda(
#       lambda image: tf.image.rgb_to_grayscale(image), name='grey_scale'
#     )(processed_input)
#     
#     # Define the convoluational layers
#     # From VGG16
#     num_filters = hp.Int("base_filters", min_value=4, max_value=64, step=2, sampling='log')
#     cnn_dropout = hp.Float('cnn_dropout', min_value=0.1, max_value=0.6, step=0.1)
#     
#     # Block 1
#     # conv 1.1
#     x = keras.layers.Conv2D(
#         num_filters, (3, 3), padding="same", name="block1_conv1",
#     )(grey_scale_input)
#     if hp.Boolean('cnn_batch_normalization'):
#         x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.Activation("relu")(x)
#     
#     # conv 1.2
#     x = keras.layers.Conv2D(
#         num_filters, (3, 3), padding="same", name="block1_conv2"
#     )(x)
#     if hp.Boolean('cnn_batch_normalization'):
#         x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.Activation("relu")(x)
#     
#     # pool 1    
#     x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)
#     x = keras.layers.Dropout(cnn_dropout)(x)
#     
# 
#     # Block 2
#     # conv 2.1
#     x = keras.layers.Conv2D(
#         num_filters * 2, (3, 3), padding="same", name="block2_conv1"
#     )(x)
#     if hp.Boolean('cnn_batch_normalization'):
#         x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.Activation("relu")(x)
#     
#     # conv 2.2    
#     x = keras.layers.Conv2D(
#         num_filters * 2, (3, 3), padding="same", name="block2_conv2"
#     )(x)
#     if hp.Boolean('cnn_batch_normalization'):
#         x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.Activation("relu")(x)
#     
#     # pool 2    
#     x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)
#     x = keras.layers.Dropout(cnn_dropout)(x)
#     
# 
#     # Block 3
#     # conv 3.1
#     x = keras.layers.Conv2D(
#         num_filters * 4, (3, 3), padding="same", name="block3_conv1"
#     )(x)
#     if hp.Boolean('cnn_batch_normalization'):
#         x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.Activation("relu")(x)
#     
#     # conv 3.2    
#     x = keras.layers.Conv2D(
#         num_filters * 4, (3, 3), padding="same", name="block3_conv2"
#     )(x)
#     if hp.Boolean('cnn_batch_normalization'):
#         x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.Activation("relu")(x)
#     
#     # conv 3.3
#     x = keras.layers.Conv2D(
#         num_filters * 4, (3, 3), padding="same", name="block3_conv3"
#     )(x)
#     if hp.Boolean('cnn_batch_normalization'):
#         x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.Activation("relu")(x)
#     
#     # pool 3
#     x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)
#     x = keras.layers.Dropout(cnn_dropout)(x)
#     
# 
#     # Block 4
#     # conv 4.1
#     x = keras.layers.Conv2D(
#         num_filters * 8, (3, 3), padding="same", name="block4_conv1"
#     )(x)
#     if hp.Boolean('cnn_batch_normalization'):
#         x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.Activation("relu")(x)
#     
#     # conv 4.2
#     x = keras.layers.Conv2D(
#         num_filters * 8, (3, 3), padding="same", name="block4_conv2"
#     )(x)
#     if hp.Boolean('cnn_batch_normalization'):
#         x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.Activation("relu")(x)
#     
#     # conv 4.3
#     x = keras.layers.Conv2D(
#         num_filters * 8, (3, 3), padding="same", name="block4_conv3"
#     )(x)
#     if hp.Boolean('cnn_batch_normalization'):
#         x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.Activation("relu")(x)
#     
#     # pool 4
#     x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)
#     x = keras.layers.Dropout(cnn_dropout)(x)
#     
# 
#     # Block 5
#     # conv 5.1
#     x = keras.layers.Conv2D(
#         num_filters * 8, (3, 3), padding="same", name="block5_conv1"
#     )(x)
#     if hp.Boolean('cnn_batch_normalization'):
#         x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.Activation("relu")(x)
#     
#     # conv 5.2
#     x = keras.layers.Conv2D(
#         num_filters * 8, (3, 3), padding="same", name="block5_conv2"
#     )(x)
#     if hp.Boolean('cnn_batch_normalization'):
#         x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.Activation("relu")(x)
#     
#     # conv 5.3
#     x = keras.layers.Conv2D(
#         num_filters * 8, (3, 3), padding="same", name="block5_conv3"
#     )(x)
#     if hp.Boolean('cnn_batch_normalization'):
#         x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.Activation("relu")(x)
#     
#     # pool 5
#     x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)
#     x = keras.layers.Dropout(cnn_dropout)(x)
#     
#     # Get 1D output
#     if hp.Boolean('global_max_or_ave_pooling'):
#         x = keras.layers.GlobalMaxPooling2D()(x)
#     else:
#         x = keras.layers.GlobalAveragePooling2D()(x)
#     
#     # Get additional information
#     additional_x = keras.layers.Input(shape=(5), name="additional_input")
#     
#     # Decide if the additional information should be ignored
#     if hp.Boolean('ignore_additional_input'):
#         additional_x = keras.layers.Lambda(
#           lambda v: v * 0, name='ignore_additional_input'
#         )(additional_x)
#     
#     # Merge inputs
#     x = keras.layers.concatenate([x, additional_x])
#     
#     # Dense layers    
#     x = keras.layers.Dense(
#         hp.Int('dense_units', min_value=128, max_value=2048, step=2, sampling='log'))(x)
#     if hp.Boolean('dense_batch_normalization'):
#         x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.Dropout(hp.Float('dense_dropout', min_value=0.1, max_value=0.6, step=0.1))(x)
#     x = keras.layers.Activation("relu")(x)
#     
#     model_output = keras.layers.Dense(1, activation="relu")(x)
#     this_model = keras.Model([model_input, additional_x], model_output)
#     
#     # Compile the model
#     this_model.compile(
#       keras.optimizers.legacy.Adam(
#         learning_rate=hp.Float('learning_rate', min_value=1e-8, max_value=1e-1, step=2, sampling='log')
#       ),
#       loss="mean_squared_error",
#       metrics=[keras.metrics.RootMeanSquaredError()]
#     )
#                        
#     this_model.summary()
#     return this_model
#   
# 
# tuner = keras_tuner.BayesianOptimization(hypermodel=build_model,
#                                          objective=keras_tuner.Objective("val_root_mean_squared_error", direction="min"),
#                                          max_trials=100,
#                                          executions_per_trial=1,
#                                          overwrite=False,
#                                          directory=f"keras_tuner/tuner/{DATA_CLASS}/{INPUT_TYPE}",
#                                          project_name=f'{RES}',
#                                          max_consecutive_failed_trials=20)
# 
# final_mod = build_model(tuner.get_best_hyperparameters()[0])
# 


model_dir = os.path.join(project_dir,
                         "keras_tuner",
                         "best_models",
                         f"{DATA_CLASS}",
                         f"{INPUT_TYPE}",
                         f'final_0.keras')
                         
test_mod = keras.models.load_model(model_dir)

pickle_weight_dir = os.path.join(project_dir,
                          "keras_tuner",
                          "best_models",
                          f"{DATA_CLASS}",
                          f"{INPUT_TYPE}",
                          f'final_0_weight.pkl')
def save_weights_to_pickle(model, pickle_filename):
    # Get the weights of all layers
    all_weights = [layer.get_weights() for layer in model.layers]

    # Serialize and save the weights using pickle
    with open(pickle_filename, 'wb') as pickle_file:
        pickle.dump(all_weights, pickle_file)

save_weights_to_pickle(test_mod, pickle_weight_dir)


weight_dir = os.path.join(project_dir,
                          "keras_tuner",
                          "best_models",
                          f"{DATA_CLASS}",
                          f"{INPUT_TYPE}",
                          f'final_0.weights.h5')
test_mod.save_weights(weight_dir)

print("save test images")
test_im_dir = os.path.join(project_dir,
                           "keras_tuner",
                           "best_models",
                           f"{DATA_CLASS}",
                           f"{INPUT_TYPE}",
                           f'test_im.npy')
# test_im_local_dir = os.path.join(project_dir,
#                            "keras_tuner",
#                            "best_models",
#                            f"{DATA_CLASS}",
#                            f"{INPUT_TYPE}",
#                            f'test_local_im.npy')
np.save(test_im_dir, train_x)

print("save additional inputs")
test_additional_dir = os.path.join(project_dir,
                           "keras_tuner",
                           "best_models",
                           f"{DATA_CLASS}",
                           f"{INPUT_TYPE}",
                           f'test_additional.npy')
# test_additional_local_dir = os.path.join(project_dir,
#                            "keras_tuner",
#                            "best_models",
#                            f"{DATA_CLASS}",
#                            f"{INPUT_TYPE}",
#                            f'test_local_additional.npy')
np.save(test_additional_dir, train_x_additional)

# 
# for layer in test_mod.layers:
#     print(layer.get_weights())
# 
train_pred = test_mod.predict([train_x, train_x_additional])
print(train_pred)
print(train_y)
val_pred = test_mod.predict([val_x, val_x_additional])
print(val_pred)
print(val_y)

print(np.mean((test_mod.predict([train_x, train_x_additional]) - train_y) ** 2))
print(test_mod.evaluate([train_x, train_x_additional], train_y))
print(np.mean((test_mod.predict([val_x, val_x_additional]) - val_y) ** 2))
print(test_mod.evaluate([val_x, val_x_additional], val_y))

print("load weights")
test_mod.load_weights(weight_dir)
train_pred = test_mod.predict([train_x, train_x_additional])
print(train_pred)
print(train_y)
val_pred = test_mod.predict([val_x, val_x_additional])
print(val_pred)
print(val_y)

print(np.mean((test_mod.predict([train_x, train_x_additional]) - train_y) ** 2))
print(test_mod.evaluate([train_x, train_x_additional], train_y))
print(np.mean((test_mod.predict([val_x, val_x_additional]) - val_y) ** 2))
print(test_mod.evaluate([val_x, val_x_additional], val_y))


# new_mod = keras.models.clone_model(test_mod)
# final_mod.load_weights(weight_dir)
# final_mod.predict([train_x, train_x_additional])
# np.array_equal(np.load(test_im_dir), train_x)
# np.array_equal(np.load(test_additional_dir), train_x_additional)
