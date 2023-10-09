import sys
import subprocess
import os
import tensorflow as tf
from tensorflow import keras
import keras_tuner

# Global setttings
DATA_CLASS = 'phn_extended'
INPUT_TYPE = 'residual_plots'
RES = 32
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
                         f'{INPUT_TYPE}',
                         f'{RES}',
                         'mixed',
                         'train')
                         
                         
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
                             
def build_model(hp):
    
    # Preprocess the input image
    model_input = keras.layers.Input(shape=(RES, RES, CHANNELS))
    processed_input = keras.applications.vgg16.preprocess_input(model_input)
    grey_scale_input = keras.layers.Lambda(
      lambda image: tf.image.rgb_to_grayscale(image), name='grey_scale'
    )(processed_input)
    
    # Define the convoluational layers
    # From VGG16
    num_blocks = hp.Int("blocks", min_value=1, max_value=5)
    num_filters = hp.Int("filters", min_value=4, max_value=64, step=2, sampling='log')
    
    # Block 1
    x = keras.layers.Conv2D(
        num_filters, (3, 3), activation="relu", padding="same", name="block1_conv1",
    )(grey_scale_input)
    x = keras.layers.Conv2D(
        num_filters, (3, 3), activation="relu", padding="same", name="block1_conv2"
    )(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)
    
    if num_blocks >= 2:
        # Block 2
        x = keras.layers.Conv2D(
            num_filters * 2, (3, 3), activation="relu", padding="same", name="block2_conv1"
        )(x)
        x = keras.layers.Conv2D(
            num_filters * 2, (3, 3), activation="relu", padding="same", name="block2_conv2"
        )(x)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)
    
    if num_blocks >= 3:
        # Block 3
        x = keras.layers.Conv2D(
            num_filters * 4, (3, 3), activation="relu", padding="same", name="block3_conv1"
        )(x)
        x = keras.layers.Conv2D(
            num_filters * 4, (3, 3), activation="relu", padding="same", name="block3_conv2"
        )(x)
        x = keras.layers.Conv2D(
            num_filters * 4, (3, 3), activation="relu", padding="same", name="block3_conv3"
        )(x)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)
    
    if num_blocks >= 4:
        # Block 4
        x = keras.layers.Conv2D(
            num_filters * 8, (3, 3), activation="relu", padding="same", name="block4_conv1"
        )(x)
        x = keras.layers.Conv2D(
            num_filters * 8, (3, 3), activation="relu", padding="same", name="block4_conv2"
        )(x)
        x = keras.layers.Conv2D(
            num_filters * 8, (3, 3), activation="relu", padding="same", name="block4_conv3"
        )(x)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)
    
    if num_blocks >= 5:
        # Block 5
        x = keras.layers.Conv2D(
            num_filters * 8, (3, 3), activation="relu", padding="same", name="block5_conv1"
        )(x)
        x = keras.layers.Conv2D(
            num_filters * 8, (3, 3), activation="relu", padding="same", name="block5_conv2"
        )(x)
        x = keras.layers.Conv2D(
            num_filters * 8, (3, 3), activation="relu", padding="same", name="block5_conv3"
        )(x)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)
    
    # Define the classifier
    if hp.Boolean('max_pooling'):
        x = keras.layers.GlobalMaxPooling2D()(x)
    else:
        x = keras.layers.GlobalAveragePooling2D()(x)
        
    x = keras.layers.Dense(
        hp.Int('units', min_value=8, max_value=1024, step=2, sampling='log'),
        kernel_regularizer=keras.regularizers.L1L2(l1=hp.Float('l1', min_value=1e-6, max_value=1e-1, step=2, sampling='log'), 
                                                   l2=hp.Float('l2', min_value=1e-6, max_value=1e-1, step=2, sampling='log')))(x)
    x = keras.layers.BatchNormalization(fused=False)(x)
    x = keras.layers.Dropout(hp.Float('dropout', min_value=0.1, max_value=0.8, step=0.1))(x)
    x = keras.layers.Activation(activation="relu")(x)
    
    model_output = keras.layers.Dense(2, activation="softmax")(x)
    this_model = keras.Model(model_input, model_output)
    
    # Compile the model
    this_model.compile(
      keras.optimizers.legacy.Adam(
        learning_rate=hp.Float('learning_rate', min_value=1e-6, max_value=1e-2, step=2, sampling='log')
      ),
      loss="categorical_crossentropy",
      metrics=["categorical_accuracy"]
    )
                       
    this_model.summary()
    return this_model
  

tuner = keras_tuner.BayesianOptimization(hypermodel=build_model,
                                         objective='val_categorical_accuracy',
                                         max_trials=30,
                                         executions_per_trial=1,
                                         overwrite=False,
                                         directory=f"keras_tuner/tuner/{DATA_CLASS}/{INPUT_TYPE}",
                                         project_name=f'{RES}')

log_dir = os.path.join(project_dir,
                       "keras_tuner",
                       "logs",
                       f"{DATA_CLASS}",
                       f"{INPUT_TYPE}",
                       f'{RES}')
                       
callbacks = []
callbacks.append(keras.callbacks.EarlyStopping(
                 patience=10,
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
                 patience=3,
                 verbose=1))
                       
tuner.search_space_summary()

tuner.search(x=train_dat,
             validation_data=val_dat,
             epochs=100, 
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
