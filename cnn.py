import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.utils as utils
import tensorflow.keras.preprocessing as preprocessing
import tensorflow.keras.callbacks as callbacks
from tensorflow.data import Dataset
from tensorflow.image import rgb_to_grayscale
import numpy as np
import pickle

# Change numbers in dense layers, batch size, learning rate, data augmentation (adding more images)
train = utils.image_dataset_from_directory(
    'images',
    labels = 'inferred',
    label_mode = 'categorical',
    class_names = None,
    # Images were gray legos, but still RGB so I grayscaled them
    color_mode = 'grayscale',       
    batch_size = 32,
    image_size = (200, 200), 
    shuffle = True,
    seed = 29,
    validation_split = 0.3,
    subset = 'training',
)

test = utils.image_dataset_from_directory(
    'images',
    labels = 'inferred',
    label_mode = 'categorical',
    class_names = None,
    color_mode = 'grayscale',
    batch_size = 32,
    image_size = (200, 200),
    shuffle = True,
    seed = 29,
    validation_split = 0.3,
    subset = 'validation',
)

# # Data Augmentation Section
# # You can have multiple layers, but first one always needs the input_shape
rotation = models.Sequential([
    layers.RandomRotation(0.25, input_shape = (200, 200, 1))
])
# # train.map() applies the transformation in parentheses to each pair x,y 
# # in the dataset.  We only need to transform the x-values, we just pass
# # the y-values along passively.  Notice that the output of the lambda
# # function is a 2-tuple, which is the transformed image followed
rotated = train.map(lambda x, y: (rotation(x), y))
# # Make sure you create *all* the transformed copies before assembling
# # them into a new training set.
train = train.concatenate(rotated)



class Net():
    def __init__(self, input_shape):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(
            8, # filters (kept 8 bc I liked it)
            10, # size
            strides = 5, # step size
            activation = 'relu',
            input_shape = input_shape,   # To build model
        )) # 39 x 39 x 8 📣
        self.model.add(layers.ZeroPadding2D(
            padding = ((1, 0), (1, 0)),
            input_shape = input_shape,
        )) # 40 x 40 x 8 📣
        self.model.add(layers.MaxPool2D(
            pool_size = 2
        )) # 20 x 20 x 8 📣
        self.model.add(layers.Conv2D(
            8, # filters (kept 8 bc I liked it)
            3, # size
            strides = 1, # step size
            activation = 'relu',
        )) # 18 x 18 x 8 📣
        self.model.add(layers.MaxPool2D(
            pool_size = 2
        )) # 9 x 9 x 8 = 648 📣
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(
            512,   # Power of 2s >
            activation = 'relu',
        ))
        self.model.add(layers.Dense(
            256,
            activation = 'relu',
        ))
        self.model.add(layers.Dense(
            64,  
            activation = 'relu',
        ))
        self.model.add(layers.Dense(
            32,  
            activation = 'relu',
        ))
        self.model.add(layers.Dense(
            16,   # Number of classes
            activation = 'softmax',
        ))
        # Also try CategoricalCrossentropy()
        self.loss = losses.CategoricalCrossentropy()
        self.optimizer = optimizers.SGD(learning_rate = 0.0001)
        self.model.compile(
            loss = self.loss,
            optimizer = self.optimizer,
            metrics = ['accuracy'],
        )

    def __str__(self):
        self.model.summary()
        return ""

net = Net((200, 200, 1))    # 1 because grayscale
print(net)

# callbacks = [
#     callbacks.ModelCheckpoint(
#         'checkpoints{epoch:02d}', 
#         verbose = 1, 
#         save_freq = 80
#     )
# ]
net.model.fit(
    train,
    batch_size = 32,
    epochs = 100,
    verbose = 2,
    validation_data = test,
    validation_batch_size = 32,
    # callbacks = callbacks
)
save_path = './lego_model_save'
net.model.save(save_path)
with open(f'{save_path}/class_names.data', 'wb') as f:
    pickle.dump(train.class_names, f)