import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import activations
import gc

import golois

planes = 31
moves = 361
N = 10000
epochs = 1000
batch = 128
# filters = 32
filters = 48
trunk = 24

input_data = np.random.randint(2, size=(N, 19, 19, planes))
input_data = input_data.astype ('float32')

policy = np.random.randint(moves, size=(N,))
policy = keras.utils.to_categorical (policy)

value = np.random.randint(2, size=(N,))
value = value.astype ('float32')

end = np.random.randint(2, size=(N, 19, 19, 2))
end = end.astype ('float32')

groups = np.zeros((N, 19, 19, 1))
groups = groups.astype ('float32')

print ("getValidation", flush = True)
golois.getValidation (input_data, policy, value, end)


model = keras.models.load_model("RidaLali_V5-4-2.h5")

model.summary ()

# model.compile(optimizer=keras.optimizers.Adagrad(learning_rate=1e-3),
#               loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
#               loss_weights={'policy' : 1.0, 'value' : 1.0},
#               metrics={'policy': 'categorical_accuracy', 'value': 'mse'})

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.92, beta_2=0.9991),
              loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
              loss_weights={'policy' : 1.0, 'value' : 1.0},
              metrics={'policy': 'categorical_accuracy', 'value': 'mse'})

for i in range (1, epochs + 1):
    print ('epoch ' + str (i))
    golois.getBatch (input_data, policy, value, end, groups, i * N)
    history = model.fit(input_data,
                        {'policy': policy, 'value': value},
                        epochs=1, batch_size=batch)
    if (i % 5 == 0):
        gc.collect ()
    if (i % 20 == 0):
        golois.getValidation (input_data, policy, value, end)
        val = model.evaluate (input_data,
                              [policy, value], verbose = 0, batch_size=batch)
        print ("val =", val)
        model.save ('RidaLali_V5-4-2.h5')
