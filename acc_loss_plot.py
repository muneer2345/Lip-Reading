import numpy as np
from keras import optimizers
from keras.preprocessing import image as image_utils
import json
from keras.models import model_from_json
#from IPython.display import Image, display, SVG
#from keras.utils.visualize_util import model_to_dot
from keras.utils.vis_utils import plot_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def read_model(weights_filename='untrained_weight.h5',
               topo_filename='untrained_topo.json'):
    print("Reading Model from "+weights_filename + " and " + topo_filename)
    print("Please wait, it takes time.")
    with open(topo_filename) as data_file:
        topo = json.load(data_file)
        model = model_from_json(topo)
        model.load_weights(weights_filename)
        print("Finish Reading!")
        return model

model = read_model();

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9),
              metrics=['accuracy'])

print("compilation is done")

seed = 7
np.random.seed(seed)
input_testdata_path = "./speaker_input_train"
output_testdata_path = "./speaker_final_output_train"

fil = np.load(input_testdata_path + str(5)+".npz")#+'_'+str(j)+".npz")
X_train = fil['arr_0']
y_train = np.load(output_testdata_path + str(5)+".npy")
	# y_train = y_train[j*500:(j+1)*500]
X_train, y_train = shuffle(X_train, y_train, random_state=0)

print("Running model fit")
history = model.fit(X_train, y_train, validation_split=0.9, epochs=100, batch_size=100, verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
