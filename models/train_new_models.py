import os
import argparse
import json
import copy
import datetime
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow_model_optimization as tfmot
tf.keras.backend.set_floatx('float64')
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def str2bool(v):
    """
    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/main.py
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def make_new_models(args):

    # Load MNIST dataset
    tf.random.set_seed(args.seed)

    if args.dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Normalize pixel values to be between 0 and 1 and flatten the input
        x_train = x_train / 255.0
        x_test = x_test / 255.0
    else:
        raise NotImplementedError

    if args.layer_type == 'dense':
        x_train = x_train.reshape(-1, 784)
        x_test = x_test.reshape(-1, 784)

        # Define the model architecture
        input_layer = Input(shape=(784,), name='input')
        x = Dense(args.hidden_size, activation='relu', name='layer0')(input_layer)
        for i in range(1, args.layer_number):
            x = Dense(args.hidden_size, activation='relu', name=f"layer{i}")(x)

    elif args.layer_type == 'conv2d':
        # Define the model architecture
        input_layer = Input(shape=(28, 28, 1), name='input')  # Assuming 28x28 grayscale images

        # Convolutional layers
        x = Conv2D(args.hidden_size, (3, 3), activation='relu', padding='same', name='layer0')(input_layer)
        for i in range(1, args.layer_number):
            x = Conv2D(args.hidden_size, (3, 3), activation='relu', padding='same', name=f"layer{i}")(x)

        # Flatten the output to feed into dense layers
        x = Flatten()(x)

        if args.flattodense:
            if args.falttodense_size < 0:
                print("set flat to dense same as hidden size")
                args.falttodense_size = args.hidden_size
            # Dense layers
            x = Dense(args.falttodense_size, activation='relu', name='flattentodense')(x)
    else:
        raise NotImplementedError


    if args.lastactivation == 'softmax':
        num_classes = 10
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
    else:
        num_classes = 1
        loss = 'mean_squared_error'
        metrics = ['mean_absolute_error']

    # Output layer for multi-class as regression (non-standard approach)
    # The Flatten layer is still used to flatten the 2D feature maps into a 1D vector before feeding into the dense layers.'
    output_layer = Dense(num_classes, activation=args.lastactivation, name='output')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    print(model.summary())

    log_dir = f"models/logs/{args.layer_type}/{args.hidden_size}x{args.layer_number}/{args.lastactivation}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.compile(optimizer=Adam(), loss=loss, metrics=metrics)

    # Train the model
    model.fit(x_train, y_train, 
            epochs=args.epochs, batch_size=args.bs, validation_data=(x_test, y_test),
            callbacks=[tensorboard_callback]) 

    # Saving model path modified for MNIST
    model.save(f"{log_dir}/model.keras")

    # Predict the test set
    predictions_tr = model.predict(x_train)
    predictions = model.predict(x_test)

    if  args.lastactivation == 'softmax':
        # Round predictions to the nearest integer
        rounded_predictions_tr = np.argmax(predictions_tr, axis=1)
        rounded_predictions    = np.argmax(predictions, axis=1)
    else:
        # Round predictions to the nearest integer
        rounded_predictions_tr = np.round(predictions_tr).astype(int).flatten()
        rounded_predictions    = np.round(predictions).astype(int).flatten()


    # Calculate accuracy
    accuracy_tr = np.mean(rounded_predictions_tr == y_train)
    accuracy    = np.mean(rounded_predictions == y_test)

    args.train_acc = accuracy_tr
    args.test_acc = accuracy
    args_dict = copy.deepcopy(vars(args))
    with open(os.path.join(log_dir, 'args.json'), 'w') as fp:
        json.dump(args_dict, fp, indent=4)

    # Print 
    for l in model.layers:
        if len(l.get_weights()) > 0:
            w, b = l.get_weights()
            print("Weight: ",w)
            print("Bias: ", b)

    # Model summary
    model.summary()

    print(f"Accuracy: {accuracy_tr * 100:.2f}%, {accuracy * 100:.2f}%")

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'create model')
    parser.add_argument('--dataset', default="mnist", choices=['mnist', 'cifar10'], type=str, help='')
    parser.add_argument('--layer_type', default="dense", choices=['dense', 'conv2d'], type=str, help='')
    parser.add_argument('--hidden_size', default=8, type=int, help='')
    parser.add_argument('--layer_number', default=1, type=int, help='')
    parser.add_argument('--kernel_size', default=3, type=int, help='')
    parser.add_argument('--flattodense', default=True, type=str2bool, help='')
    parser.add_argument('--falttodense_size', default=-1, type=int, help='')
    parser.add_argument('--lastactivation', default="linear", type=str, help='')
    parser.add_argument('--epochs', default=1, type=int, help='')
    parser.add_argument('--bs', default=64, type=int, help='')
    parser.add_argument('--seed', default=42, type=int, help='')

    parser.add_argument('--load_json', default="", type=str, help='')

    args = parser.parse_args()

    if len(args.load_json) > 0:
        with open(args.load_json, "r") as f:
            json_data = json.load(f)

            # Update args with JSON data
            args.__dict__.update(json_data)

    print(args)

    make_new_models(args)

# make_new_cnn_mnist_model(8,2)
# make_new_cnn_mnist_model(16,2)
# make_new_cnn_mnist_model(32,2)
# make_new_cnn_mnist_model(64,2)

# make_new_cnn_mnist_model(8,1)
# make_new_cnn_mnist_model(16,1)
# make_new_cnn_mnist_model(32,1)
# make_new_cnn_mnist_model(64,1)
