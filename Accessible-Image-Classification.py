import os
import numpy as np
import keras.backend
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras_preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from vis.visualization import visualize_cam
from matplotlib.colors import ListedColormap
from PIL import ImageFile
from PIL import Image
from time import time
from pathlib import Path

ImageFile.LOAD_TRUNCATED_IMAGES = True


def gen_split(directory, split, width, height, batch_size, color_mode, verbosity):
    # creates training and validation sets from the provided directory using given validation split
    if verbosity > 0:
        print('Generating datasets from images...')

    # create a generator with some minor preprocessing and the given validation split
    all_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=10,
        brightness_range=[0.5, 1.5],
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=split)

    # derive separate training and validation generators
    train_generator = all_datagen.flow_from_directory(
        directory,
        target_size=(width, height),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode=color_mode,
        subset='training')
    valid_generator = all_datagen.flow_from_directory(
        directory,
        target_size=(width, height),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode=color_mode,
        subset='validation')

    if verbosity > 0:
        # tell us which class names we found and their indices
        print('In ', directory, '\nFound classes: ', train_generator.class_indices)

    return train_generator, valid_generator


def gen_whole(directory, width, height, batch_size, color_mode, verbosity):
    if verbosity > 0:
        print('Generating datasets from images...')
    # get generator for the images in directory
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    generator = datagen.flow_from_directory(
        directory,
        target_size=(width, height),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode=color_mode,
        shuffle=False)

    # tell us which class names we found and their indices
    if verbosity > 0:
        print('In ', directory, '\nFound classes: ', generator.class_indices)

    return generator


def make_model(img_width, img_height, color_mode, num_classes, verbosity):
    # builds the model, using channels_last image data format and a triple CONV-RELU-POOL layout (can be customized).
    if verbosity > 0:
        print('Building model...')

    # prepare the correct input shape
    keras.backend.set_image_data_format('channels_last')
    if color_mode == 'rgb':
        input_shape = (img_width, img_height, 3)
    elif color_mode == 'rgba':
        input_shape = (img_width, img_height, 4)
    else:  # grayscale
        input_shape = (img_width, img_height, 1)

    model = Sequential()

    model.add(Conv2D(20, (5, 5), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(100, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.40))
    model.add(Dense(num_classes, activation='softmax'))

    if verbosity > 0:
        model.summary()

    return model


def prep_model(params):
    # makes a fresh model and loads it with provided weights before compiling
    if params['verbosity'] > 0:
        print('Loading trained model...')
    model = make_model(params['img_width'], params['img_height'],
                       params['color_mode'], params['num_classes'], params['verbosity'])
    model.load_weights(params['weights'])
    compile_model(model, params['verbosity'])
    return model


def compile_model(model, verbosity):
    # compiles the model. Customizable optimizer, loss, and metrics freely
    if verbosity > 0:
        print('Compiling model...')
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])


def train(directory, params, vis):
    # gets the generators and trains them on a fresh model

    # get the generators for the provided image directory
    train_generator, valid_generator = gen_split(directory,
                                                 params['val_split'],
                                                 params['img_width'], params['img_height'],
                                                 params['batch_size'],
                                                 params['color_mode'],
                                                 params['verbosity'])

    # prepare automatic class weights to account for an unbalanced dataset
    unique, counts = np.unique(np.array(train_generator.classes), return_counts=True)
    auto_weights = sum(counts) / counts
    class_weights = dict(zip(unique, auto_weights))
    u_val, c_val = np.unique(np.array(valid_generator.classes), return_counts=True)
    if params['verbosity'] > 0:
        print('Class counts (validation set): ', dict(zip(u_val, c_val)))
        print('Class counts (training set): ', dict(zip(unique, counts)))
        print('Generated class weights: ', class_weights)

    # prepare a fresh model and compile it
    model = make_model(params['img_width'], params['img_height'],
                       params['color_mode'], params['num_classes'], params['verbosity'])
    compile_model(model, params['verbosity'])

    v = 1
    if params['verbosity'] < 1:
        v = 0
    #  get callbacks ready to save most accurate (on validation images) model and stop early if learning isn't happening
    callbacks = [ModelCheckpoint(params['weights'],
                                 monitor='val_acc',
                                 save_best_only=True,
                                 period=1,
                                 verbose=v),
                 EarlyStopping(monitor='val_loss',
                               min_delta=0,
                               patience=int(params['epochs'] / 3),
                               restore_best_weights=True,
                               verbose=v)]
    if vis:
        #  run >> tensorboard --logdir='log/' << to see this TensorBoard training metrics visualization
        log_dir_label = 'Classes'
        for key in train_generator.class_indices.keys():
            log_dir_label = log_dir_label + '_' + key
        log_dir = Path(os.getcwd() + '/logs/' + log_dir_label + params['time'])
        tensorboard = TensorBoard(log_dir=log_dir)
        callbacks.append(tensorboard)

    # do the training
    if params['verbosity'] > 0:
        print('Retraining...')
    v_fit = 0
    if params['verbosity'] == 1:
        v_fit = 2
    elif params['verbosity'] > 1:
        v_fit = 1
    history = model.fit_generator(
        train_generator,
        validation_data=valid_generator,
        epochs=params['epochs'],
        shuffle='batch',
        class_weight=class_weights,
        steps_per_epoch=np.ceil(train_generator.samples / params['batch_size']),
        validation_steps=np.ceil(valid_generator.samples / params['batch_size']),
        verbose=v_fit,
        callbacks=callbacks)

    if vis:
        # Plot all training & validation accuracy values
        fig, axes = plt.subplots(2)
        axes[0].plot((history.history['acc']))
        axes[0].plot(history.history['val_acc'])
        axes[0].set_title('Model accuracy')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].legend(['Train', 'Validation'], loc='upper left')
        # Plot all training & validation loss values
        axes[1].set_yscale('log')
        axes[1].plot((history.history['loss']))
        axes[1].plot(history.history['val_loss'])
        axes[1].set_title('Model loss')
        axes[1].set_ylabel('Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].legend(['Train', 'Validation'])
        plt.show()


def evaluate(directory, model, params):
    # returns evaluations for the images from provided generator
    if params['verbosity'] > 0:
        print('Evaluating...')
    generator = gen_whole(directory, params['img_width'], params['img_height'],
                          params['batch_size'], params['color_mode'], params['verbosity'])
    evals = model.evaluate_generator(generator, verbose=1)  # do the evaluation
    print('Evaluation Results:')
    # print the end results
    for e, n in zip(evals, model.metrics_names):
        print('\t', n, '\t', "%.3f" % e)


def predict(directory, model, params, printout, rename):
    # returns predictions for images from provided generator
    v = 1
    if params['verbosity'] < 1:
        v = 0
    if params['verbosity'] > 0:
        print('Predicting...')
    generator = gen_whole(directory, params['img_width'], params['img_height'],
                          params['batch_size'], params['color_mode'], params['verbosity'])
    pred_vals = model.predict_generator(generator, verbose=v)
    predicted_class_indices = np.argmax(pred_vals, axis=1)
    labels = dict((vl, k) for k, vl in generator.class_indices.items())
    pred_labels = [labels[k] for k in predicted_class_indices]
    # consolidate all the prediction and image file information in one place
    pred_all = []
    for p_l, p_v, f in zip(pred_labels, pred_vals, generator.filenames):
        pred_all.append((p_l, p_v, f))

    # find out which class we should be sorting by
    class_index = 0
    class_label = list(generator.class_indices.keys())[class_index]
    found = False
    for label, index in generator.class_indices.items():
        if index == params['sort_class'] or label == params['sort_class']:
            found = True
            if params['verbosity'] > 0:
                print('Sorting by class: ', label)
            class_label = label
            class_index = index
    if not found:
        print('Specified class not found for sorting, using default:\n'
              'Class index: ', class_index,
              '\nClass label: ', class_label)

    # sort the consolidated information (not super efficient, but it should work)
    pred_sorted = []
    while len(pred_all) > 0:
        i = 0
        min_ind = 0
        for pa in pred_all:
            # find the index of the image with minimum predicted value
            if pa[1][class_index] < pred_all[min_ind][1][class_index]:
                min_ind = i
            i = i + 1
        pred_sorted.append(pred_all[min_ind])  # add the info of the image with minimum prediction value to sorted list
        del pred_all[min_ind]  # delete the info of the image with minimum prediction value from the unsorted list

    # go through all the sorted information
    saved_cwd = os.getcwd()  # make sure we can go back to our home directory
    for p in pred_sorted:
        p_l = str(p[0])
        p_v = p[1]
        f = str(p[2])
        root, ext = os.path.splitext(f)
        path, name = os.path.split(f)
        if rename:
            # renaming each image file with the prediction score
            os.chdir(directory + path)  # switch to this file's own directory
            randy = np.random.random_sample()  # need a random number to ensure each name is unique
            new_name = str(
                "{:.2f}".format(100 * p_v[class_index]) + '%' + class_label + '_'
                + str(randy).split('.')[1] + ext)  # unique new name with score + random sequence + extension
            os.rename(name, 'test_name' + str(randy))  # replace existing name with a unique one
            os.rename('test_name' + str(randy), new_name)  # replace unique name with a score-bearing unique one
            os.chdir(saved_cwd)
        else:  # don't rename, just use previous name
            new_name = name
        if printout:
            # tell about each file's prediction values
            print('File name: ', os.path.join(path, new_name), '\n',
                  'Predicted Category: ', p_l.upper(), '(', path == p_l, ')', '\n',
                  'Categorical Prediction Values:')
            for v, i in zip(p_v, generator.class_indices.items()):
                print('\t', "%.2f" % (100 * v), '%', i[0].upper())
            print('-----')


def visualize(directory, model, params):
    # shows a plot of all images in provided directory overlaid with CAM heatmaps
    if params['verbosity'] > 0:
        print('Visualizing...')
    directory = os.fsencode(directory)

    # prepare an appropriately-sized plot with enough subplots
    image_count = len(os.listdir(directory))
    fig, axes2d = plt.subplots(nrows=int(np.ceil(image_count / params['images_per_row'])),
                               ncols=params['images_per_row'], squeeze=False)
    axes = axes2d.flatten()

    # prepare a partially-transparent colormap to use for the heatmap overlays
    cmap = pl.cm.get_cmap('afmhot')
    alpha_cmap = cmap(np.arange(cmap.N))
    alpha_cmap[:, -1] = 1 / (1 + np.e ** (-1 * np.linspace(-10, 10, cmap.N)))
    alpha_cmap = ListedColormap(alpha_cmap)

    for a, img_path in enumerate(os.listdir(directory)):  # go through each image in the directory
        img = image.load_img(os.path.join(directory, img_path),
                             target_size=(params['img_width'], params['img_height']),
                             color_mode=params['color_mode'])  # load the image in the appropriate size and color mode
        if params['verbosity'] > 1:
            print('Visualizing image', a + 1, 'of ', image_count, ': ', str(img_path))
        if params['color_mode'] == 'rgb' or params['color_mode'] == 'rgba':
            axes[a].imshow(img)  # show original image
        else:
            axes[a].imshow(img, cmap='gray')  # show original image (grayscale)
        for layer in params['vis_layers']:
            heat_map = visualize_cam(model, layer, None, image.img_to_array(img))  # get the CAM heatmap
            heat_map = heat_map[:, :, 0]  # reformat for custom mapping and visualization
            heat_map = Image.fromarray(heat_map)  # convert to a PIL image
            # overlay heat map
            axes[a].imshow(heat_map, cmap=alpha_cmap)
        # use file names as plot titles
        axes[a].set_title(str(img_path), fontsize='xx-small', fontstretch='ultra-condensed')
    # adjust and show finished plot
    for ax in axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('auto')
    plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0.05, left=0, bottom=0, right=1, top=0.95)
    for layer in params['vis_layers']:
        # print index and name of layer that was visualized
        if params['verbosity'] > 0:
            print(str('Visualized layer index: ' + str(layer) + '\nVisualized layer name: ' +
                      model.get_layer(index=layer).get_config().get('name')))
    plt.show()


# parameters
parameters = {
    # size for all images processed (smaller images take up less memory)
    'img_width': 250,
    'img_height': 250,
    'epochs': 300,  # how many epochs to train for (may end early due to EarlyStopping, see fit_generator call)
    'batch_size': 30,  # how many images per batch
    'num_classes': 2,  # how many different categories you want to train on
    'val_split': 0.25,  # proportion of training images to hold back for validation
    'color_mode': 'grayscale',  # how to preprocess images to change color mode: "grayscale", "rgb", or "rgba"
    'weights': 'weights.hdf5',  # name of the file where model+weights is saved
    'sort_class': 0,  # class index or name to be used for sorting predicted images
    'vis_layers': [3, 6, 9, 10, 11, 12, 13],  # indices of the layers to be visualized
    'images_per_row': 10,  # how many images per row to output during visualization
    'verbosity': 1,  # how much console output to generate (0=minimal, 1=some, 2=all)
    'time': ('/{}'.format(time()))  # getting the time for logging purposes
}
directories = {
    'train': 'train_images/',  # training images directory
    'eval': 'eval_images/',  # evaluation images directory
    'pred': 'predict_images/',  # prediction images directory
    'vis': 'vis_images/'  # visualization images directory
}

keras.backend.clear_session()  # get rid of previous sessions before starting operations

# train a fresh model on the images in training directory, with optional visualization of accuracy and loss
#  run >> tensorboard --logdir='logs/' << to see TensorBoard training metrics visualization
train(directories['train'], parameters, vis=True)

# create a fresh model and loads it with saved weights
loaded_model = prep_model(parameters)

# output an evaluation of the model on the images in evaluation directory
evaluate(directories['eval'], loaded_model, parameters)

# output the model's prediction for each image in prediction directory, by printing out to console and/or renaming files
predict(directories['pred'], loaded_model, parameters, printout=True, rename=True)

# produce a plot of images in visualization directory with class activation mapping (attention visualization) overlay
visualize(directories['vis'], loaded_model, parameters)
