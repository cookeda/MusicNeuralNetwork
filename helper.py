import os
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential, load_model
from keras.layers import Dense, Input
from keras.backend import clear_session
import matplotlib.pyplot as plt
from keras.losses import mean_squared_error, kullback_leibler_divergence
import tensorflow as tf
import warnings
import matplotlib as mpl
mpl.use('TkAgg')


def get_xy(endpoint='loudness', subset='train'):
    """
    Gets the data matrix, X, and target, y for the desired 'endpoint'.

    :param endpoint: 'loudness', 'pitch', 'timbre', 'loudness', 'chroma', 'mfcc', 'spectral
    :param subset: 'train', 'valid'
    :return:
    """
    npz_file = f'music_{endpoint}.npz'
    with np.load(npz_file) as data:
        x = data[f'x_{subset}']
        y = data[f'y_{subset}']
        if y.ndim == 1:
            y = y.reshape((-1, 1))
    return x, y


def setup_model_checkpoints(output_path):
    """
    Setup model checkpoints using the save path and frequency.

    :param output_path: The directory to store the checkpoints in
    :return: a ModelCheckpoint
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model_checkpoint = ModelCheckpoint(
        os.path.join(output_path, 'model.{epoch:05d}_{val_loss:f}.h5'),
        save_weights_only=False,
        save_freq='epoch',
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )
    return model_checkpoint


def visualize(model, endpoint='loudness', subset='valid', output_path=''):
    """
    Create a joint distribution plot that shows relationship between
    model estimates and true values.

    :param model: A trained model
    :param endpoint: The name of the target
    :param subset: Which data set to use
    :param output_path: The output directory to save the PNG
    :return: None
    """
    import seaborn as sns
    bins = 128

    x, y_true = get_xy(endpoint, subset)

    y_pred = model.predict(x)

    # check assumptions
    if y_true.shape != y_pred.shape:
        print(f'WARNING: output should have shape {y_true.shape} not {y_pred.shape}. Broadcasting output.')
        y_pred = np.broadcast_to(y_pred, y_true.shape)

    # loss should be mean squared error unless pitch target
    metric = mean_squared_error

    if endpoint in ('pitch', 'chroma'):
        # if each row of target sums to one and is nonnegative, we know it must be pitches.
        metric = kullback_leibler_divergence
        if not (y_pred >= 0).all():
            print(f'WARNING: output should be nonnegative. Setting negative values to a small positive value.')
            y_pred[y_pred < 0] = 1e-6
        if not np.allclose(a=y_pred.sum(axis=1), b=1):
            print(f'WARNING: outputs should sum to one. Scaling rows of output to sum to one.')
            y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)

    loss = tf.reduce_mean(metric(y_true, y_pred)).numpy()

    # make joint plot
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        num_targets = y_true.shape[1]
        num_cols = int(np.ceil(num_targets ** (1/2)))
        num_rows = int(np.ceil(num_targets / num_cols))
        png_file = os.path.join(output_path, f'visualize_{subset}.png')

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*3, num_rows*3),
                                 gridspec_kw=dict(hspace=0.0, wspace=0.0))
        if isinstance(axes, plt.Axes):
            axes = np.array([axes])
        for j, axi in enumerate(axes.flat[:num_targets]):
            # make joint plot
            jg = sns.jointplot(x=y_true[:, j], y=y_pred[:, j], kind='hist',
                               joint_kws=dict(bins=bins), marginal_kws=dict(bins=bins))
            if metric == mean_squared_error:
                # for MSE we can assign loss to each target
                loss_j = ((y_pred[:, j] - y_true[:, j])**2).mean()
                jg.fig.suptitle(f'{endpoint}[{j}] loss = {loss_j:8g}')
            else:
                # kullback-leibler divergence cannot be divided among targets
                jg.fig.suptitle(f'{endpoint}[{j}]')
            jg.set_axis_labels(xlabel='Actual', ylabel='Model')
            xlm = plt.xlim()
            ylm = plt.ylim()
            max_value = max(max(xlm), max(ylm))
            min_value = min(min(xlm), min(ylm))
            jg.ax_joint.plot([min_value, max_value], [min_value, max_value], color='k', linestyle='--')
            plt.tight_layout()

            # save joint plot in image
            jg.fig.canvas.draw()
            image_flat = np.frombuffer(jg.fig.canvas.tostring_rgb(), dtype='uint8')
            image = image_flat.reshape(*reversed(jg.fig.canvas.get_width_height()), 3)  #
            plt.close(jg.fig)

            # show image in subplots figure
            axi.imshow(image)
            axi.set(xticks=[], yticks=[])
        fig.suptitle(f'{endpoint} {subset} loss = {loss:8g}')
        fig.tight_layout()
        plt.savefig(fname=png_file, dpi=300)


def get_best_model(output_path):
    """
    Parses the output_path to find the best model. Relies on the ModelCheckpoint
    saving a file name with the validation loss in it. If a model was saved with
    a Normalization layer, it's provided as a custom object.

    :param output_path: The directory to scan for H5 files
    :return: The best model compiled.
    """
    min_loss = float('inf')
    best_model_file = None
    best_epoch = None
    for file_name in os.listdir(output_path):
        if file_name.endswith('.h5'):
            try:
                val_loss = float('.'.join(file_name.split('_')[1].split('.')[:-1]))
                epoch = int(file_name.split('.')[1].split('_')[0])
                if val_loss < min_loss:
                    best_model_file = file_name
                    min_loss = val_loss
                    best_epoch = epoch
            except IndexError:
                pass
    print(f'loading best model: {best_model_file}')
    model = load_model(os.path.join(
        output_path, best_model_file), compile=True)
    return model, best_epoch, min_loss


def plot_history(history, output_path='.'):
    keys = [k for k in history.history.keys() if not k.startswith('val_')]

    num_cols = int(np.ceil(len(keys) ** (1/2)))
    num_rows = int(np.ceil(len(keys) / num_cols))
    plt.figure()
    for i, k in enumerate(keys):
        ax = plt.subplot(num_rows, num_cols, i+1)
        ax.plot(history.history[k], label=k)
        val_key = f'val_{k}'
        if val_key in history.history:
            ax.plot(history.history[val_key], label=val_key)
        plt.legend()
    plt.savefig(os.path.join(output_path, 'learning_curve.png'), bbox_inches='tight')


def loudness_example():
    """
    An example applying linear regression to the loudness problem.

    This example uses early stopping on the validation loss.

    :return: None
    """
    endpoint = 'loudness'
    output_path = f'music_{endpoint}_linear'

    x_train, y_train = get_xy(endpoint, subset='train')
    x_valid, y_valid = get_xy(endpoint, subset='valid')

    clear_session()

    # setup callbacks
    model_checkpoint = setup_model_checkpoints(output_path)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    # create linear model
    model = Sequential([
        Input(129),
        Dense(1, activation='linear')
    ])

    model.compile(loss='mse', optimizer='adam')
    model.summary()

    history = model.fit(x_train, y_train, validation_data=[x_valid, y_valid],
                        callbacks=[model_checkpoint, early_stopping])
    model, best_epoch, best_loss = get_best_model(output_path)

    # plot history
    plot_history(history=history, output_path=output_path)
    visualize(model, endpoint=endpoint, subset='valid', output_path=output_path)


if __name__ == '__main__':
    loudness_example()
