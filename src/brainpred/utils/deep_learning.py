import numpy as np
import tensorflow as tf
import pandas as pd


def prepare_img_for_gan(img, img_size=(192, 224, 192), axis_channels=-1, img_offset=(0,0,0)):
    img = img[img_offset[0]:img_offset[0]+img_size[0], img_offset[1]:img_offset[1]+img_size[1], img_offset[2]:img_offset[2]+img_size[2]]
    img = np.expand_dims(np.expand_dims(img, axis=0), axis=axis_channels)
    return img

def recover_img_from_gan(img, img_size=(193, 229, 193), axis_channels=-1, img_offset=(0,0,0)):
    # res_shape = (np.array(img_size)+np.array(img_offset)).astype(np.int32)
    rec_img = np.zeros(np.array(img_size))
    if axis_channels == 1:
        img = img[0,0]
    if axis_channels == -1:
        img = img[0,...,0]

    rec_img[img_offset[0]:img_offset[0]+img.shape[0], img_offset[1]:img_offset[1]+img.shape[1], img_offset[2]:img_offset[2]+img.shape[2]] = img
    return rec_img

def get_optimizer():
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.9)
    optimizer = tf.keras.optimizers.Adam()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01)


def freeze_layers(base_model, not_frozen_layers=0):
    if not_frozen_layers < 0:
        max = 0
    else:
        max = len(base_model.layers) -not_frozen_layers
        
    for i in range(len(base_model.layers)):
        if i == max:
            break 
        base_model.layers[i].trainable = False

def convert_image_to_tensor(image):
    image = image / np.max(image)
    image = tf.expand_dims(image, axis=0)
    return image


class LearningRateMonitor:
    def __init__(self, optimizer, initial_lr=0.01, reduce_factor=0.5, min_lr=0.0001, patience=3, monitor="loss"):
        self.optimizer = optimizer
        self.lr = initial_lr
        self.reduce_factor = reduce_factor
        self.min_lr = min_lr
        self.patience = patience
        self.monitor = monitor
        self.patience_counter = 0
        self.init_best_metric()
    
    def init_best_metric(self):
        if self.monitor == "loss":
            self.best_metric = 10000000
        elif self.monitor == "acc":
            self.best_metric = 0

    def is_bad_epoch(self, metric_value):
        if self.monitor == "loss":
            return metric_value >= self.best_metric 
        elif self.monitor == "acc":
            return metric_value <= self.best_metric 

    def on_epoch_end(self, metric):
        if self.is_bad_epoch(metric):
            self.patience_counter += 1
            self.__update(metric)
        else:
            self.patience_counter = 0
            self.best_metric = metric

    def __update(self, metric):
        if self.patience_counter >= self.patience and self.lr > self.min_lr:          
            self.lr *= self.reduce_factor
            self.optimizer.learning_rate = self.lr
            print(self.optimizer.learning_rate)
            self.patience_counter = 0



class AcumulativeMetrics:
    def __init__(self, nb_metrics):
        self.nb_metrics = nb_metrics
        self.metrics_sum = np.zeros(nb_metrics)
        self.step = 0
    
    def update(self, metrics):
        self.metrics_sum += np.array(metrics)
        self.step += 1
        
    def get_mean(self):
        return self.metrics_sum / (self.step)  

    def reset(self):
        self.metrics_sum = np.zeros(self.nb_metrics)
        self.step = 0   

    def get_progress_bar_text(self, names=None):
        _mean = self.get_mean()
        text = ""
        for i in range(len(_mean)):
            text += f'{names[i]}: {_mean[i]:.5f}'
            if i < len(names)-1:
                text += '\t'
        return text


class AccumulativeClassificationMetrics:
    def __init__(self, nb_classes):
        self.nb_classes = nb_classes
        self.step = 0
        self.acumulative_y_pred = np.zeros(nb_classes)
        self.acumulative_y = np.zeros(nb_classes)
        self.acc = 0

    def update(self, y_onehot, y_pred_onehot):
        for i in range(len(y_pred_onehot)):
            self.acumulative_y_pred += y_pred_onehot[i]
            self.acumulative_y += y_onehot[i]
            self.acc += np.argmax(y_pred_onehot[i]) == np.argmax(y_onehot[i])
            self.step += 1  

    def reset(self):
        self.acumulative_y_pred = np.zeros(self.nb_classes)
        self.acumulative_y = np.zeros(self.nb_classes)
        self.step = 0  
        self.acc = 0

    # def get_classes_mean(self):
    #     return np.where(self.acumulative_y != 0, self.acumulative_y_pred / self.acumulative_y, 0)
    
    def get_acc(self):
        return self.acc/self.step
    
    def get_progress_bar_text(self):
        text = "["
        for i, (y_pred, y) in enumerate(zip(self.acumulative_y_pred, self.acumulative_y)):
            text += f'{int(y_pred)}/{int(y)}'
            if i < self.nb_classes-1:
                text += ', '
        return text + "]"

class EarlyStopping:
    def __init__(self, patience, monitor="acc", early_model_path_name=None, current_model_path_name=None):
        self.patience = patience
        self.monitor = monitor
        self.best_epoch = 0
        self.init_best_metric()
        self.patience_counter = 0
        self.early_model_path_name = early_model_path_name
        self.current_model_path_name = current_model_path_name

    def init_best_metric(self):
        if self.monitor == "loss":
            self.best_metric = 10000000
        elif self.monitor == "acc":
            self.best_metric = 0

    def is_bad_epoch(self, metric_value):
        if self.monitor == "loss":
            return metric_value > self.best_metric 
        elif self.monitor == "acc":
            return metric_value < self.best_metric 

    def on_epoch_end(self, metric, model=None):
        if self.is_bad_epoch(metric):
            self.patience_counter += 1
        else:
            self.patience_counter = 0
            self.best_metric = metric
            if model is not None and self.early_model_path_name is not None:
                model.save_weights(self.early_model_path_name)
        
        if model is not None and self.current_model_path_name is not None:
            model.save_weights(self.current_model_path_name)

        return self.patience_counter >= self.patience



class HistorySaver:
    def __init__(self, nb_epochs, path_name, columns):
        self.columns = columns
        self.data = np.zeros((nb_epochs, len(columns)))
        self.path_name = path_name
        self.current_epoch = 0

    def update_and_save(self, new_data):
        self.data[self.current_epoch] = new_data
        df = pd.DataFrame(self.data, columns=self.columns)
        df.to_csv(self.path_name, index=False)
        self.current_epoch += 1





# Segmentation using gaussian mixture model
def _prob_x_given_k_tf(data, means, stds, dtype=tf.float32):   
    data = tf.cast(data, dtype=dtype)
    means = tf.cast(means, dtype=dtype)
    stds = tf.cast(stds, dtype=dtype)
    pi = tf.constant(np.pi, dtype=dtype)

    scale_factor = 1 / (stds * tf.sqrt(2 * pi))
    exponent = -0.5 * ((data[:, tf.newaxis] - means) / stds) ** 2
    pdf = scale_factor * tf.exp(exponent)
    return pdf

def probs_from_gmm_tf(img_flatten_masked, means, stds, weights, dtype=tf.float32):
    prob_x_given_k = _prob_x_given_k_tf(img_flatten_masked, means, stds, dtype=dtype)
    w = prob_x_given_k * weights[tf.newaxis, :]
    den = tf.reduce_sum(w, axis=1)
    probs = w / den[:, tf.newaxis]
    return probs


def dice_loss(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred, axis=0)
    union = tf.reduce_sum(y_true, axis=0) + tf.reduce_sum(y_pred, axis=0)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1-dice


def gmm_dice_loss(seg_true, y_pred, means, stds, classs_weights, nb_tissues=3, dtype=tf.float32):
   
    y_pred_flat = tf.reshape(y_pred, [-1])
    seg_true_flat = tf.reshape(seg_true, [-1])

    y_pred_flat_masked = tf.boolean_mask(y_pred_flat, seg_true_flat > 0)
    seg_true_flat_masked = tf.boolean_mask(seg_true_flat, seg_true_flat > 0)-1

    seg_true_one_hot = tf.cast(tf.one_hot(seg_true_flat_masked, depth=nb_tissues), dtype)
    seg_prob_y_pred = probs_from_gmm_tf(y_pred_flat_masked, means, stds, classs_weights, dtype=dtype)

    res_dice_loss = dice_loss(seg_true_one_hot, seg_prob_y_pred)

    return res_dice_loss