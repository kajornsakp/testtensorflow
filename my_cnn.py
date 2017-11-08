import itertools
from tensorflow.contrib.session_bundle import exporter

from utils import *
from tensorflow.contrib import learn
from tensorflow.contrib.learn import *
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

def read_img(data_dir, batch_size, shuffle):
    def input_fn():
        image_list, label_list = read_labeled_image_list(data_dir)

        images = tf.convert_to_tensor(image_list, dtype=tf.string)
        labels = tf.convert_to_tensor(label_list, dtype=tf.int32)

        input_queue = tf.train.slice_input_producer([images, labels],
                                                    shuffle=shuffle,
                                                    capacity=batch_size * 5,
                                                    name="file_input_queue")

        image, label = read_images_from_disk(input_queue)

        # resize image
        image = tf.image.resize_images(image, (400, 300), tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # batch image and label to queue
        image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, capacity=batch_size * 10,
                                                  num_threads=1, name="batch_queue",
                                                  allow_smaller_final_batch=True)

        return tf.identity(image_batch, name="features"), tf.identity(label_batch, name="label")

    return input_fn


def cnn_model_fn(features, labels, mode, params):
    """
    Model function for CNN
    :param features: images features with shape (batch_size, height, width, channels)
    :param labels: images category with shape (batch_size)
    :param mode: Specifies if this training, evaluation or
                 prediction. See `model_fn_lib.ModeKey`
    :param params: dict of hyperparameters
    :return: predictions, loss, train_op, Optional(eval_op). See `model_fn_lib.ModelFnOps`
    """

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=features,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # conv3 = tf.layers.conv2d(
    #     inputs=pool2,
    #     filters=64,
    #     kernel_size=[3, 3],
    #     padding="same",
    #     activation=tf.nn.relu)
    # pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool_flat = tf.reshape(pool2, [-1, 100 * 75 * 64])
    dense = tf.layers.dense(inputs=pool_flat, units=256, activation=tf.nn.relu)
    # drop some data
    dropout = tf.layers.dropout(inputs=dense, rate=params['drop_out_rate'], training=mode == learn.ModeKeys.TRAIN)

    # map to output
    logits = tf.layers.dense(inputs=dropout, units=32)

    loss = None
    train_op = None

    #cal loss
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=32, name="onehot")
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            optimizer=tf.train.AdamOptimizer,
            learning_rate=params['learning_rate'],
            summaries=[
                "learning_rate",
                "loss",
                "gradients",
                "gradient_norm",
            ])
    # Generate Predictions
    predictions = {
        "classes": tf.argmax(
            input=logits, axis=1),
        "probabilities": tf.nn.softmax(
            logits, name="softmax_tensor")
    }

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op, eval_metric_ops={'dense': dense})


def feature_engineering_fn(features, labels):
    """
    feature_engineering_fn: Feature engineering function. Takes features and
                              labels which are the output of `input_fn` and
                              returns features and labels which will be fed
                              into `model_fn`
    """

    features = tf.to_float(features)

    # Preprocessing or Data Augmentation
    # tf.image implements most of the standard image augmentation

    # Example
    # Subtract off the mean and divide by the variance of the pixels.
    features = tf.map_fn(tf.image.per_image_standardization, features)

    return features, labels

def my_signature_fn(examples,features,predictions):
    return None,{
        "inputs":exporter.generic_signature({"features":examples}),
        "outputs":exporter.generic_signature({"score":predictions})
    }

if __name__ == '__main__':
    params = {'drop_out_rate': 0.2, 'learning_rate': 0.00001}
    cnn_classifier = learn.Estimator(
        model_fn=cnn_model_fn, model_dir="my_cnn_model_exported",
        config=RunConfig(save_summary_steps=10, keep_checkpoint_max=2, save_checkpoints_secs=30),
        feature_engineering_fn=feature_engineering_fn, params=params)
    # Configure the accuracy metric for evaluation
    metrics = {
        "accuracy":
            learn.MetricSpec(
                metric_fn=tf.metrics.accuracy, prediction_key="classes"),
    }

    train_input_fn = read_img(data_dir='leaves_data/train', batch_size=4, shuffle=True)
    monitor_input_fn = read_img(data_dir='leaves_data/validate', batch_size=4, shuffle=True)
    test_input_fn = read_img(data_dir='leaves_data/test', batch_size=4, shuffle=False)
    predict_input_fn = read_img(data_dir='input_leaves',batch_size=4,shuffle=False)

    validation_monitor = monitors.ValidationMonitor(input_fn=monitor_input_fn,
                                                    eval_steps=10,
                                                    every_n_steps=50,
                                                    metrics=metrics,
                                                    name='validation')
    hook = monitors.replace_monitors_with_hooks([validation_monitor],cnn_classifier)
    cnn_classifier.fit(input_fn=train_input_fn, steps=1000, monitors=[validation_monitor])
    # prediction = cnn_classifier.predict(input_fn=predict_input_fn)
    # y = list(itertools.islice(prediction,6))
    # x = 1r
    # for i in y:
    #     print("Input : ", x , "  Output: ",i['classes']+1," : ",max(i['probabilities']))
    #     x+= 1

    cnn_classifier.export("model_my_cnn_2/")
    # Evaluate the _model and print results
    eval_results = cnn_classifier.evaluate(input_fn=test_input_fn, metrics=metrics, steps=1,hooks=hook)
    np.save(os.getcwd() + '/embedding1.npy', eval_results['dense'])
