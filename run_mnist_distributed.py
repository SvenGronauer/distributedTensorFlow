import json
import argparse
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import dataset
import logger
import datetime
import os

FLAGS = None


class Config:
    def __init__(self, file_path='config.json'):
        self.file_path = file_path
        self.json = self._load_config()
        try:  # check if specified keys in config.json equal the expected names
            self.ps = self.json['ps']
            self.workers = self.json['workers']
        except KeyError as err:
            raise AttributeError('Please provide a value for "{0}" configuration key in the config.json file!'.format(
                err.args[0]))

    def _load_config(self):
        with open(self.file_path) as f:
            data = json.load(f)
        return data

    def get_workers_with_addresses(self):
        # print(self.workers.items()) ==>
        # {'worker:0': '192.168.2.107:5001', 'worker:1': '192.168.2.107:5002'}
        workers, addresses = zip(*self.workers.items())  # unzipping
        return workers, addresses

    def get_ps_with_addresses(self):
        ps, addresses = zip(*self.ps.items())  # unzipping
        return ps, addresses

    def get_ps_and_worker_hosts(self):
        _, ps_hosts = self.get_ps_with_addresses()
        _, worker_hosts = self.get_workers_with_addresses()
        return ps_hosts, worker_hosts


def create_model(inputs):
    """
    Model to recognize digits in the MNIST dataset.
    """
    input_layer = tf.reshape(inputs, [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    logits = tf.layers.dense(inputs=dense, units=10)
    return logits


def main(_):
    print('run main with args =', FLAGS)

    # get MNIST dataset
    batch_size = 128
    ds = dataset.train("/tmp/data/")  # download data set to directory "/tmp/data/"
    # ds = tf.data.Dataset.from_tensor_slices(
    #     (mnist.train.images, mnist.train.labels))
    ds = ds.repeat().batch(batch_size).prefetch(batch_size)
    # create iterator for dataset
    iterator = tf.data.Iterator.from_structure(ds.output_types,
                                               ds.output_shapes)
    iter_init = iterator.make_initializer(ds)
    # iterator = tf.Iterator(ds)

    # load configuration from .json file
    config = Config()
    ps_hosts, worker_hosts = config.get_ps_and_worker_hosts()

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        print('Started Parameter Server ...')
        server.join()
        print('Close Parameter Server ...')
    elif FLAGS.job_name == "worker":
        # Assigns ops to the local worker by default.
        is_chief = (FLAGS.task_index == 0)
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,
                                                      cluster=cluster)):

            # Build model...
            X, Y = iterator.get_next()
            logits = create_model(X)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=Y, logits=logits)
            global_step = tf.train.get_or_create_global_step()

            train_op = tf.train.AdamOptimizer(0.0005).minimize(loss, global_step=global_step)

        # The StopAtStepHook handles stopping after running given steps.
        max_steps = 1000
        hooks = [tf.train.StopAtStepHook(last_step=max_steps)]

        tf_config = tf.ConfigProto(allow_soft_placement=True,  # soft placement to allow flexible training on CPU/GPU
                                   intra_op_parallelism_threads=os.cpu_count(),  # speed up training time
                                   inter_op_parallelism_threads=os.cpu_count())
        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=is_chief,
                                               # checkpoint_dir="/tmp/train_logs",
                                               config=tf_config,
                                               hooks=hooks) as mon_sess:
            mon_sess.run(iter_init)
            local_step = 0
            # current_date_time = datetime.datetime.now()
            if is_chief:
                date_string = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
                log_directory = os.path.join('/tmp/distributed_logs', date_string)
                logs = logger.TensorBoardOutputFormat(dir=log_directory)
                print('Logging to', log_directory)

            while not mon_sess.should_stop():
                # Run a training step asynchronously.
                # See tf.train.SyncReplicasOptimizer for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.

                local_step += 1
                if is_chief:  # for debug prints and logging
                    _, train_loss, gstep = mon_sess.run([train_op, loss, global_step])

                    print("Worker ({}): loss = {:0.2f} (global step: {})".format(
                            FLAGS.task_index, train_loss, gstep))
                    summary = {"Global Step": gstep,
                               "Loss": train_loss}
                    # logger.logkv("Global Step", gstep)
                    # logger.logkv("Loss", train_loss)
                    # # logger.dumpkvs()
                    logs.writekvs(summary, global_step=gstep)
                else:
                    mon_sess.run(train_op)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        required=True,
        help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        required=True,
        help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
