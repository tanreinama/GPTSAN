import json
import glob
import os
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_FP16_MATMUL_USE_FP32_COMPUTE'] = '0'
os.environ['TF_FP16_CONV_USE_FP32_COMPUTE'] = '0'

import tensorflow.compat.v1 as tf
import modeling
import time

tf.disable_v2_behavior()
tf.enable_eager_execution()
tf.get_logger().setLevel("INFO")

tf.flags.DEFINE_string("vocabulary", help="vocabulary file", default="ja-swe36k.txt" )
tf.flags.DEFINE_string('parameter_file', help="parameter file", default='train_params.json')
tf.flags.DEFINE_string('input_files', help="input file", default='*.tfrecord')
tf.flags.DEFINE_string('spout_vector', help="train generate seed vector (none/uniform/onehot)", default='none')
tf.flags.DEFINE_string('tpu_nodes', help="tpu nodes", default='')
tf.flags.DEFINE_bool('use_bfloat16', help="use bfloat16 for calculate", default=False)
tf.flags.DEFINE_bool('use_mixed_precision', help="use float16 for calculate", default=False)
tf.flags.DEFINE_bool('clip_gradients', help="use clip gradients for training (set on finetun transformer)", default=False)
tf.flags.DEFINE_string("ignore_parameters", help="no train parameter names", default="" )
tf.flags.DEFINE_integer("max_training_step", help="max training steps", default=-1 )
args = tf.flags.FLAGS

def input_fn_builder(input_files,
                     max_seq_length,
                     batch_size,
                     num_cpu_threads=1):

    def input_fn(params):
        name_to_features = {
            "x":
            tf.FixedLenFeature([max_seq_length], tf.int64),
            "y":
            tf.FixedLenFeature([max_seq_length], tf.int64),
            "i":
            tf.FixedLenFeature([1], tf.int64),
            "num_input":
            tf.FixedLenFeature([1], tf.int64)
        }
        d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
        d = d.apply(
            tf.data.experimental.parallel_interleave(
                tf.data.TFRecordDataset,
                sloppy=True,
                cycle_length=num_cpu_threads))
        d = d.apply(
            tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                num_parallel_batches=num_cpu_threads,
                drop_remainder=True))
        d = d.apply(tf.data.experimental.ignore_errors())
        return d.repeat()

    return input_fn

def _decode_record(record, name_to_features):
    example = tf.parse_single_example(record, name_to_features)
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t
    return example


def main():
    assert os.path.isfile(args.vocabulary), f"vocabulary file not found in {args.vocabulary}"
    assert os.path.isfile(args.parameter_file), f"parameter file not found in {args.parameter_file}"
    assert args.ignore_parameters=='' or os.path.isfile(args.ignore_parameters), f"ignore file not found in {args.ignore_parameters}"
    num_vocabulary = len(open(args.vocabulary, encoding='utf-8').read().split('\n'))

    if not args.input_files.startswith("gs://"):
        input_train_files = glob.glob(args.input_files)
    else:
        input_train_files = tf.gfile.Glob(args.input_files)
    assert len(input_train_files)>0, f"training file(s) not found in {args.input_files}"
    tf.logging.info("input_train_files = "+str(input_train_files))
    train_params = json.loads(open(args.parameter_file).read())
    train_params["model_params"]["train_mode"] = "lm"
    train_params["model_params"]["num_vocabulary"] = num_vocabulary

    if args.ignore_parameters!='':
        ignore_parameters = json.loads(open(args.ignore_parameters).read())
    else:
        ignore_parameters = []

    try:
        if args.tpu_nodes != "":
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver(args.tpu_nodes)
        else:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        topology = tf.tpu.experimental.initialize_tpu_system(tpu)
    except:
        tpu = None
        topology = None

    assert not (args.use_bfloat16 and args.use_mixed_precision), "bfloat16 and float16 cannot use at the same time."
    assert train_params["train_params"]["checkpoint_per_steps"]%100==0, "checkpoint_per_steps needs can division by 100."

    def model_fn(features, labels, mode, params):
        x = features["x"]
        y = features["y"] if "y" in features else tf.concat([x[:,1:],tf.zeros([x.shape[0],1],dtype=x.dtype)+num_vocabulary-1],axis=1)
        num_precontext = features["num_input"] if "num_input" in features else tf.zeros([x.shape[0],1], dtype=tf.int32)
        spout = None
        if args.spout_vector == "uniform":
            v = train_params["model_params"]["num_spout"]
            i = tf.cast(features["i"], tf.float32)
            spout = tf.random.uniform(shape=[i.shape[0], v])
        elif args.spout_vector == "stateless_uniform":
            v = train_params["model_params"]["num_spout"]
            i = tf.cast(features["i"], tf.float32)
            spout = tf.map_fn(fn=lambda t: tf.random.stateless_uniform(shape=[v], seed=tf.concat([tf.cast(t,tf.int32),tf.cast(t,tf.int32)],axis=0)), elems=i)
        elif args.spout_vector == "onehot":
            v = train_params["model_params"]["num_spout"]
            i = tf.truncatemod(features["i"], v)
            i = tf.reshape(i, [i.shape[0]])
            spout = tf.one_hot(i, depth=v)
        elif args.spout_vector == "half_onehot":
            v = train_params["model_params"]["num_spout"]
            i = tf.truncatemod(features["i"], v//2)
            r = tf.random.uniform(shape=[i.shape[0], v//2])
            i = tf.reshape(i, [i.shape[0]])
            spout = tf.concat([tf.one_hot(i, depth=v//2), r], axis=1)

        model, train = modeling.model(tpu,
                                      params,
                                      train_params,
                                      True,
                                      args.use_bfloat16,
                                      args.use_mixed_precision
                                    )
        return train(model(x, num_precontext, spout=spout),
                     y,
                     train_params["train_params"],
                     ignore_names=ignore_parameters,
                     clip_gradients=args.clip_gradients)

    run_config = tf.estimator.tpu.RunConfig(
          cluster=tpu,
          master=None,
          model_dir=train_params["train_params"]["output_dir"],
          save_checkpoints_steps=train_params["train_params"]["checkpoint_per_steps"],
          tpu_config=tf.estimator.tpu.TPUConfig(
              iterations_per_loop=100,
              num_cores_per_replica=1,
              per_host_input_for_training=tf.estimator.tpu.InputPipelineConfig.BROADCAST))

    estimator = tf.estimator.tpu.TPUEstimator(
        use_tpu=tpu is not None,
        eval_on_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=train_params["train_params"]["batch_size"])

    train_input_fn = input_fn_builder(
        input_files=input_train_files,
        max_seq_length=train_params["model_params"]["num_contexts"],
        batch_size=train_params["train_params"]["batch_size"])

    if not train_params["train_params"]["output_dir"].startswith("gs://"):
        if not os.path.isdir(train_params["train_params"]["output_dir"]):
            os.mkdir(train_params["train_params"]["output_dir"])
        with open(os.path.join(train_params["train_params"]["output_dir"],"parameters.json"), "w") as f:
            f.write(json.dumps(train_params))
    else:
        with tf.io.gfile.GFile(train_params["train_params"]["output_dir"]+"/parameters.json", "w") as f:
            f.write(json.dumps(train_params))

    try:
        current_step = int(tf.train.load_variable(train_params["train_params"]["output_dir"],
                                                  tf.GraphKeys.GLOBAL_STEP))
    except (TypeError, ValueError, tf.errors.NotFoundError):
        current_step = 0

    while True:
        start_time = time.time()
        train_steps = train_params["train_params"]["checkpoint_per_steps"] * 100
        next_checkpoint = current_step + train_steps
        estimator.train(input_fn=train_input_fn, steps=train_steps)
        print("Training Loop Iterated:",current_step)
        try:
            current_checkpoint_step = int(tf.train.load_variable(train_params["train_params"]["output_dir"],
                                                      tf.GraphKeys.GLOBAL_STEP))
            if current_checkpoint_step != next_checkpoint:
                current_step = next_checkpoint
                print("Training Loop Stopping:",current_checkpoint_step)
                break
        except (TypeError, ValueError, tf.errors.NotFoundError):
            current_step = next_checkpoint
        if args.max_training_step > 0 and args.max_training_step < train_steps:
            print("Training Loop Ended:",current_step)
            break

if __name__ == "__main__":
    main()
