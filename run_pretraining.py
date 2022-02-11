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

tf.flags.DEFINE_string("vocabulary", help="vocabulary file", default="ja-swe36k.txt" )
tf.flags.DEFINE_string('parameter_file', help="parameter file", default='train_params.json')
tf.flags.DEFINE_string('input_files', help="input file", default='*.tfrecord')
tf.flags.DEFINE_string('tpu_nodes', help="tpu nodes", default='')
tf.flags.DEFINE_bool('use_bfloat16', help="use bfloat16 for calculate", default=False)
tf.flags.DEFINE_bool('use_mixed_precision', help="use float16 for calculate", default=False)
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
        return d

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
    num_vocabulary = len(open(args.vocabulary, encoding='utf-8').read().split('\n'))

    if not args.input_files.startswith("gs://"):
        input_train_files = glob.glob(args.input_files)
    else:
        input_train_files = tf.gfile.Glob(args.input_files)
    assert len(input_train_files)>0, f"training file(s) not found in {args.input_files}"
    train_params = json.loads(open(args.parameter_file).read())
    train_params["model_params"]["train_mode"] = "lm"
    train_params["model_params"]["num_vocabulary"] = num_vocabulary

    if args.tpu_nodes != "":
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(args.tpu_nodes)
        tf.config.experimental_connect_to_cluster(tpu)
        topology = tf.tpu.experimental.initialize_tpu_system(tpu)
    else:
        tpu = None
        topology = None

    assert not (args.use_bfloat16 and args.use_mixed_precision), "bfloat16 and float16 cannot use at the same time."
    assert train_params["train_params"]["checkpoint_per_steps"]%100==0, "checkpoint_per_steps needs can division by 100."

    def model_fn(features, labels, mode, params):
        x = features["x"]
        y = features["y"] if "y" in features else tf.concat([x[:,1:],tf.zeros([x.shape[0],1],dtype=x.dtype)+num_vocabulary-1],axis=1)
        num_precontext = features["num_input"] if "num_input" in features else tf.zeros([x.shape[0],1], dtype=tf.int32)
        model, train = modeling.model(tpu,
                                      params,
                                      train_params,
                                      True,
                                      args.use_bfloat16,
                                      args.use_mixed_precision
                                    )
        return train(model(x, num_precontext), y, train_params["train_params"])

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

if __name__ == "__main__":
    main()
