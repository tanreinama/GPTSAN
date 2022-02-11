import os
import shutil
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

tf.flags.DEFINE_string('tpu_nodes', help="tpu nodes", default='')
tf.flags.DEFINE_string("checkpoint", help="checkpoint folder", default="checkpoints" )
tf.flags.DEFINE_string('output', help="output directory", default='model-to-release')
args = tf.flags.FLAGS

def main():
    assert os.path.isdir(args.checkpoint), f"checkpoint directory not exists in {args.checkpoint}"
    assert not os.path.exists(args.output), f"output directory exists in {args.output}"

    if args.tpu_nodes=='':
        print('TPU node not foud. Using GPU device.')

    if args.tpu_nodes != "":
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(args.tpu_nodes)
        tf.config.experimental_connect_to_cluster(tpu)
        topology = tf.tpu.experimental.initialize_tpu_system(tpu)
    else:
        tpu = None
        topology = None

    os.mkdir(args.output)

    with tf.device('/TPU:0' if args.tpu_nodes!='' else ''):
        sess = tf.Session()
        model = tf.train.latest_checkpoint(args.checkpoint)

        assert os.path.isfile(model+".meta"), f"checkpoint meta file not exists in {model}"

        n_saver = tf.train.import_meta_graph(model+".meta")
        n_saver.restore(sess, model)

        g = tf.get_default_graph()
        gv = g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        gvf = [v for v in gv if not v.name.split("/")[-1].startswith("adam_")]
        sv = tf.train.Saver(var_list=gvf)
        sv.save(sess, os.path.join(args.output,model[len(args.checkpoint):]))
        shutil.copy(os.path.join(args.checkpoint,"parameters.json"), args.output)


if __name__ == "__main__":
    main()
