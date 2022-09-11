import os
import json
import shutil
import numpy as np
from numpy import random
import tensorflow as tf
import tensorflow.compat.v1 as tf1

tf1.flags.DEFINE_string("checkpoint", help="checkpoint folder", default="GPTSAN-2.8B-spout_is_uniform" )
tf1.flags.DEFINE_string('output', help="output directory", default='GPTSAN-finetune-3B')
tf1.flags.DEFINE_integer('add_extra_stage', help="append layer num", default=14)
args = tf1.flags.FLAGS

def main():
    assert os.path.isdir(args.checkpoint) or args.checkpoint.startswith("gs://"), f"checkpoint directory not exists in {args.checkpoint}"
    assert not os.path.exists(args.output), f"output directory exists in {args.output}"

    os.mkdir(args.output)

    with tf.device("/CPU:0"):
        reader = tf.train.load_checkpoint(args.checkpoint)
        shapes = reader.get_variable_to_shape_map()
        dtypes = reader.get_variable_to_dtype_map()
        param = json.loads(open(os.path.join(args.checkpoint,"parameters.json")).read())
        ignore = json.loads(open(os.path.join(args.checkpoint,"ignore-params.json")).read())

        total_layer = param["model_params"]["num_switch_layers"] + param["model_params"]["num_ext_layers"]
        num_contexts = param["model_params"]["num_contexts"]
        num_hidden = param["model_params"]["num_hidden"]
        num_header = param["model_params"]["num_header"]
        num_spout = param["model_params"]["num_spout"]

        ignore_names = [i+"/adam_m" for i in ignore] + [i+"/adam_v" for i in ignore]
        ignore_names += ['pasts/out/kernel', 'pasts/out/kernel/adam_m', 'pasts/out/kernel/adam_v']
        vals = [tf.Variable(reader.get_tensor(k), name=k) for k,v in shapes.items() if k not in ignore_names]
        names = [k for k,v in shapes.items() if k not in ignore_names]

        addn = []
        def get_val(size, type, name):
            addn.append(name)
            if type==0:
                return tf.Variable(np.zeros(size,dtype=np.float32), name=name)
            elif type==1:
                return tf.Variable(np.ones(size,dtype=np.float32), name=name)
            else:
                return tf.Variable(random.standard_normal(size=size).astype(np.float32)*0.02, name=name)

        for suf in ["","/adam_m","/adam_v"]:
            vals.append(get_val((num_spout, (total_layer+args.add_extra_stage)*2*num_hidden), 2 if suf == "" else 0, name='pasts/out/kernel'+suf))
            for layer in range(total_layer,total_layer+args.add_extra_stage):
                z = 0
                o = 1 if suf == "" else 0
                t = 2 if suf == "" else 0
                vals.append(get_val((num_hidden,), z, name='model/an%d/b'%layer+suf))
                vals.append(get_val((num_hidden,), o, name='model/an%d/g'%layer+suf))
                vals.append(get_val((num_header,num_hidden//num_header,num_hidden), t, name='model/att%d/o/kernel'%layer+suf))
                vals.append(get_val((num_hidden,3,num_header,num_hidden//num_header), t, name='model/att%d/qkv/kernel'%layer+suf))
                vals.append(get_val((num_hidden,), z, name='model/ln%d/b'%layer+suf))
                vals.append(get_val((num_hidden,), o, name='model/ln%d/g'%layer+suf))
                vals.append(get_val((num_hidden,num_hidden*4), t, name='model/mlp%d/p1/kernel'%layer+suf))
                vals.append(get_val((num_hidden*4,), t, name='model/mlp%d/p1/bias'%layer+suf))
                vals.append(get_val((num_hidden*4,num_hidden), t, name='model/mlp%d/p2/kernel'%layer+suf))
                vals.append(get_val((num_hidden,), t, name='model/mlp%d/p2/bias'%layer+suf))
            if param["model_params"]["num_ext_layers"]==0 and args.add_extra_stage>0:
                vals.append(get_val((num_contexts, num_hidden), z, name='model/ete'+suf))

        param["model_params"]["num_ext_layers"] += args.add_extra_stage
        total_params = 0
        model_params = 0
        for v in vals:
            a = 1
            for s in list(v.shape):
                a *= s
            if not (v.name.endswith("/adam_m:0") or v.name.endswith("/adam_v:0")):
                model_params += a
            total_params += a
        print("total_params:",total_params)
        print("model_params:",model_params)

        saver = tf1.train.Saver(var_list=vals)
        saver.save(sess=None, save_path=os.path.join(args.output,"model"))

        with open(os.path.join(args.output,"checkpoint"), "w") as wf:
            wf.write("model_checkpoint_path: \"model\"\nall_model_checkpoint_paths: \"model\"\n")

        with open(os.path.join(args.output,"parameters.json"), "w") as wf:
            param["model_params"]["num_spout"] = num_spout
            param["train_params"]["output_dir"] = os.path.basename(args.output)
            wf.write(json.dumps(param))

        with open(os.path.join(args.output,"ignore-params.json"), "w") as wf:
            wf.write(json.dumps(ignore))

        with open(os.path.join(args.output,"transformer-params.json"), "w") as wf:
            wf.write(json.dumps(names))

        with open(os.path.join(args.output,"all-params.json"), "w") as wf:
            wf.write(json.dumps(names+addn))


if __name__ == "__main__":
    main()
