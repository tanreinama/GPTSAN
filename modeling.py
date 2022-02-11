import os
import numpy as np
import tensorflow.compat.v1 as tf
import mesh_tensorflow as mtf
from mesh_tensorflow.auto_mtf.api import layout as auto_layout
from mesh_tensorflow.transformer import moe as moe_lib
import optimization as optimization_lib

import mesh_tensorflow.optimize as mtf_optimize

tf.disable_v2_behavior()
tf.enable_eager_execution()

def model(tpu, params, train_params, is_training, use_bfloat, use_mixed_precision):
    num_contexts = train_params["model_params"]["num_contexts"]
    num_layers = train_params["model_params"]["num_layers"]
    num_hidden = train_params["model_params"]["num_hidden"]
    num_header = train_params["model_params"]["num_header"]
    num_vocabulary = train_params["model_params"]["num_vocabulary"]
    use_moe = train_params["model_params"]["use_moe"]
    num_experts = train_params["model_params"]["num_experts"]
    num_pallarelizm = train_params["model_params"]["num_pallarelizm"]

    dim_contexts = mtf.Dimension("max_contexts", num_contexts)
    dim_layers = mtf.Dimension("layers", num_layers)
    dim_hidden = mtf.Dimension("hidden", num_hidden)
    dim_header = mtf.Dimension("header", num_header)
    dim_kernel = mtf.Dimension("kernel", num_hidden//num_header)
    dim_vocabulary = mtf.Dimension("vocabulary", num_vocabulary)
    dim_intermediate = mtf.Dimension("intermediate", num_hidden * 4)
    dim_keyvalue = mtf.Dimension("keyvalue", 2)
    dim_store = mtf.Dimension("store", 3)
    dim_vector = mtf.Dimension("vector", num_hidden * 4)

    graph = mtf.Graph()
    if tpu is not None:
        ctx = params["context"]
        device_list = [ctx.tpu_host_placement_function(host_id=t) for t in range(ctx.num_hosts)]
        num_repricates = tpu.num_accelerators()['TPU']
        mesh_shape = mtf.Shape([("batch", num_repricates//num_pallarelizm),("model", num_pallarelizm)])
        layout_rules = [('batch', 'batch'),('header', 'model'),('experts', 'model')]
        vp = mtf.utils.BalancedVariablePlacer(device_list, [600 * 1000000 * num_repricates] + [0]*(ctx.num_hosts-1))
        mesh = mtf.Mesh(graph, "mesh", vp)
        physical_shape = list(ctx.device_assignment.topology.mesh_shape)
        logical_to_physical = mtf.simd_mesh_impl.auto_logical_to_physical_tpu(
            mesh_shape.to_integer_list, physical_shape)
        mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
            mesh_shape, layout_rules, [""]*mesh_shape.size, ctx.device_assignment, logical_to_physical)
    else:
        num_repricates = len(tf.config.experimental.list_physical_devices('GPU'))
        mesh_shape = mtf.Shape([("batch", num_repricates//num_pallarelizm),("model", num_pallarelizm)])
        layout_rules = [('batch', 'batch'),('header', 'model'),('experts', 'model')]
        device_list = ["/device:gpu:%d"%i for i in range(num_repricates)]
        mesh = mtf.Mesh(graph, "mesh")
        mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
            mesh_shape, layout_rules, device_list)

    tf.logging.info("device_list =", device_list)

    fret_dtype = tf.float16 if use_mixed_precision else tf.float32
    dense_dtype = tf.bfloat16 if (is_training and use_bfloat) else fret_dtype
    mask_dtype = tf.float32 if use_mixed_precision else dense_dtype
    variable_dtype = mtf.VariableDType(tf.float32, tf.float32, dense_dtype)

    def run_fn(results):
        output = results['output']
        logits = results['logits']
        present = results['present']

        lowering = mtf.Lowering(graph, {mesh: mesh_impl})

        output = lowering.export_to_tf_tensor(output)
        logits = lowering.export_to_tf_tensor(logits)
        present = lowering.export_to_tf_tensor(present)

        restore_hook = mtf.MtfRestoreHook(lowering)
        predictions={'output':output,
                     'logits':logits,
                     'present':present}
        if results['vector'] is not None:
            vector = results['vector']
            vector = lowering.export_to_tf_tensor(vector)
            predictions['vector'] = vector
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            prediction_hooks=[restore_hook])

    def train_fn(results, y, train_params):
        num_batch = y.get_shape().as_list()[0]
        dim_batch = mtf.Dimension("batch", num_batch)
        num_input_contexts = y.get_shape().as_list()[1]
        dim_input_contexts = mtf.Dimension("input_contexts", num_input_contexts)
        output = results['output']
        logits = results['logits']
        present = results['present']
        vector = results['vector']
        extra_loss = results['extra_loss']
        output_dir = train_params['output_dir']
        base_lr = train_params['base_lr']
        num_warmup_steps = train_params['num_warmup_steps']
        max_to_keep_save = train_params['max_to_keep_save']
        checkpoint_per_hours = train_params['checkpoint_per_hours']
        checkpoint_per_steps = train_params['checkpoint_per_steps']

        y = mtf.import_tf_tensor(mesh, y, [dim_batch, dim_input_contexts])

        mask = mtf.cast(mtf.cast(y+1, tf.bool), tf.int32)
        labels = mtf.cast(y + (1 - mask), tf.int32)
        example_loss = mtf.layers.softmax_cross_entropy_with_logits(
                logits, labels, dim_vocabulary, z_loss=1e-4)
        mask = mtf.cast(mask, tf.float32)
        numerator = mtf.reduce_sum(mask * example_loss)
        denominator = mtf.reduce_sum(mask) + mtf.constant(mesh, 1e-5, dtype=tf.float32)
        loss = numerator / denominator
        loss = mtf.anonymize(loss)
        extra_loss = mtf.anonymize(extra_loss) if extra_loss is not None else 0

        num_params = 0
        for s in [[p.size for p in t.shape.dims] for t in mesh.graph.trainable_variables]:
            d = 1
            for i in s:
                d *= i
            num_params += d
        tf.logging.info("num_params =",num_params)

        _, update_ops = optimization_lib.create_optimizer(
            loss + extra_loss,
            base_lr,
            num_warmup_steps,
            optimizer="adam",
            grad_scale=2**9 if use_mixed_precision else 1.0,
            clip_gradients=False)

        outputs = (output, present, logits, vector)

        lowering = mtf.Lowering(graph, {mesh: mesh_impl})

        tf_loss = tf.cast(lowering.export_to_tf_tensor(loss), tf.float32)

        global_step = tf.train.get_global_step()
        tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
        tf_update_ops.append(tf.assign_add(global_step, 1))
        train_op = tf.group(tf_update_ops)

        with mtf.utils.outside_all_rewrites():
            restore_hook = mtf.MtfRestoreHook(lowering)
            saver = tf.train.Saver(
                tf.global_variables(),
                sharded=True,
                max_to_keep=max_to_keep_save,
                keep_checkpoint_every_n_hours=checkpoint_per_hours,
                defer_build=False,
                save_relative_paths=True)
            tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
            saver_listener = mtf.MtfCheckpointSaverListener(lowering)
            saver_hook = tf.train.CheckpointSaverHook(
                output_dir,
                save_steps=checkpoint_per_steps,
                saver=saver,
                listeners=[saver_listener])

        res = tf.estimator.tpu.TPUEstimatorSpec(
            tf.estimator.ModeKeys.TRAIN,
            loss=tf_loss,
            train_op=train_op,
            training_hooks=[restore_hook, saver_hook])
        return res

    def model_fn(x, num_precontext=None, pos_vector=None, pasts=None):
        assert x.shape.ndims == 2  # x Should be [batch, sequence]
        num_batch = x.get_shape().as_list()[0]
        dim_batch = mtf.Dimension("batch", num_batch)
        num_input_contexts = x.get_shape().as_list()[1]
        dim_input_contexts = mtf.Dimension("input_contexts", num_input_contexts)
        dim_memory_contexts = mtf.Dimension("memory_contexts", num_input_contexts)

        num_pasts_contexts = 0
        if pasts is not None:
            num_pasts_contexts = pasts.get_shape().as_list()[4]
        num_output_contexts = num_input_contexts+num_pasts_contexts
        dim_output_contexts = mtf.Dimension("memory_contexts", num_output_contexts)
        pasts_dims = [dim_batch, dim_layers, dim_keyvalue, dim_header, mtf.Dimension("memory_contexts", num_pasts_contexts), dim_kernel]
        presents_dims = [dim_batch, dim_layers, dim_keyvalue, dim_header, dim_output_contexts, dim_kernel]

        x = mtf.import_tf_tensor(mesh, x, [dim_batch, dim_input_contexts])
        if num_precontext is not None:
            assert num_precontext.shape.ndims == 2 and num_precontext.shape[1]==1  # num_precontext Should be [batch,1]
            num_precontext = tf.reshape(num_precontext, [-1])
        else:
            num_precontext = tf.zeros([num_batch])
        if pasts is not None:
            pasts = mtf.import_tf_tensor(mesh, pasts, pasts_dims)
        if pos_vector is not None:
            pos_vector = mtf.import_tf_tensor(mesh, tf.reshape(pos_vector, [-1]), [dim_batch])

        def normalization(x, scope, axis=-1, epsilon=1e-5):
            assert x.shape.ndims == 3  # x Should be [batch, sequence, features]
            with tf.variable_scope(scope):
                n_state = x.shape[-1]
                g = mtf.get_variable(mesh, 'g', [n_state], initializer=tf.ones_initializer(), dtype=variable_dtype)
                b = mtf.get_variable(mesh, 'b', [n_state], initializer=tf.zeros_initializer(), dtype=variable_dtype)
                if use_mixed_precision:
                    x = mtf.cast(x, tf.float32)
                    x -= mtf.reduce_mean(x, reduced_dim=dim_hidden)
                    s = mtf.reduce_mean(mtf.square(x), reduced_dim=dim_hidden)
                    x *= mtf.rsqrt(s + epsilon)
                    x = mtf.cast(x, dense_dtype)
                else:
                    x -= mtf.reduce_mean(x, reduced_dim=dim_hidden)
                    s = mtf.reduce_mean(mtf.square(x), reduced_dim=dim_hidden)
                    x *= mtf.rsqrt(s + epsilon)
                return x*g + b

        def multiheadattention(x, scope, nh, mask, past=None, w_init_stdev=0.02):
            assert x.shape.ndims == 3  # x Should be [batch, sequence, features]
            with tf.variable_scope(scope):
                qkv = mtf.layers.dense(
                    x,
                    reduced_dims=[dim_hidden],
                    new_dims=[dim_store, dim_header, dim_kernel],
                    kernel_initializer=tf.random_normal_initializer(stddev=w_init_stdev),
                    name="qkv",
                    use_bias=False,
                    variable_dtype=variable_dtype)
                query, key, value = mtf.split(qkv, split_dim=dim_store, num_or_size_splits=3)
                query = mtf.reshape(query, [dim_batch, dim_input_contexts, dim_header, dim_kernel])
                key = mtf.reshape(key, [dim_batch, dim_memory_contexts, dim_header, dim_kernel])
                value = mtf.reshape(value, [dim_batch, dim_memory_contexts, dim_header, dim_kernel])
                query = mtf.transpose(query, [dim_batch, dim_header, dim_input_contexts, dim_kernel])
                key = mtf.transpose(key, [dim_batch, dim_header, dim_memory_contexts, dim_kernel])
                value = mtf.transpose(value, [dim_batch, dim_header, dim_memory_contexts, dim_kernel])

                present = mtf.stack([key, value], "memory_keyvalue", 1)
                present = mtf.replace_dimensions(present, present.shape.dims[1], dim_keyvalue)
                # present shuld be [batch, 2, heads, sequence, hidden]

                if past is not None:
                    target_dim = presents_dims[4]
                    pk, pv = mtf.unstack(past, dim_keyvalue)
                    # pk, pv shuld be [batch, heads, sequence, hidden]
                    key = mtf.concat([pk, key], "memory_contexts")
                    value = mtf.concat([pv, value], "memory_contexts")

                scores = mtf.einsum([query, key], reduced_dims=[dim_kernel])
                if use_mixed_precision:
                    scores = mtf.cast(scores, mask_dtype)
                scores *= dim_kernel.size ** -0.5
                scores *= mask
                scores -= (1-mask) * 10000.0
                probs = mtf.softmax(scores, reduced_dim=scores.shape.dims[-1])
                if use_mixed_precision:
                    probs = mtf.cast(probs, dense_dtype)
                output = mtf.einsum([probs, value], reduced_dims=[probs.shape.dims[-1]])
                output = mtf.layers.dense(
                    output,
                    reduced_dims=[dim_header, dim_kernel],
                    new_dims=[dim_hidden],
                    kernel_initializer=tf.random_normal_initializer(stddev=w_init_stdev),
                    name="o",
                    use_bias=False,
                    variable_dtype=variable_dtype)

                return output, present

        def mlp(x, scope, w_init_stdev=0.02):
            assert x.shape.ndims == 3  # x Should be [batch, sequence, features]
            with tf.variable_scope(scope):
                intermediate = mtf.layers.dense(
                    x, reduced_dims=[dim_hidden],
                    new_dims=[dim_intermediate],
                    activation=mtf.swish,
                    kernel_initializer=tf.random_normal_initializer(stddev=w_init_stdev),
                    name="p1", use_bias=True,
                    variable_dtype=variable_dtype)
                return mtf.layers.dense(
                    intermediate,
                    reduced_dims=[dim_intermediate],
                    new_dims=[dim_hidden],
                    kernel_initializer=tf.random_normal_initializer(stddev=w_init_stdev),
                    name="p2", use_bias=True,
                    variable_dtype=variable_dtype)

        def moe(x, scope):
            assert x.shape.ndims == 3  # x Should be [batch, sequence, features]
            with tf.variable_scope(scope):
                hparams = moe_lib.HParams()
                moe_lib.set_default_moe_hparams(hparams)
                hparams.moe_gating = "switch"
                hparams.moe_num_experts = num_experts
                hparams.moe_hidden_size = num_hidden*8
                hparams.moe_word_embed_mode = None
                hparams.moe_min_expert_capacity = 4
                hparams.moe_switch_policy_train = "input_jitter"
                hparams.moe_switch_policy_eval = "input_jitter"
                hparams.moe_switch_jitter = 1e-2
                hparams.moe_use_second_place_expert_prob = None
                hparams.moe_z_loss = None
                hparams.moe_dropout_rate = 0.0
                hparams.moe_use_experts_attention = False
                layer_output, loss = moe_lib.transformer_moe_layer_v1(
                    inputs=x,
                    output_dim=dim_hidden,
                    hparams=hparams,
                    train=is_training,
                    variable_dtype=tf.float32,
                    layout=layout_rules,
                    mesh_shape=mesh_shape,
                    nonpadding=1,
                    activation=mtf.relu)
                layer_output += mtf.layers.dense(
                    x, reduced_dims=[dim_hidden],
                    new_dims=[dim_hidden],
                    activation=mtf.tanh,
                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                    name="softmlp", use_bias=False,
                    variable_dtype=variable_dtype)
                return layer_output, loss


        def make_attention_mask(total_seq, output_seq, input_len):
            i = tf.range(total_seq)[:,None]
            j = tf.range(output_seq)
            m = i >= j - output_seq + total_seq
            lm_mask = tf.cast(m, mask_dtype) # language model mask
            lm_mask = tf.reshape(lm_mask, [total_seq, output_seq])
            # lm_mask shuld be [sequence, sequence]
            weight = tf.transpose(tf.range(output_seq)[:,None] < input_len, [1,0])
            weight = tf.cast(weight, mask_dtype) # Masked language model mask
            # weight shuld be [batch, sequence]
            mlm_mask = tf.cast(tf.reshape(weight, [1, -1, output_seq]), mask_dtype)
            mlm_ones = tf.ones(shape=[total_seq, 1, 1], dtype=mask_dtype)
            mlm_mask = mlm_ones * mlm_mask
            mlm_mask = tf.transpose(mlm_mask, [1,0,2])
            # mlm_mask shuld be [batch, sequence, sequence]
            mask = tf.cast((lm_mask + mlm_mask) > 0, mask_dtype)
            # mask shuld be [batch, sequence, sequence]
            mask = mtf.import_tf_tensor(mesh, mask, [dim_batch, dim_input_contexts, dim_output_contexts])
            return mask

        with tf.variable_scope('model'):
            wpe = mtf.get_variable(mesh, 'wpe', mtf.Shape([dim_contexts, dim_hidden]),
                                 initializer=tf.random_normal_initializer(stddev=0.01), dtype=tf.float32)
            wte = mtf.get_variable(mesh, 'wte', mtf.Shape([dim_vocabulary, dim_hidden]),
                                 initializer=tf.random_normal_initializer(stddev=0.02), dtype=tf.float32)

            x = mtf.gather(wte, x, dim_vocabulary)

            if pasts is None:
                pasts = [None] * num_layers
                start = 0
                pos = mtf.import_tf_tensor(mesh, tf.range(num_input_contexts), [dim_input_contexts])
            else:
                pasts = mtf.unstack(pasts, dim_layers)
                pos = tf.range(num_input_contexts) + num_pasts_contexts
                pos = mtf.import_tf_tensor(mesh, tf.clip_by_value(pos, num_pasts_contexts, num_contexts-1), [dim_input_contexts])

            x += mtf.gather(wpe, pos, dim_contexts)

            x = mtf.cast(x, dense_dtype)

            # Transformer
            presents = []
            assert len(pasts) == num_layers

            atten_mask = make_attention_mask(num_input_contexts, num_output_contexts, num_precontext)

            extra_loss = []
            extra_vector = []
            for layer, past in enumerate(pasts):
                h, present = multiheadattention(x, 'att%d'%layer, num_header, mask=atten_mask, past=past)
                x = normalization(h, 'an%d'%layer) + x
                if use_moe:
                    h, exloss = moe(x, 'moe%d'%layer)
                    extra_loss.append(exloss)
                else:
                    h = mlp(x, 'mlp%d'%layer)
                x = normalization(h, 'ln%d'%layer) + x
                presents.append(present)
                if (layer+1+num_layers%4) % (num_layers // 4) == 0:
                    if pos_vector is not None:
                        extra_vector.append(mtf.gather(x, pos_vector, dim_input_contexts))

            extra_loss = None if len(extra_loss) == 0 else mtf.cast(mtf.add_n(extra_loss), tf.float32)
            present = mtf.stack(presents, "memory_layers", 1)
            present = mtf.replace_dimensions(present, present.shape.dims[1], dim_layers)

            wob = mtf.get_variable(mesh, 'wob', mtf.Shape([dim_vocabulary]),
                                 initializer=tf.zeros_initializer(), dtype=tf.float32)
            logits = mtf.layers.dense(x,
                    reduced_dims=[dim_hidden],
                    new_dims=[dim_hidden],
                    activation=mtf.swish,
                    kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                    use_bias=True,
                    variable_dtype=variable_dtype)
            logits = mtf.cast(logits, tf.float32)
            logits = mtf.einsum([logits, wte], reduced_dims=[dim_hidden]) + wob

            if pos_vector is not None:
                vector = mtf.concat(extra_vector, 'hidden')
                vector = mtf.reshape(vector, [dim_batch, dim_vector])
            else:
                vector = None

            results = {}
            results['extra_loss'] = extra_loss
            results['output'] = x
            results['present'] = present
            results['logits'] = logits
            results['vector'] = vector
            results['graph'] = graph
            results['mesh'] = mesh

            return results

    if is_training:
        return model_fn, train_fn
    else:
        return model_fn, run_fn
