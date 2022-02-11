import functools
import re
import mesh_tensorflow as mtf
import mesh_tensorflow.optimize as mtf_optimize
from mesh_tensorflow import layers
from mesh_tensorflow import ops_with_redefined_builtins as mtf_lib
import tensorflow.compat.v1 as tf

class AdamWeightDecayOptimizer(object):

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None):
        """Constructs a AdamWeightDecayOptimizer."""

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_grads(self, grads, variables, grad_scale=1.0):
        ops = []
        for grad, var in zip(grads, variables):
            ops.extend(self.apply_grad(grad, var, grad_scale))
        if not ops:
            return ops
        return variables[0].graph.combine_assignments(ops)

    def apply_grad(self, grad, var, grad_scale=1.0):
        if grad is None:
            tf.logging.warning("Gradient is None for variable %s" % var.name)
            return []

        grad = mtf.cast(grad, tf.float32) / grad_scale

        assignments = []

        m = mtf_lib.get_variable(
                var.mesh, var.name + "/adam_m", var.shape,
                dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=False)

        v = mtf_lib.get_variable(
                var.mesh, var.name + "/adam_v", var.shape,
                dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=False)

        # Standard Adam update.
        next_m = self.beta_1 * m + (1.0 - self.beta_1) * grad
        next_v = self.beta_2 * v + (1.0 - self.beta_2) * mtf_lib.square(grad)

        update = next_m / (mtf_lib.sqrt(next_v) + self.epsilon)

        if self._do_use_weight_decay(var.name):
            update += self.weight_decay_rate * mtf.cast(var.value, tf.float32)

        next_sub = self.learning_rate * update

        assignments.extend(
                [mtf_lib.assign_sub(var, next_sub),
                 mtf_lib.assign(m, next_m),
                 mtf_lib.assign(v, next_v)])
        return assignments

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

def clip_by_global_norm(grads, clip_norm):
    """Clip the grads by global norm."""
    global_norm = mtf.sqrt(
            mtf.add_n([mtf.reduce_sum(mtf.square(mtf.cast(t, tf.float32))) for t in grads if t is not None
                                ]))
    multiplier = clip_norm / mtf.maximum(global_norm, clip_norm)
    clipped_grads = [None if t is None else mtf.cast(mtf.cast(t, tf.float32) * multiplier, t.dtype) for t in grads]
    return clipped_grads, global_norm

def create_optimizer(loss, base_lr, num_warmup_steps,
                     max_optimized_variable_size=None,
                     optimizer="adam",
                     grad_scale=1.0,
                     clip_gradients=True):
    """Creates an optimizer training op."""
    global_steps = tf.train.get_or_create_global_step()
    mesh = loss.mesh

    # “inverse square root” learning rate schedule start with base_lr; https://arxiv.org/abs/1910.10683
    # note: if use small batch size, base_lr needs to be small
    global_steps_float = tf.cast(global_steps, tf.float32)
    decay_steps = tf.constant((1/base_lr)**2, dtype=tf.float32)
    decay_steps_float = tf.math.maximum(decay_steps, global_steps_float)
    learning_rate = 1.0 / tf.math.sqrt(decay_steps_float)

    # Linear warm-up equivalent to RADAM; https://arxiv.org/abs/1908.03265
    if num_warmup_steps:
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int64)
        warmup_steps_float = tf.constant(num_warmup_steps, dtype=tf.float32)
        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = base_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps < warmup_steps_int, tf.float32)
        learning_rate = ((1.0 - is_warmup) * learning_rate +
                       is_warmup * warmup_learning_rate)

    mtf_learning_rate = mtf.import_tf_tensor(mesh, learning_rate, [])

    if optimizer == "adam":
        optimizer = AdamWeightDecayOptimizer(
                learning_rate=mtf_learning_rate,
                weight_decay_rate=0.01,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8,
                exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
    else:
        raise ValueError("unknown optimizer")

    trainable_variables = mesh.graph.trainable_variables
    if max_optimized_variable_size:
        trainable_variables = [t for t in trainable_variables
                                     if t.shape.size <= max_optimized_variable_size]

    var_grads = mtf.gradients(
            [loss*grad_scale], [v.outputs[0] for v in trainable_variables])

    # This is how the model was pre-trained.
    if clip_gradients:
        (var_grads, _) = clip_by_global_norm(
                var_grads, clip_norm=mtf.constant(mesh, 1.0, dtype=tf.float32))

    update_ops = optimizer.apply_grads(var_grads, trainable_variables, grad_scale)

    return learning_rate, update_ops
