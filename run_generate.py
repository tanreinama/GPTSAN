import json
import glob
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import numpy as np
import tensorflow.compat.v1 as tf
import modeling
import time
from encode_swe import SWEEncoder_ja as Encoder
import modeling

tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()
tf.enable_eager_execution()

tf.flags.DEFINE_string("model", help="model folder", default="checkpoints" )
tf.flags.DEFINE_string('context', help="input context", default='')
tf.flags.DEFINE_string('vocabulary', help="vocabulary file", default='ja-swe36k.txt')
tf.flags.DEFINE_integer('top_k', help="top_k selection", default=40)
tf.flags.DEFINE_integer('max_generate', help="max generate token num", default=300)
tf.flags.DEFINE_integer('num_generate', help="generate sentence num", default=1)
tf.flags.DEFINE_string('tpu_nodes', help="tpu nodes", default='')
args = tf.flags.FLAGS

def main():
    assert os.path.isdir(args.model), f'model not found; {args.model}'
    assert os.path.isfile(os.path.join(args.model,'parameters.json')), f'parameter file not found in {args.model}'
    assert os.path.isfile(os.path.join(args.model,'checkpoint')), f'checkpoint not found in {args.model}'
    assert os.path.isfile(args.vocabulary), f'vocabulary file not found; {args.vocabulary}'
    assert args.top_k>=0, 'invalid top_k parameter'

    if args.tpu_nodes=='':
        print('TPU node not foud. Using GPU device.')

    with open(os.path.join(args.model,'parameters.json'), encoding='utf-8') as f:
        saved_params = json.loads(f.read())
    with open(args.vocabulary, encoding='utf-8') as f:
        bpe = f.read().split('\n')
    with open('emoji.json', encoding='utf-8') as f:
        emoji = json.loads(f.read())
    enc = Encoder(bpe, emoji)

    NUM_CTX = saved_params['model_params']['num_contexts']
    MODE = saved_params['model_params']['train_mode']
    NUM_TOKENS = len(bpe)
    SOT_TOKEN = NUM_TOKENS-7
    MSK_TOKEN = NUM_TOKENS-6
    SEP_TOKEN = NUM_TOKENS-5
    NOT_TOKEN = NUM_TOKENS-4
    BAG_TOKEN = NUM_TOKENS-3
    SEG_TOKEN = NUM_TOKENS-2
    EOT_TOKEN = NUM_TOKENS-1

    assert MODE in ["lm","hybrid"], "invalid mode"

    DOT_TOKENS = enc.encode("。｡．.？！?!:：;；")
    NL_TOKEN = enc.encode("\n")[0]
    LAST_TOKEN = enc.encode("<|byte0|>")[0]-1
    TOKEN_IS_DOT_NL = [(t in DOT_TOKENS or t==NL_TOKEN) for t in range(NUM_TOKENS)]

    if MODE=="lm":
        pre_input = [SOT_TOKEN,SEG_TOKEN] + enc.encode(args.context)
    else:
        pre_input = [SOT_TOKEN] + enc.encode(args.context) + [SEG_TOKEN]

    generated = []

    if args.tpu_nodes != "":
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(args.tpu_nodes)
        tf.config.experimental_connect_to_cluster(tpu)
        topology = tf.tpu.experimental.initialize_tpu_system(tpu)
    else:
        tpu = None
        topology = None
        saved_params['model_params']['num_pallarelizm'] = min(saved_params['model_params']['num_pallarelizm'],
                                                              len(tf.config.experimental.list_physical_devices('GPU')))

    def model_fn(features, labels, mode, params):
        x = features["x"]
        num_precontext = features["num_precontext"]
        model, run = modeling.model(tpu, params, saved_params, False, False, False)
        return run(model(x=x, num_precontext=num_precontext))

    def input_fn():
        def input_gen():
            while True:
                if len(generated) > 0 and (generated[-1] == EOT_TOKEN or len(generated) >= args.max_generate):
                    raise StopIteration
                input_tokens = pre_input+generated
                input_tokens = input_tokens[-NUM_CTX:]
                pre_length = max(len(input_tokens)-len(generated), 0)
                print(f'{len(generated)} token generated...', end='\r')
                yield {"x":[input_tokens+[EOT_TOKEN]*(NUM_CTX-len(input_tokens))],
                       "num_precontext":[[pre_length if MODE=="hybrid" else 0]]}, [0]
        dataset = tf.data.Dataset.from_generator(input_gen,
                                                 output_types=({"x":tf.int32,
                                                                "num_precontext":tf.int32},tf.int32),
                                                 output_shapes=({"x":[1,NUM_CTX],
                                                                 "num_precontext":[1,1]},1))
        return dataset

    def select_fn(result):
        output_pos = min(len(pre_input)+len(generated)-1, NUM_CTX-1)
        logits = result['logits'][output_pos]
        out = np.argmax(logits)
        if out == SEP_TOKEN:
            logits = [(logits[l] if TOKEN_IS_DOT_NL[l] else -1e10) for l in range(NUM_TOKENS)]
        if out != NOT_TOKEN:
            ind = np.arange(NUM_TOKENS)
            log = np.array(logits)
            log[NOT_TOKEN] = -1e10
            log = np.exp(log - np.max(log)) / np.sum(np.exp(log)) # softmax
            k = np.sort(log)[-args.top_k]
            log[np.where(log < k)] = 1e-10
            out = np.random.choice(ind, 1, p=log/np.sum(log))[0]
        else:
            out = SEP_TOKEN
        return int(out)

    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=args.model)

    for num in range(args.num_generate):
        generated = []
        for result in estimator.predict(input_fn=input_fn):
            out = select_fn(result)
            generated.append(out)
        generated_nobag = []
        for token in generated:
            if token == BAG_TOKEN:
                if len(generated_nobag) > 0:
                    bagged = generated_nobag[-1]
                    generated_nobag.append(bagged)
                    generated_nobag.append(bagged)
            elif token < LAST_TOKEN:
                generated_nobag.append(token)

        if num==0:
            print("====[start generate]====")
        print(enc.decode(generated_nobag))
        if num == args.num_generate-1:
            print("====[end generate]====")
        else:
            print("====[next generate]====")


if __name__ == "__main__":
    main()
