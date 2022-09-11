import json
import glob
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import numpy as np
import tensorflow
import tensorflow.compat.v1 as tf
import modeling
import time
from encode_swe import SWEEncoder_ja as Encoder
import modeling

# Mesh-tensorflow用の設定
tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()
tf.enable_eager_execution()

# プログラム引数
tf.flags.DEFINE_string("model", help="model folder", default="gptsan-backbone-2.8B" )
tf.flags.DEFINE_string('context', help="input context", default='')
tf.flags.DEFINE_string('mask', help="mask sentence to fill context", default='[MASK]')
tf.flags.DEFINE_integer('pos_vector', help="token position to pull internal vector", default=-1)
tf.flags.DEFINE_string('output', help="output json file", default='')
tf.flags.DEFINE_string('vocabulary', help="vocabulary file", default='ja-swe36k.txt')
tf.flags.DEFINE_string('tpu_nodes', help="tpu nodes", default='')
args = tf.flags.FLAGS

def main():
    # 引数チェック
    assert os.path.isdir(args.model), f'model not found; {args.model}'
    assert os.path.isfile(os.path.join(args.model,'parameters.json')), f'parameter file not found in {args.model}'
    assert os.path.isfile(os.path.join(args.model,'checkpoint')), f'checkpoint not found in {args.model}'
    assert os.path.isfile(args.vocabulary), f'vocabulary file not found; {args.vocabulary}'
    assert args.output=='' or not os.path.isfile(args.output), f'file is exists in {args.output}'

    if args.tpu_nodes=='':
        print('TPU node not foud. Using GPU device.')

    # テキストエンコーダー作成
    with open(os.path.join(args.model,'parameters.json'), encoding='utf-8') as f:
        saved_params = json.loads(f.read())
    with open(args.vocabulary, encoding='utf-8') as f:
        bpe = f.read().split('\n')
    with open('emoji.json', encoding='utf-8') as f:
        emoji = json.loads(f.read())
    enc = Encoder(bpe, emoji)

    # モデル設定
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
    # マルチモーダル（画像など→テキスト）用のベクトル入力パラメーター
    TOTAL_LAYERS = saved_params['model_params']['num_switch_layers']+saved_params['model_params']['num_ext_layers']
    NUM_HEADERS = saved_params['model_params']['num_header']
    NUM_CHANNELS = saved_params['model_params']['num_hidden'] // NUM_HEADERS
    EXT_INPUTS = TOTAL_LAYERS*NUM_HEADERS*NUM_CHANNELS*2

    # モデル実行モード
    assert MODE in ["lm","hybrid"], "invalid mode"

    # SEP_TOKENが文章の区切りを意味するので、出力時に変換するための変数
    DOT_TOKENS = enc.encode("。｡．.？！?!:：;；")
    NL_TOKEN = enc.encode("\n")[0]
    LAST_TOKEN = enc.encode("<|byte0|>")[0]-1
    TOKEN_IS_DOT_NL = [(t in DOT_TOKENS or t==NL_TOKEN) for t in range(NUM_TOKENS)]

    # 続きを生成する場合の、直前の文章
    if MODE=="lm":
        pre_input = [SOT_TOKEN] + enc.encode(args.context)
        connected_inputs = 0 # hybridで入力するトークン列数
    else:
        if args.mask in args.context:
            pre_inps = args.context.split(args.mask)
            inp_token = [enc.encode(inp)+[MSK_TOKEN] for inp in pre_inps]
            inp_token = sum(inp_token,[])[:-1]
            pre_input = [SOT_TOKEN] + inp_token
            connected_inputs = len(inp_token) # hybridで入力するトークン列数
        else:
            inp_token = enc.encode(args.context)
            pre_input = [SOT_TOKEN] + inp_token
            connected_inputs = len(inp_token) # hybridで入力するトークン列数

    # 実行環境設定
    if args.tpu_nodes != "":
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(args.tpu_nodes)
        tf.config.experimental_connect_to_cluster(tpu)
        topology = tf.tpu.experimental.initialize_tpu_system(tpu)
    else:
        tpu = None
        topology = None
        saved_params['model_params']['num_pallarelizm'] = min(saved_params['model_params']['num_pallarelizm'],
                                                              len(tf.config.experimental.list_physical_devices('GPU')))

    # Mesh-tensorflowを使うので、Estimatorでモデルを読み込むので、その関数
    def model_fn(features, labels, mode, params):
        x = features["x"]
        pos_vector = features["pos_vector"]
        num_precontext = features["num_precontext"]
        model, run = modeling.model(tpu, params, saved_params, False, False, False)
        return run(model(x=x, num_precontext=num_precontext, pos_vector=pos_vector))

    # Mesh-tensorflowを使うので、Estimatorでデータを読み込むので、その関数
    def input_fn(params):
        input_size = min(len(pre_input), NUM_CTX) # Transformerへ入力する長さ
        def input_gen(): # モデルへの入力を一つずつ返す
            input_tokens = pre_input[:input_size] # モデルの最大入力数まで
            pos_vector = args.pos_vector if args.pos_vector>=0 else len(input_tokens)-1 # 内部のデータを取り出す位置
            yield {"x":[input_tokens+[EOT_TOKEN]*(input_size-len(input_tokens))],
                   "pos_vector":[[pos_vector]],
                   "num_precontext":[[connected_inputs]]}, [0]
        output_type = {"x":tf.int32,
                       "pos_vector":tf.int32,
                       "num_precontext":tf.int32}
        output_shape = {"x":[1,input_size],
                        "pos_vector":[1,1],
                        "num_precontext":[1,1]}
        dataset = tf.data.Dataset.from_generator(input_gen,
                                                 output_types=(output_type,tf.int32),
                                                 output_shapes=(output_shape,1))
        return dataset

    # モデルの実行を定義
    if tpu is not None:
        run_config = tf.estimator.tpu.RunConfig(
              cluster=tpu,
              master=None,
              model_dir=args.model,
              tpu_config=tf.estimator.tpu.TPUConfig(
                  iterations_per_loop=1,
                  num_cores_per_replica=1,
                  per_host_input_for_training=tf.estimator.tpu.InputPipelineConfig.BROADCAST))
        estimator = tf.estimator.tpu.TPUEstimator(use_tpu=True, model_fn=model_fn, config=run_config, train_batch_size=1, predict_batch_size=1)
    else:
        estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=args.model)

    # 文章生成を実行
    result = list(estimator.predict(input_fn=input_fn))
    pred_token, pred_score = [], []
    for i in range(len(pre_input)-1):
        p = int(result[0]['logits'][i].argmax()) if pre_input[i+1]==MSK_TOKEN else pre_input[i+1]
        s = float(result[0]['logits'][i][pre_input[i+1]])
        pred_token.append(p)
        pred_score.append(s)
    print("{OUTPUT TEXTS}")
    print(enc.decode(pred_token))
    print("{OUTPUT TOKENS}")
    print(pred_token)
    print("{OUTPUT SCORES}")
    print(pred_score)
    print("{OUTPUT VECTOR SHAPE}")
    print(result[0]['vector'].shape)

    if args.output!='':
        with open(args.output, "w") as wf:
            wf.write(json.dumps({"output_text":enc.decode(pred_token),
                                 "output_tokens":pred_token,
                                 "output_scores":pred_score,
                                 "output_vector":result[0]['vector'].astype(float).tolist(),
                                 "input_text":args.context,
                                 "input_tokens":pre_input,
                                }))


if __name__ == "__main__":
    main()
