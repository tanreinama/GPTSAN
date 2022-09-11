import json
import glob
import os
import copy

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
tf.flags.DEFINE_string('vocabulary', help="vocabulary file", default='ja-swe36k.txt')
tf.flags.DEFINE_integer('top_k', help="top_k selection", default=120)
tf.flags.DEFINE_integer('beam_width', help="beam search width", default=4)
tf.flags.DEFINE_integer('max_generate', help="max generate token num", default=300)
tf.flags.DEFINE_integer('num_generate', help="generate sentence num", default=1)
tf.flags.DEFINE_string('tpu_nodes', help="tpu nodes", default='')
tf.flags.DEFINE_string('spout', help="spout data (none or uniform or onhot class num or sqfile)", default='none')
args = tf.flags.FLAGS

def main():
    # 引数チェック
    assert os.path.isdir(args.model), f'model not found; {args.model}'
    assert os.path.isfile(os.path.join(args.model,'parameters.json')), f'parameter file not found in {args.model}'
    assert os.path.isfile(os.path.join(args.model,'checkpoint')), f'checkpoint not found in {args.model}'
    assert os.path.isfile(args.vocabulary), f'vocabulary file not found; {args.vocabulary}'
    assert args.top_k>0, 'invalid top_k parameter'

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

    # SEP_TOKENが文章の区切りを意味するので、出力時に変換するための変数
    DOT_TOKENS = enc.encode("。｡．.？！?!:：;；")
    NL_TOKEN = enc.encode("\n")[0]
    LAST_TOKEN = enc.encode("<|byte0|>")[0]-1
    TOKEN_IS_DOT_NL = [(t in DOT_TOKENS or t==NL_TOKEN) for t in range(NUM_TOKENS)]

    # 最大生成トークン列数
    MAX_GENERATE = args.max_generate
    # ビームサーチは1バッチ分でビームを生成する
    BATCH_SIZE = 1 if args.beam_width<=0 else args.beam_width

    # 続きを生成する場合の、直前の文章
    pre_input = [SOT_TOKEN,SEG_TOKEN] + enc.encode(args.context) # lm、hybridならSOT+context+SEG
    connected_inputs = 0 # hybridで入力するトークン列数

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

    # spout
    spout_dim = saved_params["model_params"]["num_spout"]
    if args.spout == "none" or args.spout == "":
        spout_data = None
    elif args.spout == "uniform":
        spout_data = np.random.uniform(size=[BATCH_SIZE, spout_dim])
    elif args.spout == "zeros":
        spout_data = np.zeros([BATCH_SIZE, spout_dim])
    else:
        try:
            vclass = int(args.spout)
            spout_data = np.zeros([BATCH_SIZE, spout_dim])
            spout_data[:,vclass] = 1
        except:
            assert False, f"unknown spout value {args.spout}"

    # Mesh-tensorflowを使うので、Estimatorでモデルを読み込むので、その関数
    def model_fn(features, labels, mode, params):
        nonlocal spout_data
        pasts = None
        x = features["x"]
        num_precontext = features["num_precontext"]
        model, run = modeling.model(tpu, params, saved_params, False, False, False)
        spout = tf.constant(spout_data, dtype=tf.float32) if spout_data is not None else None
        return run(model(x=x, num_precontext=num_precontext, pasts=pasts, spout=spout))

    # Mesh-tensorflowを使うので、Estimatorでデータを読み込むので、その関数
    def input_fn(params):
        nonlocal generated_all, generated_scores
        input_size = min(MAX_GENERATE+len(pre_input), NUM_CTX) # Transformerへ入力する長さ
        def input_gen(): # モデルへの入力を一つずつ返す
            while True:
                endednum = 0 # 全ビームで終了かチェック
                for generated in generated_all[:BATCH_SIZE]:
                    if len(generated) > 0 and (generated[-1] == EOT_TOKEN or len(generated) >= MAX_GENERATE):
                        endednum += 1 # EOTなら終了
                if endednum == BATCH_SIZE:
                    return # 全ビームで終了なら生成終わり
                gen_x, gen_num = [], []
                for generated in generated_all[:BATCH_SIZE]:
                    input_tokens = pre_input+generated # 一つ前までの生成文を入力し、次のトークンを得る
                    input_tokens = input_tokens[-input_size:] # モデルの最大入力数まで
                    nocon_length = (len(pre_input) - connected_inputs) + len(generated)
                    con_length = connected_inputs - max(nocon_length - (input_size-connected_inputs), 0) # hybridで入力するトークン列数
                    gen_x.append(input_tokens+[EOT_TOKEN]*(input_size-len(input_tokens)))
                    gen_num.append([0])
                print(f'{len(generated_all[0])} token generated...', end='\r')
                # モデルへの次の入力＝一つ前までの生成文、マルチモーダル用ベクトル入力、Hybridの部分の長さ
                yield {"x":gen_x,
                       "num_precontext":gen_num}, [0]
        output_type = {"x":tf.int32,
                       "num_precontext":tf.int32}
        output_shape = {"x":[BATCH_SIZE,input_size],
                        "num_precontext":[BATCH_SIZE,1]}
        dataset = tf.data.Dataset.from_generator(input_gen,
                                                 output_types=(output_type,tf.int32),
                                                 output_shapes=(output_shape,1))
        return dataset

    # モデルの出力からトークンを選択する関数
    def select_fn(result, batch_dim):
        nonlocal generated_all, generated_scores
        input_size = min(MAX_GENERATE+len(pre_input), NUM_CTX) # Transformerへ入力する長さ
        # 一つ前までの生成文を入れて、1つ多いトークンが出てくるので、出てきた場所を計算
        output_pos = min(len(pre_input)+len(generated_all[batch_dim])-1, input_size-1)
        logits = result['logits'][output_pos] # 新しく出てきたトークン1つ分
        out = np.argmax(logits)
        #if out == SEP_TOKEN: # SEP_TOKENは文章の区切り文字に変換
        #    logits = [(logits[l] if TOKEN_IS_DOT_NL[l] else -1e10) for l in range(NUM_TOKENS)]
        if out != EOT_TOKEN: # TOP_Kロジックで選択
            ind = np.arange(NUM_TOKENS)
            log = np.array(logits)
            #log = (log - np.max(log)) / (np.max(log)-np.min(log))
            log[NOT_TOKEN] = -1e10
            log[SEP_TOKEN] = -1e10
            exp = np.exp(log)
            log = exp / np.sum(exp) # softmax
            k = np.sort(log)[-args.top_k]
            p = copy.copy(log)
            log[np.where(log < k)] = 1e-10
            out = np.random.choice(ind, 1, p=log/np.sum(log))[0]
            rank = np.sum(log > log[out])
        else: # NOT_TOKENは無視するトークン
            rank = 0
        generated_all[batch_dim].append(int(out))
        generated_scores[batch_dim].append(logits[int(out)]) # 生成文のスコア
        generated_ranks[batch_dim].append(rank) # 生成文のスコア

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
    for num in range(args.num_generate):
        # 生成するトークン列の数だけ繰り返しモデルを実行
        generated_all = [[] for _ in range(BATCH_SIZE)]
        generated_scores = [[] for _ in range(BATCH_SIZE)]
        generated_ranks = [[] for _ in range(BATCH_SIZE)]
        for pos, result in enumerate(estimator.predict(input_fn=input_fn)):
            select_fn(result, pos%BATCH_SIZE) # 新しく増えたトークンを選択
            if (pos+1)%BATCH_SIZE == 0: # バッチ終了時にビームを評価
                beam_scores = [(np.mean(generated_scores[s]) if EOT_TOKEN != generated_all[s][-1] else -1e10) for s in range(BATCH_SIZE)]
                best_beam = np.argmax(beam_scores) # この時点で終了しておらず最も良かった生成文
                if beam_scores[best_beam] != -1e10:
                    for batch_dim in range(BATCH_SIZE): # 1バッチ生成時のビームの内容
                        if EOT_TOKEN == generated_all[batch_dim][-1]: # 終了したらFixして保存しておく
                            fixed_beam = copy.copy(generated_all[batch_dim])
                            fixed_score= copy.copy(generated_scores[batch_dim])
                            fixed_rank= copy.copy(generated_ranks[batch_dim])
                            generated_all.append(fixed_beam) # BATCH_SIZE以上の次元にあるデータはFixした生成文
                            generated_scores.append(fixed_score) # BATCH_SIZE以上の次元にあるデータはFixした生成文
                            generated_ranks.append(fixed_rank) # BATCH_SIZE以上の次元にあるデータはFixした生成文
                            generated_all[batch_dim] = copy.copy(generated_all[best_beam]) # 空いたバッチで終了していないのの続きを試す
                            generated_scores[batch_dim] = copy.copy(generated_scores[best_beam]) # 空いたバッチで終了していないのの続きを試す
                            generated_ranks[batch_dim] = copy.copy(generated_ranks[best_beam]) # 空いたバッチで終了していないのの続きを試す

        # 最も良かったビーム内のトークン列を取得
        last_scores = []
        for scores, generated, rank in zip(generated_scores, generated_all, generated_ranks):
            if EOT_TOKEN in generated:
                endpos = generated.index(EOT_TOKEN)
                scores = scores[:endpos]
                generated = generated[:endpos]
                rank = rank[:endpos]
            # 最も良かった場所から取得したトークンのスコアから外れ生成を判定
            cs = [s for s,g,r in zip(scores,generated,rank) if g<LAST_TOKEN and r==0]
            cs = cs if len(cs)> 0 else [-1e10]
            ss = scores if len(scores)> 0 else [-1e10]
            last_scores.append(-1e10 if np.mean(cs)>0 else np.median(ss))

        # 生成文を選択
        for generated in generated_all:
            # 特殊トークンの処理
            generated_nobag = []
            for token in generated:
                if token == BAG_TOKEN: # BAG_TOKENは直前のトークンの繰り返し
                    if len(generated_nobag) > 0: # 個数の指定は無いのでとりあえず3個
                        bagged = generated_nobag[-1]
                        generated_nobag.append(bagged)
                        generated_nobag.append(bagged)
                elif token < LAST_TOKEN: # 元NOT_TOKEN等無視するトークンは入れない
                    generated_nobag.append(token)

            # 結果を表示
            if num==0:
                print("\033[32m====[start generate]====\033[0m")
            print(enc.decode(generated_nobag))
            if num == args.num_generate-1:
                print("\033[32m====[end generate]====\033[0m")
            else:
                print("\033[32m====[next generate]====\033[0m")

if __name__ == "__main__":
    main()
