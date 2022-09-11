import numpy as np
import pandas as pd
import re
import argparse
import shutil
import os
import json
import pickle
import uuid
import glob
import copy
from multiprocessing import Pool
import tensorflow as tf

from encode_swe import SWEEncoder_ja as Encoder

parser = argparse.ArgumentParser()
parser.add_argument("--input_files", help="input text files (like 'data/*.txt' or csv or list file)", default="" )
parser.add_argument("--mode", help="training mode ('lm' for language model, 'mlm' for masked language model, 'hybrid' for hybrid model)", default="lm" )
parser.add_argument('--no_offset_mlm', help="no offset output for masked language model", action='store_true')
parser.add_argument("--vocabulary", help="vocabulary file", default="ja-swe36k.txt" )
parser.add_argument("--num_process", help="process num", type=int, default=8 )
parser.add_argument("--num_context", help="context token length", type=int, default=1280 )
parser.add_argument("--num_mask", help="mask insertion rate for masked language model", type=float, default=0.08 )
parser.add_argument("--max_mask", help="max mask insertion num", type=int, default=100 )
parser.add_argument('--use_nextsent', help="use nextsentence training for masked language model", action='store_true')
parser.add_argument("--hybrid_rate", help="train hybrid mode vs lm mode in mode='hybrid'", type=float, default=1.0 )
args = parser.parse_args()

with open(args.vocabulary, encoding='utf-8') as f:
    bpe = f.read().split('\n')
with open('emoji.json', encoding='utf-8') as f:
    emoji = json.loads(f.read())
enc = Encoder(bpe, emoji)

NUM_CTX = args.num_context
NUM_TOKENS = len(bpe)
SOT_TOKEN = NUM_TOKENS-7
MSK_TOKEN = NUM_TOKENS-6
SEP_TOKEN = NUM_TOKENS-5
NOT_TOKEN = NUM_TOKENS-4
BAG_TOKEN = NUM_TOKENS-3
SEG_TOKEN = NUM_TOKENS-2
EOT_TOKEN = NUM_TOKENS-1

DOT_TOKENS = enc.encode("。｡．.？！?!:：;；")
NL_TOKEN = enc.encode("\n")[0]
LAST_TOKEN = enc.encode("<|byte0|>")[0]-1
TOKEN_IS_DOT = [(t in DOT_TOKENS) for t in range(NUM_TOKENS)]

KIGOU_TOKEN = enc.encode("〜×÷¥！＂＃＄％＆＇（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀>｛｜｝～!\"#$%&'()*+,-./:;<=?@[\\]^_`{|}~、。・‥…「」『』【】〔〕〒○●◎→←↑↓〃△▲▽▼□■◇◆☆★♂♀§〓♭♪†¶<BR><SP><TAB><URL><EMAIL><TEL><DATE><PRICE><BLOCK><KIGOU><U2000U2BFF><|emoji1|><|emoji2|><|emoji3|><|emoji4|><|emoji5|><|emoji6|><|emoji7|><|emoji8|><|emoji9|><|emoji10|><|emoji11|><|emoji12|>")
TOKEN_IS_KIGOU = [(t in KIGOU_TOKEN) for t in range(NUM_TOKENS)]

assert args.mode in ["lm","mlm","hybrid"], "invalid mode"
assert args.input_files != "", "empty input files"

if os.path.isfile(args.input_files):
    head = open(args.input_files).readline().strip()
    if head=="filename,fileid":
        df = pd.read_csv(args.input_files)
        array_file = list(df.filename)
        file_id = {k:int(v) for k,v in zip(df.filename,df.fileid)}
    elif head.split()==2 and os.path.isfile(d.split()[0]):
        df = open(args.input_files).readlines()
        df = [d.split() for d in df]
        array_file = [d[0] for d in df]
        file_id = {d[0]:int(d[1].strip()) for d in df}
    elif os.path.isfile(head):
        df = open(args.input_files).readlines()
        array_file = [d.split() for d in df]
        file_id = None
    else:
        assert False, "valid input list file"
else:
    array_file = glob.glob(args.input_files)
    file_id = None
assert len(array_file) > 0, "input file number is 0"

np.random.shuffle(array_file)

text_indexs = np.array_split(np.arange(len(array_file)), args.num_process)
text_files = [[array_file[i] for i in p] for p in text_indexs]

def _append(i, tokens, target, nextsent, num_input, nextsent_target):
    tokens, target, nextsent = copy.copy(tokens), copy.copy(target), copy.copy(nextsent)

    lm_targetlen = len(target)
    tokens = [SOT_TOKEN] + nextsent + tokens
    end_input = 1 + len(nextsent) + num_input
    start_input = 1 + len(nextsent)
    target = [-1] * end_input + target + [-1]

    assert len(tokens)==len(target), "token length mismatch"
    assert len(tokens) - num_input - len(nextsent) == lm_targetlen + 2, "target length mismatch"
    assert len(tokens) - end_input == lm_targetlen + 1, "end_input mismatch"

    target.append(EOT_TOKEN)
    tokens.append(EOT_TOKEN)

    tokens = tokens[:NUM_CTX]
    target = target[:NUM_CTX]

    num_context = len(tokens)

    while len(tokens) < NUM_CTX:
        target.append(-1)
        tokens.append(NOT_TOKEN)

    assert end_input < NUM_CTX, "num_input is oversize"
    assert len(nextsent)==0 or nextsent[-1] == SEP_TOKEN, "nextsent miss"
    assert end_input==1 or tokens[end_input] == SEG_TOKEN, "segmentation miss"

    num_mask = min(int(args.num_mask * num_input), args.max_mask)
    posisions = np.random.permutation(num_input) + start_input

    if len(nextsent) > 0:
        pos = start_input - 1
        num_mask = max(num_mask-1, 0)
        target[pos-1] = SEP_TOKEN if nextsent_target==1 else EOT_TOKEN
        assert tokens[pos] == SEP_TOKEN, "nextsent miss"

    for p in range(num_mask):
        pos = posisions[p]
        if tokens[pos] < SOT_TOKEN:
            target[pos-1] = tokens[pos]
            if np.random.random() < 0.8:
                tokens[pos] = MSK_TOKEN
            else:
                if np.random.random() < 0.5:
                    tokens[pos] = np.random.randint(LAST_TOKEN)

    if args.mode == 'mlm' and args.no_offset_mlm:
        tokens = [SOT_TOKEN] + tokens[2:] + [EOT_TOKEN]

    features = {
        "x":tf.train.Feature(int64_list=tf.train.Int64List(value=tokens)),
        "y":tf.train.Feature(int64_list=tf.train.Int64List(value=target)),
        "i":tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
        "num_input":tf.train.Feature(int64_list=tf.train.Int64List(value=[end_input]))
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()

def _target(token):
    if len(token) < 3:
        return [(SEP_TOKEN if TOKEN_IS_DOT[e] else e) for e in token], copy.copy(token)

    bags = []
    for i in range(len(token)-2):
        if token[i]<=LAST_TOKEN and token[i]==token[i+1] and token[i+1]==token[i+2]:
            bags.append(i)
    kigou = [-1] * len(token)
    for i in bags:
        kigou[i] = token[i]
        kigou[i+1] = token[i+1]
        kigou[i+2] = token[i+2]

    bef = -1
    encoded_token, target_token = [], []
    for i in range(len(token)):
        if bef!=kigou[i] or bef==-1:
            if kigou[i] != -1:
                encoded_token.append(token[i])
                encoded_token.append(BAG_TOKEN)
                target_token.append(token[i])
                target_token.append(BAG_TOKEN)
            else:
                encoded_token.append(token[i])
                target_token.append(token[i])
        bef = kigou[i]

    encoded_token = [(e if not TOKEN_IS_DOT[e] else SEP_TOKEN) for e in encoded_token]
    return encoded_token, target_token

def _proc(i):
    np.random.seed(i)
    filep = [None for f in text_files[i]]
    filename = [f for f in text_files[i]]
    fileend = []
    filesize = [f[2] for f in text_files[i]]
    num_write_data = 0
    with tf.io.TFRecordWriter("%s_%04d.tfrecord"%(args.mode,i)) as writer:
        while len(filep) > 0:
            pos = np.random.choice(np.arange(len(filep)))
            if filep[pos] is None:
                filep[pos] = open(filename[pos])
            line = filep[pos].readline()
            hybrid_lm = np.random.random() >= args.hybrid_rate
            if not line:
                filep[pos].close()
                filep.pop(pos)
                fileend.append(filename.pop(pos))
                filesize.pop(pos)
            else:
                isnextsent = args.use_nextsent and args.mode != 'lm' and (args.mode != 'hybrid' or not hybrid_lm)
                nextsenttoken = []
                nextsent_target = -1
                appendid = -1
                if isnextsent:
                    if np.random.random() < 0.5:
                        nextsent = line
                        line = filep[pos].readline()
                        if line:
                            nextsent_target = 1
                        else:
                            line = nextsent
                            isnextsent = False
                            nextsenttoken = []
                            nextsent_target = -1
                    else:
                        anotherpos = np.random.choice(np.arange(len(filep)))
                        if filep[anotherpos] is None:
                            filep[anotherpos] = open(filename[anotherpos])
                        nextsent = filep[anotherpos].readline()
                        nextsent_target = 0
                        if not nextsent:
                            isnextsent = False
                            nextsenttoken = []
                            nextsent_target = -1
                        filep[anotherpos].close()
                        filep[anotherpos] = None

                    if isnextsent:
                        nextsenttoken = enc.encode(nextsent.strip(), clean=True)
                        if len(nextsenttoken) >= NUM_CTX // 2 - 1:
                            nextsenttoken = nextsenttoken[:NUM_CTX // 2 - 1]
                        nextsenttoken = nextsenttoken + [SEP_TOKEN]

                cur_token = []
                cur_target = []
                if args.mode == 'lm':
                    num_input = 0
                elif args.mode == 'mlm':
                    num_input = NUM_CTX-len(nextsenttoken)-3
                elif args.mode == 'hybrid':
                    num_input = 0 if hybrid_lm else np.random.randint(NUM_CTX-len(nextsenttoken)-2)
                else:
                    assert False, 'imvalid mode'
                while line:
                    line = line.replace('\r\n','\n')
                    line = line.replace('\r','\n')

                    if len(cur_token) < num_input:
                        encoded_token = enc.encode(line.strip(), clean=False)
                        encoded_token.append(NL_TOKEN)
                        if len(cur_token)+len(encoded_token) < num_input:
                            cur_token.extend(encoded_token)
                        else:
                            addlen = num_input-len(cur_token)
                            cur_token.extend(encoded_token[:addlen])
                            cur_target.extend(encoded_token[addlen:])
                            assert len(cur_token) == num_input, "num_input mismatch"
                            cur_token.append(SEG_TOKEN)
                            cur_token.extend(encoded_token[addlen:])
                    else:
                        if len(cur_token) == 0 and num_input == 0:
                            cur_token.append(SEG_TOKEN)
                        encoded_token = enc.encode(line.strip(), clean=True)
                        encoded_token, target_token = _target(encoded_token)
                        assert len(encoded_token) == len(target_token), "target len mismatch"

                        encoded_token.append(NL_TOKEN)
                        target_token.append(NL_TOKEN)

                        cur_token.extend(encoded_token)
                        cur_target.extend(target_token)
                        assert (len(cur_token) - num_input) == (len(cur_target) + 1), "num_target mismatch"

                    if len(cur_token) >= NUM_CTX-len(nextsenttoken):
                        appendid = file_id[filename[pos]] if file_id is not None else (num_write_data*args.num_process + i)
                        break

                    line = filep[pos].readline()
                    if not line:
                        appendid = file_id[filename[pos]] if file_id is not None else (num_write_data*args.num_process + i)
                        filep[pos].close()
                        filep.pop(pos)
                        fileend.append(filename.pop(pos))
                        filesize.pop(pos)
                        break

                if len(cur_token) > num_input:
                    assert (len(cur_token) - num_input) == (len(cur_target) + 1), "ending mismatch"
                    writer.write(_append(appendid, cur_token, cur_target, nextsenttoken, num_input, nextsent_target))
                    writer.flush()
                    num_write_data += 1
    print(f"process {i} wrote {num_write_data} datas and complete epoch")

with Pool(args.num_process) as p:
    p.map(_proc, list(range(args.num_process)))
