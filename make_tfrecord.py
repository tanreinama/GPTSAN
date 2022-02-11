import numpy as np
import re
import argparse
import shutil
import os
import json
import pickle
import uuid
import copy
from multiprocessing import Pool
import tensorflow as tf

from encode_swe import SWEEncoder_ja as Encoder

parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="training mode ('lm' for language model, 'mlm' for masked language model, 'hybrid' for hybrid model)", default="lm" )
parser.add_argument('--no_offset_mlm', help="no offset output for masked language model", action='store_true')
parser.add_argument("--vocabulary", help="vocabulary file", default="ja-swe36k.txt" )
parser.add_argument("--num_process", help="process num", type=int, default=32 )
parser.add_argument("--num_context", help="context token length", type=int, default=1280 )
parser.add_argument("--num_separator", help="separator insertion rate", type=float, default=0.08 )
parser.add_argument("--max_separator", help="max separator insertion num", type=int, default=8 )
parser.add_argument("--num_mask", help="mask insertion rate for masked language model", type=float, default=0.08 )
parser.add_argument("--max_mask", help="max mask insertion num", type=int, default=100 )
parser.add_argument('--use_nextsent', help="use nextsentence training for masked language model", action='store_true')
parser.add_argument("--nextsent_rate", help="train nextsentence rate", type=float, default=0.33 )
parser.add_argument("--hybrid_rate", help="train hybrid mode vs lm mode in mode='hybrid'", type=float, default=1.0 )
parser.add_argument("--data_per_file", help="num data per file", type=int, default=1000000 )
parser.add_argument('--restore_last', action='store_true')
parser.add_argument('--test', action='store_true')
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

if args.restore_last:
    start = 0
    end = args.data_per_file
    files = os.path.listdir(".")
    while True:
        isfile = False
        for f in files:
            if f.startswith("%s_restore_0_%d-"%(args.mode,start)) and f.endswith(".json"):
                isfile = True
        if not isfile:
            break
        else:
            start += args.data_per_file
            end += args.data_per_file
    text_files = []
    text_file_pos = []
    output_start = 0
    if start != 0:
        start -= args.data_per_file
        end -= args.data_per_file
        output_start = end
    for i in range(args.num_process):
        assert os.path.isfile("%s_restore_%d_%d-%d.json"%(args.mode,i,start,end)), "restore file not found"
        with open("%s_restore_%d_%d-%d.json"%(args.mode,i,start,end)) as f:
            j = json.loads(f.read())
            text_files.extend(j["files"])
            text_file_pos.extend(j["tell"])

else:
    array_file = []
    if args.test:
        array_file.extend([("sep/"+f,False,os.path.getsize("sep/"+f)) for f in os.listdir("sep")])
        array_file.extend([("nosep/"+f,True,os.path.getsize("nosep/"+f)) for f in os.listdir("nosep")])
    else:
        src_dir  = ["C4/","CC100/","extra_content/"]
        for dir in src_dir:
            if os.path.isdir(dir):
                array_file.extend([(dir+f,False,os.path.getsize(dir+f)) for f in os.listdir(dir)])
        src_dir  = ["corpus2010/"]
        for dir in src_dir:
            if os.path.isdir(dir):
                array_file.extend([(dir+f,True,os.path.getsize(dir+f)) for f in os.listdir(dir)])

    np.random.shuffle(array_file)
    np.random.shuffle(array_file)
    np.random.shuffle(array_file)

    text_indexs = np.array_split(np.arange(len(array_file)), args.num_process)
    text_files = [[array_file[i] for i in p] for p in text_indexs]
    text_file_pos = [[0] * len(text_files[i]) for i in range(args.num_process)]
    output_start = 0

    proc = True
    while proc:
        trainsize = [np.sum([f[2] for f in t]) for t in text_files]
        maxindex = np.argmax(trainsize)
        minindex = np.argmin(trainsize)
        minofmax = np.min([f[2] for f in text_files[maxindex]])
        proc = trainsize[maxindex] - minofmax > trainsize[minindex]
        if proc:
            minofmaxidx = np.argmin([f[2] for f in text_files[maxindex]])
            text_files[minindex].append(text_files[maxindex].pop(minofmaxidx))
            np.random.shuffle(text_files[minindex])

def _append(i, tokens, target, nextsent, num_input, nextsent_target, isfilesep):
    tokens, target, nextsent = copy.copy(tokens), copy.copy(target), copy.copy(nextsent)

    if not isfilesep:
        n = max(int(args.num_separator * len(target)), args.max_separator)
        positions = np.random.permutation(len(tokens) - 1 - num_input)
        for p in positions[:n]:
            if tokens[p+1+num_input] < SOT_TOKEN:
                tokens.insert(p+1+num_input, SEP_TOKEN)
                target.insert(p, NOT_TOKEN)

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

def _checkpoint(i, filename, fileend, p, n=args.data_per_file):
    start = output_start
    end = output_start + n
    with open("%s_restore_%d_%d-%d.json"%(args.mode,i,start,end), "w") as wf:
        wf.write(json.dumps({"files":filename, "tell":p, "completed":fileend}))

def _proc(i):
    np.random.seed(i)
    filep = [open(f[0]) for f in text_files[i]]
    filename = [f for f in text_files[i]]
    fileend = []
    filesep = [f[1] for f in text_files[i]]
    filesize = [f[2] for f in text_files[i]]
    for fp, rp in zip(filep, text_file_pos[i]):
        fp.seek(rp)
    num_write_data = 0
    with tf.io.TFRecordWriter("%s_%s_%d_%d-%d.tfrecord"%(args.mode,"test" if args.test else "train",i,output_start,output_start + args.data_per_file)) as writer:
        while len(filep) > 0:
            weight = np.array(filesize) / np.sum(filesize)
            pos = np.random.choice(np.arange(len(filep)), p=weight)
            line = filep[pos].readline()
            isfilesep = filesep[pos]
            hybrid_lm = np.random.random() >= args.hybrid_rate
            if not line:
                filep[pos].close()
                filep.pop(pos)
                fileend.append(filename.pop(pos))
                filesep.pop(pos)
                filesize.pop(pos)
            else:
                isnextsent = (np.random.random() < args.nextsent_rate) and args.use_nextsent and args.mode != 'lm' and (args.mode != 'hybrid' or not hybrid_lm)
                nextsenttoken = []
                nextsent_target = -1
                if isnextsent:
                    if np.random.random() < 0.5:
                        nextsent = line
                        line = filep[pos].readline()
                        if line:
                            nextsent_target = 1
                        else:
                            while not line:
                                filep[pos].close()
                                filep.pop(pos)
                                fileend.append(filename.pop(pos))
                                filesep.pop(pos)
                                filesize.pop(pos)
                                if len(filep) == 0:
                                    break
                                weight = np.array(filesize) / np.sum(filesize)
                                pos = np.random.choice(np.arange(len(filep)), p=weight)
                                isfilesep = filesep[pos]
                                line = filep[pos].readline()
                            nextsent_target = 0
                            if not line or len(filep) == 0:
                                break
                    else:
                        anotherweight = [(0 if i==pos else filesize[i]) for i in range(len(filesize))]
                        if len(anotherweight) != 1:
                            anotherweight = np.array(anotherweight) / np.sum(anotherweight)
                            anotherpos = np.random.choice(np.arange(len(filep)), p=anotherweight)
                        else:
                            anotherpos = pos
                            nextsent_target = 1
                        nextsent = filep[anotherpos].readline()
                        while not nextsent:
                            filep[anotherpos].close()
                            filep.pop(anotherpos)
                            fileend.append(filename.pop(anotherpos))
                            filesep.pop(anotherpos)
                            filesize.pop(anotherpos)
                            if len(filep) == 0:
                                break
                            anotherweight = [(0 if i==pos else filesize[i]) for i in range(len(filesize))]
                            anotherweight = np.array(anotherweight) / np.sum(anotherweight)
                            anotherpos = np.random.choice(np.arange(len(filep)), p=anotherweight)
                            nextsent = filep[anotherpos].readline()
                        if len(filep) == 0:
                            break
                        nextsent_target = 0
                        if not nextsent:
                            isnextsent = False
                            nextsenttoken = []
                            nextsent_target = -1

                    if isnextsent:
                        nextsenttoken = enc.encode(nextsent.strip(), clean=True)
                        if len(nextsenttoken) >= NUM_CTX // 2 - 1:
                            nextsenttoken = nextsenttoken[:NUM_CTX // 2 - 1]
                        nextsenttoken = nextsenttoken + [SEP_TOKEN]

                cur_token = []
                cur_target = []
                iswrote = False
                if args.mode == 'lm':
                    num_input = 0
                elif args.mode == 'mlm':
                    num_input = NUM_CTX-len(nextsenttoken)-3
                elif args.mode == 'hybrid':
                    num_input = 0 if hybrid_lm else np.random.randint(NUM_CTX-len(nextsenttoken)-2)
                else:
                    assert False, 'imvalid mode'
                while line and len(filesep)>0:
                    line = line.replace('\r\n','\n')
                    line = line.replace('\r','\n')
                    if line == '\n' or len(line.strip())==0:
                        if len(cur_token) > num_input:
                            writer.write(_append(i, cur_token, cur_target, nextsenttoken, num_input, nextsent_target, isfilesep))
                            writer.flush()
                            num_write_data += 1
                            if num_write_data >= args.data_per_file:
                                _checkpoint(i, filename, fileend, [f.tell() for f in filep])
                                print(f"process {i} wrote {num_write_data} datas")
                                return
                            iswrote = True
                            break

                    if len(cur_token) < num_input:
                        encoded_token = enc.encode(line.strip(), clean=False)
                        if filesep[pos]:
                            encoded_token.append(SEP_TOKEN)
                        else:
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

                        encoded_token.append(SEP_TOKEN)
                        target_token.append(SEP_TOKEN if filesep[pos] else NL_TOKEN)

                        cur_token.extend(encoded_token)
                        cur_target.extend(target_token)
                        assert (len(cur_token) - num_input) == (len(cur_target) + 1), "num_target mismatch"

                    if len(cur_token) >= NUM_CTX-len(nextsenttoken):
                        writer.write(_append(i, cur_token, cur_target, nextsenttoken, num_input, nextsent_target, isfilesep))
                        writer.flush()
                        num_write_data += 1
                        if num_write_data >= args.data_per_file:
                            _checkpoint(i, filename, fileend, [f.tell() for f in filep])
                            print(f"process {i} wrote {num_write_data} datas")
                            return
                        iswrote = True
                        break

                    line = filep[pos].readline()
                    if not line:
                        filep[pos].close()
                        filep.pop(pos)
                        fileend.append(filename.pop(pos))
                        filesep.pop(pos)
                        filesize.pop(pos)

                if len(cur_token) > num_input and not iswrote:
                    assert (len(cur_token) - num_input) == (len(cur_target) + 1), "ending mismatch"
                    writer.write(_append(i, cur_token, cur_target, nextsenttoken, num_input, nextsent_target, isfilesep))
                    writer.flush()
                    num_write_data += 1
                    if num_write_data >= args.data_per_file:
                        _checkpoint(i, filename, fileend, [f.tell() for f in filep])
                        print(f"process {i} wrote {num_write_data} datas")
                        return
    _checkpoint(i, filename, fileend, [f.tell() for f in filep], num_write_data)
    if num_write_data != args.data_per_file:
        shutil.move("%s_%s_%d_%d-%d.tfrecord"%(args.mode,"test" if args.test else "train",i,output_start,output_start + args.data_per_file),
                    "%s_%s_%d_%d-%d.tfrecord"%(args.mode,"test" if args.test else "train",i,output_start,output_start + num_write_data))
    print(f"process {i} wrote {num_write_data} datas and complete epoch")

with Pool(args.num_process) as p:
    p.map(_proc, list(range(args.num_process)))
