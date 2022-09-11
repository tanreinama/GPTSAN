# GPTSAN Tools

# make_tfrecord_simple.py

学習用の文章→トークン変換プログラムです。

TPUを使って学習するために、tfrecord形式にエンコードします。

## 事前学習のためのエンコード

```sh
$ python make_tfrecord_simple.py --input_files 'data/*.txt' --mode lm
```

「--mode」オプションについては、[モデルの学習モード](../report/model.md)を参照。

出力されるファイル名は「学習モード-プロセス番号」の形になります。

エンコード高速化のためにマルチプロセスで動作するので、使用するCPUの数の分だけファイル数が出来ます。

## クラス分類されたテキストをエンコード

```sh
$ cat list.csv
filename,fileid
data/A.txt,0
data/B.txt,1
data/C.txt,1
data/D.txt,2
・・・
$ python make_tfrecord_simple.py --input_files list.csv --mode lm
```

ファイル名と、テキストの種類を指定したCSVファイルを入力します。

CSVの最初の行は「filename,fileid」である必要があります。

詳しくは[Spout入力とは](../report/finetune.md#sqout)を参照。

# run_modelconvert.py

ファインチューニング用のモデル構造の変換プログラムです。

Switch-Transformer層の下に、新しいTransformer層を追加します。

詳しくは[ファインチューニング](../report/finetune.md)を参照。

```sh
$ python run_modelconvert.py --checkpoint ../GPTSAN-2.8B-spout_is_uniform --output ../GPTSAN-finetune-3B --add_extra_stage 14
```

「--add_extra_stage」で追加する層の数を指定します。Switch-Transformer層と追加のTransformer層の間に、新しい位置埋め込み層も1つ増えます。

2.8Bパラメーター数のモデルに14層追加すると、3Bパラメーター数のモデルになります。
