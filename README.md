# GPTSAN

![model](report/logo-bk.png)


# GPTSANとは

なんにでも使える汎用日本語言語モデルを目指して作成したSwitch Transformerモデルです。

特徴として、[1-GPUでファインチューニング可能](report/finetune.md#gpu)だったり、[生成文章のクラスを指定可能](report/finetune.md#sqout)だったりします。


# モデルのダウンロード


現在、28億パラメーター（10layer-1024ch-16header-16experts-1280contexts）のモデルが公開されています。

[こちら](https://drive.google.com/file/d/1Maq_EZNOnzKDiWpDACT3zXbzsmnvJ9BG/view?usp=sharing)のGoogle Driveフォルダから、モデルをダウンロードし、以下のコマンドで解凍します。

```sh
$ tar xfj GPTSAN-2.8B-spout_is_uniform.tar.bz2
```

# Dockerで使ってみる

標準環境の構築。

```sh
$ docker build .
ビルド後にコンテナIDが表示される
$ docker run --gpus all -it --rm -v `pwd`/GPTSAN-2.8B-spout_is_uniform:/tf/GPTSAN/GPTSAN-2.8B-spout_is_uniform <コンテナID> python run_generate.py --model GPTSAN-2.8B-spout_is_uniform/ --context "武田信玄は、戦国 時代ファンならぜひ押さえておきたい名将の一人。天下統一を目指し勢いに乗る織田信長からも、一目置かれていたと"
```

# とりあえず使ってみる

文章生成。

```sh
$ python run_generate.py --model GPTSAN-2.8B-spout_is_uniform/ --context "武田信玄は、戦国 時代ファンならぜひ押さえておきたい名将の一人。天下統一を目指し勢いに乗る織田信長からも、一目置かれていたと"
```

## 生成オプション

### Top_K

「--top_k」で指定可能。変えると割と変わる。

```sh
$ python run_generate.py --model GPTSAN-2.8B-spout_is_uniform/ --context "武田信玄は、戦国 時代ファンならぜひ押さえておきたい名将の一人。天下統一を目指し勢いに乗る織田信長からも、一目置かれていたと" --top_k 100
```

### 最大生成文字数

「--max_generate」で指定可能。小さくすると速く動く。

```sh
$ python run_generate.py --model GPTSAN-2.8B-spout_is_uniform/ --context "武田信玄は、戦国 時代ファンならぜひ押さえておきたい名将の一人。天下統一を目指し勢いに乗る織田信長からも、一目置かれていたと" --max_generate 100
```

### 検索木の枝数

「--beam_width」で指定可能。小さくすると速く動く。

```sh
$ python run_generate.py --model GPTSAN-2.8B-spout_is_uniform/ --context "武田信玄は、戦国 時代ファンならぜひ押さえておきたい名将の一人。天下統一を目指し勢いに乗る織田信長からも、一目置かれていたと" --beam_width 1
```


# ファインチューニング

1-GPUでファインチューニングな大規模言語モデル。

[ファインチューニング方法](report/finetune.md)


# 言語モデルとして実行

テキスト穴埋め問題をBERT風に解く。

[Masked Language Modelを実行](report/model.md#mlm)


# 文章のベクトル化

内部層の情報も含んだ文章ベクトル生成。

[任意のトークン位置からステータスを抽出](report/model.md#vectorize)
