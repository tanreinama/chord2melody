# Chord2Melody - 音楽自動生成AI



[English](README.md)

[デモンストレーション]()

[サンプル](samples/)



## Chord2Melodyとは



音楽を作曲してくれるAIです。MIDIでアウトプットされます。

GPT-2をベースに作成されました。任意の長さの音楽を生成出来て、さらにコード進行を指定して生成させることも可能です。

あるいは、作りかけの曲の続きを作曲してくれます。

出力された音楽は、著作権や利用範囲の制限のないフリーコンテンツとして利用出来ます。



### 学習済みモデル



学習済みのモデルは、出力トラック数が5トラックの「base_5tr」と、17トラックを出力する「base_17tr」の2種類あります。



| モデル                                                       | 出力トラック                                                 | 総パラメーター数 |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------- |
| [base_5tr](https://www.nama.ne.jp/models/chord2melody-base_5tr.tar.gz) | Drums, Piano, Guitar, Bass, Strings                          | 86167296         |
| [base_17tr](https://www.nama.ne.jp/models/chord2melody-base_17tr.tar.gz) | Drums, Piano, Chromatic Percussion, <br />Organ, Guitar, Bass, Strings, Ensemble, <br />Brass, Reed, Pipe, Synth Lead, Synth Pad, <br />Synth Effects, Ethnic, Percussive, Sound Effects | 86941440         |



## 使い方



まず、GitHubからクローンします。

```sh
$ git clone https://github.com/tanreinama/chord2melody
$ cd chord2melody
```

上記のリンクから学習済みのモデルをダウンロードして展開します。

```sh
$ wget https://www.nama.ne.jp/models/chord2melody-base_5tr.tar.bz2
$ tar xvfj chord2melody-base_5tr.tar.bz2
```

モデルを指定し、chord2melody.pyを起動すると、MIDIファイルが作成されます。

```sh
$ python3 chord2melody.py --model base_5tr
```

出力可能な音楽の長さに制限はありません。合計で何小節分の音楽を生成するかは、「--num_bars」オプションで指定します。

```sh
$ python3 chord2melody.py --num_bars 48
```



### コード進行→メロディー



コード進行を指定するには、「--chord」オプションを使用します。1小節に何個のコードを入れるかは「--chordbeat」オプションで指定します。

```sh
$ python3 chord2melody.py --chord "C|C|C|C|Dm|Dm|Dm|Dm|G7|G7|G7|G7|Am|Am|Am|Am" --chordbeat 4
```

「--chord」オプションには、[利用出来るコード一覧](chordlist.txt)にあるものか「auto」を、「|」で繋げて指定します。



### 曲の続きを作曲



「melody2melody.py」プログラムは、作りかけの曲の続きを自動で作曲してくれます。「melody2melody.py」に「--input」オプションで、続きを作りたいMIDIファイルを指定します。

```sh
$ python3 melody2melody.py --input halfway.mid
```



### 曲の揺らぎを指定



「--top_p」を指定することで、曲の揺らぎを指定出来ます。

```sh
$ python3 chord2melody.py --chord "C|C|C|C|Dm|Dm|Dm|Dm" --top_p 45,7
```

「--top_p」には、一つか二つの数字を入れます。一つ目の数字はコード進行がある時に使用され、二つ目は（指定されていれば）コード進行が「auto」の時に使用されます。



## 学習方法



学習のためのデータは、 [Lakh Pianoroll Dataset](https://salu133445.github.io/lakh-pianoroll-dataset/)を使用します。予め、[lpd_5_full.tar.gz](https://drive.google.com/u/0/open?id=1tZKMhYazSWapFTUt7H6abHSo-QKH9ATC)又は、[lpd-17-full.tar.gz](https://drive.google.com/uc?export=download&id=1bJAC2SKhdKbKvpLL1V1l66cCgWX8eQEm)をダウンロードして展開しておきます。

trainディレクトリに移動して、「encode.py」を実行して、学習データファイルを作成します。

「--da」オプションで、変調によるData Augmentationを指定出来ます。ランダムに変調を行ったデータで、学習データを増やします。

```sh
$ cd train
$ python3 encode.py --dataset lpd_5 --output lpd_5_dataset
```

「--dataset」にデータセットの種類（lpd_5/lpd_7）を、「--input」にエンコードされた学習データファイルを指定して、「train.py」を実行します。

```sh
$ python3 train.py --dataset lpd_5 --input lpd_5_dataset
```



### ファインチューニング



自前のデータを使ってファインチューニングするには、まずデータを5トラック or 17トラックのMIDIデータに編集する必要があります。

そして、トラックの並び順を、元のモデルと同じ順番にして、pypianoroll形式で保存します。

その後、「encoder.py」で学習データファイルを作成し、「--restore_from」にオリジナルの学習済みのモデルを指定してやると、ファインチューニングすることが出来ます。

```sh
$ python3 train.py --dataset lpd_5 --input lpd_5_dataset --restore_from ../base_5tr
```



