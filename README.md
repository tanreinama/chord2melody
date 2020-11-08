# Chord2Melody - Automatic Music Generation AI



[日本語](README_ja.md)

[demonstration1](http://ailab.nama.ne.jp/#chord2melody) | [demonstration2](http://ailab.nama.ne.jp/#melody2melody)

[samples](samples/)



## What is Chord2Melody?



It is an AI that composes music, with MIDI output.

It is based on GPT-2. You can generate music of arbitrary length, and you can also specify the chord progression to generate music.

Or they can compose a continuation of a music they've been working on.

The output music can be used as free content without any copyright or usage restrictions.



### Pretrained Models



There are two models that have been trained: the "base_5tr" with 5 output tracks and the "base_17tr" with 17 tracks.



| Model Name                                                   | output tracks                                                | total number of parameters |
| ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------- |
| [base_5tr](https://www.nama.ne.jp/models/chord2melody-base_5tr.tar.gz) | Drums, Piano, Guitar, Bass, Strings                          | 86167296                   |
| [base_17tr](https://www.nama.ne.jp/models/chord2melody-base_17tr.tar.gz) | Drums, Piano, Chromatic Percussion, <br />Organ, Guitar, Bass, Strings, Ensemble, <br />Brass, Reed, Pipe, Synth Lead, Synth Pad, <br />Synth Effects, Ethnic, Percussive, Sound Effects | 86941440                   |



## Usage



First, clone chord2melody from GitHub.

```sh
$ git clone https://github.com/tanreinama/chord2melody
$ cd chord2melody
```

Then, download and extract the pretrained model from the link above.

```sh
$ wget https://www.nama.ne.jp/models/chord2melody-base_5tr.tar.bz2
$ tar xvfj chord2melody-base_5tr.tar.bz2
```

Launch chord2melody.py with specifying the model , a MIDI file is created.

```sh
$ python3 chord2melody.py --model base_5tr
```

There is no limit to the length of music that can be output. The total number of bars of music to be generated is specified with the "--num_bars" option.

```sh
$ python3 chord2melody.py --num_bars 48
```



### Chord to Melody



To specify the chord progression, use the "--chord" option, and use the "--chordbeat" option to specify how many chords to put in a measure.

```sh
$ python3 chord2melody.py --chord "C|C|C|C|Dm|Dm|Dm|Dm|G7|G7|G7|G7|Am|Am|Am|Am" --chordbeat 4
```

Chord" option, you can specify from [Available Chord](chordlist.txt) or "auto" connected with "|".



###Compose a continuation of a music 



The program "melody2melody.py" will automatically compose a continuation of a music you have been working on. Use the "--input" option in "melody2melody.py" to specify the MIDI file you want to create a continuation of.

```sh
$ python3 melody2melody.py --input halfway.mid
```



### Specifies the fluctuation of the melody



By specifying "--top_p", you can specify the fluctuation of the song.

```sh
$ python3 chord2melody.py --top_k 25 --top_p 0
```

Put one or two numbers in "--top_p". The first number is used when there's a chord progression and the second (if specified) when the chord progression is "auto".



## Learning Methods



The data for training is from [Lakh Pianoroll Dataset](https://salu133445.github.io/lakh-pianoroll-dataset/). To train, download [lpd_5_full.tar.gz](https://drive.google.com/u/0/open?id=1tZKMhYazSWapFTUt7H6abHSo-QKH9ATC) or [lpd-17-full.tar.gz](https://drive. google.com/uc?export=download&id=1bJAC2SKhdKbKvpLL1V1l66cCgWX8eQEm) and extract it.

Next, go to the train directory and run "encode.py" to create a training data file.

The "--da" option allows you to specify data augmentation by modulation. The randomly modulated data is used to increase the training data.

```sh
$ cd train
$ python3 encode.py --dataset lpd_5 --output lpd_5_dataset
```

Run "train.py" with specify the type of dataset (lpd_5/lpd_7) in "--dataset" option and the encoded training data file in "--input" option.

```sh
$ python3 train.py --dataset lpd_5 --input lpd_5_dataset
```



### Fine Tuning



To fine-tune your own data, you must first edit the data to 5 or 17 tracks of MIDI data.

Then save the data in pypianoroll format, with the tracks in the same order as the original model.

Then you can create a training data file in encoder.py and fine tune it by specifying the original trained model in the "--restore_from" field.

```sh
$ python3 train.py --dataset lpd_5 --input lpd_5_dataset --restore_from ../base_5tr
```

