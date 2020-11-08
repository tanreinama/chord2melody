# coding=utf-8
# Copyright (c) 2020 Toshiyuki Sakamoto, Released under the MIT license
# https://opensource.org/licenses/mit-license.php
import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams
import argparse
import pypianoroll
from tqdm import tqdm
from train.src.sample import sample_sequence
import train.src.model as model

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='base_5tr')
parser.add_argument('--input', type=str)
parser.add_argument('--num_bars', type=int, default=4)
parser.add_argument('--temperature', type=float, default=1)
parser.add_argument('--output', type=str, default='output.mid')
parser.add_argument('--top_p', type=int, default=7)
args = parser.parse_args()

if '_5tr' in args.model:
    tracks = ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']
    programs = [0,0,24,32,40]
elif '_17tr' in args.model:
    tracks = ['Drums', 'Piano', 'Chromatic Percussion', 'Organ', 'Guitar', 'Bass', 'Strings', 'Ensemble', 'Brass', 'Reed', 'Pipe', 'Synth Lead', 'Synth Pad', 'Synth Effects', 'Ethnic', 'Percussive', 'Sound Effects']
    programs = [0,0,8,16,24,33,40,48,64,72,80,88,96,104,120]
else:
    print('invalid model name.')
    exit()

trc_len = len(tracks)
trc_idx = sorted(list(range(trc_len)), key=lambda x:0 if tracks[x]=='Bass' else 1)
note_size = 84
note_offset = 24
time_note = note_size*trc_len + 1
end_note = note_size*trc_len + 2
hparams = HParams(**{
  "n_vocab": end_note+1,
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12
})

top_p = args.top_p
temperature = args.temperature

def get_sequence(sess, context, context_tokens, tops):
    output = sample_sequence(
        hparams=hparams, length=1024,
        context=context,
        temperature=temperature, top_k=tops[0], top_p=tops[1]
    )
    out = sess.run(output, feed_dict={
        context: [context_tokens]
    })[0, len(context_tokens):]
    return out

def main():
    pre_melody = pypianoroll.load(args.input)
    pre = []
    step = pre_melody.beat_resolution // 4  # 16 beat minimum
    pianoroll = np.zeros((pre_melody.get_max_length(),128,len(programs)))
    for track in pre_melody.tracks:
        if track.is_drum:
            dst_index = 0
        else:
            dst_index = 1
            for i in range(1,len(programs),1):
                if track.program >= programs[i] and (len(programs) == i+1 or track.program < programs[i+1]):
                    dst_index = i
                    break
        pianoroll[0:track.pianoroll.shape[0],:,dst_index] += track.pianoroll
    pianoroll = pianoroll[:,note_offset:note_offset+note_size,trc_idx]
    p = np.where(pianoroll != 0)
    current_seq = []
    def _current(cur_seq):
        cur = []
        for c in sorted(cur_seq):
            if not (c >= note_size and c < note_size*2):
                cur.append(c)
        for c in sorted(cur_seq):
            if (c >= note_size and c < note_size*2):
                cur.append(c)
        return cur # Bass, Piano, etc..., Drums
    pos = 0
    for i in np.argsort(p[0]):
        if p[0][i] % step != 0:
            continue
        if pos < p[0][i]:
            for _ in range(pos,p[0][i],step):
                pre.extend(_current(current_seq))
                pre.append(time_note)
                current_seq = []
        pos = p[0][i]
        j = p[1][i]
        t = p[2][i]
        note = t*note_size + j
        current_seq.append(note)
    pre.extend(_current(current_seq))
    if len(pre) == 0 or pre[-1] != time_note:
        pre.append(time_note)
    if len(pre) > 512:
        pre = pre[-512:]

    cur_top = (0,top_p)
    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [1, None])
        output = model.model(hparams=hparams, X=context)
        vars = [v for v in tf.trainable_variables() if 'model' in v.name]

        saver = tf.train.Saver(var_list=vars)
        ckpt = tf.train.latest_checkpoint(args.model)
        saver.restore(sess, ckpt)

        pianoroll = np.zeros((trc_len, args.num_bars*16, 128))

        seq = get_sequence(sess, context, pre, cur_top)
        pos = 0
        firstnote = False
        print('Generating Melody...')
        progress = tqdm(total=pianoroll.shape[1])
        while pos < pianoroll.shape[1]:
            for note in seq:
                if (not firstnote) and note >= time_note:
                    continue
                else:
                    firstnote = True
                pre.append(note)
                if note == time_note:
                    pos += 1
                    progress.update(1)
                    if pos >= pianoroll.shape[1]:
                        break
                elif note < time_note:
                    trc = trc_idx.index(note // note_size)
                    mid = note % note_size + note_offset
                    if mid < 128:
                        pianoroll[trc,pos,mid] = 100
            seq = get_sequence(sess, context, pre[-512:], cur_top)

        pr = []
        for i,(t,p) in enumerate(zip(tracks,programs)):
            pr.append(pypianoroll.Track(pianoroll=pianoroll[i], program=p, is_drum=(t=='Drums')))
        mt = pypianoroll.Multitrack(tracks=pr, tempo=120.0, beat_resolution=4)
        mt.write(args.output)

if __name__ == '__main__':
    main()
