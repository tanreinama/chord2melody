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
parser.add_argument('--chord', type=str, default='')
parser.add_argument('--chordbeat', type=int, default=4)
parser.add_argument('--num_bars', type=int, default=4)
parser.add_argument('--temperature', type=float, default=1)
parser.add_argument('--output', type=str, default='output.mid')
parser.add_argument('--top_p', type=str, default='45,7')
parser.add_argument('--tempo', type=float, default=120.0)
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

top_p = [int(p) for p in args.top_p.split(',')] if ',' in args.top_p else [int(args.top_p),int(args.top_p)]
temperature = args.temperature
chords = [c for c in args.chord.split('|') if len(c)>0]
for c in chords:
    if not(c == 'auto' or c[0] in 'ABCDEFG'):
        print('invalid chord name.')
        exit()

def chord2tokens(chord):
    if chord is None or chord=='auto':
        return [time_note]
    else:
        base = ['E','F','G','A','B','C','D'].index(chord[0])
        basenote = [4,5,7,9,11,12,14][base]  # Bass
        chordtype = chord[1:]
        if len(chord) > 1 and chord[1] == '#':
            basenote += 1
            chordtype = chord[2:]
        offset = basenote+note_size*2+24  # Piano notes
        if len(chordtype) == 0:
            return [basenote,offset,offset+4,offset+7]
        elif chordtype == 'm':
            return [basenote,offset,offset+3,offset+7]
        elif chordtype == '7':
            return [basenote,offset,offset+4,offset+7,offset+10]
        elif chordtype == 'm7':
            return [basenote,offset,offset+3,offset+7,offset+10]
        elif chordtype == 'M7':
            return [basenote,offset,offset+4,offset+7,offset+11]
        elif chordtype == 'm7-5':
            return [basenote,offset,offset+3,offset+6,offset+10]
        elif chordtype == 'dim':
            return [basenote,offset,offset+3,offset+6,offset+9]
        elif chordtype == 'sus4':
            return [basenote,offset,offset+5,offset+7]
        elif chordtype == '7sus4':
            return [basenote,offset,offset+5,offset+7,offset+10]
        elif chordtype == 'aug':
            return [basenote,offset,offset+4,offset+8]
        elif chordtype == 'm6':
            return [basenote,offset,offset+3,offset+7,offset+9]
        elif chordtype == '7(9)':
            return [basenote,offset,offset+4,offset+7,offset+10,offset+14]
        elif chordtype == 'm7(9)':
            return [basenote,offset,offset+3,offset+7,offset+10,offset+14]
        elif chordtype == 'add9':
            return [basenote,offset,offset+4,offset+7,offset+14]
        elif chordtype == '6':
            return [basenote,offset,offset+4,offset+7,offset+9]
        elif chordtype == 'mM7':
            return [basenote,offset,offset+3,offset+7,offset+11]
        elif chordtype == '7-5':
            return [basenote,offset,offset+4,offset+6,offset+10]
        else:
            return [basenote]

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
    cur_top = (0,top_p[1])
    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [1, None])
        output = model.model(hparams=hparams, X=context)
        vars = [v for v in tf.trainable_variables() if 'model' in v.name]

        saver = tf.train.Saver(var_list=vars)
        ckpt = tf.train.latest_checkpoint(args.model)
        saver.restore(sess, ckpt)

        pianoroll = np.zeros((trc_len, args.num_bars*16, 128))

        pospchord = 16 // args.chordbeat
        pre = [end_note]
        if len(chords) > 0:
            pre.extend(chord2tokens(chords.pop(0)))
            cur_top = (0,top_p[0])
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
                    if pos % pospchord == 0 and len(chords) > 0:
                        c = chords.pop(0)
                        pre.extend(chord2tokens(c))
                        if c != 'auto':
                            cur_top = (0,top_p[0])
                            break
                        elif cur_top != (0,top_p[1]):
                            cur_top = (0,top_p[1])
                            break
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
        mt = pypianoroll.Multitrack(tracks=pr, tempo=args.tempo, beat_resolution=4)
        mt.write(args.output)

if __name__ == '__main__':
    main()
