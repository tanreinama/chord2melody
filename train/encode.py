# coding=utf-8
# This software includes the work that is distributed in the MIT License
# https://opensource.org/licenses/mit-license.php
import pypianoroll
import glob
import pickle
import gc
import os
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="source dir", default='lpd_5' )
parser.add_argument("--process", help="num process", type=int, default=8 )
parser.add_argument("--da", help="modulation for data augumantation (0-11)", type=int, default=5 )
parser.add_argument("--combine", help="combine data", type=int, default=50000 )
parser.add_argument("--output", help="output name", required=True )
args = parser.parse_args()

assert args.dataset=='lpd_5' or args.dataset=='lpd_17', 'Dataset requid "lpd_5" or "lpd_7"'

if args.dataset=='lpd_5':
    tracks = ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']
    if os.path.isdir('lpd_5/lpd_5_full'):
        datas = 'lpd_5/lpd_5_full/*/*.npz'
    elif os.path.isdir('lpd_5/lpd_5_cleansed'):
        datas = 'lpd_5/lpd_5_cleansed/*/*/*/*/*.npz'
    else:
        print('invalid dataset')
        exit()
else:
    tracks = ['Drums', 'Piano', 'Chromatic Percussion', 'Organ', 'Guitar', 'Bass', 'Strings', 'Ensemble', 'Brass', 'Reed', 'Pipe', 'Synth Lead', 'Synth Pad', 'Synth Effects', 'Ethnic', 'Percussive', 'Sound Effects']
    if os.path.isdir('lpd_17/lpd_17_full'):
        datas = 'lpd_17/lpd_17_full/*/*.npz'
    elif os.path.isdir('lpd_17/lpd_17_cleansed'):
        datas = 'lpd_17/lpd_17_cleansed/*/*/*/*/*.npz'
    else:
        print('invalid dataset')
        exit()

trc_len = len(tracks)
trc_idx = sorted(list(range(trc_len)), key=lambda x:0 if tracks[x]=='Bass' else 1) # fist 3 is: Bass, Drums, Piano
trc_avg_c = [[0,0] for _ in tracks]

pianoroll_files = []
note_size = 84
note_offset = 24
time_note = note_size*trc_len + 1
end_note = note_size*trc_len + 2
print('step 1/3.')
for fn in tqdm(glob.glob(datas)):
    pianoroll_files.append(fn)
    m = pypianoroll.load(fn)
    pr = m.get_stacked_pianoroll()
    pr = pr[:,:,trc_idx]
    p = np.where(pr != 0)
    for i in np.argsort(p[0]):
        trc_avg_c[p[2][i]][0] += p[1][i]
        trc_avg_c[p[2][i]][1] += 1
    del m, pr, p

trc_avg = [((trc_avg_c[i][0] / trc_avg_c[i][1]) if trc_avg_c[i][1] > 0 else 60) for i in range(trc_len)]
del trc_avg_c
gc.collect()

print('step 2/3.')
def _run(fn):
    seq = []
    m = pypianoroll.load(fn)
    step = m.beat_resolution // 4  # 16 beat minimum
    pr = m.get_stacked_pianoroll()
    pr = pr[:,:,trc_idx]
    p = np.where(pr != 0)
    cur_avg_c = [[0,0] for _ in tracks]
    for i in range(len(p[0])):
        cur_avg_c[p[2][i]][0] += p[1][i]
        cur_avg_c[p[2][i]][1] += 1
    cur_avg = [((cur_avg_c[i][0] / cur_avg_c[i][1]) if cur_avg_c[i][1] > 0 else 60) for i in range(trc_len)]
    modulation = [0] + (np.random.permutation(11)+1).tolist()
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
    for s in modulation[:args.da+1]:  # Data augumantation
        pos = 0
        for i in np.argsort(p[0]):
            if p[0][i] % step != 0:
                continue
            if pos < p[0][i]:
                for _ in range(pos,p[0][i],step):
                    seq.extend(_current(current_seq))
                    seq.append(time_note)
                    current_seq = []
            pos = p[0][i]
            j = p[1][i]
            t = p[2][i]

            shift = 0
            if t != 1:
                if cur_avg[t] + s < trc_avg[t] + 6:
                    shift = s
                else:
                    shift = s-12

            j = j + shift
            if j < 0:
                j += 12
            if j > 0x7f:
                j -= 12
            j -= note_offset
            if j < 0:
                j = 0
            if j > note_size:
                j = note_size-1
            note = t*note_size + j
            current_seq.append(note)
        seq.extend(_current(current_seq))
        seq.append(time_note)
        seq.append(end_note)
        current_seq = []
    with open(fn+'.tmp', mode='wb') as f:
        pickle.dump(seq, f)

with tqdm(total=len(pianoroll_files)) as t:
    with Pool(args.process) as p:
        for _ in p.imap_unordered(_run, pianoroll_files):
            t.update(1)

numfiles = 0
chunks = []
current_chunk = []
print('step 3/3.')
for fn in tqdm(pianoroll_files):
    with open(fn+'.tmp', mode='rb') as f:
        file_chunk = pickle.load(f)
    for note in file_chunk:
        current_chunk.append(note)
        if len(current_chunk) > args.combine:
            chunks.append(np.stack(current_chunk))
            current_chunk = []
    if len(chunks) > 32000:
        numfiles += 1
        np.savez_compressed(args.output+'_%02d.npz'%numfiles, *chunks)
        chunks = []
numfiles += 1
chunks.append(current_chunk)
np.savez_compressed(args.output+'_%02d.npz'%numfiles, *chunks)
chunks = []
with open(args.output+'.trc', 'w') as f:
    f.write(','.join([tracks[t] for t in trc_idx]))
