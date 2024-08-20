import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

import os
import pandas as pd
import numpy as np

import mne
from mne.minimum_norm import apply_inverse, make_inverse_operator
from mne.decoding import GeneralizingEstimator

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering, KMeans

from scipy.stats import spearmanr

import multiprocess as mp
import dill
# Required for multiprocess to work in Jupyter notebook
dill.settings['recurse'] = True

import spacy

@ignore_warnings(category=ConvergenceWarning) # So scikit wont print thousands of convergence warnings
def get_perf_timecourse(X, y, decoder, perf_metric, n_splits=5):
    n_times = X.shape[-1]
    scores = np.zeros(n_times)
    kf = StratifiedKFold(n_splits, shuffle=True)

    for t in range(n_times):
        t_scores = []

        for train_indices, test_indices in kf.split(X, y):
            decoder = decoder.fit(X[train_indices, :, t], y[train_indices])
            y_pred = decoder.predict_proba(X[test_indices, :, t])[:,1]

            score = perf_metric(y[test_indices], y_pred)
            #acc = decoder.score(X[train_indices, :, t], y[train_indices])
            t_scores.append(score)
        
        scores[t] = sum(t_scores) / len(t_scores)

        print(f"Score is {round(scores[t], 3)} at t={t}", end="\r")

    return scores

def as_df(x, y):
    return pd.DataFrame({'x': x, 'y': y})

nlp = spacy.load('en_core_web_sm')

def get_pos(sentence):
    words = sentence.split(' ')
    doc = nlp(sentence)
    pos = []
    
    i = -1
    
    for token in doc:
        in_current_word = token.text in words[i]
        
        if i + 1 < len(words):
            in_current_word = in_current_word and token.text not in words[i + 1]
        
        if in_current_word and i > 0:
            pos[i] = pos[i] + '-' + token.pos_
        else:
            i += 1
            pos.append(token.pos_)

    return pos

def get_pos_col(word_info):
    full_text = ' '.join(word_info['word'].tolist())
    return pd.Series(get_pos(full_text))
    # sentences = word_info.groupby(["wav_file", "sentence_number"])["word"].agg(lambda words: ' '.join(words))
    # # Get POS and flatten grouped values into a list
    # pos = np.concatenate(sentences.apply(get_pos).values)
    # return pos

def get_data(sub_id, segment='phoneme'):
    fif_name = 'start_phoneme_-1000_-1000_dcm10_BLNone_hpf60_rep0-epo.fif'
    features = ['phoneme', 'phonation', 'manner', 'place', 'frontback', 'roundness', 'centrality']
    
    if segment == 'word':
        fif_name = 'start_word_-1000_-1000_dcm5_BLNone_hpf60_rep0-epo.fif'
        features = ['pos']

    base_path = f'./story_sub_data/{sub_id}/{segment}_epochs'

    fif = mne.read_epochs(f'{base_path}/{fif_name}')
    
    metadata = pd.read_csv(f'{base_path}/{segment}-info.csv', keep_default_na=False)

    if segment == 'word':
        metadata['pos'] = get_pos_col(metadata)
    
    stim_features = metadata[features].to_numpy()
    sub_data = fif.get_data(picks=["meg"])
    
    del fif

    return stim_features, sub_data

# Useful constants for the rest of the code
sub_ids = os.listdir("./story_sub_data")
# Not sure why but this subject's data isn't loading properly
sub_ids.remove("A0281")
n_subs = len(sub_ids)

initial_stim_features, initial_sub_data = get_data(sub_ids[0], segment="word")
tpoints = initial_sub_data.shape[-1]
pos_types = ['DET', 'ADJ'] #np.unique(initial_stim_features[:,0])

# Recordings were -1000ms to 1000ms, relative to the phoneme/word presented, collected at 161 points
t = np.linspace(-1000, 1000, tpoints)

# Helper function to get first index where condition is true
def index_of(cond):
    indices = np.where(cond)[0]
    
    if len(indices) == 0:
        return -1
    else:
        return indices[0]

# Helper for finding index of a particular point in time
def at_t(t_point):
    return index_of(t == t_point)

def get_pos_scores(sub_id):    
    stim_features, sub_data = get_data(sub_id, segment='word')
    
    logistic_decoder = make_pipeline(
        StandardScaler(),
        LogisticRegression()
    )
    
    pos = stim_features[:, 0]
    
    sub_scores = { }

    for pos_type in pos_types:
        sub_scores[pos_type] = get_perf_timecourse(sub_data, (pos == pos_type).astype(int), logistic_decoder, roc_auc_score)
    
    del pos, sub_data, stim_features

    return sub_scores

MERGED_SCORES_LOCATION = "merged_pos_scores.npy"

def save_merged_scores(merged_scores):
    np.save(MERGED_SCORES_LOCATION, merged_scores)

def load_merged_scores():
    empty = not os.path.isfile(MERGED_SCORES_LOCATION)
    merged_sub_scores = {
        pos_type: np.zeros((n_subs, tpoints)) for pos_type in pos_types
    }
    
    if not empty:
        try:
            merged_sub_scores = np.load(MERGED_SCORES_LOCATION, allow_pickle=True).item()
        except Exception:
            empty = True
            return merged_sub_scores, empty
    
    return merged_sub_scores, empty

merged_sub_scores, empty = load_merged_scores()

# If merged scores have not already been saved, generate them (this takes a long time)
if empty:
    output = []

    try:
        pool = mp.Pool(8)
        output = pool.map(get_pos_scores, sub_ids)
        # for sub_id in sub_ids:
        #     output.append(get_pos_scores(sub_id))

    except KeyboardInterrupt as e:
        pass
    finally:
        pool.terminate()
        pool.join()

    for i in range(n_subs):
        for pos_type in pos_types:
            merged_sub_scores[pos_type][i] = output[i][pos_type]
            
    save_merged_scores(merged_sub_scores)