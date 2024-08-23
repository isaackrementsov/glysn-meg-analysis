import os
import sys
from multiprocessing import Pool
import pandas as pd
import numpy as np

import mne
from mne.decoding import SlidingEstimator, cross_val_multiscore

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

import dill
# Required for multiprocess to work in Jupyter notebook
dill.settings['recurse'] = True

import spacy

def warn(*args, **kwargs):
    pass

import warnings

warnings.warn = warn
warnings.filterwarnings("ignore", category=ConvergenceWarning)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" 

MERGED_SCORES_LOCATION = "merged_pos_scores.npy"
# Generate suffixed path if this file is duplicated
def gen_location(i):
    if i == 0:
        return MERGED_SCORES_LOCATION
    else:
        return MERGED_SCORES_LOCATION.replace(".npy", f"-{i}.npy")

SUB_DATA_DIR = os.environ['SCRATCH']
N_THREADS = 32
N_SPLITS = 5
T_LIMIT = None

# Useful constants for the rest of the code
sub_ids = os.listdir(SUB_DATA_DIR)
# Not sure why but this subject's data isn't loading properly
#sub_ids.remove("A0281")
n_subs = len(sub_ids)

N_SUB_THREADS = 2
N_CLASS_THREADS = 1
N_DECODING_THREADS = N_THREADS // (N_SUB_THREADS * N_CLASS_THREADS) + 1

def cutoff(n_times):
    if T_LIMIT:
        return min(n_times, T_LIMIT)
    else:
        return n_times

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
            pos[i] = token.pos_
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

    base_path = f'{SUB_DATA_DIR}/{sub_id}/{segment}_epochs'

    fif = mne.read_epochs(f'{base_path}/{fif_name}')
    
    metadata = pd.read_csv(f'{base_path}/{segment}-info.csv', keep_default_na=False)

    if segment == 'word':
        metadata['pos'] = get_pos_col(metadata)
    
    stim_features = metadata[features].to_numpy()
    sub_data = fif.get_data(picks=["meg"])
    
    del fif

    return stim_features, sub_data

print("Getting initial sub data")
initial_stim_features, initial_sub_data = get_data(sub_ids[0], segment="word")
print("Done!")

pos_types = np.unique(initial_stim_features[:,0])

tpoints = initial_sub_data.shape[-1]
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

@ignore_warnings(category=ConvergenceWarning)
def get_sub_scores(sub_id, segment="word", feature=0, classes=None):
    logistic_decoder = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1e-4)
    )

    estimator = SlidingEstimator(logistic_decoder, n_jobs=N_DECODING_THREADS, scoring='roc_auc') # check whether this can be here
    validator = StratifiedKFold(n_splits=N_SPLITS)

    stim_features, sub_data = get_data(sub_id, segment=segment)
    labels = stim_features[:, feature]

    if classes is None:
        classes = np.unique(labels)

        # If it's already a binary classification problem, we only need to check for one class
        if len(classes) == 2:
            classes = classes[:1]
    
    del stim_features
    
    sub_scores = { }

    for cl in classes:
        binary_labels = (labels == cl).astype(int)
        t_slice = cutoff(sub_data.shape[-1])
        estimated_scores = cross_val_multiscore(estimator, X=sub_data[:,:,:t_slice], y=binary_labels, cv=validator)
        sub_scores[cl] = estimated_scores.mean(axis=0)  

    return sub_scores

def blank_sub_scores():
    return {
        pos_type: np.zeros((n_subs, tpoints)) for pos_type in pos_types
    }
    
def save_merged_scores(merged_scores):
    i = 0
    location = MERGED_SCORES_LOCATION

    while os.path.isfile(location):
        i += 1
        location = gen_location(i)
    
    np.save(location, merged_scores)


def sub_scores_task(sub_id):
    if len(sys.argv) > 1:
        return get_sub_scores(sub_id, classes=sys.argv[1:])
    else:
        return get_sub_scores(sub_id)

def generate_merged_scores():
    merged_sub_scores = blank_sub_scores()

    pool = Pool(N_SUB_THREADS)
    output = pool.map(sub_scores_task, sub_ids)

    for i in range(n_subs):
        for cl in output[i]:
            merged_sub_scores[cl][i] = output[i][cl]
            
    save_merged_scores(merged_sub_scores)
    pool.terminate()
    pool.join()

if __name__ == '__main__':
    generate_merged_scores()
