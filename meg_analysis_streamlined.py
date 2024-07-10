import matplotlib
import matplotlib.pyplot as plt

import os
import pandas as pd
import numpy as np

import mne
from mne.minimum_norm import apply_inverse, make_inverse_operator
from mne.decoding import GeneralizingEstimator

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.decomposition import PCA

from scipy.stats import spearmanr

import multiprocessing
import pickle

def get_perf_timecourse(X, y, decoder, perf_metric, n_splits=5):
    classifying = len(np.unique(y)) <= 2

    n_times = X.shape[-1]

    scores = np.zeros(n_times)
    pvalues = np.zeros(n_times)

    kf = KFold(n_splits, shuffle=True)
    split = lambda X, y: kf.split(X)

    if classifying:
        kf = StratifiedKFold(n_splits, shuffle=True)
        # Account for the fact that StratifiedKFold takes 2 arguments to split
        split = lambda X, y: kf.split(X, y)

    max_acc = 0

    for t in range(n_times):
        t_scores = []
        t_pvalues = []

        for train_indices, test_indices in split(X, y):
            decoder = decoder.fit(X[train_indices, :, t], y[train_indices])
            y_pred = decoder.predict(X[test_indices, :, t])

            if len(np.unique(y_pred)) > 1:
                score = perf_metric(y_pred, y[test_indices])
                acc = decoder.score(X[train_indices, :, t], y[train_indices])

                # if acc > max_acc:
                #     max_acc = acc
                #     print(acc)

                if classifying:
                    t_scores.append(score)
                    t_pvalues.append(0)
                else:
                    t_scores.append(score.statistic)
                    t_pvalues.append(score.pvalue)

        scores[t] = sum(t_scores) / len(t_scores)
        pvalues[t] = max(t_pvalues)

    return scores, pvalues

def get_data(filename):
    fif = mne.read_epochs(f"./sub_data/{filename}")
    
    metadata = fif.metadata
    stim_features = metadata[["freq", "condition", "trial_type"]].to_numpy()

    sub_data = fif.get_data(picks=["meg"])
    
    del fif

    return stim_features, sub_data

filters = {
    "pure": lambda features: (features[:,1] == "pure"),
    "complex": lambda features: (features[:,1] == "partial"),
    "ambiguous": lambda features: (features[:,1] == "shepard")
}

def get_sub_scores(filename):    
    stim_features, sub_data = get_data(filename)
    
    ridge_decoder = make_pipeline(
        StandardScaler(),
        Ridge()
    )
    
    freqs = stim_features[:, 0]
    del stim_features

    sub_scores = { }
    
    for condition in filters:
        condition_filter = filters[condition](stim_features)

        print(f"Finding {condition} scores for {filename}...", end="\r")
        condition_scores, _ = get_perf_timecourse(sub_data[condition_filter], freqs[condition_filter], ridge_decoder, spearmanr)
        
        sub_scores[condition] = condition_scores
    
    del freqs, sub_data

    return sub_scores
    
def get_gen_scores(filename):
    stim_features, sub_data = get_data(filename)
    freqs = stim_features[:, 0]

    ridge_decoder = make_pipeline(
        StandardScaler(),
        Ridge()
    )

    for condition in filters:
        condition_filter = filters[condition](stim_features)

        print(f"Finding generalized {condition} scores for {filename}...", end="\r")
        spearman_scorer = make_scorer(spearmanr, greater_is_better=True)

        time_gen = GeneralizingEstimator(ridge_decoder, scoring=spearman_scorer, n_jobs=-1)
        time_gen.fit(X=sub_data[condition_filter], y=freqs[condition_filter])
        output = time_gen.score(X=sub_data[condition_filter], y=freqs[condition_filter])
        
        first_elem = lambda tup: tup[0]
        first_elem_vectorized = np.vectorize(first_elem)
        sub_scores_gen = first_elem_vectorized(output)


if __name__ == "__main__":
    sub_files = os.listdir("./sub_data")
    output = None

    try:
        pool = multiprocessing.Pool(8)
        output = pool.map(get_sub_scores, sub_files)
    except KeyboardInterrupt as e:
        pass
    finally:
        pool.terminate()
        pool.join()

    print(output)

    with open("output.pkl", "w") as file:
        pickle.dump(output, file)
        file.close()

