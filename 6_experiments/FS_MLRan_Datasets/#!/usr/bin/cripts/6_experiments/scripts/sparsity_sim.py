#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

train = pd.read_csv('MLRan_X_train_RFE.csv')
test  = pd.read_csv('MLRan_X_test_RFE.csv')
META = ['sample_id','sample_type','family_label','type_label']
FEATURES = [c for c in train.columns if c not in META]
assert len(FEATURES) == 483, f'expected 483 features, got {len(FEATURES)}'

X_train = train[FEATURES].to_numpy(dtype=np.int8)
y_train = train['sample_type'].to_numpy(dtype=np.int8)
X_test  = test[FEATURES].to_numpy(dtype=np.int8)
y_test  = test['sample_type'].to_numpy(dtype=np.int8)

MODELS = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RNG_SEED),
    'RandomForest':       RandomForestClassifier(n_estimators=100, random_state=RNG_SEED, n_jobs=-1),
    'ExtraTrees':         ExtraTreesClassifier(n_estimators=100, random_state=RNG_SEED, n_jobs=-1),
    'DecisionTree':       DecisionTreeClassifier(random_state=RNG_SEED),
}
for name, m in MODELS.items(): m.fit(X_train, y_train)

FRACTIONS = [0.0, 0.25, 0.50, 0.75, 0.90, 1.00]
F = len(FEATURES)

def zero_random(X, f):
    Z = X.copy()
    if f <= 0: return Z
    k = int(round(f * F))
    for i in range(len(Z)):
        idx = rng.choice(F, size=k, replace=False)
        Z[i, idx] = 0
    return Z

ransom_means = X_train[y_train == 1].mean(axis=0)
rarity_order = np.argsort(-ransom_means)

def zero_targeted(X, f):
    Z = X.copy()
    k = int(round(f * F))
    if k: Z[:, rarity_order[:k]] = 0
    return Z

def evaluate(X_zeroed):
    rows = []
    for name, m in MODELS.items():
        proba = m.predict_proba(X_zeroed)
        pred  = m.predict(X_zeroed)
        rows.append({
            'model': name, 'mean_p_goodware': float(proba[:, 0].mean()),
            'ransomware_recall': float((pred[y_test == 1] == 1).mean()),
            'goodware_fpr': float((pred[y_test == 0] == 1).mean()),
        })
    return rows

rows_r, rows_t = [], []
for f in FRACTIONS:
    for r in evaluate(zero_random(X_test, f)):
        r['fraction_zeroed'] = f; r['mode'] = 'random'; rows_r.append(r)
    for r in evaluate(zero_targeted(X_test, f)):
        r['fraction_zeroed'] = f; r['mode'] = 'targeted'; rows_t.append(r)

cols = ['fraction_zeroed','model','mean_p_goodware','ransomware_recall','goodware_fpr','mode']
pd.DataFrame(rows_r)[cols].to_csv('sparsity_results.csv', index=False)
pd.DataFrame(rows_t)[cols].to_csv('sparsity_results_targeted.csv', index=False)
