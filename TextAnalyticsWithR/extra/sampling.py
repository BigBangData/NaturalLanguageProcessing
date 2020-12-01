from sklearn.utils.random import sample_without_replacement
from sklearn.model_selection import train_test_split

# sample 5%
pct_ = round(X_lemm_bow.shape[0]/20, 0)

ix = sample_without_replacement(n_population=X_lemm_bow.shape[0],
                                n_samples=pct_, random_state=42)

X_lemm_bow_sample = X_lemm_bow[ix,]
y_sample = y[ix,]

# check that target class is balanced
sum(y_sample) / len(y_sample)

# sanity checks
X_lemm_bow_sample, len(y_sample)

# split sampled set 
X_train, X_test, y_train, y_test = \
train_test_split(X_lemm_bow_sample, y_sample, test_size=0.2, random_state=42)