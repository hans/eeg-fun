from numbers import Number
from typing import Awaitable, Tuple, List, Dict, Optional, Union
import warnings

import mne
from mne.decoding import ReceptiveField
import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, GridSearchCV
import sklearn.model_selection
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV


# Stolen from mne.decoding.receptive_field
# Create a correlation scikit-learn-style scorer
def _corr_score(y_true, y, multioutput=None):
    from scipy.stats import pearsonr
    assert multioutput == 'raw_values'
    for this_y in (y_true, y):
        if this_y.ndim != 2:
            raise ValueError('inputs must be shape (samples, outputs), got %s'
                             % (this_y.shape,))
    return np.array([pearsonr(y_true[:, ii], y[:, ii])[0]
                     for ii in range(y.shape[-1])])


def mean_corr_score(estimator: TransformedTargetRegressor, X, Y):
    """
    Scalar score function for ReceptiveField estimator
    which averages correlation scores over output sensors.
    """
    n_times, n_outputs = Y.shape
    Y_pred = estimator.predict(X)
    
    # HACK: extract valid_samples_ from TRF within regression pipeline
    trf = estimator.regressor_.named_steps["trf"]
    if hasattr(trf, "coef_"):
        # TRF has been fit. extract valid samples
        valid_samples = trf.valid_samples_
        Y_pred = Y_pred[valid_samples]
        Y = Y[valid_samples]

    # Re-vectorize and call scorer
    Y = Y.reshape([-1, n_outputs])
    Y_pred = Y_pred.reshape([-1, n_outputs])
    assert Y.shape == Y_pred.shape
    scores = _corr_score(Y, Y_pred, multioutput='raw_values')
    return scores.mean()


def downsample_data(data: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample data along first dimension by `factor`, by taking means
    over a sliding window.
    """
    mean_kernel = np.ones((factor,) + (1,) * (data.ndim - 1)) / factor
    ret = convolve(data, mean_kernel, mode="constant")
    
    # Drop `factor - 1` incompatible rows at end
    assert ret.shape[0] == data.shape[0]
    ret = ret[:data.shape[0]]
    
    # Now subsample.
    ret = ret[::factor]
    
    return ret


class Residualizer(BaseEstimator, TransformerMixin):
    
    def __init__(self, residualize_features: List[int],
                 sparse=False, use_base_features=False):
        """
        When `sparse` is `True`, the features are assumed to be sparse / impulse
        features, where each feature is nonzero iff the other features are
        nonzero. Only nonzero elements are used to train residualizers.
        (This sparsity correspondence is never checked.)
        
        When `use_base_features` is `False`, the first feature is predicted
        from all features no included in `residualize_features`. Otherwise,
        the first feature is not actually residualized, but the second feature
        is predicted using just the first feature; the third feature is
        predicted using the first and second; and so on.
        """
        super().__init__()
        self.residualize_features = residualize_features
        self.sparse = sparse
        self.use_base_features = use_base_features
        
        if not self.use_base_features and len(self.residualize_features) == 1:
            raise ValueError("Instructed to not use base features, but "
                             "there is just one feature in "
                             "`residualize_features`. This doesn't make "
                             "sense -- check usage.")
            
    def _sanity_check_sparsity(self, X, sparse_mask, threshold=0.4):
        """
        Run a sanity check: when first sparse feature is active (nonzero),
        latter sparse features should also be active a good proportion of the
        time. If not, issue a warning.
        """
        
        nonzero_means = (X[sparse_mask][:, self.residualize_features[1:]] != 0).mean(axis=0)
        failed = nonzero_means < threshold
        if failed.any():
            desc = [
                (feat_idx, np.round(feat_mean, 3))
                for feat_idx, feat_mean, feat_failed
                in zip(self.residualize_features[1:], nonzero_means, failed)
                if feat_failed]
            warnings.warn(
                "Some sparse features in residualizer are zero for "
               f">{threshold * 100}% of the time that the "
                "first feature is nonzero. Are you sure you want to do "
               f"this?\n\tProportions: {desc}")
        
    def fit(self, X, y=None):
        """
        Fit stagewise regressions for each feature to be residualized.
        Let fr_i be each feature in `self.residualize_features` and fb_j
        be each other feature not in `self.residualize_features`.
        
        For each fr_i, fit a regression
        `fr_i ~ \sum_{k < i} fr_k + \sum_j fb_j`
        """
        
        self.residualizers_ = []
        X_resid_ = X.copy()
        if not self.residualize_features:
            return X_resid_
        
        estimator = LinearRegression(fit_intercept=False)
        
        def make_residualizer(i, estimator_i, feature_idxs):
            def residualize(X):
                pred = estimator_i.predict(X[:, feature_idxs])
                return X[:, i] - pred
            return residualize
        
        sparse_mask = slice(None)
        if self.sparse:
            # Learn estimators only on sparse mask.
            sparse_mask = X[:, self.residualize_features[0]] != 0
            self._sanity_check_sparsity(X, sparse_mask)
        
        if self.use_base_features:
            base_features = np.array([idx for idx in range(X.shape[1])
                                      if idx not in self.residualize_features])
            to_residualize = self.residualize_features
        else:
            first_feature = self.residualize_features[0]
            base_features = np.array([first_feature])
            to_residualize = self.residualize_features[1:]
            
            # Add dummy residualizer for first feature.
            self.residualizers_.append(lambda X: X[:, first_feature])

        for resid_idx in to_residualize:
            # Learn regressor from current partially-residualized features.
            X_i = X_resid_[sparse_mask][:, base_features]
            y_i = X[sparse_mask, resid_idx]
            
            estimator_i = clone(estimator).fit(X_i, y_i)
            
            self.residualizers_.append(
                make_residualizer(resid_idx, estimator_i, base_features))
            X_resid_[sparse_mask, resid_idx] -= estimator_i.predict(X_i)
            
            base_features = np.concatenate([base_features, [resid_idx]])
            
        return self
    
    def transform(self, X):
        X_resid = X.copy()
        
        sparse_mask = slice(None)
        if self.sparse:
            sparse_mask = X[:, self.residualize_features[0]] != 0
        
        for resid_idx, residualize_fn in zip(self.residualize_features, self.residualizers_):
            X_resid[sparse_mask, resid_idx] = residualize_fn(X_resid[sparse_mask])

        return X_resid
        

class TRFEstimator(object):
    """
    Dask-based nested cross validation for per-subject TRF estimation.
    """
    
    def __init__(self, name, feature_names, mne_info: mne.Info,
                 hparam_grid: Dict[str, List[Number]],
                 fit_channels: Optional[List[Union[str, int]]] = None,
                 residualize_features: Optional[List[str]] = None,
                 tmin: float = 0.0, tmax: float = 1.0,
                 downsample_factor=None,
                 n_outer_folds=4, n_inner_folds=4):
        self.name = name
        self.feature_names = feature_names
        self.mne_info = mne_info
        self.hparam_grid = hparam_grid
        self.fit_channels = fit_channels
        self.residualize_features = residualize_features or []
        self.tmin = tmin
        self.tmax = tmax
        self.downsample_factor = downsample_factor
        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds
        
        self.sfreq = self.mne_info["sfreq"] if self.downsample_factor is None \
            else self.mne_info["sfreq"] / self.downsample_factor
        assert int(self.sfreq) == self.sfreq
        self.n_delays = int((self.tmax - self.tmin) * self.sfreq) + 1
        
        # Prepare result containers
        self.hparam_keys = list(hparam_grid.keys())
        self.best_hparams_ = np.zeros((n_outer_folds, len(self.hparam_keys)))
        
    def _prepare_pipeline(self):
        trf_estimator = ReceptiveField(self.tmin, self.tmax, self.sfreq,
                                       feature_names=self.feature_names,
                                       estimator=None, scoring="corrcoef",
                                       n_jobs=1, verbose=False)
        
        steps = []
        
        if self.residualize_features:
            residualize_features = [self.feature_names.index(feature)
                                    for feature in self.residualize_features]
            residualizer = Residualizer(residualize_features,
                                        use_base_features=False,
                                        sparse=True)
            steps.append(("residualizer", residualizer))
            
        steps += [
            ("scaler", StandardScaler()),
            ("trf", trf_estimator)
        ]
        
        pipeline = Pipeline(steps)
        
        # Scale Ys.
        pipeline = TransformedTargetRegressor(regressor=pipeline,
                                              transformer=StandardScaler())

        return pipeline
        
    def fit(self, X, Y):
        # Subset to channels of interest.
        if self.fit_channels is not None:
            self.fit_channels_ = np.array(
                [self.mne_info.ch_names.index(identifier) if isinstance(identifier, str) else identifier
                 for identifier in self.fit_channels])
            
            Y = Y[:, self.fit_channels_]
        else:
            self.fit_channels_ = np.arange(Y.shape[1])
        
        self.n_channels_ = Y.shape[1]
        self.n_features_ = len(self.feature_names)
        assert self.n_features_ == X.shape[1]
        
        # Preprocess and convert inputs and outputs.
        if self.downsample_factor:
            X = downsample_data(X, self.downsample_factor)
            Y = downsample_data(Y, self.downsample_factor)
        
        pipeline = self._prepare_pipeline()
        
        outer_cv = KFold(self.n_outer_folds, shuffle=False).split(Y)
        fold_results: List[Tuple[GridSearchCV, np.ndarray]] = []
        for i_split, (train, test) in enumerate(outer_cv):
            hparam_search = GridSearchCV(estimator=pipeline,
                                         param_grid=self.hparam_grid,
                                         cv=self.n_inner_folds,
                                         scoring=mean_corr_score,
                                         n_jobs=30,
                                         refit=True)
            fold_results.append((hparam_search.fit(X[train], Y[train]), test))

        #########
        # Prepare result containers.
        self.best_hparams_ = np.zeros((self.n_outer_folds, len(self.hparam_keys)))
        self.coefs_ = np.zeros((self.n_outer_folds, self.n_channels_, 
                                self.n_features_, self.n_delays))
        self.scores_ = np.zeros((self.n_outer_folds, self.n_channels_))
        
        for i_split, (hparam_search, test_indices) in enumerate(fold_results):
            est = hparam_search.best_estimator_
        
            rf = est.regressor_.named_steps["trf"]
            self.delays_ = rf.delays_
            self.coefs_[i_split, :, :, :] = rf.coef_
            
            self.scores_[i_split] = est.score(np.array(X[test_indices]), np.array(Y[test_indices]))
            self.best_hparams_[i_split] = np.array(
                [hparam_search.best_params_[key]
                 for key in self.hparam_keys])
            
    def to_pandas(self):
        if not hasattr(self, "scores_"):
            raise ValueError("Call `fit` first.")
            
        df_data = []

        for i, split in enumerate(self.coefs_):
            for j, (channel, channel_idx) in enumerate(zip(split, self.fit_channels_)):
                channel_name = self.mne_info.ch_names[channel_idx]
                for k, feature in enumerate(channel):
                    for l, delay in enumerate(feature):
                        df_data.append((i, channel_name, self.feature_names[k], l, self.coefs_[i, j, k, l]))
        df = pd.DataFrame(df_data, columns=["split", "sensor", "feature", "trf_sample", "coef"])
        df["epoch_time"] = df.trf_sample.map(dict(enumerate(self.delays_ / self.sfreq)))
        df["subject"] = self.name

        return df