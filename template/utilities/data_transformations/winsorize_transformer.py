import warnings

import numpy as np
import pandas as pd
import scipy
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    _OneToOneFeatureMixin,
    _ClassNamePrefixFeaturesOutMixin,
)
from sklearn.utils.validation import (
    check_is_fitted,
    check_random_state,
    _check_sample_weight,
    FLOAT_DTYPES,
)


class WinsorizationTransformer(_OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """
    Winsorize features by replacing values outside the specified percentiles with those percentiles.

    The percentiles are calculated independently on each feature in the training set. 
    The upper and lower percentiles are then stored to be used on later data using
    :meth:`transform`.

    Winsorization of a dataset is a common approach for dealing with outliers. 
    Dealing with outliers is a requirement for a number of machine learning
    estimators: they might behave badly if the outliers heavily influence the training process.

    Parameters
    ----------
    copy : bool, default=True
        If False, try to avoid a copy and do inplace winsorization instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.

    lower_lim : float, default=None
        The lower cut off for winsorization - this is a value between 0 and 1. E.g. 0.05 would
        correspond to a lower cut off at the 5th percentile.

    upper_lim : float, default=None
        The upper cut off for winsorization - this is a value between 0 and 1. E.g. 0.05 would
        correspond to a upper cut off at the 95th percentile.

    Attributes
    ----------
    lower_percentile_ : ndarray of shape (n_features,) or None
        Per feature relative scaling of the data to achieve zero mean and unit
        variance. Generally this is calculated using `np.sqrt(var_)`. If a
        variance is zero, we can't achieve unit variance, and the data is left
        as-is, giving a scaling factor of 1. `scale_` is equal to `None`
        when `with_std=False`.

    upper_percentile_ : ndarray of shape (n_features,) or None
        The mean value for each feature in the training set.
        Equal to ``None`` when ``with_mean=False``.

    var_ : ndarray of shape (n_features,) or None
        The variance for each feature in the training set. Used to compute
        `scale_`. Equal to ``None`` when ``with_std=False``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_samples_seen_ : int or ndarray of shape (n_features,)
        The number of samples processed by the estimator for each feature.
        If there are no missing samples, the ``n_samples_seen`` will be an
        integer, otherwise it will be an array of dtype int. If
        `sample_weights` are used it will be a float (if no missing data)
        or an array of dtype float that sums the weights seen so far.
        Will be reset on new calls to fit, but increments across
        ``partial_fit`` calls.

    See Also
    --------
    scale : Equivalent function without the estimator API.

    :class:`~sklearn.decomposition.PCA` : Further removes the linear
        correlation across features with 'whiten=True'.

    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform.

    We use a biased estimator for the standard deviation, equivalent to
    `numpy.std(x, ddof=0)`. Note that the choice of `ddof` is unlikely to
    affect model performance.

    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

    Examples
    --------
    >>> from data_transformations.winsorize_transformer import WinsorizationTransformer
    >>> data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    >>> wins = WinsorizationTransformer()
    >>> print(wins.fit(data))
    WinsorizationTransformer()
    >>> print(scaler.lower_percentile)
    [0.5 0.5]
    >>> print(scaler.transform(data))
    [[-1. -1.]
     [-1. -1.]
     [ 1.  1.]
     [ 1.  1.]]
    >>> print(scaler.transform([[2, 2]]))
    [[3. 3.]]
    """
    
    def __init__(self, *, copy=True, lower_lim=None, upper_lim=None, 
                 incl_lower=True, incl_upper=True, nan_policy='propagate'):
        self.lower_lim = lower_lim
        self.upper_lim = upper_lim
        self.copy = copy
        self.nan_policy = nan_policy
        self.incl_lower = incl_lower
        self.incl_upper = incl_upper
        
    def _reset(self):
        """Reset internal data-dependent state of the winsorization transformer, if necessary.
        __init__ parameters are not touched.
        """
        # Upper and lower percentiles are set independently and thus both need to be checked
        if hasattr(self, "lower_percentile_"):
            del self.lower_percentile_
        if hasattr(self, "upper_percentile_"):
            del self.upper_percentile_
            
            
    def fit(self, X, y=None):
        """Compute the upper and lower percentiles to be used for later winsorization.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the upper and lower percentiles
            used for later winsorization along the features axis.
        y : None
            Ignored.
        Returns
        -------
        self : object
            Fitted winsoration transformer.
        """
        # Reset internal state before fitting
        self._reset()
        try: 
            n_features = X.shape[1]
        except IndexError: 
            n_features = 1 
            
        X = self._validate_data(
            X,
            accept_sparse=("csr", "csc"),
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan"
        )
        
        if n_features == 1:
            self._winsorize1D_fit(X.ravel())
        else:
            np.ma.apply_along_axis(self._winsorize1D_fit, axis=0, arr=X)
           
        if self.upper_lim is None:
            self.upper_percentile_ = np.empty(n_features, dtype=object)
        if self.lower_lim is None:
            self.lower_percentile_ = np.empty(n_features, dtype=object)
                 
        return self
    
    def _winsorize1D_fit(self, a):
        n = a.shape[0] 
        idx = a.argsort()
          
        contains_nan = scipy.stats.stats._contains_nan(a, self.nan_policy)

        if contains_nan[0]:
            nan_count = np.count_nonzero(np.isnan(a))
            self._general_warning("NaN values found in input array. "
                          "Default behaviour is to propagate NaN values")
            
        if self.lower_lim is not None:
            if self.incl_lower:
                low_idx = int(self.lower_lim * n)
            else:
                low_idx = np.round(self.lower_lim * n).astype(int)
            if contains_nan and self.nan_policy == 'omit':
                low_idx = min(low_idx, n-nan_count-1)
            if hasattr(self, "lower_percentile_"):
                self.lower_percentile_ = np.append(self.lower_percentile_, a[idx[low_idx]])
            else:
                self.lower_percentile_ = a[idx[low_idx]]
                
        if self.upper_lim is not None:
            if self.incl_upper:
                up_idx = n - int(n * self.upper_lim)
            else:
                up_idx = n - np.round(n * self.upper_lim).astype(int)
            if hasattr(self, "upper_percentile_"):
                self.upper_percentile_ = np.append(self.upper_percentile_, a[idx[up_idx - 1]])
            else:
                self.upper_percentile_ = a[idx[up_idx - 1]]
                
    def transform(self, X, copy=None, axis=0):
        """Perform winsorization using fitted percentile values.
        Parameters
        ----------
        X : {array-like, sparse matrix of shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, default=None
            Copy the input X or not.
        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        check_is_fitted(self)
        n_features = X.shape[1]

        copy = copy if copy is not None else self.copy
        X = self._validate_data(
            X,
            reset=False,
            accept_sparse="csr",
            copy=copy,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )
        
        if axis is None or n_features == 1:
            X = self._winsorize1D_transform(X.ravel(), 
                                            lower_val=self.lower_percentile_.ravel()[0],
                                            upper_val=self.upper_percentile_.ravel()[0])
        else:
            for i, (lower_val, upper_val) in enumerate(zip(self.lower_percentile_, self.upper_percentile_)):
                X[:, i] = self._winsorize1D_transform(X[:, i], copy=copy, 
                                                      lower_val=lower_val, upper_val=upper_val)

        return X
       
    def _winsorize1D_transform(self, a, copy=None, lower_val=None, upper_val=None):
        """Perform standardization by centering and scaling.
        Parameters
        ----------
        X : {array-like, sparse matrix of shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, default=None
            Copy the input X or not.
        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
    
        if self.lower_lim is not None:
            a[a < lower_val] = lower_val
            
        if self.upper_lim is not None:
            a[a > upper_val] = upper_val
            
        return a 
    
    def _general_warning(self, message):
        warnings.warn(message, Warning, stacklevel=2)
        