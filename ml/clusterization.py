from sklearn.cluster import Birch, SpectralClustering
from typing import Any, Callable, Dict, List, Literal, Optional, Union
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
import plotly.graph_objects as go


def find_best_n(
    X: Union[np.ndarray, pd.DataFrame],
    n_range: List[int],
    score_method: Callable,
    model: Literal['kmeans', 'gaussian', 'agglomerative'] = 'kmeans',
    model_params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
    verbose: int = 1
) -> Union[pd.DataFrame, tuple]:
    """
    Find optimal number of clusters using specified clustering model and scoring method.

    Parameters
    ----------
    X : array-like or DataFrame
        Input data for clustering
    n_range : List[int]
        Range of cluster numbers to evaluate
    score_method : Callable
        Scoring function with signature (X, labels) -> float
    model : str, default='kmeans'
        Clustering model: 'kmeans', 'gaussian', 'agglomerative', 'dbscan', 'birch', 'spectral'
    model_params : Optional[Dict[str, Any]], default=None
        Additional parameters for the clustering model
    random_state : int, default=42
        Random seed for reproducibility
    verbose : int, default=1
        0: no output, 1: plot

    Returns
    -------
    pd.DataFrame or tuple
        DataFrame with scores or (DataFrame, list of fitted models)
    """

    # Validate inputs
    if not isinstance(n_range, (list, range, np.ndarray)):
        raise TypeError("n_range must be list, range or numpy array")

    if len(n_range) == 0:
        raise ValueError("n_range must contain at least one value")

    if not all(isinstance(n, (int, np.integer)) and n > 0 for n in n_range):
        raise ValueError("All values in n_range must be positive integers")

    if not callable(score_method):
        raise TypeError("score_method must be callable")

    model_params = model_params or dict()

    # Storage for results
    n_scores = {
        "n": [],
        "score": [],
    }

    for n in n_range:
        try:
            # Model initialization with error handling
            if model == 'kmeans':
                m = KMeans(n_clusters=n, init='k-means++', **model_params, random_state=random_state)
            elif model == 'gaussian':
                m = GaussianMixture(n_components=n, **model_params, random_state=random_state)
            elif model == 'agglomerative':
                m = AgglomerativeClustering(n_clusters=n, **model_params)
            elif model == 'dbscan':
                # For DBSCAN, n is used as eps parameter
                m = DBSCAN(eps=n, **model_params)
            elif model == 'birch':
                m = Birch(n_clusters=n, **model_params)
            elif model == 'spectral':
                m = SpectralClustering(n_clusters=n, **model_params, random_state=random_state)
            else:
                raise ValueError(f'Unknown model: {model}. Available: kmeans, gaussian, agglomerative, dbscan, birch, spectral')

            # Fit and predict
            labels = m.fit_predict(X)
            if model == 'dbscan':
                # For DBSCAN, count actual clusters (excluding noise = -1)
                n_actual_clusters = len(np.unique(labels[labels != -1]))
            else:
                n_actual_clusters = n

            # Calculate score with error handling
            try:
                score = score_method(X, labels)
            except Exception as e:
                if verbose > 1:
                    print(f"Warning: Score calculation failed for n={n}: {e}")
                score = np.nan

            # Store results
            n_scores["n"].append(n_actual_clusters)
            n_scores["score"].append(score)

        except Exception as e:
            if verbose > 0:
                print(f"Error for n={n}: {e}")
            n_scores["n"].append(n)
            n_scores["score"].append(np.nan)

    # Create DataFrame
    results_df = pd.DataFrame(n_scores)
    results_df.dropna(subset=['score'], inplace=True)

    # Visualization
    if verbose > 0 and not results_df['score'].isna().all():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results_df['n'], y=results_df['score']))

        fig.update_layout(
            title=f'Cluster Evaluation: {score_method.__name__}<br>Model: {model}',
            title_x=0.5,
            xaxis_title_text='n', yaxis_title_text='Score',
            xaxis=dict(tickmode='linear', dtick=1),
            width=800, height=500,
            margin=dict(t=80, r=20, l=20, b=20),
            showlegend=False,
        )

        if verbose > 1:
            fig.show()

    # Return results
    return results_df
