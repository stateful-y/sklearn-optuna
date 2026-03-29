# How to Visualize Optimization Results

!!! tip "Interactive notebook"
    See the companion notebook for a runnable example.
    [View](/examples/visualization/) · [Open in marimo](/examples/visualization/edit/)

This guide shows you how to plot optimization history and parameter
relationships from a completed `OptunaSearchCV` search. Use this when you need
to inspect convergence behavior or identify important parameter regions.

## Prerequisites

- sklearn-optuna installed ([Getting Started](../tutorials/getting-started.md))
- A completed `OptunaSearchCV` search
- Plotly installed (`pip install plotly`)

## Access the Study

After calling `fit()`, the Optuna study is available as `search.study_`:

```python
search.fit(X, y)
study = search.study_
```

Pass this study object to any of Optuna's built-in visualization functions.

## Plot Optimization History

Show how the objective value changed over trials:

```python
import optuna

fig = optuna.visualization.plot_optimization_history(study)
fig.show()
```

This helps identify whether the search converged or would benefit from more
trials.

## Plot Parameter Contours

Visualize the relationship between two parameters and the objective:

```python
fig = optuna.visualization.plot_contour(study)
fig.show()
```

Contour plots highlight regions of the parameter space that produce the best
scores.

## Plot Parameter Importances

Rank parameters by their impact on the objective:

```python
fig = optuna.visualization.plot_param_importances(study)
fig.show()
```

Use this to decide which parameters are worth tuning further and which can be
fixed.
