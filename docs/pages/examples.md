# Examples

Explore real-world applications of Sklearn-Optuna through interactive examples. Each notebook is available in two formats: **View** for static reading, and **Editable** for an interactive WASM notebook that runs entirely in your browser (no server needed).

## Getting Started

### OptunaSearchCV Quickstart ([View](/examples/quickstart/) | [Editable](/examples/quickstart/edit/))

Learn how to run a fast hyperparameter search and read the best parameters and score. This example shows the fastest path from data to optimized model using familiar Scikit-Learn patterns.

This walkthrough uses a compact search space and a small number of trials to keep results quick and readable.

### Study Management and Reproducibility ([View](/examples/study_management/) | [Editable](/examples/study_management/edit/))

Reuse an Optuna study to continue optimization runs and keep experiments reproducible with seeded samplers. This notebook demonstrates how to resume trials and compare total trial counts.

This example focuses on study reuse and reproducibility without changing the core Scikit-Learn API flow.

### Optuna Visualizations ([View](/examples/visualization/) | [Editable](/examples/visualization/edit/))

Turn completed studies into visual summaries of optimization history and parameter importance. This helps explain how the search progressed and which hyperparameters mattered most.

The notebook uses `study_` to generate interactive plots that make diagnostics easy to interpret.

## Advanced Patterns

### Early Stopping with Callbacks ([View](/examples/callbacks/) | [Editable](/examples/callbacks/edit/))

Stop unneeded work early by adding Optuna callbacks to your search. This example shows how to integrate a max-trials callback for quick experimentation.

It demonstrates how to pass a dictionary of callbacks through `OptunaSearchCV` without changing your estimator code.

### Nested OptunaSearchCV in Pipelines ([View](/examples/nested_pipeline/) | [Editable](/examples/nested_pipeline/edit/))

Discover advanced nested optimization patterns where `OptunaSearchCV` serves as a final estimator in a pipeline tuned by another `OptunaSearchCV`. This example shows how to tune preprocessing choices and sampler parameters simultaneously.

This walkthrough demonstrates tuning sampler parameters like `n_startup_trials` using nested parameter syntax and highlights the computational trade-offs of nested searches.

### Metadata Routing with sample_weight ([View](/examples/metadata_routing/) | [Editable](/examples/metadata_routing/edit/))

Handle imbalanced datasets by routing `sample_weight` through `OptunaSearchCV` to both model fitting and scoring. This example shows how to configure metadata routing for estimators, scorers, and pipelines.

The notebook demonstrates multi-metric scoring with different routing preferences and proper handling of metadata in pipeline components.

## Running Examples Locally

All examples are [marimo](https://marimo.io/) notebooks. Open any example interactively:

```bash
# Via just (pass the filename without .py)
just example quickstart
just example advanced_spaces
just example callbacks

# Or directly with marimo
uv run marimo edit examples/quickstart.py
```

## Next Steps

- Browse the [API Reference](api-reference.md) for detailed documentation
- Check the [User Guide](user-guide.md) to understand core concepts
