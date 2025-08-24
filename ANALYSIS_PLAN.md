# TO-DO

* make notebooks with all the logic
* modularize the code in a pipeline
* unit-tests with pytest
* mlflow
* typing
* shap


# Production-Ready Causal HTE Analysis Plan

This document outlines a structured approach for conducting a Heterogeneous Treatment Effect (HTE) analysis using DoWhy and EconML, transitioning from initial exploration to a reproducible, production-ready pipeline.

## 1. Recommended Project Structure

A well-organized folder structure is crucial for maintainability and reproducibility.

```
icecream-causal/
├── .gitignore
├── ANALYSIS_PLAN.md      # This file
├── config/
│   └── analysis_config.yaml  # Parameters, file paths, model settings
├── data/
│   ├── ice_cream_sales.csv # Raw data
│   └── ...
├── notebooks/
│   ├── 01-EDA.ipynb        # (As-is) Initial exploration
│   ├── 02-Causal-DAG.ipynb # (As-is) Causal graph definition and exploration
│   └── 03-Analysis-Orchestration.ipynb # New: High-level notebook to run analysis & view results
├── results/
│   ├── plots/              # Output plots (e.g., HTE distributions)
│   └── tables/             # Output tables (e.g., summary statistics)
├── src/
│   ├── __init__.py
│   ├── data_processing.py  # Data loading, cleaning, feature engineering
│   ├── causal_modeling.py  # DoWhy model definition and refutation logic
│   ├── hte_estimation.py   # EconML model training and effect estimation
│   └── utils.py            # Helper functions (e.g., plotting, saving results)
├── tests/
│   ├── __init__.py
│   └── test_data_processing.py # Unit tests for data processing logic
├── main.py                 # Main script to run the end-to-end pipeline
├── pyproject.toml
├── requirements.txt
└── ...
```

## 2. Step-by-Step Analysis Workflow

### Step A: Configuration (`config/analysis_config.yaml`)

Centralize all parameters to avoid hardcoding them in scripts or notebooks.

**Example `analysis_config.yaml`:**
```yaml
data:
  raw_path: "data/ice_cream_sales.csv"
  processed_path: "data/ice_cream_sales_processed.csv"

causal_assumptions:
  treatment: "Price"
  outcome: "Sales"
  common_causes: ["Temperature", "Weekend"]
  effect_modifiers: ["Temperature", "Weekend", "Location"]
  instrumental_variables: ["Promotion"] # If applicable

estimation_params:
  estimator: "CausalForestDML"
  econml_params:
    n_estimators: 100
    min_samples_leaf: 5

output:
  plot_path: "results/plots/"
  table_path: "results/tables/"
```

### Step B: Data Processing (`src/data_processing.py`)

Create a robust script to handle data loading and preparation.

- **Function:** `load_and_process_data(config)`
- **Actions:**
    1. Load data from `config['data']['raw_path']`.
    2. Perform cleaning steps identified in `01-EDA.ipynb` (e.g., handle missing values, correct data types).
    3. Engineer features if necessary.
    4. Save the processed data to `config['data']['processed_path']`.
    5. Return the cleaned DataFrame.

### Step C: Causal Modeling (`src/causal_modeling.py`)

Formalize the causal graph and assumptions using DoWhy.

- **Function:** `build_causal_model(data, config)`
- **Actions:**
    1. Extract variable names (`treatment`, `outcome`, `common_causes`, etc.) from the `config`.
    2. Construct the causal graph string (`digraph G{...}`).
    3. Instantiate `dowhy.CausalModel` with the data, graph, and variable names.
    4. Return the `CausalModel` object.

### Step D: HTE Estimation (`src/hte_estimation.py`)

Isolate the EconML estimation logic.

- **Function:** `estimate_heterogeneous_effects(model, data, config)`
- **Actions:**
    1. Identify the causal effect using `model.identify_effect()`.
    2. Select the EconML estimator specified in `config['estimation_params']['estimator']`.
    3. Instantiate the estimator (e.g., `econml.dml.CausalForestDML`) with parameters from the config.
    4. Fit the estimator: `estimator.fit(Y, T, X, W)`.
    5. Return the fitted estimator object.

### Step E: Pipeline Orchestration (`main.py`)

Create a master script to run the entire analysis from start to finish. This is your "production" script.

- **Logic:**
    1. Load the `analysis_config.yaml`.
    2. Call `load_and_process_data()` to get the data.
    3. Call `build_causal_model()` to define the causal structure.
    4. Call `estimate_heterogeneous_effects()` to train the HTE model.
    5. Use the fitted estimator to get conditional average treatment effects (CATEs): `estimator.const_marginal_effect(X)`.
    6. Generate and save summary tables and plots (e.g., distribution of CATEs) to the `results/` directory using helper functions from `src/utils.py`.

### Step F: Interactive Analysis (`notebooks/03-Analysis-Orchestration.ipynb`)

Use a new, clean notebook to orchestrate the analysis and visualize the results in an interactive way. This notebook should import and use the functions from the `src/` directory, not repeat their logic.

- **Cells:**
    1. **Setup:** Load config and import necessary functions from `src`.
    2. **Run Pipeline:** Execute the core functions (`load_and_process_data`, `build_causal_model`, `estimate_heterogeneous_effects`).
    3. **Interpret Results:**
        - Get CATEs for the dataset.
        - Use EconML's interpretation tools, like `estimator.cate_interpreter_`.
        - Plot feature importances for HTEs.
        - Visualize the CATE distribution (histogram, boxplots).
        - Plot CATEs against key features (e.g., "How does the price effect change with temperature?").
    4. **Refutation (Optional but Recommended):** Use the `CausalModel` object to run refutation tests (e.g., placebo treatment, random common cause) to validate the model's robustness.

By following this structure, you create a clear, maintainable, and reproducible causal analysis pipeline that is easy to update, validate, and share.
