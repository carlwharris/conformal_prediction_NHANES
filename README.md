# Uncertainty quantification in waist circumference estimation using conformal prediction

### System requirements/installation
Built using Python 3.11.15, package requirements in `requirements.txt`. Should be installed in <5 minutes.

Requirements can be installed in a conda environment via:
```
pip install -r requirements.txt
```

Tested on Apple M2 Max laptop (Sonoma 14.0 OS) with 32 GB RAM. There is no required non-standard hardware, nor GPU requirement.

### Instructions
To reproduce the primary analysis for the Look AHEAD cohort, run `main_results.ipynb` (setting the file path for the source data to `data.csv`) without running cell [8]. To reproduce the resampling analysis, run the first part of `check_coverage_resampling.ipynb`.

For the NHANES results, the corresponding notebooks are in `main_results.ipynb` and *including* cell [8] and the latter part of `check_coverage_resampling.ipynb`. Unfortunately, due to data use restrictions, we are unable to release the source data for those analyses (though the notebooks have not been cleared, which should provide sufficient context to understand the analyses that were run.)

For point prediction models, the corresponding notebook is `XG_boost_v2.ipynb`.

The expected output is shown in the notebooks.

## Data
Data for the NHANES dataset is included in this repo, `data.csv`. To run the code on separate data, follow the exmplate in `results_LOOK_AHEAD.ipynb`. Specifically, the `CP` (conformal prediction class, located in `conformal_prediction.py`) will be useful -- important methods are `hyperparam_search`, `train`, `calc_qhat` and `predict_cp_quantiles`.