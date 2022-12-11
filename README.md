# FRE7773 Final Project Group6

## Credit Card Approval Prediction

`requirements.txt` holds the Python dependencies for running the notebook and all the python scripts. 
The `dataset` folder contains two dataset csv files that are needed for running scripts. 


### Notebooks

* The `notebooks` folder contains the jupyter notebook `creditcard_approval.ipynb`, which shows our steps and thinking procedures in building the optimized final model. It also demonstrates how we analyzed the dataset, performed feature engineering, and how we changed among several different types of models. 


### Metaflow

* The `app` folder contains `app.py` and `flow.py`. 
* `app.py` has a flask app that retrieve the model artifacts produced by the latest run from `flow.py`. As an interactive user interface, it  will respond to requests such as: URL/predict?DAYS_EMPLOYED=-3000. DAYS_EMPLOYED can be replaced with AMT_INCOME_TOTAL, DAYS_BIRTH, etc.
* `flow.py` contains the full pipeline for predicting whether a person can be approved for credit card in metaflow format. 
* Run `flow.py` first, install the `requirements.txt` in your virtual environment, and then flask run the `app.py`.

* `comet.py` has the same content as `flow.py` except it has additional lines for logging the experiment in Comet. 




