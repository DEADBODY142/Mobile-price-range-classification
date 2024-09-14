# Readme

## Data information
- Information about the question is stored in `info.txt`.
- Site through which it is pulled it `https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification/data`.
- The train and test dataset are separated in two different files namely `train.csv` and `test.csv` respectively.

## Beginning of process
- The main file where model is trained is in `attempt.ipynb`.
- I have used four ml models `Decision Tree`,`KNN`,`Logistic Regression` and `Random Forest`.
- I also had perfomed cross validation on all models except KNN since they got 100% train accuracy which may be due to overfitting.
- Finally, `logistic regression` had the highest accuracy so we chose it.
- The predicted price of test dataset was saved in `output.csv`.
- `loading.py` is generated through AI to get to know how things is done and it's output is saved in `predictions.csv`.