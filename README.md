# Stock Price Prediction

# Prepare data for predict
First you have to download data file and put it in repo's root folder. The model will use data as history data to predict 2022 prices.
[Google Drive Link Download](https://drive.google.com/file/d/1--oClO1sXXiyE9LSsJLJbYeNRqIfWgDO/view?usp=sharing)

You can look my training steps is in training.ipynb.
For testing my app just run
```
python3 api.py
python3 app.py
```
Look at the app in http://localhost:8501/
Type in your symbol and choose top N symbol then predict

![App](./app.png)

The result is 2 table of the chosen symbol prices and top N grow rate prices in 2022
![result](./result.png)
