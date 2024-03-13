# OASIS-INFOBYTE-car-price-prediction
Car Price Prediction
This project aims to predict the selling price of cars based on various features using machine learning techniques. It utilizes a dataset containing information about cars, including their selling price, present price, year of purchase, kilometers driven, fuel type, selling type, transmission type, and number of previous owners.

Requirements
To run this project, you need:

Python 3.x
Libraries: pandas, scikit-learn, seaborn, matplotlib
Usage
Clone the repository:

bash
Copy code
git clone <repository_url>
Install the required libraries:

bash
Copy code
pip install -r requirements.txt
Run the car_price_prediction.py script:

bash
Copy code
python car_price_prediction.py
Follow the prompts to input data or adjust parameters as necessary.

Files
car_price_prediction.py: Main Python script containing the code for data preprocessing, model training, and evaluation.
car data.csv: CSV file containing the dataset used for training and testing the model.
README.md: This file providing information about the project.
Model Training
The project utilizes a Random Forest Regressor to predict car prices. The model is trained on the provided dataset using the following steps:

Data preprocessing, including handling missing values, feature engineering, and encoding categorical variables.
Feature importance analysis using Extra Trees Regressor.
Splitting the dataset into training and testing sets.
Hyperparameter tuning using RandomizedSearchCV to optimize the Random Forest Regressor.
Evaluating the model performance using various metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
Results
The model achieves reasonable performance in predicting car prices based on the provided features. Evaluation metrics such as MAE, MSE, and RMSE are calculated and displayed to assess the model's performance.
