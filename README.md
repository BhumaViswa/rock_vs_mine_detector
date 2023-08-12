# rock_vs_mine_detector
To detect the difference between rock and mine in submarine
# Rock vs. Mine Classification using Sonar Data

This repository contains Python code for performing rock vs. mine classification using logistic regression on sonar data. The code reads the sonar data from a CSV file, preprocesses it, trains a logistic regression model, and evaluates its performance. It also includes data visualization to understand the dataset.

## Getting Started

These instructions will help you understand and run the code on your local machine.

### Prerequisites

Make sure you have the following software installed:

- Python (version x.y.z)
- Required Python packages: numpy, pandas, matplotlib, scikit-learn

### Installation

##Usage
Place your sonar data CSV file in the repository directory.

Open a terminal and navigate to the repository directory.

##Run the Python script:
python rock_vs_mine_detection for submarine.ipynb

The script will display various visualizations and the accuracy of the logistic regression model.
Customization
You can customize the code to suit your needs:

Modify file paths to match your data location.
Adjust parameters in the train-test split and logistic regression sections.


## BRIEF EXPLINATION OF THE PROJECT STEPS
 This code aims to differentiate between rocks and mines based on some features. Let's go through the code step by step to understand what it's doing:

1. **Importing Libraries:** The code begins by importing the necessary libraries, including NumPy for numerical operations, pandas for data manipulation, and Matplotlib for plotting.

2. **Loading Data:** The code reads a CSV file containing the sonar data using the `pd.read_csv` function. The data is assumed to be in a file named "sonar data.csv". The `header=None` parameter indicates that the data file doesn't have a header row. The data is loaded into a pandas DataFrame (`df`), and the first few rows are displayed using `df.head()`.

3. **Data Exploration:**
   - The shape of the DataFrame is printed using `df.shape`.
   - Null values in the DataFrame are checked using `df.isnull().sum()`.
   - A summary of the data's statistics is obtained with `df.describe()`.
   - The data is grouped by the last column (column index 60) using `df.groupby(60).size()`.

4. **Data Visualization:**
   - Histograms of the data are plotted using `df.hist(...)`, displaying the distribution of each feature.
   - Density plots are created using `df.plot(...)` to visualize the distribution of each feature.

5. **Data Preprocessing:**
   - The feature matrix `X` is extracted from the DataFrame by excluding the last column using `X=df.iloc[:,:-1].values`.
   - The target vector `y` is extracted from the last column using `y=df.iloc[:,-1].values`.
   - The target vector `y` is encoded into binary labels (0 and 1) using Label Encoding from `sklearn.preprocessing.LabelEncoder`.

6. **Data Visualization (Scatter Plots):**
   - For each of the 60 features, scatter plots are created to visualize the relationship between that feature and the target variable (`y`).

7. **Sample Data Preparation:**
   - A sample data point for "rock" is selected from the feature matrix `X[5, :]` and reshaped to match the input shape expected by the model (`sample_for_rock = sample_for_rock.reshape(1, -1)`).

8. **Train-Test Split:**
   - The data is split into training and testing sets using the `train_test_split` function from `sklearn.model_selection`.

9. **Logistic Regression:**
   - A logistic regression model is instantiated using `LogisticRegression()` from `sklearn.linear_model`.
   - The model is trained on the training data using `model.fit(X_train, y_train)`.
   - Predictions are made on the test data using `model.predict(X_test)`.

10. **Accuracy Calculation:**
    - The accuracy of the model's predictions is calculated using `accuracy_score(y_test, y_pred)` from `sklearn.metrics`.

11. **Prediction and Probability:**
    - The `predict_proba` method of the model is used to get the probability estimates for the classes for the `sample_for_rock`.
    - The `predict` method can be used to directly predict the class of the sample.




