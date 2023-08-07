# poly_regression
polynomial regression example - Regression model (Supervised Machine Learning )
# README

This code implements polynomial regression to predict salaries based on position levels. It uses the scikit-learn library for regression analysis and data preprocessing. Here's a breakdown of the code and its functionality:

1. Importing the necessary libraries:
   - `numpy` for numerical operations
   - `matplotlib.pyplot` for data visualization
   - `pandas` for data manipulation and analysis
   - `train_test_split` from `sklearn.model_selection` to split the dataset into training and testing sets
   - `LinearRegression` from `sklearn.linear_model` to create a linear regression model
   - `StandardScaler` from `sklearn.preprocessing` for feature scaling
   - `PolynomialFeatures` from `sklearn.preprocessing` to create polynomial features
   - `LogisticRegression` from `sklearn.linear_model` for logistic regression (currently commented out)
   - `ListedColormap` from `matplotlib.colors` for colormap customization

2. Loading the dataset:
   - The code reads the CSV file named "Position_Salaries.csv" using `pd.read_csv()` from the pandas library.
   - The dataset is assumed to have three columns: position level (input) and salary (output).

3. Splitting the dataset:
   - The code splits the dataset into training and testing sets using `train_test_split()`.
   - It assigns 70% of the data to the training set and 30% to the testing set.

4. Feature Scaling:
   - The code uses `StandardScaler()` to scale the training and testing sets of the input variables.

5. Fitting the polynomial regression to the dataset:
   - The code uses `PolynomialFeatures()` to transform the input variables into polynomial features.
   - It fits the transformed features and the output variable to a linear regression model using `fit()`.

6. Visualizing the polynomial regression:
   - The code creates a scatter plot of the original data points using `plt.scatter()`.
   - It plots the polynomial regression curve using `plt.plot()` with the transformed input variables.
   - The x-axis represents the position level, and the y-axis represents the salary.

Note: Make sure to update the file path in `dataset=pd.read_csv("Position_Salaries.csv")` to the correct location of your "Position_Salaries.csv" file.

This code provides a basic implementation of polynomial regression for salary prediction based on position levels. Feel free to modify and build upon it for your specific requirements.
