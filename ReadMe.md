# Dash User Data Analysis Application
This is a Python Dash application that allows users to analyze their own data by reading csv, excel, and txt files. The application has three main sections: Variable Descriptions, Descriptive Analytics, and Predictive Analytics. 

<span style="color:red"><strong> Please see caveats at the end! </strong></span>

### Variable Descriptions
This section provides information on the characteristics of the variables in the user's data. The user can view: the number of unique values in a column, the Percentage of null values, the data type, and the Mode/Median value of each variable.
<ol>
  <li>Number of unique values in a column</li>
  <li>Percentage of null values</li>
  <li>Data Type</li>
  <li>Mode/Median value in selected column</li>
</ol>

### Descriptive Analytics
This section allows the user to select columns from their data to create:
<ol>
  <li><strong>Scatter Plots</strong>: A graph displaying the relationship between two variables, represented by dots placed on the x and y axes, which helps to visualize patterns in data.</li>
  <li><strong>Distribution Plots</strong>: A visual representation of the distribution of a single variable, showing how frequently the values occur, which helps to identify patterns, such as skewness, kurtosis, and outliers.</li>
  <li><strong>Feature Importance Plot</strong>: A representation of the importance of each feature (predictor variable) in a Machine Learning model, which helps to determine which features are the most influential in making accurate predictions.</li>
</ol>

### Predictive Analytics
This section allows the user to run regression and classification models on their data. The models available include Multilinear Regression, Random Forest Regression, K Nearest Neighbors Classification, and Random Forest Classification:
<ol>
  <li><strong>(Multi) Linear Regression</strong>: A statistical model used for predicting a continuous dependent variable based on an independent variable(s), with the assumption that the relationship between the features and target variable(s) is linear.</li>
  <li><strong>Random Forest (Regressor & Classification): </strong>: an ensemble machine learning model that uses multiple Decision Trees to make predictions by leveraging a random sample for each Decision Tree and averaging the predictions across trees.</li>
  <li><strong>K Nearest Neighbors</strong>: A non-parametric algorithm (does not assume any distribution of data) that finds the "K" closest data points to a given unseen sample and making predictions based on the majority class .</li>
</ol>


### Caveats:

<ol>
  <li>This application is still under development.</li>
  <li>The Machine Learning models are built with a priority on application performance; they are simple models to better understand your data. These models are not highly optimized or tuned. 
    <ol>
      <li>KNN (Classification): Trained on a value of K between 1 and 10. Chooses best K based on accuracy and uses that model for prediction.</li>
    </ol>
  </li>
  <li>Automatically imputes mean values for missing continuous features and mode for discreet features.</li>
  <li>Automatically drops any features with high cardinality (more than 4 distinct values) to increase efficiency of preprocessing and reduce the impact of high dimensionality.</li>
  <li>Still working through some bugs that have been displayed since production, one of which includes the inability to inverse Label Encoders for Boolean values.</li>
  <li>This was a project made for fun to test the waters of the (free version) Dash framework (https://plotly.com/), and I didn't take it too seriously, so neither should you! </li>
</ol>


### Deployed using Dash Render: https://crdash-app.onrender.com


