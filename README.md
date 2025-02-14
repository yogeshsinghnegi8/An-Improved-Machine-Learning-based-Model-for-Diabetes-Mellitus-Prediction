# An-Improved-Machine-Learning-based-Model-for-Diabetes-Mellitus-Prediction

Diabetes Prediction Using Machine Learning

Millions of people worldwide suffer from diabetes, a chronic metabolic disease that, if left untreated, may cause serious health problems. Proactive healthcare measures may be greatly aided by early diabetes prediction. The goal of this project is to create an effective machine learning-based prediction model that leverages clinical data to correctly categorize people as either diabetes or non-diabetic.

Dataset Collection & Preprocessing

The research makes use of the Pima Indians Diabetes Dataset (PIDD), which covers essential physiological and biochemical characteristics associated with diabetes diagnosis. Data pretreatment processes include addressing missing values by imputing median (for numerical features) and mode (for categorical characteristics), feature scaling with MinMaxScaler, and feature selection with SelectKBest (k=7). To overcome class imbalance, the SMOTETomek hybrid resampling approach is used for balanced model training.

Tools & Implementation

Python is used to carry out the study, with Jupyter Notebook and Google Colab serving as execution platforms. Key libraries include scikit-learn, LightGBM, XGBoost, CatBoost, imbalanced-learn, NumPy, and Pandas. To achieve accurate performance assessment, the model is evaluated using stratified K-fold cross-validation (3, 5, and 10 folds).

Machine Learning Models

The research tests many categorization models, including:

  * Traditional classifiers include Decision Trees, K-Nearest Neighbors (KNN), Support Vector Machines (SVMs), Gaussian Naïve Bayes, and 
    Logistic Regression.
  
  * Ensemble-based models include Random Forest, AdaBoost, Gradient Boosting Machine (GBM), Extreme Gradient Boosting (XGBoost), 
    LightGBM, and CatBoost.

RandomizedSearchCV is used to tune hyperparameters, with the goal of improving the best-performing classifiers.

Results & Conclusion

LightGBM with 10-Fold Stratified Cross-Validation outperformed all other models in diabetes prediction, with an accuracy of 93.42%. The results emphasize the importance of feature selection, data balance, and hyperparameter adjustment in enhancing classification performance.

Accurate diabetes prediction models may help healthcare practitioners with early diagnosis, risk assessment, and individualized treatment plans. The suggested technique, which combines machine learning and systematic data preparation, provides a scalable way for incorporating predictive analytics into clinical decision-making.






