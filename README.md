# feature_search
This project is to be used for classification problems to enable one to find  the most optimum features to use hyperparameters.

-This project carries out feature selection methods such as variance threshold, selectKbest and  Recursive feature elimination with cross validation so as to attain the best features to use ( minimizing overfitting ) This project is to be used for classification problems.

- After feature selection has taken place, hyperparameter optimization comes next. Random Forest Classifier and Gradient boosting are the models that are used.
- Rfc is used due to its robustness to overfitting due to the multiple Decision trees created.

- for the hyperparameter optimization, Bayesian optimization is used so as to reduce training time and precision as opposed to Grid search, manual search and random search

- A dict containing the two models is then created, and the best hyperparameters are to be input for model stacking to take place.
- Note: The stacking classifier to be used is gbc ( Gradient boosting ) 

- A classification report is then created, enabling on eot see the Recall, precision, accuracy and F1 score of their model 

