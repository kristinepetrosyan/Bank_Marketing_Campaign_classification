# Helper functions

import matplotlib.pyplot as plt


def sampled(X_train, y_train):
    
    """Returns sampled X_train, y_train by using smote().
    Returns sampled X_test, y_test by using RandomOverSampler().
    Parameters:
    X_train (pd.dataframe): X_train set to be sampled
    X_test (pd.dataframe): X_test set to be sampled
    y_train (pd.series): y_train set to be sampled
    y_test (pd.series): y_test set to be sampled
    Returns:
    X_train (pd.dataframe): sampled X_train
    X_test (pd.dataframe): sampled X_test
    y_train (pd.series): sampled y_train
    y_test (pd.series): sampled y_test
    """
    
    from imblearn.over_sampling import SMOTE
    #from imblearn.over_sampling import RandomOverSampler
    
    smote = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=33)
    #X_train, y_train = smote.fit_sample(X_train, y_train) 
    #X_test, y_test = smote.fit_sample(X_test, y_test) 
    
    #return X_train, X_test, y_train, y_test

    # Create instance of SMOTE
    #smote = SMOTE()

    # Apply smote
    X_train_resampled, y_train_resampled = smote.fit_sample(X_train, y_train)
    return X_train_resampled, y_train_resampled

def print_accuracy_report(X_train, X_test, y_train, y_test, model):
    
    """Takes in  X_train, X_test, y_train, y_test, model and calculates and prints accuracy_score, 
    f1_score, confusion_matrix, classification_report for X_train, X_test, y_train, y_test
    
    Parameters:
    
    X_train (pd.dataframe): 
    X_test (pd.dataframe):
    y_train (pd.series):
    y_test (pd.series):
    model: classifier
    Returns:
    
    Returns accuracy_score, f1_score, confusion_matrix, classification_report for X_train, X_test,
    y_train, y_test
    """
    
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

    training_preds = model.predict(X_train)
    y_pred = model.predict(X_test)
    training_accuracy = accuracy_score(y_train, training_preds)
    val_accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
 
    
    print("\n\nTraining Accuracy: {:.4}%".format(training_accuracy * 100))
    print("Validation Accuracy: {:.4}%".format(val_accuracy * 100))
    print("F1 Score: {:.4}%". format(f1))

    # Classification report
    
    print("\n\n\nClassification Report:")
    print("---------------------")
    print(classification_report(y_test, y_pred))

   # Plot a confusion matrix    

def plot_con_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues): 
    """Returns confusion matrix for y_test and y_pred 
    Parameters: 
    y_test (pd.Series):
    y_pred (pd.series):
    class_names: names of the target variable
    cmap: colormap for the matrix
    Returns:
    
    Returns confusion matrix.
    """
    
    
  
    #Add Normalization Option
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
    
def plot_roc_curve(fpr, tpr):
    
    """ Plots ROC curve.
    
    Parameters:
    fpr: false positive rate
    tpr: true positive rate
    
    Returns:
    Returns ROC curve.
    """
    
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=4, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

def grid_search(X_train, X_test, y_train, y_test, model, model_name, params, cv=3):
    
    """Returns confusion matrix for y_test and y_pred 
    Parameters: 
    y_test (pd.Series):
    y_pred (pd.series):
    class_names: names of the target variable
    cmap: colormap for the matrix
    Returns:
    
    Returns confusion matrix.
    """
    
    from sklearn.model_selection import GridSearchCV
    # Create grid search object
    gridsearch = GridSearchCV(
        model,
        param_grid = params,
        cv = cv,
        return_train_score=True
    )

    # Fit on data
    gridsearch.fit(X_train, y_train)
    
    best_params = gridsearch.best_params_
    print(f'Optimal parameters: {best_params}')

    print('\n')
    print(f"Training Accuracy: {gridsearch.best_score_ :.2%}")

    print('\n')
    print(f"Test Accuracy: {gridsearch.score(X_test, y_test) :.2%}")
    
    import joblib
    model_name = model_name.lower().replace(' ', '_')
    joblib.dump(gridsearch.best_estimator_, f'{model_name}_gridsearch_output_model.pkl', compress = 1)
    print('\n')
    print('Model saved successfully!')

    return best_params
