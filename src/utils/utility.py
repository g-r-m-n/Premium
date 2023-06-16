# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
import os
import seaborn as sns
from sklearn.metrics import  make_scorer,auc
from matplotlib import pyplot as plt
from lightgbm import  log_evaluation, early_stopping 
from sklearn.model_selection import RandomizedSearchCV
from pandas.api.types import is_numeric_dtype
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import TweedieRegressor
from lightgbm import LGBMRegressor, log_evaluation, early_stopping 
from scipy.stats import   pearsonr

def train_model(X_train, y_train, df, model_type = 'tweedie', TUNE = False, output_folder_model ='', LOAD = 0, X_val = None, y_val = None, Y_TRANSFORM = 'None'):
    """function to tune and train the model"""


    # adjust the predict function to account for possible transformations of y
    def predict_y(self, *args, **kwargs):  
        # passing arbitrary arguments and/or keyword arguments via *args and **kwargs   
        y_pred = self.predict(*args, **kwargs) 
        if self.Y_TRANSFORM == 'log' :
            y_pred = np.exp(y_pred) - 1
            y_pred[y_pred<0] = 0

        return y_pred
    
    # get file name of the trained model:
    file_name_trained_model = output_folder_model + model_type+ '_model.pkl'

    # check if trained model already exists:
    if (os.path.isfile(file_name_trained_model)) and LOAD:
        # load the trained model:        
        #model  = keras.models.load_model(folder_name_trained_model)        
        model = pickle.load(open(file_name_trained_model, "rb"))

        # add transformation type to the model instance:
        model.Y_TRANSFORM = Y_TRANSFORM
        import types
        # add the new method to the model instance:
        model.predict_y = types.MethodType( predict_y, model )       

        return model 

    # transform y
    if Y_TRANSFORM =='log':
        print("\nUsing {Y_TRANSFORM} for y.")
        y_train = np.log(y_train.copy()+1)          

    # initialize optional weights and arguments for the fitting process:
    weights = {}
    fit_args = {}

    # Use an LGBM or RF ML model:    
    if model_type in ['lgbm','rf','lgbm_tw']:
        n_estimators = 50 
        params = {'subsample': 0.5, 'num_leaves': 30, 'max_depth': 10, 'learning_rate': 0.3} #'boosting_type' : 'dart'}
        if model_type == 'rf':
                params =  {'num_leaves': 20, 'max_depth': 5, 'feature_fraction' : 0.8, 'learning_rate': 1, 'boosting_type' : 'rf', 'bagging_freq' : 1, 'subsample_freq' : 1, 'bagging_fraction' : 0.8 } 
                # {'subsample': 0.5, 'num_leaves': 31, 'max_depth': 5, 'learning_rate': 0.01}                
        if model_type == 'lgbm_tw':
                params =  params  | {'objective': 'tweedie', 'metric': 'tweedie'}     

        clf = LGBMRegressor( random_state=42, n_estimators=n_estimators, **params)
        
        tuning_dict = { 
                                'max_depth': [3,  10, 15, -1], #'max_depth': [3, 5, 15, 20, 30],
                                'num_leaves': [5,  20, 30, 40], #'num_leaves': [5, 10, 20, 30],
                                #'subsample': [0.3, 0.5, 1] #'subsample': [0.1, 0.2, 0.8, 1]                  
            }
        if model_type in ['lgbm','lgbm_tw']:
            tuning_dict = tuning_dict | {'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3,],} # 'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        if model_type in ['rf']:
            tuning_dict = tuning_dict | {'feature_fraction' : [0.1, 0.2, 0.5, 0.8, 1]   }
        if model_type in ['lgbm_tw']:
            tuning_dict = tuning_dict | { 'tweedie_variance_power': [ 1.1, 1.2, 1.5, 1.7, 1.8, 1.9],} # constraints: 1.0 <= tweedie_variance_power < 2.0,  see https://lightgbm.readthedocs.io/en/latest/Parameters.html
        
        fit_args = {'eval_metric' : ['neg_mean_absolute_error','neg_root_mean_squared_error'], 
               }  
        if not ((X_val is None) or (y_val is None)) :
            fit_args = fit_args | {'callbacks': [ log_evaluation(n_estimators), early_stopping(2)],
                    'eval_set' : [[X_val, y_val]], }

        weights = {'sample_weight' : df.loc[X_train.index,"Exposure"]}
        
    # Use a tweedie GLM model:    
    if model_type in ['tweedie']:
        params =  {'power': 1.2, 'alpha': 0.1} # {'power': 0.5, 'alpha': 0.05} # {'power': 1.9, 'alpha': 0.1} # link ='auto'

        clf = TweedieRegressor( **params, solver='newton-cholesky')

        weights = {'sample_weight' : df.loc[X_train.index,"Exposure"]}
        
        tuning_dict = { 
                     'power': [0, 1.1, 1.2, 1.5, 1.7, 1.9, 2, 2.5, 3], 
                     'alpha': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3]               
            }


    #create, train and do inference of the model
    if TUNE:
        # Tune hyper-parameters and final model using cv cross-validation with n_iter parameter settings sampled from random search. Random search can cover a larger area of the parameter space with the same number of consider setting compared to e.g. grid search.
        rs = RandomizedSearchCV(clf, tuning_dict, 
            scoring= {'MAE': make_scorer(metrics.mean_absolute_error), 'RMSE':  make_scorer(metrics.mean_squared_error)}, #'f1', 'balanced_accuracy' Overview of scoring parameters: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
                                # default: accuracy_score for classification and sklearn.metrics.r2_score for regression
            refit= 'MAE',
            cv=10, 
            return_train_score=False, 
            n_iter = 40,
            verbose = False,
            random_state = 888
           
        )
        print("\nTuning hyper-parameters ..")
        rs.fit(X_train, y_train,  **weights , **fit_args)    
        
        print("\nTuned hyper-parameters :(best score)     ",rs.best_score_)
        print("\nTuned hyper-parameters :(best parameters) ",rs.best_params_)
        
        model = clf
        clf.set_params(**rs.best_params_)
    else:
        model = clf
        
    # show the model parameters used by the model:    
    print("\nUsed model parameters : ",model.get_params())

    # fit the model:
    model.fit(X_train, y_train, **weights )

    # save the trained model 
    pickle.dump(model, open(file_name_trained_model, "wb"))

    # add transformation type to the model instance:
    model.Y_TRANSFORM = Y_TRANSFORM

    import types
    # add the new method to the model instance:
    model.predict_y = types.MethodType( predict_y, model )
    


    return model



def error_statistics(y_true, y_pred, df, PRINT = 0, ADDITIONAL_STATS=0, RND =3, colname='Error Statistics'):
    """
    Get error statistics of the true and predicted values
    :param y_true: (vector) true values.
    :param y_pred: (vector) predicted values.
    :param ADDITIONAL_STATS: (binary) indicates whether to show additional error statistics.
    :param RND: (int) indicates the number of digits of the printed error statistics.
    """
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)

    # 'Number of cases': len(y_true),
    res_stats =pd.DataFrame.from_dict({ 
        'Root mean squared error': round(np.sqrt(mse), RND),
        'Mean absolute error':round(mean_absolute_error, RND), 
        } , orient='index',columns=[colname]) 
    if 1:
        Total_observed_ClaimAmount  = sum( df.loc[y_true.index,"ClaimAmount"].values )
        Total_predicted_ClaimAmount = sum( df.loc[y_true.index,"Exposure"].values * y_pred) 
        Percentage_error_Total_ClaimAmount = (Total_predicted_ClaimAmount - Total_observed_ClaimAmount)/Total_observed_ClaimAmount*100
        res_stats.loc['Total observed ClaimAmount '] =  round(Total_observed_ClaimAmount, RND)
        res_stats.loc['Total predicted ClaimAmount'] =  round(Total_predicted_ClaimAmount, RND)
        res_stats.loc['Percentage error total ClaimAmount'] =  round(Percentage_error_Total_ClaimAmount, RND)


    if ADDITIONAL_STATS:
        median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
        mean_absolute_percentage_error =metrics.mean_absolute_percentage_error(y_true, y_pred)
        r2 = metrics.r2_score(y_true, y_pred)
        explained_variance = metrics.explained_variance_score(y_true, y_pred)
        correlation, _ = pearsonr(y_true, y_pred)
        #mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)

        res_stats.loc['Mean squared error '] =  round(mse, RND)
        res_stats.loc['Median absolute error'] =  round(median_absolute_error, RND)
        res_stats.loc['Mean absolute prctg error'] =  round(mean_absolute_percentage_error, RND)
        res_stats.loc['Explained variance'] =  round(explained_variance, RND)
        res_stats.loc['R2']                =  round(r2, RND)
        res_stats.loc['Correlation']       =  round(correlation,RND)

     
    return res_stats



def get_split(df1,  y_var = 'Premium', test_size = 0.2, validation_size = 0.1, stratify=None):
    """function to split the data into training and test sets"""
    X = df1.drop(y_var, axis=1)
    y = df1[y_var]
    
    # For non-numeric data, set them to the categorial data type: 
    for i in X.columns:
        if not is_numeric_dtype(X.loc[:,i]):
            X[i] = X.loc[:,i].astype("category")        

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=test_size+validation_size, random_state=42, stratify= stratify)

    X_val = None
    y_val = None
    if validation_size>0:
        X_test, X_val, y_test, y_val = train_test_split( X_test, y_test, test_size=validation_size, random_state=42)

    return X_train, X_test, y_train, y_test, X_val, y_val



def plot_obs_pred(df, feature, weight, observed, predicted, bins = 10, y_label=None, title=None, ax=None, fill_legend=False, SHOW_FEATURE_DISTRIBUTION = False):
    """Plot observed and predicted - aggregated per feature level.

    # License: BSD 3 clause. Adopted from python code from Christian Lorentzen, Roman Yurchak and Olivier Grisel.
    Parameters
    ----------
    df : DataFrame
        input data
    feature: str
        a column name of df for the feature to be plotted
    weight : str
        column name of df with the values of weights or exposure
    observed : str
        a column name of df with the observed target
    predicted : DataFrame
        a dataframe, with the same index as df, with the predicted target
    fill_legend : bool, default=False
        whether to show fill_between legend
    SHOW_FEATURE_DISTRIBUTION  : bool, default=False
        whether to show the feature distribution  
    """
    # aggregate observed and predicted variables by feature level
    df_ = df.loc[:, [feature, weight]].copy()
    # group the feature into bins:
    if bins > 0 and is_numeric_dtype(df_[feature]):
        df_[feature] = pd.cut(df_[feature], bins=bins)
    df_["observed"] = df[observed] * df[weight]
    df_["predicted"] = predicted * df[weight]
    df_ = (
        df_.groupby([feature])[[weight, "observed", "predicted"]]
        .sum()
        .assign(observed=lambda x: x["observed"] / x[weight])
        .assign(predicted=lambda x: x["predicted"] / x[weight])
    )

    ax = df_.loc[:, ["observed", "predicted"]].plot(style=['.','-'], ax=ax)
    y_max = df_.loc[:, ["observed", "predicted"]].values.max() * 0.8
    
    if SHOW_FEATURE_DISTRIBUTION:
        p2 = ax.fill_between(
            df_.index,
            0,
            y_max * df_[weight] / df_[weight].values.max(),
            color="g",
            alpha=0.1,
        )
        if fill_legend:
            ax.legend([p2], ["{} distribution".format(feature)])

    ax.set(
        ylabel=y_label if y_label is not None else None,
        title=title if title is not None else "Train: Observed vs Predicted",
    )
    return df_ 



def plot_model_results(trained_model, X_test = None, y_test=  None, PLOT=0, output_folder_plots = '', title1='Prediction', title2= 'Variable importances', SAVE_OUTPUT = 0, TOP_IMPORTANCES = 10):
    """function to predict and plot model results"""
    

    #plot Observed vs prediction 
    if PLOT: 
        predictions = trained_model.predict_y(X_test)
        ix = np.argsort(y_test.values)

        plt.rcParams.update({'figure.figsize':(24,5)})
        plt.plot(pd.Series(y_test.values[ix],index=range(0,len(ix))), color='red')
        plt.plot(pd.Series(predictions[ix], index=range(0,len(ix))), color='green')
        plt.tick_params('x', labelrotation=45)
        plt.title('Observed vs predicted')
        plt.legend(labels=['Observed', 'Predicted'])
        #plt.xlabel('Date')
        plt.grid()       
        plt.show()

        # Saving plot to pdf and png file
        if SAVE_OUTPUT:
            plt.savefig(output_folder_plots  +title1+'_'+'.pdf', dpi=100,bbox_inches="tight")
            #plt.title(title1, fontsize=20)
            plt.savefig(output_folder_plots  +title1+'_'+ '.png', dpi=100,bbox_inches="tight")
        plt.show()

    #create a dataframe with the variable importances of the model
    df_importances = pd.DataFrame({
        'feature': trained_model.feature_name_,
        'importance': trained_model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    #plot variable importances of the model
    plt.title(title2, fontsize=16)
    sns.barplot(x=df_importances.importance[:TOP_IMPORTANCES-1], y=df_importances.feature[:TOP_IMPORTANCES-1], orient='h')
    # Saving plot to pdf and png file
    if SAVE_OUTPUT:
        plt.savefig(output_folder_plots  +title2+'.pdf', dpi=100,bbox_inches="tight")
        #plt.title(title1, fontsize=20)
        plt.savefig(output_folder_plots  +title2+ '.png', dpi=100,bbox_inches="tight")    
    plt.show()



def plot_feature_fit(model, df, X_train, X_test, x_vars_show):
    fig, ax = plt.subplots(nrows=len(x_vars_show), ncols=2, figsize=(24, 7))
    fig.subplots_adjust(hspace=0.7, wspace=0.2)
    for ix, i in enumerate(x_vars_show):
        df_ = plot_obs_pred(df =df.loc[X_train.index,:], feature=i, weight='Exposure', observed='Premium', predicted=model.predict_y(X_train), title='Training', ax = ax[ix,0] )
        df_ = plot_obs_pred(df =df.loc[X_test.index,:], feature=i, weight='Exposure', observed='Premium', predicted=model.predict_y(X_test) , title='Test', ax = ax[ix,1] )



def lorenz_curve(y_true, y_pred, exposure):
    """ 
    Get the lorenz curve of a model.
    
    # License: BSD 3 clause. Adopted from python code from Christian Lorentzen, Roman Yurchak and Olivier Grisel.
    """    
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    return cumulated_samples, cumulated_claim_amount


def show_lorenz_curves(df, X_val, gbm_model, rf_model, glm_tr, gbm_tw_model):
    """ 
    Show the lorenz curve of the models.

    # License: BSD 3 clause. Adopted from python code from Christian Lorentzen, Roman Yurchak and Olivier Grisel.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    for label, y_pred in [
        ("GBM", gbm_model.predict_y(X_val)),
        ("RF", rf_model.predict_y(X_val)),
        ("GLM", glm_tr.predict_y(X_val)),
        ("GBM-TW", gbm_tw_model.predict_y(X_val)),
    ]:
        ordered_samples, cum_claims = lorenz_curve(
            df.loc[X_val.index,"Premium"], y_pred, df.loc[X_val.index,"Exposure"]
        )
        gini = 1 - 2 * auc(ordered_samples, cum_claims)
        label += " (Gini index: {:.3f})".format(gini)
        ax.plot(ordered_samples, cum_claims, linestyle="-", label=label)

    # Oracle model: y_pred == y_test
    ordered_samples, cum_claims = lorenz_curve(
        df.loc[X_val.index,"Premium"], df.loc[X_val.index,"Premium"], df.loc[X_val.index,"Exposure"]
    )
    gini = 1 - 2 * auc(ordered_samples, cum_claims)
    label = "Ranking by observed claims (Gini index: {:.3f})".format(gini)
    ax.plot(ordered_samples, cum_claims, linestyle="-.", color="gray", label=label)

    # Random baseline
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")
    ax.set(
        title="Lorenz Curves",
        xlabel="Fraction of policyholders\n(ordered by model from safest to riskiest)",
        ylabel="Fraction of total claim amount",
    )
    ax.legend(loc="upper left")
    plt.plot()

