
# Bank Marketing using ML classification models
## Table of Contents
* [Problem statement](#problem-statement)
* [Data](#data)
* [Technologies](#technologies)
* [Models' Results](#models'-results)
* [Insights](#insights)
* [Recommendations](#recommendations)
* [Further Studies](#further-studies)
## Problem statement

In module 3 project,  a famous dataset 'Bank Marketing' from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) is chosen to work on. The aim of this project is to predict the result of target variable (subscribing bank deposit) by applying several machine learning classification models. 

## Data

The bank marketing dataset consists of 21 columns and 41187 rows. The features of the data are:

1 - age (numeric)

2 - job : type of job (categorical: "admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown")

3 - marital : marital status (categorical: "divorced","married","single","unknown"; note: "divorced" means divorced or widowed)

4 - education (categorical: "basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown")

5 - default: has credit in default? (categorical: "no","yes","unknown")

6 - housing: has housing loan? (categorical: "no","yes","unknown")

7 - loan: has personal loan? (categorical: "no","yes","unknown")

8 - contact: contact communication type (categorical: "cellular","telephone")

9 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")

10 - day_of_week: last contact day of the week (categorical: "mon","tue","wed","thu","fri")

11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)

13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means clients was not previously contacted)

14 - previous: number of contacts performed before this campaign and for this client (numeric)

15 - poutcome: outcome of the previous marketing campaign (categorical: "failure","nonexistent","success")

16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)

17 - cons.price.idx: consumer price index - monthly indicator (numeric)

18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)

19 - euribor3m: euribor 3-month rate - daily indicator (numeric)

20 - nr.employed: number of employees - quarterly indicator (numeric)

Output variable (desired target): 21 - y - has the client subscribed a term deposit? (binary: "yes","no")

## Technologies
This project was created using the following languages and libraries. An environment with the correct versions of the following libraries will allow re-production and improvement on this project. 

* Python version: 3.6.9
* Matplotlib version: 3.0.3
* Seaborn version: 0.9.0
* Pandas version: 0.24.2
* Numpy version: 1.16.2
* Sklearn version: 0.20.3
* XGBoost version: 0.90

## Models' Results
In the project:

- Logistic Regression,
- Decision Tree,
- Random Forest,
- XGBoost, 
- Naive Bayes

classifiers are used with GridSearch. 

The best performing machine learning model is XGBoost used with:

 "booster": "dart", 
"colsample_bytree": 0.8, 
"gamma": 2, 
"grow_policy": "depthwise", 
"learning_rate": 0.007, 
"max_depth": 5, 
"min_child_weight": 10,
"n_estimators": 300, 
"random_state": 33, 
"subsample": 0.6, 
"tree_method": "hist"

  parameters. 

<img src="https://github.com/kristinepetrosyan/Bank_Marketing_Campaign_classification/blob/master/images/Screen%20Shot%202020-07-11%20at%203.11.43%20PM.png">


 While Naive Bayes classifier has the lowest training time, XGBoost has the highest training time. Hence, it is the computationally most expensive one. In terms of training, test, f1 and recall score, XGBoost has the highest score, Naive Bayes has the lowest. Both Random Forest and Logistic Regression's score are very close to XGBoost's scores, but their training time is much shorter compared to XGBoost.
 
<img src="https://github.com/kristinepetrosyan/Bank_Marketing_Campaign_classification/blob/master/images/Screen%20Shot%202020-07-15%20at%203.36.05%20PM.png">


<img src="https://github.com/kristinepetrosyan/Bank_Marketing_Campaign_classification/blob/master/images/Screen%20Shot%202020-07-15%20at%203.36.27%20PM.png">

## Insights:

* The marketing campaign mostly targeted people who:

- have administrative jobs,
- are married,
- have university degree,
- have credit default,
- have housing loan,
- doesn't have any personal loan,
- have mobile number.

* It can be concluded that the telemarketing campaign mostly targeted the customer profiles which are more likely to subscribe term deposit.
* Also, May, July and August are the months that the campaign mostly took place.
* As the duration increases customers are more likely to accept term deposit.
* Customers who are mostly never contacted or contacted after high pdays are more likely to refuse term deposit offer.
* As the number of campaign calls passes 10 customers are more likely to reject term deposit offer.
* Lower Euribor3m rates leads to positive response from customers.
* Lower Emp_var_rate result with positive response from customers.
* Knowing that 999 means that a customer never contacted previously, customers contacted after high pdays or never contacted before are more likely to refuse term deposit offer.
* Between the nr_employed rates 4963.6 and 5099.1 percentage of yes labels are higher than no label, between 5191.0 and 5228.1 no labels are higher than yes labels.
* Knowing that there is high correlation between nr_employed and emp_var_rate, euribor3m, cons_price_idx are highly correlated, it can be concluded that cons_price_index is also an important feature affecting our target variable.
* Besides duration, it is observed that monthly or quarterly economic indicators like cons_conf_idx, emp_var_rate euribor3m, nr_employed are among the most important features. Here, it can be concluded that economic indicators are influential on people's financial decisions.

## Recommendations:
* Customers are more likely to subscribe term deposits under good economic conditions.
* Customers contacted after low pdays are more likely to accept term deposit offer.
* Over focused telemarketing campaigns may affect customers negatively. Number of campaigns shouldn't be exceeding 10.
* Longer duration indicates positive response from customers in terms of term deposits. 

## Further studies:
* Further analysis on consumer behaviour could be conducted to gain more insights. 
* More concrete insights can be reached with broader datasets.
* Regression models can be applied to predict duration as future work.