
import pandas as pd
import numpy as np
from sklearn.ensemble         import RandomForestRegressor
from sklearn.linear_model     import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.tree             import DecisionTreeRegressor
from sklearn.svm              import SVR
from sklearn.ensemble         import GradientBoostingRegressor,ExtraTreesRegressor,AdaBoostRegressor,BaggingRegressor
from sklearn.neural_network   import MLPRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
# Read the CSV file
data = pd.read_excel('somanshi_cs.xlsx')
from sklearn.neighbors import KNeighborsRegressor

# Separate the data into input X and output Y
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
rf_model = RandomForestRegressor(n_estimators=100, max_depth=None)
lr_model = LinearRegression()
gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
knn_model = KNeighborsRegressor(n_neighbors=5)
etr_model = ExtraTreesRegressor(random_state=42)
dtr_model = DecisionTreeRegressor(random_state=42)
adaboost_model = AdaBoostRegressor(random_state=42)
bagging_model = BaggingRegressor(random_state=42)
ridge_model= Ridge()
lasso_model= Lasso()
svr_model = SVR()
enet_model =ElasticNet()
xgb_model = XGBRegressor(random_state=42)

y_pred_knn= knn_model.fit(X_train.values,y_train.values).predict(X_test.values)
y_pred_etr= etr_model.fit(X_train.values,y_train.values).predict(X_test.values)
y_pred_dtr= dtr_model.fit(X_train.values,y_train.values).predict(X_test.values)
y_pred_br= bagging_model.fit(X_train.values,y_train.values).predict(X_test.values)
y_pred_ar= adaboost_model.fit(X_train.values,y_train.values).predict(X_test.values)
y_pred_xg= xgb_model.fit(X_train.values,y_train.values).predict(X_test.values)
y_pred_lr= lr_model.fit(X_train.values,y_train.values).predict(X_test.values)
y_pred_rf= rf_model.fit(X_train.values,y_train.values).predict(X_test.values)
y_pred_gbr= gbr_model.fit(X_train.values,y_train.values).predict(X_test.values)
y_pred_ridge=ridge_model.fit(X_train.values,y_train.values).predict(X_test.values)
y_pred_lasso=lasso_model.fit(X_train.values,y_train.values).predict(X_test.values)
y_pred_svr=svr_model.fit(X_train.values,y_train.values).predict(X_test.values)
y_pred_net=enet_model.fit(X_train.values,y_train.values).predict(X_test.values)

		

# rf gb lr ar lasso ridge svr xg et net 
import joblib

joblib.dump(rf_model,'./rf_cs.joblib')
joblib.dump(gbr_model,'./gbr_cs.joblib')
joblib.dump(lr_model,'./lr_cs.joblib')
joblib.dump(adaboost_model,'./ar_cs.joblib')
joblib.dump(lasso_model,'./lasso_cs.joblib')
joblib.dump(ridge_model,'./ridge_cs.joblib')
joblib.dump(svr_model,'./svr_cs.joblib')
joblib.dump(xgb_model,'./xg_cs.joblib')
joblib.dump(enet_model,'./net_cs.joblib')
joblib.dump(etr_model,'./etr_cs.joblib')
