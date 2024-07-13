import streamlit as st
import pandas as pd
import joblib

# Load the trained models
rf_cs = joblib.load('./rf_cs.joblib')
rf_ts = joblib.load('./rf_ts.joblib')
lr_cs = joblib.load('./lr_cs.joblib')
lr_ts = joblib.load('./lr_ts.joblib')
etr_cs = joblib.load('./etr_cs.joblib')
etr_ts = joblib.load('./etr_ts.joblib')
ar_cs = joblib.load('./ar_cs.joblib')
ar_ts = joblib.load('./ar_ts.joblib')
gbr_cs = joblib.load('./gbr_cs.joblib')
gbr_ts = joblib.load('./gbr_ts.joblib')
lasso_cs = joblib.load('./lasso_cs.joblib')
lasso_ts = joblib.load('./lasso_ts.joblib')
net_cs = joblib.load('./net_cs.joblib')
net_ts = joblib.load('./net_ts.joblib')
ridge_cs = joblib.load('./ridge_cs.joblib')
ridge_ts = joblib.load('./ridge_ts.joblib')
svr_cs = joblib.load('./svr_cs.joblib')
svr_ts = joblib.load('./svr_ts.joblib')
xg_cs = joblib.load('./xg_cs.joblib')
xg_ts = joblib.load('./xg_ts.joblib')


# Title of the web app
st.title('Compressive/Tensile Strength Hybrid Prediction')

# Sidebar with input fields
st.sidebar.title('Input Features')

U = st.sidebar.slider('Cement (kg/m3)', min_value=200.1, max_value=500.1, step=0.1, value=232.5)
H = st.sidebar.slider('Fly Ash (kg/m3)', min_value=200.1, max_value=500.1, step=1.5, value=270.6)
D = st.sidebar.slider('Fine Aggregates (kg/m3)', min_value=800.1, max_value=1200.1, step=1.5, value=845.5)
Fr = st.sidebar.slider('Water (kg/m3)', min_value=150.1, max_value=300.1, step=1.5, value=180.0)
d50 = st.sidebar.slider('Coarse Aggregates (kg/m3)', min_value=500.1, max_value=1000.1, step=1.5, value=542.2)
HD_ratio = st.sidebar.slider('Water to Binder Ratio', min_value=0.01, max_value=1.0, step=0.01, value=0.12)
Dd50_ratio = st.sidebar.slider('Water to Powder Ratio', min_value=0.01, max_value=1.0, step=0.01, value=0.05)
DsD_ratio = st.sidebar.slider('Superplastisizer (kg/m3)', min_value=0.01, max_value=10.0, step=0.01, value=0.18)
Days = st.sidebar.slider('Curing Days', min_value=1, max_value=30, step=1, value=7)

# Function to make predictions using the models
def make_predictions(rf_cs,rf_ts,lr_cs,lr_ts,etr_cs,etr_ts,ar_ts,ar_cs,xg_cs,xg_ts,svr_ts,svr_cs,ridge_cs,ridge_ts,lasso_ts,lasso_cs,net_cs,net_ts,gbr_cs,gbr_ts, U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days):
    # Make predictions using the models
    rf_cs_pred = rf_cs.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days]])[0]
    rf_ts_pred = rf_ts.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days]])[0]
    lr_cs_pred = lr_cs.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days]])[0]
    lr_ts_pred = lr_ts.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days]])[0]
    etr_cs_pred = etr_cs.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days]])[0]
    etr_ts_pred = etr_ts.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days]])[0]
    ar_cs_pred = ar_cs.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days]])[0]
    ar_ts_pred = ar_ts.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days]])[0]
    xg_cs_pred = xg_cs.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days]])[0]
    xg_ts_pred = xg_ts.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days]])[0]
    svr_cs_pred = svr_cs.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days]])[0]
    svr_ts_pred = svr_ts.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days]])[0]
    ridge_cs_pred = ridge_cs.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days]])[0]
    ridge_ts_pred = ridge_ts.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days]])[0]
    lasso_cs_pred = lasso_cs.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days]])[0]
    lasso_ts_pred = lasso_ts.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days]])[0]
    net_cs_pred = net_cs.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days]])[0]
    net_ts_pred = net_ts.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days]])[0]
    gbr_cs_pred = gbr_cs.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days]])[0]
    gbr_ts_pred = gbr_ts.predict([[U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days]])[0]
    return rf_cs_pred, rf_ts_pred,lr_cs_pred,lr_ts_pred,etr_cs_pred,etr_ts_pred,ar_cs_pred,ar_ts_pred,xg_cs_pred,xg_ts_pred,svr_cs_pred,svr_ts_pred,ridge_cs_pred,ridge_ts_pred,lasso_cs_pred,lasso_ts_pred,net_cs_pred,net_ts_pred,gbr_cs_pred,gbr_ts_pred

# Make predictions using the input values
rf_cs_pred, rf_ts_pred,lr_cs_pred,lr_ts_pred,etr_cs_pred,etr_ts_pred,ar_cs_pred,ar_ts_pred,xg_cs_pred,xg_ts_pred,svr_cs_pred,svr_ts_pred,ridge_cs_pred,ridge_ts_pred,lasso_cs_pred,lasso_ts_pred,net_cs_pred,net_ts_pred,gbr_cs_pred,gbr_ts_pred = make_predictions(rf_cs,rf_ts,lr_cs,lr_ts,etr_cs,etr_ts,ar_ts,ar_cs,xg_cs,xg_ts,svr_ts,svr_cs,ridge_cs,ridge_ts,lasso_ts,lasso_cs,net_cs,net_ts,gbr_cs,gbr_ts, U, H, D, Fr, d50, HD_ratio, Dd50_ratio, DsD_ratio,Days)

rf_gb_lr_cs= (rf_cs_pred+gbr_cs_pred+lr_cs_pred)/3
rf_ar_ridge_cs= (rf_cs_pred+ar_cs_pred+ridge_cs_pred)/3
gb_ar_lasso_cs= (gbr_cs_pred+ar_cs_pred+lasso_cs_pred)/3
et_svr_gbr_cs= (etr_cs_pred+svr_cs_pred+gbr_cs_pred)/3
rf_xg_lr_cs= (lr_cs_pred+rf_cs_pred+xg_cs_pred)/3
gb_xg_ridge_cs= (gbr_cs_pred+xg_cs_pred+ridge_cs_pred)/3
ar_xg_lasso_cs= (ar_cs_pred+xg_cs_pred+lasso_cs_pred)/3
rf_et_net_cs= (rf_cs_pred+etr_cs_pred+net_cs_pred)/3

rf_gb_lr_ts= (rf_ts_pred+gbr_ts_pred+lr_ts_pred)/3
rf_ar_ridge_ts= (rf_ts_pred+ar_ts_pred+ridge_ts_pred)/3
gb_ar_lasso_ts= (gbr_ts_pred+ar_ts_pred+lasso_ts_pred)/3
et_svr_gbr_ts= (etr_ts_pred+svr_ts_pred+gbr_ts_pred)/3
rf_xg_lr_ts= (lr_ts_pred+rf_ts_pred+xg_ts_pred)/3
gb_xg_ridge_ts= (gbr_ts_pred+xg_ts_pred+ridge_ts_pred)/3
ar_xg_lasso_ts= (ar_ts_pred+xg_ts_pred+lasso_ts_pred)/3
rf_et_net_ts= (rf_ts_pred+etr_ts_pred+net_ts_pred)/3




# Display predictions

st.title("Compressive Strength")
st.write('### Random Forest, Gradient Boosting and Linear Prediction (cs):', rf_gb_lr_cs)
st.write('### Random Forest,  AdaBoost and Ridge Prediction (cs):', rf_ar_ridge_cs)
st.write('### Adaboost, Gradient Boosting and Lasso Prediction (cs):', gb_ar_lasso_cs)
st.write('### Extra tree, Gradient Boosting and Support Vector Machine Prediction (cs):', et_svr_gbr_cs)
st.write('### Random Forest, Extream Gradient Boosting and Linear Prediction (cs):', rf_xg_lr_cs)
st.write('### Extream Gradient Boosting, Gradient Boosting and Ridge Prediction (cs):', gb_xg_ridge_cs)
st.write('### Adaboost, Extream Gradient Boosting and Lasso Prediction (cs):', ar_xg_lasso_cs)
st.write('### Random Forest, Extra Tree and Extream Net Prediction (cs):', rf_et_net_cs)


st.title("Tensile Strength")
st.write('### Random Forest, Gradient Boosting and Linear Prediction (ts):', rf_gb_lr_ts)
st.write('### Random Forest,  AdaBoost and Ridge Prediction (ts):', rf_ar_ridge_ts)
st.write('### Adaboost, Gradient Boosting and Lasso Prediction (ts):', gb_ar_lasso_ts)
st.write('### Extra tree, Gradient Boosting and Support Vector Machine Prediction (ts):', et_svr_gbr_ts)
st.write('### Random Forest, Extream Gradient Boosting and Linear Prediction (ts):', rf_xg_lr_ts)
st.write('### Extream Gradient Boosting, Gradient Boosting and Ridge Prediction (ts):', gb_xg_ridge_ts)
st.write('### Adaboost, Extream Gradient Boosting and Lasso Prediction (ts):', ar_xg_lasso_ts)
st.write('### Random Forest, Extra Tree and Extream Net Prediction (ts):', rf_et_net_ts)

