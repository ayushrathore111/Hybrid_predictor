import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load the data from an Excel file
data = pd.read_excel('azas_ds.xlsx')

# Extract features and target variable
X = data.drop(columns=['ds (m)'])
y = data['ds (m)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
rf_r2 = r2_score(y_test, y_pred_rf)
rf_mse = mean_squared_error(y_test, y_pred_rf)

# Train and evaluate Gaussian Process Regressor
kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-3, 1e3))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, random_state=42)
gpr.fit(X_train_scaled, y_train)
y_pred_gpr, sigma = gpr.predict(X_test_scaled, return_std=True)
gpr_r2 = r2_score(y_test, y_pred_gpr)
gpr_mse = mean_squared_error(y_test, y_pred_gpr)

# Print the evaluation metrics
print(f"Random Forest Regressor R^2: {rf_r2:.4f}")
print(f"Random Forest Regressor MSE: {rf_mse:.4f}")
print(f"Gaussian Process Regressor R^2: {gpr_r2:.4f}")
print(f"Gaussian Process Regressor MSE: {gpr_mse:.4f}")

# Save the models
joblib.dump(rf, '/home/proayush/Desktop/ML_Analysis/Analysis 2024/shailza_work/random_forest_model.joblib')
joblib.dump(gpr, '/home/proayush/Desktop/ML_Analysis/Analysis 2024/shailza_work/gpr_model.joblib')
