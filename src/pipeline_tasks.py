# Contents of: src/pipeline_tasks.py
# (Paths actualizados para Docker)

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import genpareto
import warnings
import os

# --- File Paths (CORRECTED FOR DOCKER) ---
# Estos son los paths ABSOLUTOS *DENTRO* del contenedor de Docker,
# tal como se definieron en los 'volumes' de docker-compose.yaml
BASE_PATH = '/opt/airflow'
RAW_DATA_PATH = os.path.join(BASE_PATH, 'data/raw/demanding_forecast.csv')
MODEL_PATH = os.path.join(BASE_PATH, 'models/gbr_model.joblib')
INTERIM_DATA_PATH = os.path.join(BASE_PATH, 'data/interim/model_core_validation.csv')
PROCESSED_FORECAST_PATH = os.path.join(BASE_PATH, 'data/processed/demand_forecasts_2024.csv')
FINAL_REPORT_PATH = os.path.join(BASE_PATH, 'data/processed/final_inventory_policy_2024.csv')

REPORTS_DIR = os.path.join(BASE_PATH, 'reports/figures')
os.makedirs(REPORTS_DIR, exist_ok=True)

BACKTEST_NET_PROFIT_FIG = os.path.join(REPORTS_DIR, 'backtest_profit_net.png')
BACKTEST_COSTS_FIG = os.path.join(REPORTS_DIR, 'backtest_total_costs.png')
TRACEABILITY_FIG = os.path.join(REPORTS_DIR, 'final_traceability_gbr.png')


# -----------------------------------------------------------------
# HELPER FUNCTION: FEATURE ENGINEERING
# -----------------------------------------------------------------

def create_features(data):
    """
    Creates time-series features from the input dataframe.
    """
    df_feat = data.sort_values(['prod_id', 'fecha']).copy()
    
    df_feat['year'] = df_feat['fecha'].dt.year
    df_feat['month'] = df_feat['fecha'].dt.month
    df_feat['lag_12'] = df_feat.groupby('prod_id')['ventas'].shift(12)
    df_feat['lag_13'] = df_feat.groupby('prod_id')['ventas'].shift(13)
    df_feat['lag_24'] = df_feat.groupby('prod_id')['ventas'].shift(24)
    df_feat['rolling_mean_3_lag12'] = df_feat.groupby('prod_id')['lag_12'].transform(lambda x: x.rolling(3).mean())
    df_feat['precio_lag_12'] = df_feat.groupby('prod_id')['precio_promedio'].shift(12)
    
    return df_feat

# -----------------------------------------------------------------
# HELPER FUNCTION: FINANCIAL SIMULATION
# -----------------------------------------------------------------

def simulate_scenario(df, order_col_name):
    """
    Calculates the financial results for a given order quantity policy.
    """
    actual_sales = df['ventas']
    effective_sales = np.minimum(actual_sales, df[order_col_name])
    excess_inventory = np.maximum(0, df[order_col_name] - actual_sales)
    lost_sales = np.maximum(0, actual_sales - df[order_col_name])
    
    gross_profit = effective_sales * df['unit_margin']
    spoilage_cost = excess_inventory * df['unit_cost']
    opportunity_cost = lost_sales * df['unit_margin']
    net_profit = gross_profit - spoilage_cost
    
    return pd.Series({
        'Total_Net_Profit': net_profit.sum(),
        'Total_Spoilage_Cost': spoilage_cost.sum(),
        'Total_Opportunity_Cost': opportunity_cost.sum(),
        'Total_Effective_Sales_Units': effective_sales.sum()
    })

# -----------------------------------------------------------------
# TASK 1: VALIDATION & ECONOMIC BACKTEST (From Notebook 01 & 02)
# -----------------------------------------------------------------

def run_validation_task():
    """
    Runs the logic from Notebooks 01 and 02.
    1. Trains the GBR model on (T-1) data.
    2. Saves the model artifact.
    3. Runs the full economic backtest (proving GBR Base is the winner).
    4. Saves the backtest results (CSVs, figures).
    """
    print("--- Running Validation & Backtest Task ---")
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # --- Notebook 01 Logic: Train GBR Model ---
    print(f"Loading raw data from: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    print("Engineering features...")
    df_model = create_features(df)
    df_model = df_model.dropna()

    TARGET_VARIABLE = 'ventas'
    FEATURES = [
        'month', 'year', 'prod_id', 
        'lag_12', 'lag_13', 'lag_24', 
        'rolling_mean_3_lag12', 'precio_lag_12'
    ]
    
    train = df_model[df_model['year'] < 2023].copy()
    val = df_model[df_model['year'] == 2023].copy()
    print(f"Training on {len(train)} rows, validating on {len(val)} rows.")

    gbr = GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=7,
        random_state=42, n_iter_no_change=20, validation_fraction=0.1
    )
    
    print("Fitting GBR model...")
    gbr.fit(train[FEATURES], train[TARGET_VARIABLE])
    
    print(f"Saving model artifact to: {MODEL_PATH}")
    joblib.dump(gbr, MODEL_PATH)
    
    print("Calculating WAPE...")
    val_preds = gbr.predict(val[FEATURES])
    val_comparativo = val.copy()
    val_comparativo['pred_gbr'] = val_preds
    
    wape_gbr = np.sum(np.abs(val_comparativo['ventas'] - val_comparativo['pred_gbr'])) / np.sum(np.abs(val_comparativo['ventas']))
    wape_actual = np.sum(np.abs(val_comparativo['ventas'] - val_comparativo['modelo_actual'])) / np.sum(np.abs(val_comparativo['ventas']))
    wape_improvement = (wape_actual - wape_gbr) / wape_actual
    print(f"WAPE Improvement: {wape_improvement:.2%}")
    
    print(f"Saving interim validation data to: {INTERIM_DATA_PATH}")
    val_comparativo.to_csv(INTERIM_DATA_PATH, index=False)
    print("GBR model trained and saved. Validation data saved.")

    # --- Notebook 02 Logic: Economic Backtest ---
    print("Starting Economic Backtest...")
    
    # 1. Load data for robust median calculation
    historical_df = pd.read_csv(RAW_DATA_PATH)
    median_sales_map = historical_df.fillna(0).groupby('prod_id')['ventas'].median()
    print("Historical median sales map calculated.")
    
    # 2. Calculate Robust Heuristic (V7 Logic) - IN-MEMORY ONLY
    validation_df = val_comparativo
    validation_df['residuo'] = validation_df['ventas'] - validation_df['pred_gbr']
    
    SERVICE_LEVEL_TARGET = 0.98
    ERROR_QUANTILE_THRESHOLD = 0.90
    SAFETY_STOCK_MONTHS_CAP = 3.0

    safety_stock_map = {}
    unique_products = validation_df['prod_id'].unique()
    
    for pid in unique_products:
        positive_residuals = validation_df[
            (validation_df['prod_id'] == pid) & (validation_df['residuo'] > 0)
        ]['residuo']
        
        calculated_safety_stock = 0
        
        if not positive_residuals.empty:
            fallback_security = positive_residuals.quantile(ERROR_QUANTILE_THRESHOLD)
            umbral = fallback_security
            calculated_safety_stock = fallback_security # Start with fallback
            
            excesos = positive_residuals[positive_residuals > umbral] - umbral
            
            if (not excesos.empty) and (umbral > 0):
                try:
                    c, loc, scale = genpareto.fit(excesos, floc=0)
                    prob_gpd = (SERVICE_LEVEL_TARGET - ERROR_QUANTILE_THRESHOLD) / (1 - ERROR_QUANTILE_THRESHOLD)
                    stock_extra_gpd = genpareto.ppf(prob_gpd, c, loc=loc, scale=scale)
                    calculated_safety_stock = umbral + stock_extra_gpd
                except Exception as e:
                    calculated_safety_stock = fallback_security
            
        median_sale = median_sales_map.get(pid, 0)
        security_cap = median_sale * SAFETY_STOCK_MONTHS_CAP
        
        final_stock = calculated_safety_stock
        if security_cap > 0:
            final_stock = min(calculated_safety_stock, security_cap)
            
        final_stock = max(0, final_stock)
        safety_stock_map[pid] = np.round(np.nan_to_num(final_stock))
    
    print("Robust heuristic safety stock calculated in-memory for backtest.")
    df_safety_stock = pd.DataFrame(list(safety_stock_map.items()), columns=['prod_id', 'safety_stock'])
    
    # 3. Run Financial Simulation
    backtest_df = pd.merge(validation_df, df_safety_stock, on='prod_id', how='left')
    backtest_df['safety_stock'] = backtest_df['safety_stock'].fillna(0)
    
    COGS_PERCENTAGE = 0.60
    backtest_df['unit_cost'] = backtest_df['precio_promedio'] * COGS_PERCENTAGE
    backtest_df['unit_margin'] = backtest_df['precio_promedio'] * (1 - COGS_PERCENTAGE)
    
    backtest_df['order_qty_actual'] = np.round(backtest_df['modelo_actual'])
    backtest_df['order_qty_gbr_base'] = np.round(backtest_df['pred_gbr'])
    backtest_df['order_qty_hybrid'] = np.round(backtest_df['pred_gbr'] + backtest_df['safety_stock'])
    
    results = []
    scenarios = {
        '1. Client Model': 'order_qty_actual',
        '2. GBR Base Model (Precision)': 'order_qty_gbr_base',
        '3. Hybrid Model (Risk)': 'order_qty_hybrid'
    }
    for scenario_name, col_name in scenarios.items():
        scenario_result = simulate_scenario(backtest_df, col_name)
        scenario_result.name = scenario_name
        results.append(scenario_result)

    df_results = pd.DataFrame(results)
    
    print("Economic backtest simulation complete.")
    
    # 4. Save Backtest Figures
    df_plot = df_results / 1e9 # Plot in Billions
    df_plot_melt = df_plot.reset_index().melt('index', var_name='Metric', value_name='Value (Billions $)')

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df_plot_melt[df_plot_melt['Metric'] == 'Total_Net_Profit'], 
        x='index', y='Value (Billions $)', palette='Greens_d',
        order=['1. Client Model', '2. GBR Base Model (Precision)', '3. Hybrid Model (Risk)']
    )
    plt.title('2023 Simulation: Total Net Profit by Scenario')
    plt.xlabel('Scenario')
    plt.ylabel('Net Profit (Billions $)')
    plt.savefig(BACKTEST_NET_PROFIT_FIG, bbox_inches='tight')
    plt.close()

    df_costos = df_plot_melt[df_plot_melt['Metric'].isin(['Total_Spoilage_Cost', 'Total_Opportunity_Cost'])]
    plt.figure(figsize=(14, 7))
    sns.barplot(
        data=df_costos, x='index', y='Value (Billions $)', hue='Metric',
        palette={'Total_Spoilage_Cost': 'orange', 'Total_Opportunity_Cost': 'red'},
        order=['1. Client Model', '2. GBR Base Model (Precision)', '3. Hybrid Model (Risk)']
    )
    plt.title('2023 Simulation: Key Costs by Scenario')
    plt.xlabel('Scenario')
    plt.ylabel('Cost (Billions $)')
    plt.legend(title='Cost Type')
    plt.savefig(BACKTEST_COSTS_FIG, bbox_inches='tight')
    plt.close()
    
    print(f"Backtest figures saved.")
    print("--- Validation & Backtest Task Finished ---")


# -----------------------------------------------------------------
# TASK 2: INFERENCE & REPORTING (From Notebook 03 & 04)
# -----------------------------------------------------------------

def run_inference_task():
    """
    Runs the logic from Notebooks 03 and 04.
    1. Re-trains the GBR model on 100% of data (T).
    2. Generates the 2024 forecast.
    3. Generates the final report and traceability plot.
    """
    print("--- Running Final Inference Task ---")
    
    # --- Notebook 03 Logic: Re-train and Predict ---
    print("Loading full historical data for re-training...")
    df_history = pd.read_csv(RAW_DATA_PATH)
    df_history['fecha'] = pd.to_datetime(df_history['fecha'])
    
    print("Loading model parameters...")
    gbr_validated = joblib.load(MODEL_PATH)
    model_params = gbr_validated.get_params()

    print("Engineering features on full dataset...")
    df_model = create_features(df_history)
    df_model = df_model.dropna()

    TARGET_VARIABLE = 'ventas'
    FEATURES = [
        'month', 'year', 'prod_id', 
        'lag_12', 'lag_13', 'lag_24', 
        'rolling_mean_3_lag12', 'precio_lag_12'
    ]
    
    train_full = df_model[df_model['year'] <= 2023]
    print(f"Re-training final model on {len(train_full)} rows...")
    
    gbr_final = GradientBoostingRegressor(**model_params)
    gbr_final.set_params(n_iter_no_change=None) # Disable early stopping
    
    gbr_final.fit(train_full[FEATURES], train_full[TARGET_VARIABLE])
    print("Final model re-training complete.")

    print("Generating 2024 forecast scaffold...")
    future_dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='MS')
    prod_ids = df_history['prod_id'].unique()
    
    df_future_rows = []
    for pid in prod_ids:
        for date in future_dates:
            df_future_rows.append({'fecha': date, 'prod_id': pid, 'ventas': np.nan, 'precio_promedio': np.nan})
    df_future = pd.DataFrame(df_future_rows)
    
    df_full_history = pd.concat([df_history, df_future], axis=0) 
    df_full_features = create_features(df_full_history) 
    
    df_2024 = df_full_features[df_full_features['fecha'].dt.year == 2024].copy()
    
    print("Predicting 2024 demand...")
    df_2024[FEATURES] = df_2024[FEATURES].fillna(0)
    df_2024['prediccion_ventas'] = gbr_final.predict(df_2024[FEATURES])
    
    output_cols = ['fecha', 'prod_id', 'prediccion_ventas']
    df_2024[output_cols].to_csv(PROCESSED_FORECAST_PATH, index=False)
    print(f"2024 forecast saved to: {PROCESSED_FORECAST_PATH}")

    # --- Notebook 04 Logic: Generate Final Report ---
    print("Generating final client report...")
    df_forecast = df_2024[output_cols].copy()
    
    df_final_report = df_forecast.copy()
    df_final_report['stock_total_recomendado'] = np.round(df_final_report['prediccion_ventas'])
    final_columns = ['fecha', 'prod_id', 'prediccion_ventas', 'stock_total_recomendado']
    df_final_report = df_final_report[final_columns]
    
    df_final_report.to_csv(FINAL_REPORT_PATH, index=False)
    print(f"Final client-ready report saved to: {FINAL_REPORT_PATH}")
    
    print("Generating final traceability plot...")
    PRODUCT_ID_TO_PLOT = 0
    df_plot_hist = df_history[df_history['prod_id'] == PRODUCT_ID_TO_PLOT]
    df_plot_fcst = df_final_report[df_final_report['prod_id'] == PRODUCT_ID_TO_PLOT]

    plt.figure(figsize=(20, 8))
    plt.plot(df_plot_hist['fecha'], df_plot_hist['ventas'], label='Actual Sales (History)', color='blue', alpha=0.7)
    plt.plot(df_plot_fcst['fecha'], df_plot_fcst['stock_total_recomendado'], label='GBR Forecast (Recommended Policy 2024)', color='green', linestyle='--', marker='o')
    plt.axvline(pd.to_datetime('2024-01-01'), color='black', linestyle=':', linewidth=2, label='Forecast Start')
    plt.title(f'Historical Traceability & 2024 Forecast - Product {PRODUCT_ID_TO_PLOT}')
    plt.ylabel('Sales Units')
    plt.xlabel('Year')
    plt.xlim(pd.to_datetime('2017-01-01'), pd.to_datetime('2024-12-01'))
    plt.legend()
    plt.grid(True)
    
    plt.savefig(TRACEABILITY_FIG, bbox_inches='tight')
    plt.close() # Close plot to save memory
    print(f"Final traceability plot saved to: {TRACEABILITY_FIG}")
    print("--- Final Inference Task Finished ---")