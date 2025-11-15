# **Demand Forecasting for Agricultural Products**
*Inventory Optimization with Hybrid Model (GBR + GPD)*
--
## 1. Project Summary
This project is a Data Science solution for an agricultural products distributor. The objective is to solve the critical business problems of excess inventory and lost sales (stockouts), which were being caused by an inaccurate forecasting model.

The solution is not just a single model but a hybrid decision pipeline that separates forecasting from risk management:

1. **Core Model (GBR)**: A GradientBoostingRegressor (GBR) machine learning model that accurately predicts the base demand (the mean).

2. **Risk Model (GPD)**: An advanced statistical model, Generalized Pareto Distribution (GPD), that models extreme events (demand spikes) and calculates a dynamic safety stock for each product individually.

The final result is a complete inventory policy that tells the business not just what they are likely to sell, but what they must order to profitably achieve a 98% service level.

## 2. The Business Problem
The client was facing two primary, costly challenges:

1. **Excess Inventory**: The previous model (modelo_actual) often overestimated demand for predictable products, leading to high storage costs and spoilage of perishable goods.

2. **Lost Sales**: The same model underestimated demand for volatile products, failing to predict demand spikes. This resulted in stockouts, lost revenue, and customer dissatisfaction.

## 3. Key Results
The new hybrid model outperformed the existing modelo_actual on all key metrics during validation (using all of 2023 data):

- **Improved Accuracy**: The Core Model (GBR) was 6.31% more accurate (as measured by WAPE) than the existing model_actual.

- **Per-Product Risk Management**: We moved from a "one-size-fits-all" safety stock to a dynamic, statistically-driven buffer calculated for all 1,000 products based on their individual volatility.

- **Actionable Inventory Policy**: The final deliverable (politica_inventario_final_2024.csv) is a production-ready file that combines the mean forecast (prediccion_ventas) with the calculated risk (stock_de_seguridad).

## 4. Solution Visualization

**2024 Inventory Policy (Sample Product)**
This chart shows the base forecast (blue line) and the safety stock buffer (green area) calculated by the GPD model to achieve a 98% Service Level.

**Historical Traceability + Forecast**
This chart demonstrates the solution's impact by plotting historical actual sales (2017-2023) and seamlessly connecting them to the new inventory policy for 2024 (forecast + risk).

## 5. Repository Structure 

The project is modularized into a 4-notebook pipeline:

```bash 
project-repository/
├── data/
│   ├── raw/
│   │   └── demanding_forecast.csv     (Original raw data)
│   ├── interim/
│   │   └── model_core_validation.csv  (Output of N-01 -> Input for N-02)
│   └── processed/
│       ├── stock_seguridad_por_producto.csv (Output of N-02)
│       ├── predicciones_demanda_2024.csv    (Output of N-03)
│       └── politica_inventario_final_2024.csv (FINAL DELIVERABLE)
├── models/
│   └── gbr_model.joblib               (Trained GBR model, Output of N-01)
├── notebooks/
│   ├── 01-model_core.ipynb            (GBR Training & Validation)
│   ├── 02-model_risk_gpd.ipynb        (Per-Product Safety Stock Calculation)
│   ├── 03-model_inference.ipynb       (Re-training & 2024 Forecasting)
│   └── 04-generar_reporte_final.ipynb (Final Report Assembly & Visualization)
├── reports/
│   └── figures/
│       ├── politica_inventario_prod_0.png
│       └── trazabilidad_historica_prod_0.png
├── requirements.txt                   (Project dependencies)
└── README.md                          (This file)
```

**Notebook Descriptions**:
1. 01-model_core.ipynb:
    - Trains the GradientBoostingRegressor (GBR) on data up to 2022.
    - Validates the model against 2023 data and compares it to the modelo_actual (achieving a 6.31% WAPE improvement).
    - Saves the validated model (gbr_model.joblib) and the validation results (model_core_validation.csv).

2. 02-model_risk_gpd.ipynb:
    - Loads the validation results.
    - Calculates the GBR model's residuals (errors).
    - Iterates through each prod_id, fits a GPD model to its positive-error-tail, and calculates the required safety stock for a 98% service level.
    - Saves the per-product safety stock map (stock_seguridad_por_producto.csv).

3. 03-model_inference.ipynb:
    - Loads the hyperparameters from the validated model (gbr_model.joblib).
    - Re-trains the GBR model on the full historical dataset (2012-2023) to ensure maximum accuracy for the future.
    - Generates the base forecast for all 12 months of 2024 (predicciones_demanda_2024.csv).

4. 04-generar_reporte_final.ipynb:
    - Loads the base forecast (from N-03) and the safety stock map (from N-02).
    - Merges them to create the final, actionable report: politica_inventario_final_2024.csv.
    - Generates the project's key visualizations.

## 6. How to Run the Project
1. Clone the repository.

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
.\venv\Scripts\activate   # Windows
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Jupyter notebooks in order (from 01 to 04).

--- 
## 7. Detailed Analysis
---
### **Model Construction & Validation**
1. **Hybrid Model (GBR + GPD)** Our solution consists of two components:

    - **Core Model (GBR)**: A GradientBoostingRegressor was chosen for its high performance in handling seasonality and complex interactions across a large set of 1,000 time series.

    - **Risk Model (GPD)**: We apply Generalized Pareto Distribution (GPD), a technique from Extreme Value Theory (EVT), to the model's positive residuals (under-forecasts). This allows us to precisely model the probability of extreme demand spikes and calculate the exact inventory needed to meet a target service level (98%).

2. **Validation Metrics**

    - **Precision (WAPE)**: We used WAPE (Weighted Absolute Percentage Error) as the primary metric, as it is directly interpretable by the business. On 2023 data, our GBR model was 6.31% more accurate than the client's model_actual.

    - **Risk (Backtesting)**: We validated the GPD model by backtesting it against 2023, confirming that the calculated per-product safety stock would have covered the majority of unexpected demand spikes, reducing lost sales.

3. **Forecast Window (12 Months)**: A 12-month window (all of 2024) was selected for the projection. Given the strong annual seasonality of agricultural products (harvests, consumption seasons), a 12-month window is essential for strategic planning.

### **Measuring Economic Benefit ($)**
To quantify the financial value of this solution, we conducted an Economic Backtest using 2023 data.

- **Metric 1**: Reduced Spoilage Cost (Over-stocking) We simulated inventory decisions month-by-month. Any time a forecast was higher than the actual demand, we calculated the cost of the resulting excess inventory.

    - *Spoilage_Cost = (Forecast - Actual_Sales) * Unit_Cost*

    - Result: Our model, by being more precise and applying risk buffers only where needed, dramatically reduces this cost.

- **Metric 2**: Reduced Opportunity Cost (Lost Sales) This is the key benefit of the GPD Risk Model. We simulated every instance where actual demand exceeded the total ordered stock.

    - *Opportunity_Cost = (Actual_Sales - Total_Stock_Ordered) * Unit_Margin*

    - Result: The model_actual (which only orders the mean forecast) loses all sales from demand spikes. Our hybrid model (Base Forecast + Safety Stock) captures a significant percentage of this lost demand, converting it directly into revenue.

### **Production Methodology (MLOps)**
A model only provides value when it is reliably in production. We propose the following deployment and monitoring (MLOps) framework:

#### **Phase 1: Shadow Mode Deployment**

- Action: For the first month, the pipeline runs in production, but its outputs are not used for purchasing decisions. It runs in parallel with the model_actual.

- Purpose: To compare our model's live predictions against the old model and against reality, validating its performance without operational risk.

#### **Phase 2: Canary Deployment**

- Action: In month two, the model is activated to make real purchasing decisions, but only for a limited, controlled group of products (e..g., 10% of SKUs).

- Purpose: To verify the real-world impact on inventory levels and costs in a contained environment.

#### **Phase 3: Continuous Monitoring & Maintenance Once fully deployed, the model is monitored via automated dashboards**:

- **Performance Monitoring (Model Drift)**: WAPE is tracked monthly. If the error rate begins to climb, an alert is triggered for a Data Scientist to review the model.

- **Data Monitoring (Data Drift)**: Alerts are set to detect if the input data changes suddenly (e.g., a precio_promedio spikes, or a new prod_id appears).

- **Retraining Plan**:

    - **Core Model (GBR)**: Automatically retrained monthly with the latest sales data (as built in notebook 03).

    - **Risk Model (GPD)**: Manually retrained every 6-12 months, as a product's underlying risk profile is more stable than its sales trend.