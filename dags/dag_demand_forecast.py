# Contents of: airflow/dags/dag_demand_forecast.py

import sys
import os
from datetime import datetime
from airflow.decorators import dag, task

# --- ROBUST PATHING FOR AIRFLOW (CORRECTED) ---
# __file__ is /opt/airflow/dags/dag_demand_forecast.py
DAG_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# Navigate up ONE level (from /opt/airflow/dags -> /opt/airflow)
PROJECT_ROOT = os.path.abspath(os.path.join(DAG_FILE_DIR, '..'))

# Now, join with 'src' to get /opt/airflow/src
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')

# Add 'src' to the path if it's not already there
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
# --- END OF PATHING BLOCK ---

try:
    # This import will now work
    from pipeline_tasks import run_validation_task, run_inference_task
except ImportError as e:
    print(f"Error importing pipeline_tasks: {e}")
    raise e

# -----------------------------------------------------------------
# DAG DEFINITION
# -----------------------------------------------------------------

@dag(
    dag_id='demand_forecast_pipeline',
    description='MLOps pipeline for agricultural product demand forecasting.',
    schedule_interval='0 1 1 * *',  # 1:00 AM on the 1st of every month
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['mlops', 'forecasting', 'production']
)
def forecast_dag():
    """
    ### Demand Forecast MLOps Pipeline
    Orchestrates the full ML pipeline based on the findings from
    the notebook-based research (economic backtesting).
    """

    @task(task_id='run_validation_and_backtest')
    def validation_task():
        """ Wraps the logic from notebooks 01 and 02. """
        print("Starting: Task 'run_validation_and_backtest'")
        run_validation_task()
        print("Finished: Task 'run_validation_and_backtest'")

    @task(task_id='run_final_inference')
    def inference_task():
        """ Wraps the logic from notebooks 03 and 04. """
        print("Starting: Task 'run_final_inference'")
        run_inference_task()
        print("Finished: Task 'run_final_inference'")

    # --- 3. Define Task Dependencies ---
    validation_task() >> inference_task()

# --- 4. Instantiate the DAG ---
forecast_dag()