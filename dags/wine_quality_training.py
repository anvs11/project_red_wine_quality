from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from config import config  # noqa: E402

MODEL_TO_TRAIN = config.get("active_model", "random_forest")

default_args = {
    'owner': 'vjacheslav-andreev-njw9544',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'wine_quality_training',
    default_args=default_args,
    description='Everyday model training and saving artefacts in DVC',
    schedule_interval='@daily',
    catchup=False
)

dvc_pull_data = BashOperator(
    task_id='dvc_pull_data',
    bash_command='cd /opt/airflow/project && dvc pull data/winequality-red.csv',
    dag=dag
)

# работаю внутри смонтированной папки /opt/airflow/project
train_task = BashOperator(
    task_id='train_model',
    bash_command=f'cd /opt/airflow/project && python src/train.py --model {MODEL_TO_TRAIN}',
    dag=dag
)

dvc_add_and_push = BashOperator(
    task_id='dvc_add_and_push',
    bash_command=('cd /opt/airflow/project &&'
                  'dvc add -f models/rf_model.pkl &&'
                  'dvc add -f models/model_metadata.json &&'
                  'dvc push'),
    dag=dag
)

dvc_pull_data >> train_task >> dvc_add_and_push
