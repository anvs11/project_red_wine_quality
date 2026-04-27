from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

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

# работаю внутри смонтированной папки /opt/airflow/project
train_task = BashOperator(
    task_id='train_model',
    bash_command='cd /opt/airflow/project && python src/train.py --model random_forest',
    dag=dag
)

dvc_push_task = BashOperator(
    task_id='dvc_push',
    bash_command='cd /opt/airflow/project && dvc push',
    dag=dag
)

train_task >> dvc_push_task