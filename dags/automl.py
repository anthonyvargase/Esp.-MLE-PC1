import airflow
import os
import zipfile
import pandas as pd
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from kaggle.api.kaggle_api_extended import KaggleApi
from pycaret.classification import *
from joblib import dump, load

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 7, 14),
    'email': ['anthony.vaescobedo@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_workflow_pc1_mle_ed3',
    default_args=default_args,
    description='Pipeline de PC1',
    schedule_interval='0 17 * * *',
)

def GetDataKaggle():
    os.environ['KAGGLE_USERNAME'] = 'anthonyvargase'
    os.environ['KAGGLE_KEY'] = 'ed0859047e119f6ab37904ecde9f84dd'
    os.environ['KAGGLE_CONFIG_DIR'] = os.path.join(os.getcwd(), '.kaggle')

    api = KaggleApi()
    api.authenticate()

    competition_name = 'playground-series-s4e6'
    download_path = 'Data/'
    api.competition_download_files(competition_name, path=download_path)

    # Descomprimir el archivo zip
    zip_file_path = os.path.join(download_path, f"{competition_name}.zip")

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(download_path)


def AutoML_PyCaret(ti):
    train_data = pd.read_csv('Data/train.csv')
    test_data = pd.read_csv('Data/test.csv')

    # Realiza todas las transformaciones y preparaciones necesarias en los datos
    # Los datos preprocesados se almacenan internamente en pycaret
    clf1 = setup(data=train_data, target='Target', 
                 normalize=True,
                 transformation=True,
                 combine_rare_levels=True,
                 remove_multicollinearity=True,
                 multicollinearity_threshold=0.95,
                 ignore_low_variance=True)

    # Entrena y evalua diferentes modelos quedandose con el mejor
    best_model = compare_models()
    final_model = finalize_model(best_model)

    # Guarda el modelo
    model_path = '/opt/airflow/dags/data/best_model.joblib'
    dump(final_model, model_path)

    # Aplica las mismas transformaciones del best_model a los datos de prueba    
    preprocessed_test_data = predict_model(final_model, data=test_data)
    preprocessed_test_data.to_csv('Data/test_preprocessed.csv', index=False)

    # Guarda la ruta del modelo usando XCom
    ti.xcom_push(key='model_path', value=model_path)



def SubmitKaggle(ti):
    model_path = ti.xcom_pull(task_ids='AutoML_PyCaret', key='model_path')

    api = KaggleApi()
    api.authenticate()

    api.competition_submit(file_name="submissions/submission_0.csv",
    message="PC1 MLE ed3 - Anthony Vargas Escobedo",
    competition="playground-series-s4e4")


GetDataKaggle_task = PythonOperator(
    task_id='GetDataKaggle',
    python_callable=GetDataKaggle,
    provide_context=True,
    dag=dag,
)

AutoML_PyCaret_task = PythonOperator(
    task_id='AutoML_PyCaret',
    python_callable=AutoML_PyCaret,
    provide_context=True,
    dag=dag,
)

SubmitKaggle_task = PythonOperator(
    task_id='SubmitKaggle',
    python_callable=SubmitKaggle,
    provide_context=True,
    dag=dag,
)

GetDataKaggle_task >> AutoML_PyCaret_task >> SubmitKaggle_task