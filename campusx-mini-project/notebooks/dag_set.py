import mlflow
mlflow.set_tracking_uri('https://dagshub.com/ayushkotadiya5499/mlops-mini-project.mlflow')

import dagshub
dagshub.init(repo_owner='ayushkotadiya5499', repo_name='mlops-mini-project', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)