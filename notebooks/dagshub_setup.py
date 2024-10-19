import mlflow
import dagshub

mlflow.set_tracking_uri('https://dagshub.com/pratik2524/Mlops_mini_project.mlflow')
dagshub.init(repo_owner='pratik2524', repo_name='Mlops_mini_project', mlflow=True)


with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)