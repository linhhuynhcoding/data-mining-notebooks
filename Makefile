
load-env: source ~/python/venv/bin/activate

install-package:
	pip install "apache-airflow[postgres]==${AIRFLOW_VERSION}"   --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
PHOYE: load-env