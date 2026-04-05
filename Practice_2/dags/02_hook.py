from datetime import datetime
import json
import logging
import os

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.timetables.interval import CronDataIntervalTimetable
from hooks import CarsHook  # ← импортируем из plugins/hooks/

# from hooks import MovielensHook

# with DAG(
#     dag_id="02_hook",
#     description="Fetches ratings from the Movielens API using a custom hook.",
#     start_date=datetime(2023, 1, 1),
#     end_date=datetime(2023, 1, 10),
#     schedule=CronDataIntervalTimetable("@daily", "UTC"),
#     catchup=True,
# ):

#     def _fetch_ratings(conn_id:str, templates_dict:dict, batch_size:int=1000, **_):
#         logger = logging.getLogger(__name__)

#         start_date = templates_dict["start_date"]
#         end_date = templates_dict["end_date"]
#         output_path = templates_dict["output_path"]

#         logger.info(f"Fetching ratings for {start_date} to {end_date}")
#         hook = MovielensHook(conn_id=conn_id)
#         ratings = list(hook.get_ratings(start_date=start_date, end_date=end_date, batch_size=batch_size))
#         logger.info(f"Fetched {len(ratings)} ratings")

#         logger.info(f"Writing ratings to {output_path}")

#         # Make sure output directory exists.
#         output_dir = os.path.dirname(output_path)
#         os.makedirs(output_dir, exist_ok=True)

#         with open(output_path, "w") as file_:
#             json.dump(ratings, fp=file_)

#     PythonOperator(
#         task_id="fetch_ratings",
#         python_callable=_fetch_ratings,
#         op_kwargs={"conn_id": "movielens"},
#         templates_dict={
#             "start_date": "{{data_interval_start | ds}}",
#             "end_date": "{{data_interval_end | ds}}",
#             "output_path": "/data/custom_hook/{{data_interval_start | ds}}.json",
#         },
#     )


def _fetch_cars(conn_id: str, templates_dict: dict, batch_size: int = 1000, **_):
    logger = logging.getLogger(__name__)
    output_path = templates_dict["output_path"]

    logger.info("Fetching all cars from the API...")
    hook = CarsHook(conn_id=conn_id)
    cars = list(hook.get_cars(batch_size=batch_size))
    logger.info(f"Fetched {len(cars)} car records")

    # Убедимся, что директория существует
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(cars, f)

    logger.info(f"Saved cars to {output_path}")

def _clean_cars_data(templates_dict: dict, **_):

    logger = logging.getLogger(__name__)

    input_path = templates_dict["input_path"]
    output_path = templates_dict["output_path"]

    with open(input_path, "r") as f:
        cars = json.load(f)

    cars = [c for c in cars if None not in c.values()]

    unique_cars = [dict(t) for t in {tuple(sorted(d.items())) for d in cars}]

    fuel_map = {}
    transmission_map = {}

    def encode(value, mapping):
        if value not in mapping:
            mapping[value] = len(mapping)
        return mapping[value]

    for car in unique_cars:
        if "Fuel_type" in car:
            car["Fuel_type"] = encode(car["Fuel_type"], fuel_map)
        if "Transmission" in car:
            car["Transmission"] = encode(car["Transmission"], transmission_map)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(unique_cars, f)

with DAG(
    dag_id="02_hook",
    description="Fetches car data from the custom API using a custom hook.",
    start_date=datetime(2026, 2, 3),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
) as dag:

    fetch_task = PythonOperator(
        task_id="fetch_cars",
        python_callable=_fetch_cars,
        op_kwargs={"conn_id": "carsapi"},
        templates_dict={
            "output_path": "/data/custom_hook/cars.json",
        },
    )

    clean_task = PythonOperator(
        task_id="clean_cars_data",
        python_callable=_clean_cars_data,
        templates_dict={
            "input_path": "/data/custom_hook/cars.json",
            "output_path": "/data/cleaned/cars_cleaned.json",
        },
    )

fetch_task >> clean_task
