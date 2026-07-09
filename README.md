# MLOps-Integrated-Travel
Integrates Flight price prediction, Hotel recommendation, Gender classification - MLFlow, Streamlit, Rest API, Airflow, CI/CD, Kubernetes

## Repository Structure

```
MLOps-Integrated-Travel/
├── Colab-files/                            # training notebooks (regression, classification, recommender)
├── data/                                    # canonical datasets: flights.csv, hotels.csv, users.csv
├── docs/                                    # documentation outline, video script, K8s walkthrough
├── Hotel_Recommendation_app/                # Streamlit app: hybrid hotel recommender
├── Productionization_flight_price_prediction/  # Flask REST API: flight price regression
│   ├── Dockerfile, Jenkinsfile
│   ├── k8s/                                # deployment.yaml, service.yaml, hpa.yaml
│   ├── pkl_files/, templates/, tests/, mlruns/
│   └── flights.csv                         # intentional duplicate, see note below
├── infra/
│   ├── airflow/                            # Airflow (docker compose): schedules batch price scoring
│   └── jenkins/                            # local Jenkins (docker compose): CI/CD demo environment
├── environment.yml                          # conda env covering all three apps
└── .gitignore
```

Two things that look like clutter at a glance but are deliberate:

- **`Productionization_flight_price_prediction/flights.csv`** duplicates
  `data/flights.csv`. It has to live there: the app's `Dockerfile` builds with that
  folder as its context (`COPY . /app`), so the file must be physically present
  inside it for the Docker image to include it.
- **`infra/jenkins/`** is a self-contained local Jenkins (Docker Compose +
  Configuration-as-Code) used to build and verify
  `Productionization_flight_price_prediction/Jenkinsfile` end-to-end without needing
  a real Jenkins server. It's supporting tooling, not itself a graded deliverable —
  the actual CI/CD pipeline definition is the `Jenkinsfile`.
