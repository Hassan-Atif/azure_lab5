import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Environment

SUBSCRIPTION_ID = "a00dcbea-fd05-4973-82dc-120208b60116"
RESOURCE_GROUP = "rg-60103194"
WORKSPACE_NAME = "goodreads-dbx-60103194"
ENDPOINT_NAME = "tumor-ga-endpoint-60103194"
DEPLOYMENT_NAME = "ga-rf-deployment"
MODEL_NAME = "tumor_ga_model"      
MODEL_STAGE = "Development"       
ENV_NAME = "lab5-ga-env"
ENV_VERSION = "1"                  
INSTANCE_TYPE = "Standard_DS2_v2"
INSTANCE_COUNT = 1


def get_ml_client() -> MLClient:
    cred = DefaultAzureCredential()
    return MLClient(
        credential=cred,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )


def get_latest_model(ml_client: MLClient):
    models = list(ml_client.models.list(name=MODEL_NAME))
    if not models:
        raise RuntimeError(f"No models found with name '{MODEL_NAME}'")
    models_sorted = sorted(
        models,
        key=lambda m: m.creation_context.created_at,
        reverse=True,
    )
    latest = models_sorted[0]
    print(f"Using model: {latest.name}:{latest.version}")
    return latest


def create_or_update_endpoint(ml_client: MLClient):
    endpoint = ManagedOnlineEndpoint(
        name=ENDPOINT_NAME,
        auth_mode="key",  
    )
    endpoint = ml_client.begin_create_or_update(endpoint).result()
    print(f"Endpoint '{ENDPOINT_NAME}' is ready.")
    return endpoint


def create_or_update_deployment(ml_client: MLClient, model_id: str):
    env = Environment(name=ENV_NAME, version=ENV_VERSION)

    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=model_id,
        environment=env,
        code_path="src",         
        scoring_script="score.py",
        instance_type=INSTANCE_TYPE,
        instance_count=INSTANCE_COUNT,
    )

    deployment = ml_client.begin_create_or_update(deployment).result()
    print(f"Deployment '{DEPLOYMENT_NAME}' is ready.")
    endpoint = ml_client.online_endpoints.get(name=ENDPOINT_NAME)
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    endpoint = ml_client.begin_create_or_update(endpoint).result()
    print(f"Updated endpoint traffic: {endpoint.traffic}")


def main():
    ml_client = get_ml_client()
    latest_model = get_latest_model(ml_client)

    create_or_update_endpoint(ml_client)
    create_or_update_deployment(ml_client, model_id=latest_model.id)


if __name__ == "__main__":
    main()
