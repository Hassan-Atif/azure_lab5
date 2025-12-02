from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, dsl, load_component

SUBSCRIPTION_ID = "a00dcbea-fd05-4973-82dc-120208b60116"
RESOURCE_GROUP = "rg-60103194"
WORKSPACE_NAME = "goodreads-dbx-60103194"

COMPUTE_NAME = "lab-5-cluster"
FEATURES_DATA_ASSET = "azureml:azureml_olden_tooth_qkdfkgrpr3_output_data_output_parquet:1"

FEATURE_SET_VERSION = "2"


def get_ml_client() -> MLClient:
    """Create an MLClient for the workspace."""
    cred = DefaultAzureCredential()
    return MLClient(
        credential=cred,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )

feature_retrieval_component = load_component(source="components/feature_retrieval.yml")
feature_selection_component = load_component(source="components/feature_selection.yml")
train_eval_component = load_component(source="components/train_eval.yml")

@dsl.pipeline(
    compute=COMPUTE_NAME,
    description="Tumor MRI feature pipeline: retrieval -> feature selection -> training",
)
def tumor_pipeline_job(features_parquet: Input(type="uri_file")):

    retrieval_step = feature_retrieval_component(
        features_parquet=features_parquet
    )

    feature_selection_step = feature_selection_component(
        train_parquet=retrieval_step.outputs.train_parquet
    )

    train_eval_step = train_eval_component(
        train_parquet=retrieval_step.outputs.train_parquet,
        test_parquet=retrieval_step.outputs.test_parquet,
        selected_features_json=feature_selection_step.outputs.selected_features_json,
        feature_set_version=FEATURE_SET_VERSION,
    )
    retrieval_step.compute = COMPUTE_NAME
    feature_selection_step.compute = COMPUTE_NAME
    train_eval_step.compute = COMPUTE_NAME

    return {
        # From A
        "train_parquet": retrieval_step.outputs.train_parquet,
        "test_parquet": retrieval_step.outputs.test_parquet,
        # From B
        "selected_features_json": feature_selection_step.outputs.selected_features_json,
        "baseline_metrics_json": feature_selection_step.outputs.baseline_metrics_json,
        "ga_metrics_json": feature_selection_step.outputs.ga_metrics_json,
        # From C
        "training_outputs": train_eval_step.outputs.output_dir,
    }

if __name__ == "__main__":
    ml_client = get_ml_client()

    features_input = Input(
        type="uri_file",
        path=FEATURES_DATA_ASSET,
    )

    pipeline_job = tumor_pipeline_job(
        features_parquet=features_input
    )

    submitted_job = ml_client.jobs.create_or_update(
        pipeline_job,
        experiment_name="tumor_mri_pipeline",
    )

    print("Pipeline submitted!")
    print("Job name:", submitted_job.name)
    if hasattr(submitted_job, "studio_url"):
        print("Studio URL:", submitted_job.studio_url)
