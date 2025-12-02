# Brain Tumor MRI MLOps Lab (Azure)

Complete Azure ML MLOps lifecycle for binary tumor (yes/no) MRI image classification. Includes baseline model and a Genetic Algorithm (GA) based feature/model optimization.

## High-Level Flow
Ingest Images → Extract & Store Features → Feature Selection → Train (Baseline + GA) → Evaluate → Register/Deploy → Test Endpoint

## Quick Start (Azure ML Studio)
1. Create Azure ML Studio workspace (subscription & resource group).
2. Set up environments from `environments/` (e.g. create Conda env in Studio or attach spec to components).
3. Run `src/ingest_images.py` with your storage/auth credentials to stage data.
4. Define pipeline components from `components/` YAML in Azure ML (feature extraction, selection, training, eval).
5. Submit job from `jobs/` (e.g. feature extraction job) to produce features.
6. Execute remaining `src/` scripts sequentially or via a pipeline (`pipeline_job.py`) to complete all phases.
7. Use `scripts/deploy_endpoint.py` to deploy, then `scripts/test_endpoint.py` to verify predictions.

## Directory Summary
- `data/`: Raw MRI images (`yes/`, `no/`).
- `src/`: Core phase scripts (ingest, extract features, select features, GA optimization inside training, scoring/pipeline).
- `components/`: Azure ML component YAML specs.
- `jobs/`: Job submission YAML (e.g., feature extraction job).
- `featurestore/`: Entity & feature set specs (for consistent offline/online features).
- `scripts/`: Operational helpers (materialize features, deploy endpoint, test endpoint).
- `environments/`: Environment definitions (Conda / Azure ML runtime).





