# BrainRT Analytics

BrainRT Analytics is a no-code Streamlit platform for healthcare professionals to build, evaluate, compare, validate, and apply prediction models in radiotherapy and oncology. It supports clinical model development, voxel-based analysis, a structured knowledge base, and an established-model workflow for model search, comparison, risk calculation, external validation, documentation, and collaboration.

## Main modules

### 1. Clinical Model Development
This module helps users build prediction models from patient-level clinical datasets.

Workflow:
1. **Start / Open clinical project** — create or reopen a local clinical project folder.
2. **Upload Excel** — load and QC the patient-level clinical dataset.
3. **Variables & outcomes** — select predictors, baseline/follow-up variables and derived outcomes.
4. **Treatment groups** — select treatment/grouping variables.
5. **Statistics** — descriptive, inferential and timepoint analyses.
6. **Machine learning** — regression, random forest and XGBoost model development.
7. **Model comparison** — compare generated risk models and export to Established Model.

### 2. Voxel-Based Analysis
This module supports image/dose/mask workflows for voxel-based analysis in radiotherapy.

Workflow:
1. **Start / Open project**
2. **Load patient clinical data**
3. **Load images / masks**
4. **Normalisation**
5. **Reference image / CCS**
6. **Batch registration**
7. **Registration QC**
8. **Warp to reference space**
9. **Dose normalisation**
10. **VBA-ready dataset / Final QC**
11. **Statistical analysis**

### 3. Knowledge Base
The Knowledge Base stores documents and notes under:
- Outcome assessment tools
- Model development tools
- Model evaluation tools
- Clinical applications
  - Brain
  - Head and neck
  - Thorax and abdomen
  - Pelvis
  - Other
- Other

Saved documents can be edited or deleted from the document cards.

### 4. Established Model
The Established Model module supports model reuse and external validation.

Workflow:
1. **Search** — filter and select established models.
2. **Compare** — compare selected models side-by-side.
3. **Risk calculator** — calculate patient-level risk from one selected model.
4. **Validate model using my data** — externally validate one model using a local dataset.
5. **External validation results** — publish and view institute-level validation results.
6. **Documentation** — store model documentation.
7. **Collaborate** — record collaboration notes.

## File structure

The app is currently maintained as a single Streamlit file (`app.py`) for simple deployment. Large section headers divide the file by module and tab. Use `Ctrl+F` to jump to page blocks such as:

- `clinical_start_project`
- `clinical_upload`
- `clinical_model_comparison`
- `voxel_analysis_home`
- `voxel_batch_registration`
- `voxel_statistical_analysis`
- `established_model_search`
- `established_model_validate`
- `knowledge_base`

## Installation

Create a virtual environment and install requirements:

```powershell
py -m venv .venv
.\.venv\Scriptsctivate
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

## Run locally

```powershell
py -m streamlit run app.py
```

## Persistent storage

By default, runtime data are stored in:

```text
datarainrt_persistent_store.sqlite3
```

For a deployed environment, set `BRAINRT_STORAGE_DIR` to a persistent writable folder.

## GitHub upload / push

From the project folder:

```powershell
git init
git add .
git commit -m "Initial BrainRT Analytics app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the remote URL with your own GitHub repository URL.

## Notes for maintenance

- Keep changes modular and patch one page/module at a time.
- Clinical module pages start with `clinical_`.
- VBA pages start with `voxel_`.
- Established Model pages start with `established_model_`.
- Knowledge Base pages are under `knowledge_base` and `kb_` sections.
- The persistent store uses SQLite/pickle helpers near the top of `app.py`.
