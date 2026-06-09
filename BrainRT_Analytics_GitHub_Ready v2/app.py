"""
BrainRT Analytics
=================

No-code Streamlit application for healthcare professionals to build, compare,
validate, and apply clinical prediction models, voxel-based radiotherapy analyses,
and knowledge-base resources.

Code organisation map
---------------------
1. Imports and page setup
2. Persistent storage helpers
3. Session state and navigation
4. Clinical module helpers
5. Voxel-based analysis helpers
6. Knowledge Base helpers/pages
7. Established Model helpers/pages
8. Page render blocks

Maintenance note
----------------
This app is intentionally kept as a single Streamlit file for easy deployment.
Large section headers mark each module and each user-facing tab so the file can be
searched quickly. Use Ctrl+F for page names such as:
- clinical_start_project
- clinical_upload
- voxel_analysis_home
- voxel_batch_registration
- established_model_search
- established_model_validate
- knowledge_base
"""

import os
import re
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shutil
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False
from matplotlib.path import Path as MplPath

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except Exception:
    NIBABEL_AVAILABLE = False

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except Exception:
    PYDICOM_AVAILABLE = False

try:
    import SimpleITK as sitk
    SIMPLEITK_AVAILABLE = True
except Exception:
    SIMPLEITK_AVAILABLE = False

from scipy import stats

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="BrainRT Analytics",
    page_icon="🧠",
    layout="wide",
)



# ============================================================
# PERSISTENT STORAGE
# ============================================================

# GitHub stores the app code, not runtime data. These helpers persist shared app
# state to a SQLite database so exported models, knowledge-base entries, and
# aggregate validation summaries survive app restarts on deployments with a
# persistent writable volume.
#
# For production cloud deployment, set BRAINRT_STORAGE_DIR to a persistent
# mounted folder. If not set, the app uses ./data.

PERSISTENT_STORAGE_DIR = Path(os.getenv("BRAINRT_STORAGE_DIR", "data"))
PERSISTENT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
PERSISTENT_DB_PATH = PERSISTENT_STORAGE_DIR / "brainrt_persistent_store.sqlite3"


def init_persistent_store():
    """Create the persistent key-value table if it does not exist."""
    with sqlite3.connect(PERSISTENT_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_state (
                key TEXT PRIMARY KEY,
                value BLOB NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def persistent_save(key, value):
    """Save a Python object to SQLite using pickle."""
    init_persistent_store()
    blob = pickle.dumps(value)
    with sqlite3.connect(PERSISTENT_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO app_state (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
            """,
            (key, blob),
        )
        conn.commit()


def persistent_load(key, default_value):
    """Load a Python object from SQLite. Return default_value if missing/corrupt."""
    init_persistent_store()
    try:
        with sqlite3.connect(PERSISTENT_DB_PATH) as conn:
            row = conn.execute(
                "SELECT value FROM app_state WHERE key = ?",
                (key,),
            ).fetchone()

        if row is None:
            return default_value

        return pickle.loads(row[0])
    except Exception:
        return default_value


def load_persistent_state_once():
    """Load shared persistent state into st.session_state once per session."""
    if st.session_state.get("_brainrt_persistent_state_loaded", False):
        return

    st.session_state.established_calculators = persistent_load(
        "established_calculators",
        st.session_state.get("established_calculators", {}),
    )

    established_workflow_state = persistent_load("established_model_workflow_state", {})
    if isinstance(established_workflow_state, dict):
        for key, value in established_workflow_state.items():
            st.session_state[key] = value

    st.session_state.knowledge_base_documents = persistent_load(
        "knowledge_base_documents",
        st.session_state.get(
            "knowledge_base_documents",
            {
                "Outcome assessment tools": [],
                "Model development tools": [],
                "Model evaluation tools": [],
                "Clinical applications": [],
                "Other": [],
            },
        ),
    )

    st.session_state._brainrt_persistent_state_loaded = True


def save_established_calculators_persistent():
    """Persist the Established Model library."""
    persistent_save(
        "established_calculators",
        st.session_state.get("established_calculators", {}),
    )


def save_established_model_workflow_state_persistent():
    """Persist user activity/state from the Established Model workflow.

    This keeps selections, comparison sets, risk-calculator outputs and pending
    external-validation results available after a browser refresh or app restart.
    Patient-level data are only stored here when the user explicitly runs a
    validation/risk calculation in the Established Model workflow.
    """
    state_keys = [
        "established_search_has_run",
        "est_search_name",
        "est_search_site",
        "est_search_outcome",
        "est_search_method",
        "est_result_sort_order",
        "established_selected_models_for_compare",
        "established_models_for_risk_calculator",
        "established_risk_selected_model",
        "established_risk_calculator_last_result",
        "established_external_validation_selected_model",
        "established_validate_selected_model",
        "established_local_validation_results",
        "established_local_validation_predictions",
        "established_local_validation_results_path",
        "established_local_validation_model_key",
        "established_pending_external_validation_result",
        "established_pending_external_validation_predictions",
        "established_last_exported_model",
    ]
    workflow_state = {key: st.session_state.get(key) for key in state_keys if key in st.session_state}
    persistent_save("established_model_workflow_state", workflow_state)


def save_established_model_everything_persistent():
    """Persist the Established Model library and current workflow state together."""
    save_established_calculators_persistent()
    save_established_model_workflow_state_persistent()


def save_knowledge_base_documents_persistent():
    """Persist the Knowledge Base document registry."""
    persistent_save(
        "knowledge_base_documents",
        st.session_state.get("knowledge_base_documents", {}),
    )



# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "page": "home",
    "df": None,
    "input_variables": [],
    "baseline_variable": "",
    "baseline_variables": [],
    "followup_variable": "",
    "followup_variables": [],
    "primary_followup_variable": "",
    "outcome_mapping_rows": [],
    "primary_derived_outcome": "",
    "decline_threshold": 2.0,
    "treatment_variable": "",
    "treatment_options": [],
    "analytics_method": "Regression",
    "trained_pipeline": None,
    "trained_predictors": [],
    "trained_model_name": "",
    "trained_models": {},
    "established_calculators": {},
    "include_treatment_in_model": True,
    "code_viewer_return_page": "home",
    "code_viewer_target_page": "home",
    "clinical_project_folder": "",
    "clinical_project_setup": {},
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

load_persistent_state_once()

# Safety reset: if Streamlit remembers an old page name from a previous app version,
# reset to home so the page does not appear blank.
valid_pages = {
    "home",
    "clinical_start_project",
    "clinical_upload",
    "clinical_variables",
    "clinical_treatment",
    "clinical_analysis",
    "clinical_descriptive",
    "clinical_inferential",
    "clinical_timepoint",
    "clinical_spider_chart",
    "clinical_model_selection",
    "clinical_model_calculation",
    "clinical_model_comparison",
    "risk_calculator_only",
    "generated_risk_calculator_page",
    "model_export_page",
    "established_model",
    "established_model_search",
    "established_model_compare",
    "established_model_risk_calculator",
    "established_model_validate",
    "established_model_external_validation",
    "established_model_documentation",
    "established_model_collaborate",
    "model_development_home",
    "supporting_documentation",
    "knowledge_base",
    "kb_outcome_assessment_tools",
    "kb_model_development_tools",
    "kb_model_evaluation_tools",
    "kb_clinical_applications",
    "kb_other",
    "voxel_analysis_home",
    "voxel_start_project",
    "voxel_load_patient_data",
    "voxel_load_images",
    "voxel_registration_alignment",
    "voxel_batch_registration",
    "voxel_registration_qc",
    "voxel_warp_to_ccs",
    "voxel_dose_normalisation",
    "voxel_vba_ready_dataset",
    "voxel_statistical_analysis",
    "voxel_reference_ccs",
    "page_code_viewer",
}

if st.session_state.page not in valid_pages:
    st.session_state.page = "home"



# ============================================================
# HELPER FUNCTIONS
# ============================================================

# ============================================================
# PAGE CODE / STEPS VIEWER
# ============================================================

PAGE_STEP_GUIDES = {
    "home": [
        "Open the app home page.",
        "Choose Model Development or Established Model.",
        "The left workflow path appears after entering a workflow."
    ],
    "model_development_home": [
        "Choose whether to develop a new model or work with an established model.",
        "Select Clinical Model or Voxel-based Analysis.",
        "The workflow path on the left updates according to the selected module."
    ],
    "clinical_start_project": [
        "Create or open a clinical model project.",
        "Create local folders matching the clinical workflow tabs.",
        "Reload saved clinical datasets when reopening a project.",
        "Save project-level notes and output locations."
    ],
    "clinical_upload": [
        "Upload the patient-level Excel file.",
        "Preview the uploaded data.",
        "Check that patient ID, predictors and outcome columns are present.",
        "Proceed to variable and outcome selection."
    ],
    "clinical_variables": [
        "Select input predictor variables.",
        "Select baseline and follow-up outcome variables.",
        "Create derived decline outcomes where needed.",
        "Preview derived variables and continue to treatment selection."
    ],
    "clinical_treatment": [
        "Select the treatment/grouping column.",
        "Choose which treatment groups to include.",
        "Store treatment settings for statistics and machine learning."
    ],
    "clinical_analysis": [
        "Open the statistics hub.",
        "Choose descriptive statistics, inferential statistics, timepoint analysis, or domain-specific decline proportion.",
        "Machine learning is kept separate in Step 5."
    ],
    "clinical_descriptive": [
        "Select variables to summarise.",
        "Calculate descriptive statistics.",
        "Check distributions and visual summaries."
    ],
    "clinical_inferential": [
        "Select variables and treatment groups.",
        "Run appropriate statistical tests.",
        "Review p-values and significant comparisons."
    ],
    "clinical_timepoint": [
        "Select baseline and follow-up timepoints.",
        "Calculate change from baseline.",
        "Compare change between treatment groups.",
        "Generate timepoint plots."
    ],
    "clinical_spider_chart": [
        "Select baseline domain variables.",
        "Match each baseline domain to a follow-up domain.",
        "Define decline threshold.",
        "Compare percentage of patients with decline in each domain across groups."
    ],
    "clinical_model_selection": [
        "Choose Regression, Random Forest or XGBoost.",
        "Proceed to model development.",
        "The selected model uses outcomes and predictors from earlier steps."
    ],
    "clinical_model_comparison": [
        "View all generated clinical risk models.",
        "Compare training and validation performance side-by-side.",
        "Enter one patient profile and compare predicted risks across models.",
        "Save the model comparison table inside the clinical project folder."
    ],
    "established_model": [
        "Search established models.",
        "Filter by model type, method, site and outcome.",
        "Select a model.",
        "Open model description and risk calculator."
    ],
    "supporting_documentation": [
        "Load supporting publications.",
        "Load TRIPOD or AI assessment documentation.",
        "Link documentation to established models."
    ],
    "voxel_analysis_home": [
        "Open voxel-based analysis workflow.",
        "Start a project, load patient data, load images/masks, then prepare preprocessing."
    ],
    "voxel_start_project": [
        "Start or open a VBA project.",
        "Create a new project folder or open an existing project folder.",
        "Save or reload project-level metadata and output locations."
    ],
    "voxel_load_patient_data": [
        "Upload patient-level clinical Excel data.",
        "Check patient ID and covariate/outcome structure.",
        "Run data quality checks.",
        "Map patient ID, covariates, baseline and follow-up outcomes."
    ],
    "voxel_load_images": [
        "Select DICOM or NIfTI input format.",
        "Load patient image folders/files.",
        "Run filename and patient-ID matching checks.",
        "Select the reference image/plane for batch processing.",
        "Prepare the viewer after reference selection."
    ],
    "voxel_registration_qc": [
        "Review the batch registration results before warping dose and masks.",
        "Check patient-level registration status, readable outputs and geometry consistency.",
        "Use visual overlay review and mark each patient as approved or needing manual review.",
        "Save the registration QC summary inside the project folder."
    ],
    "voxel_dose_normalisation": [
        "Read registered dose outputs from batch registration.",
        "Select dose scaling or unit conversion.",
        "Save dose-normalised outputs and a summary CSV inside the project folder.",
        "Preview the saved dose-normalisation table before continuing."
    ],
    "voxel_vba_ready_dataset": [
        "Combine clinical data, warped images/masks and dose-normalised outputs.",
        "Check patient-level readiness for voxel-based analysis.",
        "Save the VBA-ready dataset manifest inside the project folder.",
        "Review missing files and geometry mismatches before running voxel-wise statistics."
    ],
    "voxel_statistical_analysis": [
        "Select the VBA statistic to calculate.",
        "Select multiple-comparison correction.",
        "Choose voxel filters or masks.",
        "Select adjustment variables and run the statistical analysis with progress tracking."
    ],
}


def get_page_source_block(page_name):
    """
    Extract the source code block that renders the current Streamlit page.

    This is intentionally a lightweight viewer, not a full IDE.
    """
    try:
        source_path = Path(__file__)
        app_source = source_path.read_text(encoding="utf-8")
    except Exception as error:
        return f"Could not read app source file: {error}"

    page_marker = f'elif st.session_state.page == "{page_name}":'

    if page_name == "home":
        # Home usually starts with an if rather than elif.
        candidates = [
            'if st.session_state.page == "home":',
            'elif st.session_state.page == "home":',
        ]
    else:
        candidates = [page_marker, f'if st.session_state.page == "{page_name}":']

    start = -1
    for marker in candidates:
        start = app_source.find(marker)
        if start != -1:
            break

    if start == -1:
        return f"No specific page block was found for page: {page_name}"

    next_page = app_source.find('\nelif st.session_state.page == "', start + 1)
    next_section = app_source.find("\n# ============================================================", start + 1)

    candidates_end = [
        pos for pos in [next_page, next_section]
        if pos != -1 and pos > start
    ]

    end = min(candidates_end) if candidates_end else min(len(app_source), start + 8000)

    block = app_source[start:end].strip()

    if len(block) > 12000:
        block = block[:12000] + "\n\n# ... code preview truncated ..."

    return block


def open_code_steps_page(target_page=None):
    """
    Open the full-page code/steps viewer for the current page.
    """
    if target_page is None:
        target_page = st.session_state.get("page", "home")

    if target_page != "page_code_viewer":
        st.session_state.code_viewer_return_page = target_page
        st.session_state.code_viewer_target_page = target_page

    go_to("page_code_viewer")


def render_code_icon_button(location="sidebar"):
    """
    Small code icon button. Opens a full page instead of an expander.
    """
    current_page = st.session_state.get("page", "home")

    if current_page == "page_code_viewer":
        return

    container = st.sidebar if location == "sidebar" else st

    if location == "sidebar":
        col1, col2 = container.columns([0.25, 0.75])
        with col1:
            if st.button("💻", help="Open page code and steps", key=f"open_code_steps_{current_page}"):
                open_code_steps_page(current_page)
        with col2:
            st.caption("Page code / steps")
    else:
        if st.button("💻 Code / steps", help="Open page code and steps", key=f"open_code_steps_main_{current_page}"):
            open_code_steps_page(current_page)


def go_to(page_name):
    st.session_state.page = page_name
    st.rerun()


def existing_columns(df, columns):
    """
    Keep only columns that exist in the current dataframe.
    This prevents old session-state variables from breaking a new uploaded Excel file.
    """
    if df is None:
        return []
    return [col for col in columns if col in df.columns]


def get_columns():
    if st.session_state.df is None:
        return []
    return list(st.session_state.df.columns)


def reset_trained_model():
    st.session_state.trained_pipeline = None
    st.session_state.trained_predictors = []
    st.session_state.trained_model_name = ""
    st.session_state.trained_models = {}


def simplify_distribution_label(distribution_text):
    """
    Convert Step 4A distribution wording to a simple label:
    Normal / Non-normal / Not assessed.

    This function must be defined before Step 4A runs.
    """
    text = str(distribution_text).strip().lower()

    if "non" in text:
        return "Non-normal"

    if "normal" in text:
        return "Normal"

    return "Not assessed"


def get_current_workflow():
    """
    Identify which workflow is active so the navigation panel shows the correct path.
    """
    page = st.session_state.get("page", "home")

    if page.startswith("voxel_"):
        return "Voxel-based Analysis"

    if page.startswith("clinical_"):
        return "Clinical Model"

    if page == "page_code_viewer":
        return "Page code / steps"

    if page in ["model_development_home"]:
        return "Model Development"

    if page == "knowledge_base" or page.startswith("kb_"):
        return "Knowledge Base"

    if page == "established_model" or page.startswith("established_model_") or page == "supporting_documentation":
        return "Established Model"

    return "Home"


def get_navigation_steps():
    """
    Context-sensitive navigation path.

    The first item is always Home, as requested.
    """
    page = st.session_state.get("page", "home")

    home_step = ("🏠 Home", "home")

    if page.startswith("voxel_"):
        return [
            home_step,
            ("🧪 Model Development", "model_development_home"),
            ("🧠 Voxel-based Analysis", "voxel_analysis_home"),
            ("🚀 Start / Open project", "voxel_start_project"),
            ("📁 Load patient clinical data", "voxel_load_patient_data"),
            ("🖼️ Load images / masks", "voxel_load_images"),
            ("🛠️ Normalisation", "voxel_registration_alignment"),
            ("🧭 Reference image / CCS", "voxel_reference_ccs"),
            ("⚙️ Batch registration", "voxel_batch_registration"),
            ("🔎 Registration QC", "voxel_registration_qc"),
            ("🌀 Warp to reference space", "voxel_warp_to_ccs"),
            ("🧮 Dose normalisation", "voxel_dose_normalisation"),
            ("✅ VBA-ready dataset / Final QC", "voxel_vba_ready_dataset"),
            ("📊 Statistical analysis", "voxel_statistical_analysis"),
        ]

    if page.startswith("clinical_"):
        return [
            home_step,
            ("🧪 Model Development", "model_development_home"),
            ("📊 Clinical Model", "clinical_start_project"),
            ("🚀 Start / Open clinical project", "clinical_start_project"),
            ("1. Upload Excel", "clinical_upload"),
            ("2. Variables & outcomes", "clinical_variables"),
            ("3. Treatment", "clinical_treatment"),
            ("4. Statistics", "clinical_analysis"),
            ("5. Machine learning", "clinical_model_selection"),
            ("6. Model comparison", "clinical_model_comparison"),
                                ]

    if page == "model_development_home":
        return [
            home_step,
            ("🧪 Model Development", "model_development_home"),
            ("📊 Clinical Model", "clinical_start_project"),
            ("🧠 Voxel-based Analysis", "voxel_analysis_home"),
            ("📚 Knowledge Base", "knowledge_base"),
        ]

    if page == "knowledge_base" or page.startswith("kb_"):
        return [
            home_step,
            ("📚 Knowledge Base", "knowledge_base"),
            ("🧠 Outcome assessment tools", "kb_outcome_assessment_tools"),
            ("🛠️ Model development tools", "kb_model_development_tools"),
            ("📏 Model evaluation tools", "kb_model_evaluation_tools"),
            ("🏥 Clinical applications", "kb_clinical_applications"),
            ("📦 Other", "kb_other"),
        ]

    if page == "established_model" or page.startswith("established_model_") or page == "supporting_documentation":
        return [
            home_step,
            ("📌 Established Model", "established_model"),
            ("🔎 Search", "established_model_search"),
            ("📊 Compare", "established_model_compare"),
            ("🧮 Risk calculator", "established_model_risk_calculator"),
            ("✅ Validate model using my data", "established_model_validate"),
            ("🌍 External validation results", "established_model_external_validation"),
            ("📎 Documentation", "established_model_documentation"),
            ("🤝 Collaborate", "established_model_collaborate"),
        ]

    if page == "page_code_viewer":
        return [
            home_step,
            ("💻 Code / steps", "page_code_viewer"),
        ]

    return [home_step]


def render_step_sidebar():
    """
    Navigation panel with full workflow path.
    It is hidden on the Home page and appears once the user opens Clinical Model,
    Voxel-based Analysis, Model Development, or Established Model.
    """
    current_workflow = get_current_workflow()

    st.sidebar.title("🧠 BrainRT")
    st.sidebar.caption(current_workflow)
    st.sidebar.caption(f"Storage: {PERSISTENT_DB_PATH}")

    trained_models = st.session_state.get("trained_models", {})

    def is_step_complete(page_name):
        if page_name == "home":
            return True
        if page_name == "model_development_home":
            return True
        if page_name == "voxel_analysis_home":
            return True
        if page_name == "clinical_start_project":
            return bool(st.session_state.get("clinical_project_setup", {})) or bool(st.session_state.get("clinical_project_folder", ""))
        if page_name == "voxel_start_project":
            return bool(st.session_state.get("voxel_project_setup", {}))
        if page_name == "voxel_load_patient_data":
            return st.session_state.get("voxel_patient_data", None) is not None
        if page_name == "voxel_load_images":
            file_df = st.session_state.get("voxel_loaded_image_files_df", pd.DataFrame())
            return file_df is not None and not file_df.empty
        if page_name == "voxel_registration_alignment":
            # Normalisation is complete when a non-empty results table exists in memory
            # or when the saved project CSV exists. Use direct paths here because the
            # sidebar is rendered before the later normalisation helper functions are defined.
            norm_df = st.session_state.get("vbv_normalisation_results", pd.DataFrame())
            if norm_df is not None and isinstance(norm_df, pd.DataFrame) and not norm_df.empty:
                return True
            results_csv = st.session_state.get("vbv_normalisation_results_csv", "")
            if results_csv and Path(str(results_csv)).exists():
                return True
            project_folder = st.session_state.get("voxel_project_folder", "")
            if project_folder:
                normalisation_csv = Path(project_folder) / "03_Normalisation" / "normalisation_results.csv"
                if normalisation_csv.exists():
                    return True
            return False
        if page_name == "voxel_reference_ccs":
            return bool(st.session_state.get("voxel_reference_ccs_setup", {})) or bool(st.session_state.get("vbv_reference_setup", {}))
        if page_name == "voxel_batch_registration":
            batch_df = st.session_state.get("vbv_batch_registration_summary", pd.DataFrame())
            if batch_df is not None and isinstance(batch_df, pd.DataFrame) and not batch_df.empty:
                return True
            project_folder = st.session_state.get("voxel_project_folder", "")
            if project_folder:
                batch_csv = Path(project_folder) / "05_Batch_Registration" / "batch_registration_summary.csv"
                if batch_csv.exists():
                    return True
            return False
        if page_name == "voxel_registration_qc":
            qc_df = st.session_state.get("vbv_registration_qc_summary", pd.DataFrame())
            if qc_df is not None and isinstance(qc_df, pd.DataFrame) and not qc_df.empty:
                return True
            project_folder = st.session_state.get("voxel_project_folder", "")
            if project_folder:
                qc_csv = Path(project_folder) / "06_Registration_QC" / "registration_qc_summary.csv"
                if qc_csv.exists():
                    return True
            return False
        if page_name == "voxel_warp_to_ccs":
            warp_df = st.session_state.get("vbv_warp_manifest", pd.DataFrame())
            if warp_df is not None and isinstance(warp_df, pd.DataFrame) and not warp_df.empty:
                return True
            project_folder = st.session_state.get("voxel_project_folder", "")
            if project_folder:
                warp_csv = Path(project_folder) / "07_Warp_To_Reference_Space" / "warp_to_reference_manifest.csv"
                if warp_csv.exists():
                    return True
            return False
        if page_name == "voxel_dose_normalisation":
            dose_df = st.session_state.get("vbv_dose_normalisation_summary", pd.DataFrame())
            if dose_df is not None and isinstance(dose_df, pd.DataFrame) and not dose_df.empty:
                return True
            project_folder = st.session_state.get("voxel_project_folder", "")
            if project_folder:
                dose_csv = Path(project_folder) / "08_Dose_Normalisation" / "dose_normalisation_summary.csv"
                if dose_csv.exists():
                    return True
            return False
        if page_name == "voxel_vba_ready_dataset":
            ready_df = st.session_state.get("vbv_vba_ready_manifest", pd.DataFrame())
            if ready_df is not None and isinstance(ready_df, pd.DataFrame) and not ready_df.empty:
                return True
            project_folder = st.session_state.get("voxel_project_folder", "")
            if project_folder:
                ready_csv = Path(project_folder) / "09_VBA_Ready_Dataset_Final_QC" / "vba_ready_dataset_manifest.csv"
                if ready_csv.exists():
                    return True
            return False
        if page_name == "voxel_statistical_analysis":
            stats_df = st.session_state.get("vbv_statistical_analysis_summary", pd.DataFrame())
            if stats_df is not None and isinstance(stats_df, pd.DataFrame) and not stats_df.empty:
                return True
            project_folder = st.session_state.get("voxel_project_folder", "")
            if project_folder:
                stats_csv = Path(project_folder) / "10_Statistical_Analysis" / "statistical_analysis_run_summary.csv"
                if stats_csv.exists():
                    return True
            return False
        if page_name == "clinical_upload":
            if st.session_state.df is not None:
                return True
            clinical_folder = st.session_state.get("clinical_project_folder", "")
            if clinical_folder:
                return (Path(clinical_folder) / "01_Upload_Excel" / "clinical_data_clean_copy.csv").exists()
            return False
        if page_name == "clinical_variables":
            return (
                st.session_state.df is not None
                and len(st.session_state.get("input_variables", [])) > 0
                and st.session_state.get("primary_derived_outcome", "") != ""
            )
        if page_name == "clinical_treatment":
            return st.session_state.get("treatment_variable", "") != ""
        if page_name == "clinical_analysis":
            return st.session_state.get("treatment_variable", "") != ""
        if page_name == "clinical_model_selection":
            return st.session_state.get("analytics_method", "") != ""
        if page_name == "clinical_model_comparison":
            comp_df = st.session_state.get("clinical_model_comparison_df", pd.DataFrame())
            if comp_df is not None and isinstance(comp_df, pd.DataFrame) and not comp_df.empty:
                return True
            clinical_folder = st.session_state.get("clinical_project_folder", "")
            if clinical_folder:
                return (Path(clinical_folder) / "06_Model_Comparison" / "clinical_model_comparison_table.csv").exists()
            return False
        if page_name == "clinical_train_validate":
            return st.session_state.trained_pipeline is not None
        if page_name == "clinical_risk_calculator":
            return len(trained_models) > 0 or st.session_state.trained_pipeline is not None
        if page_name == "knowledge_base" or page_name.startswith("kb_"):
            return True
        return False

    def is_step_locked(page_name):
        # Home and top-level pages are always available.
        if page_name in [
            "home",
            "model_development_home",
            "established_model",
            "established_model_search",
            "established_model_compare",
            "established_model_risk_calculator",
            "established_model_validate",
            "established_model_external_validation",
            "established_model_documentation",
                    "established_model_collaborate",
            "supporting_documentation",
            "voxel_analysis_home",
            "voxel_start_project",
            "voxel_load_patient_data",
            "voxel_load_images",
            "voxel_registration_qc",
            "voxel_warp_to_ccs",
            "voxel_dose_normalisation",
            "voxel_statistical_analysis",
            "clinical_start_project",
            "clinical_upload",
            "clinical_model_comparison",
        ]:
            return False

        # Clinical workflow locking.
        if page_name.startswith("clinical_") and st.session_state.df is None:
            return True

        if page_name in [
            "clinical_analysis",
            "clinical_model_selection",
                        ] and st.session_state.get("treatment_variable", "") == "":
            return True

        if page_name == "clinical_risk_calculator":
            if len(trained_models) == 0 and st.session_state.trained_pipeline is None:
                return True

        return False

    steps = get_navigation_steps()

    st.sidebar.markdown("---")
    st.sidebar.caption("Full path")

    for label, page_name in steps:
        locked = is_step_locked(page_name)
        complete = is_step_complete(page_name)
        current = st.session_state.page == page_name

        if current:
            icon = "▶"
        elif locked:
            icon = "🔒"
        elif complete:
            icon = "✅"
        else:
            icon = "⭕"

        if st.sidebar.button(
            f"{icon} {label}",
            use_container_width=True,
            disabled=locked,
            key=f"sidebar_nav_{page_name}_{label}_{icon}"
        ):
            go_to(page_name)

    st.sidebar.markdown("---")

    if current_workflow == "Clinical Model":
        if st.session_state.df is None:
            st.sidebar.info("Upload an Excel file to unlock the clinical workflow.")
        else:
            st.sidebar.success("Excel loaded")
            st.sidebar.write("Rows:", st.session_state.df.shape[0])
            st.sidebar.write("Columns:", st.session_state.df.shape[1])

        if st.session_state.get("treatment_variable", "") != "":
            st.sidebar.success(f"Treatment: {st.session_state.treatment_variable}")

        if len(trained_models) > 0:
            st.sidebar.success("Trained models:")
            for model_name in trained_models.keys():
                st.sidebar.write(f"✅ {model_name}")

    if current_workflow == "Voxel-based Analysis":
        file_df = st.session_state.get("voxel_loaded_image_files_df", pd.DataFrame())
        patient_df = st.session_state.get("voxel_patient_data", None)

        if patient_df is not None:
            st.sidebar.success("Patient data loaded")
        else:
            st.sidebar.info("Patient data not loaded")

        if file_df is not None and not file_df.empty:
            st.sidebar.success("Images loaded")
            st.sidebar.write("Files:", file_df.shape[0])
        else:
            st.sidebar.info("Images not loaded")

    if st.sidebar.button("Refresh page", use_container_width=True):
        st.rerun()

    if st.sidebar.button("Reset app", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.sidebar.divider()
    render_code_icon_button(location="sidebar")


def render_breadcrumb_path():
    """
    Compact path shown at the top of workflow pages.
    """
    if st.session_state.get("page", "home") == "home":
        return

    steps = get_navigation_steps()
    labels = []
    for label, page_name in steps:
        clean = str(label)
        for prefix in ["✅ ", "▶ ", "⭕ ", "🔒 "]:
            clean = clean.replace(prefix, "")
        labels.append(clean)
        if page_name == st.session_state.page:
            break

    st.caption(" › ".join(labels))


def derive_decline(df, baseline_col, followup_col, threshold):
    df = df.copy()

    baseline = pd.to_numeric(df[baseline_col], errors="coerce")
    followup = pd.to_numeric(df[followup_col], errors="coerce")

    df["Cognitive_Change"] = followup - baseline
    df["Derived_Neurocognitive_Decline"] = (
        df["Cognitive_Change"] <= -abs(threshold)
    ).astype("Int64")

    df.loc[
        df["Cognitive_Change"].isna(),
        "Derived_Neurocognitive_Decline"
    ] = pd.NA

    return df




def make_safe_column_name(name):
    """Create a safe suffix for generated column names."""
    safe = str(name).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in safe)
    return safe


def derive_declines_for_followups(df, baseline_col, followup_cols, threshold):
    """
    Create cognitive change and decline columns for multiple follow-up outcomes.

    For each follow-up variable:
    Change_<followup> = follow-up - baseline
    Decline_<followup> = 1 if change <= -threshold
    """
    df = df.copy()
    generated_outcomes = []
    generated_changes = []

    baseline = pd.to_numeric(df[baseline_col], errors="coerce")

    for followup_col in followup_cols:
        suffix = make_safe_column_name(followup_col)
        change_col = f"Cognitive_Change_{suffix}"
        decline_col = f"Decline_{suffix}"

        followup = pd.to_numeric(df[followup_col], errors="coerce")
        df[change_col] = followup - baseline
        df[decline_col] = (df[change_col] <= -abs(threshold)).astype("Int64")
        df.loc[df[change_col].isna(), decline_col] = pd.NA

        generated_changes.append(change_col)
        generated_outcomes.append(decline_col)

    # Preserve the old generic column names for compatibility using the first follow-up.
    if len(followup_cols) > 0:
        first_suffix = make_safe_column_name(followup_cols[0])
        df["Cognitive_Change"] = df[f"Cognitive_Change_{first_suffix}"]
        df["Derived_Neurocognitive_Decline"] = df[f"Decline_{first_suffix}"]

    return df, generated_changes, generated_outcomes


def derive_declines_from_mapping(df, mapping_rows, threshold):
    """
    Create change and decline columns from an outcome mapping table.

    Each mapping row contains:
    - Baseline Variable
    - Follow-up 1
    - Follow-up 2
    - Follow-up 3

    For every baseline/follow-up pair:
    Change_<baseline>_to_<followup> = follow-up - baseline
    Decline_<baseline>_to_<followup> = 1 if change <= -threshold
    """
    df = df.copy()
    generated_rows = []
    generated_changes = []
    generated_outcomes = []

    for row in mapping_rows:
        baseline_col = row.get("Baseline Variable", "")

        if baseline_col == "" or baseline_col not in df.columns:
            continue

        baseline = pd.to_numeric(df[baseline_col], errors="coerce")

        for followup_key in ["Follow-up 1", "Follow-up 2", "Follow-up 3"]:
            followup_col = row.get(followup_key, "")

            if followup_col == "" or followup_col not in df.columns:
                continue

            baseline_suffix = make_safe_column_name(baseline_col)
            followup_suffix = make_safe_column_name(followup_col)

            change_col = f"Change_{baseline_suffix}_to_{followup_suffix}"
            decline_col = f"Decline_{baseline_suffix}_to_{followup_suffix}"

            followup = pd.to_numeric(df[followup_col], errors="coerce")
            df[change_col] = followup - baseline
            df[decline_col] = (df[change_col] <= -abs(threshold)).astype("Int64")
            df.loc[df[change_col].isna(), decline_col] = pd.NA

            generated_changes.append(change_col)
            generated_outcomes.append(decline_col)
            generated_rows.append({
                "Baseline": baseline_col,
                "Follow-up": followup_col,
                "Change Variable": change_col,
                "Decline Variable": decline_col,
            })

    # Preserve old generic names for compatibility using the first generated pair.
    if len(generated_changes) > 0:
        df["Cognitive_Change"] = df[generated_changes[0]]
        df["Derived_Neurocognitive_Decline"] = df[generated_outcomes[0]]

    generated_summary = pd.DataFrame(generated_rows)
    return df, generated_changes, generated_outcomes, generated_summary


def get_treatment_options(df, column_name):
    if column_name == "" or column_name not in df.columns:
        return []

    options = (
        df[column_name]
        .dropna()
        .astype(str)
        .str.strip()
    )

    options = options[options != ""]
    return sorted(options.unique().tolist())


def is_numeric_column(df, column_name):
    converted = pd.to_numeric(df[column_name], errors="coerce")
    non_missing_original = df[column_name].dropna().shape[0]
    non_missing_numeric = converted.dropna().shape[0]

    if non_missing_original == 0:
        return False

    return non_missing_numeric / non_missing_original >= 0.8


def summarize_numeric(series):
    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) == 0:
        return "No data"

    median = values.median()
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    return f"{median:.2f} ({q1:.2f}–{q3:.2f})"


def summarize_categorical(series):
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]

    if len(values) == 0:
        return "No data"

    counts = values.value_counts()
    total = counts.sum()

    summary_parts = []
    for label, count in counts.items():
        percentage = 100 * count / total
        summary_parts.append(f"{label}: {count} ({percentage:.1f}%)")

    return "; ".join(summary_parts)


def format_mean_sd(values):
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) == 0:
        return "No data"
    mean = values.mean()
    sd = values.std(ddof=1) if len(values) > 1 else 0
    return f"{mean:.2f} ± {sd:.2f}"


def format_median_iqr(values):
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) == 0:
        return "No data"
    median = values.median()
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    return f"{median:.2f} ({q1:.2f}–{q3:.2f})"


def normality_test(values):
    """
    Run a simple normality test for numeric data.

    Uses Shapiro-Wilk for 3 to 5000 observations.
    For larger samples, uses D'Agostino-Pearson if possible.
    """
    values = pd.to_numeric(values, errors="coerce").dropna()
    n = len(values)

    if n < 3:
        return "Not enough data", np.nan, "Not assessed"

    try:
        if n <= 5000:
            _, p_value = stats.shapiro(values)
            test_name = "Shapiro-Wilk"
        else:
            _, p_value = stats.normaltest(values)
            test_name = "D'Agostino-Pearson"

        interpretation = "Approximately normal" if p_value >= 0.05 else "Non-normal"
        return test_name, p_value, interpretation

    except Exception:
        return "Normality test failed", np.nan, "Not assessed"


def summarize_numeric_extended(values):
    """
    Extended numeric descriptive statistics.
    """
    numeric_values = pd.to_numeric(values, errors="coerce").dropna()
    test_name, p_value, interpretation = normality_test(numeric_values)

    return {
        "N non-missing": int(len(numeric_values)),
        "Mean ± SD": format_mean_sd(numeric_values),
        "Median (IQR)": format_median_iqr(numeric_values),
        "Normality test": test_name,
        "Normality p-value": format_p_value(p_value),
        "Distribution": interpretation,
    }


def summarize_categorical_percent_n(values):
    """
    Categorical descriptive statistics as % (n), as requested.
    """
    clean_values = values.dropna().astype(str).str.strip()
    clean_values = clean_values[clean_values != ""]

    if len(clean_values) == 0:
        return {
            "N non-missing": 0,
            "% (n)": "No data",
            "Categories": 0,
        }

    counts = clean_values.value_counts()
    total = counts.sum()

    parts = []
    for label, count in counts.items():
        percentage = 100 * count / total
        parts.append(f"{label}: {percentage:.1f}% ({count})")

    return {
        "N non-missing": int(total),
        "% (n)": "; ".join(parts),
        "Categories": int(len(counts)),
    }


def format_p_value(p_value):
    if p_value is None or pd.isna(p_value):
        return "Not available"

    if p_value < 0.001:
        return "<0.001"

    return f"{p_value:.3f}"


def p_value_is_significant(p_value_text):
    """
    Identify significant p-values from formatted strings.
    """
    try:
        text = str(p_value_text).strip()
        if text == "<0.001":
            return True
        if text in ["Not available", "", "nan", "None"]:
            return False
        return float(text) < 0.05
    except Exception:
        return False


def style_significant_rows(row):
    """
    Bold rows with significant p-values.
    """
    if p_value_is_significant(row.get("p-value", "")):
        return ["font-weight: bold"] * len(row)
    return [""] * len(row)


def explain_statistical_test(test_name):
    """
    User-facing explanations for statistical tests used in the app.
    """
    explanations = {
        "Shapiro-Wilk": (
            "Shapiro-Wilk is a normality test. It checks whether a numeric variable looks "
            "approximately normally distributed. A p-value below 0.05 suggests that the data "
            "deviate from normality. In this app, it helps decide whether to use parametric "
            "tests such as t-test/ANOVA or non-parametric tests such as Mann-Whitney/Kruskal-Wallis."
        ),
        "D'Agostino-Pearson": (
            "D'Agostino-Pearson is a normality test for larger samples. It checks skewness and "
            "kurtosis to assess whether the distribution is approximately normal. A p-value below "
            "0.05 suggests evidence against normality."
        ),
        "Welch t-test": (
            "Welch t-test compares the mean of a numeric variable between two independent groups. "
            "It is used when both included groups appear approximately normally distributed. "
            "Welch's version is used because it does not require equal variances."
        ),
        "One-way ANOVA": (
            "One-way ANOVA compares the mean of a numeric variable across three or more independent groups. "
            "It is used when the numeric variable appears approximately normally distributed in all included groups. "
            "A significant result means at least one group mean differs, but it does not identify which groups differ."
        ),
        "Mann–Whitney U": (
            "Mann–Whitney U compares a numeric or ordinal variable between two independent groups. "
            "It is non-parametric, so it is chosen when at least one included group does not appear normally distributed "
            "or the sample is too small to assess normality reliably."
        ),
        "Kruskal–Wallis": (
            "Kruskal–Wallis compares a numeric or ordinal variable across three or more independent groups. "
            "It is the non-parametric alternative to one-way ANOVA and is chosen when at least one included group "
            "does not appear normally distributed or normality cannot be assessed reliably."
        ),
        "Chi-square": (
            "Chi-square tests whether the distribution of a categorical variable differs between groups. "
            "For example, it can test whether the proportion of patients with decline differs by treatment group."
        ),
        "Fisher exact": (
            "Fisher exact test compares proportions between two groups for a 2x2 table. "
            "It is useful when expected cell counts are small."
        ),
    }

    return explanations.get(
        test_name,
        "No explanation is available for this test yet."
    )


def explain_test_choice(test_name):
    """
    Explain why the app selected a test.
    """
    choices = {
        "Welch t-test": (
            "Chosen because there are two included treatment groups and the numeric variable appears approximately "
            "normally distributed in both groups. Welch's version is used because it does not require equal variances."
        ),
        "One-way ANOVA": (
            "Chosen because there are three or more included treatment groups and the numeric variable appears approximately "
            "normally distributed in all included groups."
        ),
        "Mann–Whitney U": (
            "Chosen because there are two included treatment groups and at least one group is non-normal, too small for "
            "normality assessment, or missing enough data to make a parametric test less appropriate."
        ),
        "Kruskal–Wallis": (
            "Chosen because there are three or more included treatment groups and at least one group is non-normal, too small "
            "for normality assessment, or missing enough data to make ANOVA less appropriate."
        ),
        "Chi-square": (
            "Chosen because the variable is categorical and the app is comparing the distribution of categories "
            "between treatment groups."
        ),
    }
    return choices.get(test_name, "The app selected this test based on variable type, number of groups, and available data.")


def sample_normality_status(values):
    """
    Assess normality of the included analysis sample for one numeric variable.

    This uses all included patients for the selected variable, rather than testing
    each treatment group separately. Therefore, if Age is normally distributed in
    the included analysis sample, the app will use a parametric test.
    """
    numeric_values = pd.to_numeric(values, errors="coerce").dropna()

    if len(numeric_values) < 3:
        return None, "Too few observations to assess normality"

    test_name, p_value, interpretation = normality_test(numeric_values)

    if p_value is None or pd.isna(p_value):
        return None, interpretation

    return p_value >= 0.05, f"{test_name} p={format_p_value(p_value)} ({interpretation})"


def choose_numeric_group_test(df, variable, treatment_variable, treatment_groups):
    """
    Choose numeric group-comparison test using normality of the included analysis sample.

    Two included treatment groups:
    - normal included sample -> Welch t-test
    - non-normal / not assessable -> Mann-Whitney U

    Three or more included treatment groups:
    - normal included sample -> One-way ANOVA
    - non-normal / not assessable -> Kruskal-Wallis
    """
    grouped_values = []
    included_values = []

    for group in treatment_groups:
        values = df.loc[
            df[treatment_variable].astype(str).str.strip() == str(group),
            variable
        ]
        values = pd.to_numeric(values, errors="coerce").dropna()

        if len(values) > 0:
            grouped_values.append(values)
            included_values.extend(values.tolist())

    if len(grouped_values) < 2:
        return "Not available", None, "Fewer than two included groups have data.", "Not available"

    is_normal, normality_note = sample_normality_status(pd.Series(included_values))

    if is_normal is True:
        distribution_label = "Normal"
    elif is_normal is False:
        distribution_label = "Non-normal"
    else:
        distribution_label = "Normality not assessed"

    try:
        if len(grouped_values) == 2:
            if is_normal is True:
                _, p_value = stats.ttest_ind(
                    grouped_values[0],
                    grouped_values[1],
                    equal_var=False,
                    nan_policy="omit"
                )
                return (
                    "Welch t-test",
                    p_value,
                    f"Chosen because {variable} appears normally distributed in the included analysis sample: {normality_note}",
                    distribution_label,
                )

            _, p_value = stats.mannwhitneyu(
                grouped_values[0],
                grouped_values[1],
                alternative="two-sided"
            )
            return (
                "Mann–Whitney U",
                p_value,
                f"Chosen because {variable} is not normally distributed, or normality could not be assessed: {normality_note}",
                distribution_label,
            )

        if is_normal is True:
            _, p_value = stats.f_oneway(*grouped_values)
            return (
                "One-way ANOVA",
                p_value,
                f"Chosen because {variable} appears normally distributed in the included analysis sample: {normality_note}",
                distribution_label,
            )

        _, p_value = stats.kruskal(*grouped_values)
        return (
            "Kruskal–Wallis",
            p_value,
            f"Chosen because {variable} is not normally distributed, or normality could not be assessed: {normality_note}",
            distribution_label,
        )

    except Exception as error:
        return "Not available", None, f"Test failed: {error}; {normality_note}", distribution_label


def inferential_numeric_summary_for_test(values, test_name):
    """
    Summary display depends on the selected test:
    - Parametric tests: mean ± SD
    - Non-parametric tests: median (IQR)
    """
    if test_name in ["Welch t-test", "One-way ANOVA"]:
        return f"Mean ± SD: {format_mean_sd(values)}"

    return f"Median (IQR): {format_median_iqr(values)}"


def inferential_categorical_summary(values):
    """
    Group summary for categorical variables in inferential tables.

    Displays categories as:
    category: percentage% (n)
    """
    summary = summarize_categorical_percent_n(values)
    return summary["% (n)"]


def run_numeric_group_test(df, variable, treatment_variable, treatment_groups):
    """
    Run appropriate numeric group-comparison test based on normality.

    Returns test_name, p_value for compatibility with existing code.
    """
    test_name, p_value, _, _ = choose_numeric_group_test(
        df,
        variable,
        treatment_variable,
        treatment_groups
    )
    return test_name, p_value


def run_categorical_group_test(df, variable, treatment_variable):
    table = pd.crosstab(df[treatment_variable], df[variable])

    if table.empty or table.shape[0] < 2 or table.shape[1] < 2:
        return "Chi-square", None

    _, p_value, _, _ = stats.chi2_contingency(table)

    return "Chi-square", p_value


# ============================================================
# MODEL FUNCTIONS
# ============================================================

def prepare_model_data(df, predictors, treatment_variable, outcome_variable):
    model_df = df.copy()

    model_predictors = predictors.copy()

    if treatment_variable != "" and treatment_variable not in model_predictors:
        model_predictors.append(treatment_variable)

    required_columns = model_predictors + [outcome_variable]
    model_df = model_df[required_columns].dropna()

    X = model_df[model_predictors]
    y = model_df[outcome_variable].astype(int)

    return X, y, model_predictors



def safe_int_setting(value, default):
    """Safely convert a model setting to int. Fall back to default if invalid."""
    try:
        if value is None:
            return int(default)
        if isinstance(value, str):
            value = value.strip()
            if value == "" or value.lower() in ["none", "automatic", "nan"]:
                return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def safe_float_setting(value, default):
    """Safely convert a model setting to float. Fall back to default if invalid."""
    try:
        if value is None:
            return float(default)
        if isinstance(value, str):
            value = value.strip()
            if value == "" or value.lower() in ["none", "automatic", "nan"]:
                return float(default)
        return float(value)
    except Exception:
        return float(default)


def safe_bool_setting(value, default=True):
    """Safely convert a model setting to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ["true", "yes", "1", "on"]:
            return True
        if value in ["false", "no", "0", "off"]:
            return False
    return bool(default)


def build_prediction_pipeline(X, method, model_settings=None):
    """
    Build preprocessing + prediction pipeline.

    model_settings is optional and is used for Random Forest and XGBoost hyperparameters.
    """
    if model_settings is None:
        model_settings = {}

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    if method == "Regression":
        model = LogisticRegression(
            max_iter=safe_int_setting(model_settings.get("max_iter", 3000), 3000),
            class_weight="balanced" if safe_bool_setting(model_settings.get("class_weight_balanced", True), True) else None
        )

    elif method == "Random Forest":
        max_depth = model_settings.get("max_depth", None)
        if max_depth in [0, "None", "Automatic", "", None]:
            max_depth = None
        else:
            max_depth = safe_int_setting(max_depth, 8)

        model = RandomForestClassifier(
            n_estimators=safe_int_setting(model_settings.get("n_estimators", 500), 500),
            max_depth=max_depth,
            min_samples_leaf=safe_int_setting(model_settings.get("min_samples_leaf", 2), 2),
            min_samples_split=safe_int_setting(model_settings.get("min_samples_split", 2), 2),
            class_weight="balanced" if safe_bool_setting(model_settings.get("class_weight_balanced", True), True) else None,
            random_state=safe_int_setting(model_settings.get("random_state", 42), 42),
            n_jobs=-1
        )

    elif method == "XGBoost":
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is not installed. Run: py -3.13 -m pip install xgboost"
            )

        model = XGBClassifier(
            n_estimators=safe_int_setting(model_settings.get("n_estimators", 400), 400),
            learning_rate=safe_float_setting(model_settings.get("learning_rate", 0.03), 0.03),
            max_depth=safe_int_setting(model_settings.get("max_depth", 3), 3),
            subsample=safe_float_setting(model_settings.get("subsample", 0.85), 0.85),
            colsample_bytree=safe_float_setting(model_settings.get("colsample_bytree", 0.85), 0.85),
            min_child_weight=safe_float_setting(model_settings.get("min_child_weight", 1.0), 1.0),
            gamma=safe_float_setting(model_settings.get("gamma", 0.0), 0.0),
            reg_alpha=safe_float_setting(model_settings.get("reg_alpha", 0.0), 0.0),
            reg_lambda=safe_float_setting(model_settings.get("reg_lambda", 1.0), 1.0),
            eval_metric="logloss",
            random_state=safe_int_setting(model_settings.get("random_state", 42), 42),
            n_jobs=-1
        )

    else:
        raise ValueError("Unknown analytics method.")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def get_feature_names_from_pipeline(pipeline, X):
    preprocessor = pipeline.named_steps["preprocessor"]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    feature_names = []
    feature_names.extend(numeric_features)

    if len(categorical_features) > 0:
        encoder = (
            preprocessor
            .named_transformers_["categorical"]
            .named_steps["encoder"]
        )

        encoded_names = encoder.get_feature_names_out(categorical_features)
        feature_names.extend(encoded_names)

    return feature_names


def run_univariable_screening(df, predictors, treatment_variable, outcome_variable, method):
    """
    Univariable screening.

    For Regression:
        coefficient and odds ratio are shown.
    For RF/XGBoost:
        single-variable AUC is shown.
    """
    results = []

    all_predictors = predictors.copy()

    if treatment_variable != "" and treatment_variable not in all_predictors:
        all_predictors.append(treatment_variable)

    for predictor in all_predictors:
        temp_df = df[[predictor, outcome_variable]].dropna()

        if temp_df.empty:
            continue

        X = temp_df[[predictor]]
        y = temp_df[outcome_variable].astype(int)

        if y.nunique() < 2:
            results.append({
                "Predictor": predictor,
                "Term": predictor,
                "Metric": "Skipped",
                "Value": np.nan,
                "Odds Ratio": np.nan,
                "Direction": "Outcome has one class only",
            })
            continue

        try:
            pipeline = build_prediction_pipeline(X, method)
            pipeline.fit(X, y)

            y_prob = pipeline.predict_proba(X)[:, 1]

            try:
                auc = roc_auc_score(y, y_prob)
            except Exception:
                auc = np.nan

            if method == "Regression":
                model = pipeline.named_steps["model"]
                feature_names = get_feature_names_from_pipeline(pipeline, X)
                coefficients = model.coef_[0]

                for feature, coef in zip(feature_names, coefficients):
                    results.append({
                        "Predictor": predictor,
                        "Term": feature,
                        "Metric": "Coefficient",
                        "Value": coef,
                        "Odds Ratio": np.exp(coef),
                        "Direction": (
                            "Increases decline risk"
                            if coef > 0
                            else "Decreases decline risk"
                        ),
                    })
            else:
                results.append({
                    "Predictor": predictor,
                    "Term": predictor,
                    "Metric": "Single-variable AUC",
                    "Value": auc,
                    "Odds Ratio": np.nan,
                    "Direction": "Higher AUC suggests stronger single-variable prediction",
                })

        except Exception as error:
            results.append({
                "Predictor": predictor,
                "Term": predictor,
                "Metric": "Model failed",
                "Value": np.nan,
                "Odds Ratio": np.nan,
                "Direction": str(error),
            })

    return pd.DataFrame(results)


def run_multivariable_model(df, predictors, treatment_variable, outcome_variable, method, model_settings=None):
    X, y, final_predictors = prepare_model_data(
        df,
        predictors,
        treatment_variable,
        outcome_variable
    )

    if X.empty:
        return None, None, None, None

    if y.nunique() < 2:
        return None, None, None, None

    try:
        pipeline = build_prediction_pipeline(X, method, model_settings)
        pipeline.fit(X, y)

        model = pipeline.named_steps["model"]
        feature_names = get_feature_names_from_pipeline(pipeline, X)

        rows = []

        if method == "Regression":
            coefficients = model.coef_[0]

            for feature, coef in zip(feature_names, coefficients):
                rows.append({
                    "Term": feature,
                    "Coefficient": coef,
                    "Odds Ratio": np.exp(coef),
                    "Importance": abs(coef),
                    "Effect Direction": (
                        "Increases decline risk"
                        if coef > 0
                        else "Decreases decline risk"
                    ),
                })

        else:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            else:
                importances = np.zeros(len(feature_names))

            for feature, importance in zip(feature_names, importances):
                rows.append({
                    "Term": feature,
                    "Coefficient": np.nan,
                    "Odds Ratio": np.nan,
                    "Importance": importance,
                    "Effect Direction": "Feature importance from tree model",
                })

        results_df = pd.DataFrame(rows)

        if not results_df.empty:
            results_df = results_df.sort_values("Importance", ascending=False)

        return pipeline, results_df, X, y

    except Exception as error:
        st.error(f"{method} model failed: {error}")
        return None, None, None, None


def evaluate_model(pipeline, X, y, validation_size, random_state=42):
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=validation_size,
            random_state=random_state,
            stratify=y
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=validation_size,
            random_state=random_state
        )

    pipeline.fit(X_train, y_train)

    def _evaluate_split(X_split, y_split):
        y_prob = pipeline.predict_proba(X_split)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        try:
            auc = roc_auc_score(y_split, y_prob)
        except Exception:
            auc = np.nan

        accuracy = accuracy_score(y_split, y_pred)
        brier = brier_score_loss(y_split, y_prob)
        precision = precision_score(y_split, y_pred, zero_division=0)
        sensitivity = recall_score(y_split, y_pred, zero_division=0)
        f1 = f1_score(y_split, y_pred, zero_division=0)

        cm = confusion_matrix(y_split, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        return {
            "Rows": X_split.shape[0],
            "AUC": auc,
            "Accuracy": accuracy,
            "Brier Score": brier,
            "Precision": precision,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "F1 Score": f1,
            "Confusion matrix": cm,
        }

    training_metrics = _evaluate_split(X_train, y_train)
    validation_metrics = _evaluate_split(X_val, y_val)

    # Keep previous validation keys for compatibility with export and established-model pages.
    metrics = {
        "Training rows": training_metrics["Rows"],
        "Validation rows": validation_metrics["Rows"],
        "AUC": validation_metrics["AUC"],
        "Accuracy": validation_metrics["Accuracy"],
        "Brier Score": validation_metrics["Brier Score"],
        "Brier score": validation_metrics["Brier Score"],
        "Precision": validation_metrics["Precision"],
        "Sensitivity": validation_metrics["Sensitivity"],
        "Specificity": validation_metrics["Specificity"],
        "F1 Score": validation_metrics["F1 Score"],
        "F1": validation_metrics["F1 Score"],

        "Training AUC": training_metrics["AUC"],
        "Training Accuracy": training_metrics["Accuracy"],
        "Training Brier Score": training_metrics["Brier Score"],
        "Training Precision": training_metrics["Precision"],
        "Training Sensitivity": training_metrics["Sensitivity"],
        "Training Specificity": training_metrics["Specificity"],
        "Training F1 Score": training_metrics["F1 Score"],

        "Validation AUC": validation_metrics["AUC"],
        "Validation Accuracy": validation_metrics["Accuracy"],
        "Validation Brier Score": validation_metrics["Brier Score"],
        "Validation Precision": validation_metrics["Precision"],
        "Validation Sensitivity": validation_metrics["Sensitivity"],
        "Validation Specificity": validation_metrics["Specificity"],
        "Validation F1 Score": validation_metrics["F1 Score"],

        "Training confusion matrix": training_metrics["Confusion matrix"],
        "Validation confusion matrix": validation_metrics["Confusion matrix"],
    }

    performance_table = []
    metric_pairs = [
        ("Patients", "Rows"),
        ("AUC", "AUC"),
        ("Accuracy", "Accuracy"),
        ("Sensitivity", "Sensitivity"),
        ("Specificity", "Specificity"),
        ("Precision", "Precision"),
        ("F1 score", "F1 Score"),
        ("Brier score", "Brier Score"),
    ]

    for label, key in metric_pairs:
        performance_table.append({
            "Result": label,
            "Training": training_metrics.get(key, np.nan),
            "Validation": validation_metrics.get(key, np.nan),
        })

    metrics["Performance table"] = performance_table

    # cm remains validation confusion matrix for compatibility.
    return pipeline, metrics, validation_metrics["Confusion matrix"]

def create_nomogram_points(model_results):
    if model_results is None or model_results.empty:
        return pd.DataFrame()

    nomogram = model_results.copy()

    if "Coefficient" in nomogram.columns and nomogram["Coefficient"].notna().any():
        nomogram["Absolute Coefficient"] = nomogram["Coefficient"].abs()
        score_source = "Absolute Coefficient"
    else:
        nomogram["Absolute Coefficient"] = nomogram["Importance"].abs()
        score_source = "Absolute Coefficient"

    max_value = nomogram[score_source].max()

    if max_value == 0 or pd.isna(max_value):
        nomogram["Nomogram Points"] = 0
    else:
        nomogram["Nomogram Points"] = (
            nomogram[score_source] / max_value * 100
        ).round(1)

    keep_cols = [
        "Term",
        "Coefficient",
        "Odds Ratio",
        "Importance",
        "Effect Direction",
        "Nomogram Points",
    ]

    keep_cols = [col for col in keep_cols if col in nomogram.columns]

    return nomogram[keep_cols]




def get_step4a_normality_for_variable(df, variable):
    """
    Use Step 4A normality result if available.
    If Step 4A has not been run, calculate the same overall normality here.
    """
    normality_map = st.session_state.get("step4a_normality_map", {})

    if variable in normality_map:
        info = normality_map[variable]
        return {
            "Distribution": simplify_distribution_label(info.get("Distribution", "Not assessed")),
            "Normality test": info.get("Normality test", "Not available"),
            "Normality p-value": info.get("Normality p-value", "Not available"),
        }

    numeric_values = pd.to_numeric(df[variable], errors="coerce").dropna()

    if len(numeric_values) < 3 or numeric_values.nunique() < 3:
        return {
            "Distribution": "Not assessed",
            "Normality test": "Not available",
            "Normality p-value": "Not available",
        }

    test_name, p_value, interpretation = normality_test(numeric_values)

    if p_value is None or pd.isna(p_value):
        distribution = "Not assessed"
    elif p_value >= 0.05:
        distribution = "Normal"
    else:
        distribution = "Non-normal"

    return {
        "Distribution": distribution,
        "Normality test": test_name,
        "Normality p-value": format_p_value(p_value),
    }


def choose_numeric_group_test_from_step4a(df, variable, treatment_variable, treatment_groups):
    """
    Choose Step 4B test using Step 4A normality result.

    Normal:
      2 groups -> Welch t-test
      3+ groups -> One-way ANOVA

    Non-normal or not assessed:
      2 groups -> Mann-Whitney U
      3+ groups -> Kruskal-Wallis
    """
    normality_info = get_step4a_normality_for_variable(df, variable)
    distribution = normality_info["Distribution"]
    normality_test_name = normality_info["Normality test"]
    normality_p_value = normality_info["Normality p-value"]

    grouped_values = []

    for group in treatment_groups:
        values = df.loc[
            df[treatment_variable].astype(str).str.strip() == str(group).strip(),
            variable
        ]
        values = pd.to_numeric(values, errors="coerce").dropna()

        if len(values) > 0:
            grouped_values.append(values)

    if len(grouped_values) < 2:
        return (
            "Not available",
            None,
            distribution,
            normality_p_value,
            "Fewer than two included treatment groups have usable numeric data.",
        )

    is_normal = distribution == "Normal"

    try:
        if len(grouped_values) == 2:
            if is_normal:
                _, p_value = stats.ttest_ind(
                    grouped_values[0],
                    grouped_values[1],
                    equal_var=False,
                    nan_policy="omit"
                )
                return (
                    "Welch t-test",
                    p_value,
                    distribution,
                    normality_p_value,
                    f"Used because Step 4A {normality_test_name} result was Normal.",
                )

            _, p_value = stats.mannwhitneyu(
                grouped_values[0],
                grouped_values[1],
                alternative="two-sided"
            )
            return (
                "Mann–Whitney U",
                p_value,
                distribution,
                normality_p_value,
                f"Used because Step 4A {normality_test_name} result was {distribution}.",
            )

        if is_normal:
            _, p_value = stats.f_oneway(*grouped_values)
            return (
                "One-way ANOVA",
                p_value,
                distribution,
                normality_p_value,
                f"Used because Step 4A {normality_test_name} result was Normal.",
            )

        _, p_value = stats.kruskal(*grouped_values)
        return (
            "Kruskal–Wallis",
            p_value,
            distribution,
            normality_p_value,
            f"Used because Step 4A {normality_test_name} result was {distribution}.",
        )

    except Exception as error:
        return (
            "Not available",
            None,
            distribution,
            normality_p_value,
            f"Test failed: {error}",
        )


def inferential_numeric_summary_for_test(values, test_name):
    """
    Summary shown depends on the test selected from Step 4A normality.
    """
    if test_name in ["Welch t-test", "One-way ANOVA"]:
        return f"Mean ± SD: {format_mean_sd(values)}"
    return f"Median (IQR): {format_median_iqr(values)}"



# ============================================================
# PLOTTING HELPERS
# ============================================================

def plot_numeric_histogram(df, variable):
    values = pd.to_numeric(df[variable], errors="coerce").dropna()
    if len(values) == 0:
        st.warning(f"No numeric data available for {variable}.")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=20)
    ax.set_title(f"Histogram: {variable}")
    ax.set_xlabel(variable)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
    plt.close(fig)


def plot_numeric_boxplot_by_group(df, variable, group_variable):
    if group_variable == "" or group_variable not in df.columns:
        st.warning("No valid treatment/group column selected.")
        return
    plot_df = df[[variable, group_variable]].copy()
    plot_df[variable] = pd.to_numeric(plot_df[variable], errors="coerce")
    plot_df[group_variable] = plot_df[group_variable].astype(str).str.strip()
    plot_df = plot_df.dropna()
    if plot_df.empty:
        st.warning(f"No data available for {variable}.")
        return

    groups = sorted(plot_df[group_variable].unique().tolist())
    data = [plot_df.loc[plot_df[group_variable] == group, variable].dropna() for group in groups]
    filtered = [(g, d) for g, d in zip(groups, data) if len(d) > 0]
    if len(filtered) == 0:
        st.warning(f"No grouped data available for {variable}.")
        return
    groups, data = zip(*filtered)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, labels=groups, showmeans=True)
    ax.set_title(f"Boxplot: {variable} by {group_variable}")
    ax.set_xlabel(group_variable)
    ax.set_ylabel(variable)
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)
    plt.close(fig)


def plot_numeric_mean_ci_by_group(df, variable, group_variable):
    if group_variable == "" or group_variable not in df.columns:
        st.warning("No valid treatment/group column selected.")
        return
    plot_df = df[[variable, group_variable]].copy()
    plot_df[variable] = pd.to_numeric(plot_df[variable], errors="coerce")
    plot_df[group_variable] = plot_df[group_variable].astype(str).str.strip()
    plot_df = plot_df.dropna()
    if plot_df.empty:
        st.warning(f"No data available for {variable}.")
        return

    summary = plot_df.groupby(group_variable)[variable].agg(["mean", "std", "count"]).reset_index()
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    summary["ci95"] = 1.96 * summary["sem"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(summary[group_variable], summary["mean"], yerr=summary["ci95"], fmt="o", capsize=5)
    ax.set_title(f"Mean with 95% CI: {variable} by {group_variable}")
    ax.set_xlabel(group_variable)
    ax.set_ylabel(variable)
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)
    plt.close(fig)


def plot_categorical_bar(df, variable):
    values = df[variable].dropna().astype(str).str.strip()
    values = values[values != ""]
    if len(values) == 0:
        st.warning(f"No categorical data available for {variable}.")
        return
    counts = values.value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(f"Bar chart: {variable}")
    ax.set_xlabel(variable)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)
    plt.close(fig)


def plot_categorical_bar_by_group(df, variable, group_variable, percent=True):
    if group_variable == "" or group_variable not in df.columns:
        st.warning("No valid treatment/group column selected.")
        return
    plot_df = df[[variable, group_variable]].dropna().copy()
    plot_df[variable] = plot_df[variable].astype(str).str.strip()
    plot_df[group_variable] = plot_df[group_variable].astype(str).str.strip()
    plot_df = plot_df[(plot_df[variable] != "") & (plot_df[group_variable] != "")]
    if plot_df.empty:
        st.warning(f"No data available for {variable}.")
        return

    table = pd.crosstab(plot_df[group_variable], plot_df[variable])
    if percent:
        table_to_plot = table.div(table.sum(axis=1), axis=0) * 100
        ylabel = "Percentage (%)"
        title = f"Percentage bar chart: {variable} by {group_variable}"
    else:
        table_to_plot = table
        ylabel = "Count"
        title = f"Count bar chart: {variable} by {group_variable}"

    fig, ax = plt.subplots(figsize=(8, 4))
    table_to_plot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(group_variable)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title=variable, bbox_to_anchor=(1.05, 1), loc="upper left")
    st.pyplot(fig)
    plt.close(fig)


def render_variable_plot(df, variable, graph_type, treatment_variable=""):
    is_numeric = is_numeric_column(df, variable)
    if is_numeric:
        if graph_type == "Histogram":
            plot_numeric_histogram(df, variable)
        elif graph_type == "Boxplot by treatment/group":
            plot_numeric_boxplot_by_group(df, variable, treatment_variable)
        elif graph_type == "Mean with 95% CI by treatment/group":
            plot_numeric_mean_ci_by_group(df, variable, treatment_variable)
        else:
            st.warning(f"{graph_type} is not suitable for numeric variable {variable}.")
    else:
        if graph_type == "Bar chart":
            plot_categorical_bar(df, variable)
        elif graph_type == "Grouped bar chart by treatment/group":
            plot_categorical_bar_by_group(df, variable, treatment_variable, percent=False)
        elif graph_type == "Percentage bar chart by treatment/group":
            plot_categorical_bar_by_group(df, variable, treatment_variable, percent=True)
        else:
            st.warning(f"{graph_type} is not suitable for categorical variable {variable}.")


# ============================================================
# TIMEPOINT ANALYSIS HELPERS
# ============================================================

def paired_test_for_change(baseline_values, followup_values):
    """
    Within-patient baseline vs follow-up test.

    Uses:
    - paired t-test if change scores are normal
    - Wilcoxon signed-rank test if change scores are non-normal or normality cannot be assessed
    """
    paired_df = pd.DataFrame({
        "baseline": pd.to_numeric(baseline_values, errors="coerce"),
        "followup": pd.to_numeric(followup_values, errors="coerce"),
    }).dropna()

    if paired_df.empty or paired_df.shape[0] < 2:
        return "Not available", None, "Too few paired observations", np.nan

    change = paired_df["followup"] - paired_df["baseline"]

    if change.dropna().shape[0] < 3 or change.nunique() < 3:
        normality_label = "Not assessed"
        normality_p = np.nan
        use_parametric = False
    else:
        test_name, normality_p, interpretation = normality_test(change)
        use_parametric = normality_p is not None and not pd.isna(normality_p) and normality_p >= 0.05
        normality_label = "Normal" if use_parametric else "Non-normal"

    try:
        if use_parametric:
            _, p_value = stats.ttest_rel(paired_df["followup"], paired_df["baseline"], nan_policy="omit")
            return "Paired t-test", p_value, normality_label, normality_p

        # Wilcoxon can fail if all differences are zero
        if np.allclose(change, 0, equal_nan=True):
            return "Wilcoxon signed-rank", 1.0, normality_label, normality_p

        _, p_value = stats.wilcoxon(paired_df["followup"], paired_df["baseline"])
        return "Wilcoxon signed-rank", p_value, normality_label, normality_p

    except Exception as error:
        return "Not available", None, f"Test failed: {error}", normality_p


def between_treatment_change_test(df, change_col, treatment_variable, treatment_groups):
    """
    Compare change scores between treatments.

    Uses:
    - Welch t-test for 2 groups if change is normal
    - Mann-Whitney U for 2 groups if change is non-normal
    - One-way ANOVA for 3+ groups if change is normal
    - Kruskal-Wallis for 3+ groups if change is non-normal
    """
    included = df[df[treatment_variable].astype(str).str.strip().isin([str(g).strip() for g in treatment_groups])].copy()
    change_values_all = pd.to_numeric(included[change_col], errors="coerce").dropna()

    if len(change_values_all) < 3 or change_values_all.nunique() < 3:
        normality_label = "Not assessed"
        normality_p = np.nan
        use_parametric = False
    else:
        _, normality_p, _ = normality_test(change_values_all)
        use_parametric = normality_p is not None and not pd.isna(normality_p) and normality_p >= 0.05
        normality_label = "Normal" if use_parametric else "Non-normal"

    grouped_values = []
    for group in treatment_groups:
        vals = included.loc[
            included[treatment_variable].astype(str).str.strip() == str(group).strip(),
            change_col
        ]
        vals = pd.to_numeric(vals, errors="coerce").dropna()
        if len(vals) > 0:
            grouped_values.append(vals)

    if len(grouped_values) < 2:
        return "Not available", None, normality_label, normality_p

    try:
        if len(grouped_values) == 2:
            if use_parametric:
                _, p_value = stats.ttest_ind(grouped_values[0], grouped_values[1], equal_var=False, nan_policy="omit")
                return "Welch t-test", p_value, normality_label, normality_p
            _, p_value = stats.mannwhitneyu(grouped_values[0], grouped_values[1], alternative="two-sided")
            return "Mann–Whitney U", p_value, normality_label, normality_p

        if use_parametric:
            _, p_value = stats.f_oneway(*grouped_values)
            return "One-way ANOVA", p_value, normality_label, normality_p

        _, p_value = stats.kruskal(*grouped_values)
        return "Kruskal–Wallis", p_value, normality_label, normality_p

    except Exception:
        return "Not available", None, normality_label, normality_p


def summarize_change_by_group(df, change_col, treatment_variable, treatment_groups, normality_label):
    """
    Summarize change score by treatment group.
    """
    rows = []
    use_mean = normality_label == "Normal"

    for group in treatment_groups:
        values = df.loc[
            df[treatment_variable].astype(str).str.strip() == str(group).strip(),
            change_col
        ]
        values = pd.to_numeric(values, errors="coerce").dropna()

        if use_mean:
            summary = format_mean_sd(values)
            summary_type = "Mean ± SD"
        else:
            summary = format_median_iqr(values)
            summary_type = "Median (IQR)"

        rows.append({
            "Treatment": group,
            "N": int(len(values)),
            "Summary shown": summary_type,
            "Change summary": summary,
        })

    return rows


def plot_timepoint_change_boxplot(df, change_col, treatment_variable, treatment_groups):
    plot_df = df[[change_col, treatment_variable]].copy()
    plot_df[change_col] = pd.to_numeric(plot_df[change_col], errors="coerce")
    plot_df[treatment_variable] = plot_df[treatment_variable].astype(str).str.strip()
    plot_df = plot_df.dropna()
    plot_df = plot_df[plot_df[treatment_variable].isin([str(g).strip() for g in treatment_groups])]

    if plot_df.empty:
        st.warning("No change-score data available to plot.")
        return

    groups = [str(g).strip() for g in treatment_groups]
    data = [plot_df.loc[plot_df[treatment_variable] == group, change_col].dropna() for group in groups]
    filtered = [(g, d) for g, d in zip(groups, data) if len(d) > 0]

    if len(filtered) == 0:
        st.warning("No treatment groups have change-score data to plot.")
        return

    groups, data = zip(*filtered)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, labels=groups, showmeans=True)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title(f"Change score by treatment: {change_col}")
    ax.set_xlabel(treatment_variable)
    ax.set_ylabel("Change from baseline")
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)
    plt.close(fig)


def plot_timepoint_mean_scores(df, baseline_col, followup_cols, treatment_variable, treatment_groups):
    """
    Plot mean baseline/follow-up scores safely.

    This function only uses columns that exist in the current uploaded Excel file.
    It avoids crashes from stale session-state variables such as old baseline columns.
    """
    if df is None or df.empty:
        st.warning("No data available for the timepoint plot.")
        return

    missing_columns = []

    if baseline_col not in df.columns:
        st.warning(
            f"Cannot plot timepoint graph because the baseline column '{baseline_col}' "
            "is not present in the current uploaded Excel file."
        )
        return

    valid_followup_cols = []
    for col in followup_cols:
        if col in df.columns:
            valid_followup_cols.append(col)
        else:
            missing_columns.append(col)

    if len(valid_followup_cols) == 0:
        st.warning("Select at least one follow-up column that exists in the current uploaded Excel file.")
        return

    if treatment_variable not in df.columns:
        st.warning(
            f"Cannot plot by treatment because '{treatment_variable}' is not present in the current uploaded Excel file."
        )
        return

    if missing_columns:
        st.info(
            "Some follow-up columns were skipped because they are not in the current Excel file: "
            + ", ".join([str(col) for col in missing_columns])
        )

    time_labels = ["Baseline"] + valid_followup_cols

    fig, ax = plt.subplots(figsize=(8, 4))

    plotted_any = False

    for group in treatment_groups:
        group_df = df[df[treatment_variable].astype(str).str.strip() == str(group).strip()]

        if group_df.empty:
            continue

        means = []

        baseline_values = pd.to_numeric(group_df[baseline_col], errors="coerce")
        means.append(baseline_values.mean())

        for followup_col in valid_followup_cols:
            vals = pd.to_numeric(group_df[followup_col], errors="coerce")
            means.append(vals.mean())

        if not all(pd.isna(means)):
            ax.plot(time_labels, means, marker="o", label=str(group))
            plotted_any = True

    if not plotted_any:
        st.warning("No numeric data were available to plot for the selected treatment groups.")
        plt.close(fig)
        return

    ax.set_title(f"Mean score over time: {baseline_col}")
    ax.set_xlabel("Timepoint")
    ax.set_ylabel("Mean score")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title=treatment_variable)
    st.pyplot(fig)
    plt.close(fig)


def plot_timepoint_spaghetti(df, id_col, baseline_col, followup_cols, treatment_variable=None, selected_group=None):
    """
    Plot individual trajectories safely, using only columns that exist in the current dataframe.
    """
    if df is None or df.empty:
        st.warning("No patient data available for spaghetti plot.")
        return

    if baseline_col not in df.columns:
        st.warning(
            f"Cannot plot spaghetti graph because the baseline column '{baseline_col}' "
            "is not present in the current uploaded Excel file."
        )
        return

    valid_followup_cols = [col for col in followup_cols if col in df.columns]

    if len(valid_followup_cols) == 0:
        st.warning("Select at least one follow-up column that exists in the current uploaded Excel file.")
        return

    if id_col and id_col not in df.columns:
        st.info(f"Patient ID column '{id_col}' is not present. Plotting rows without patient labels.")

    cols = [baseline_col] + valid_followup_cols
    time_labels = ["Baseline"] + valid_followup_cols

    plot_df = df.copy()

    if treatment_variable and treatment_variable in plot_df.columns and selected_group not in [None, "All"]:
        plot_df = plot_df[plot_df[treatment_variable].astype(str).str.strip() == str(selected_group).strip()]

    if plot_df.empty:
        st.warning("No patient data available for spaghetti plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    max_patients = min(50, plot_df.shape[0])
    plot_df = plot_df.head(max_patients)

    plotted_any = False

    for _, row in plot_df.iterrows():
        values = [
            pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            for col in cols
        ]

        if not all(pd.isna(values)):
            ax.plot(time_labels, values, marker="o", alpha=0.35)
            plotted_any = True

    if not plotted_any:
        st.warning("No numeric values were available to plot.")
        plt.close(fig)
        return

    ax.set_title(f"Individual patient trajectories: {baseline_col}")
    ax.set_xlabel("Timepoint")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# MODEL EXPORT HELPERS
# ============================================================

def build_calculator_export_package(model_info, model_label):
    """
    Build a portable Python object for the established-model calculator.

    This can be downloaded as a .pkl file and loaded later.
    """
    export_package = {
        "model_label": model_label,
        "model_type": model_info.get("model_type", "Clinical"),
        "sample_size": model_info.get("sample_size", ""),
        "method": model_info.get("method", model_label),
        "model_site": model_info.get("model_site", "Brain"),
        "outcome_group": model_info.get("outcome_group", "Neurocognitive"),
        "outcome_variable": model_info.get("outcome_variable", ""),
        "model_description": model_info.get("model_description", ""),
        "parameter_variables_filename": model_info.get("parameter_variables_filename", ""),
        "parameter_variables_bytes": model_info.get("parameter_variables_bytes", None),
        "publication_not_available": model_info.get("publication_not_available", False),
        "supporting_publication_title": model_info.get("supporting_publication_title", ""),
        "supporting_publication_reference": model_info.get("supporting_publication_reference", ""),
        "supporting_publication_filename": model_info.get("supporting_publication_filename", ""),
        "supporting_publication_bytes": model_info.get("supporting_publication_bytes", None),
        "pipeline": model_info.get("pipeline"),
        "predictors": model_info.get("predictors", []),
        "metrics": model_info.get("metrics", {}),
        "model_settings": model_info.get("model_settings", {}),
        "export_note": "BrainRT Analytics established model calculator export",
    }
    return export_package


def safe_metric_text(value):
    try:
        if value is None or pd.isna(value):
            return "Not available"
        return f"{float(value):.3f}"
    except Exception:
        return "Not available"


# ============================================================
# MODEL PARAMETER DESCRIPTIONS
# ============================================================

def parameter_description_table(method):
    """
    User-facing descriptions and suggested settings for model parameters.
    """
    if method == "Regression":
        rows = [
            {
                "Parameter": "Use class balancing",
                "What it does": "Gives more weight to the smaller outcome group, useful when decline/no-decline groups are imbalanced.",
                "Suggested setting": "On",
            },
            {
                "Parameter": "Maximum iterations",
                "What it does": "Maximum number of optimisation steps used to fit the logistic regression model.",
                "Suggested setting": "3000",
            },
            {
                "Parameter": "Validation set percentage",
                "What it does": "Percentage of patients held back to test the model after training.",
                "Suggested setting": "20%",
            },
        ]

    elif method == "Random Forest":
        rows = [
            {
                "Parameter": "Number of trees",
                "What it does": "How many decision trees are built. More trees usually improve stability but take longer.",
                "Suggested setting": "500",
            },
            {
                "Parameter": "Automatic maximum depth",
                "What it does": "Lets trees grow until stopping rules are reached. Can work well but may overfit small datasets.",
                "Suggested setting": "On initially; turn off and use depth 3-8 if overfitting",
            },
            {
                "Parameter": "Maximum tree depth",
                "What it does": "Limits how complex each tree can become. Lower depth reduces overfitting.",
                "Suggested setting": "3-8 for small/medium clinical datasets",
            },
            {
                "Parameter": "Minimum samples per leaf",
                "What it does": "Minimum number of patients allowed in a final tree leaf. Higher values make the model smoother.",
                "Suggested setting": "2-5",
            },
            {
                "Parameter": "Minimum samples to split",
                "What it does": "Minimum number of patients required before a tree node can be split.",
                "Suggested setting": "2-10",
            },
            {
                "Parameter": "Use class balancing",
                "What it does": "Adjusts for imbalance in decline/no-decline classes.",
                "Suggested setting": "On",
            },
            {
                "Parameter": "Random seed",
                "What it does": "Makes the random split and model reproducible.",
                "Suggested setting": "42",
            },
            {
                "Parameter": "Validation set percentage",
                "What it does": "Percentage of patients held back to test the model after training.",
                "Suggested setting": "20%",
            },
        ]

    elif method == "XGBoost":
        rows = [
            {
                "Parameter": "Number of boosting rounds / trees",
                "What it does": "Number of sequential boosted trees. More trees can improve learning but may overfit.",
                "Suggested setting": "300-500",
            },
            {
                "Parameter": "Learning rate",
                "What it does": "Controls how much each tree changes the model. Lower values learn more slowly.",
                "Suggested setting": "0.03-0.10",
            },
            {
                "Parameter": "Maximum tree depth",
                "What it does": "Controls complexity of each boosted tree.",
                "Suggested setting": "2-4 for clinical datasets",
            },
            {
                "Parameter": "Subsample fraction",
                "What it does": "Fraction of patients sampled for each tree. Helps reduce overfitting.",
                "Suggested setting": "0.80-0.90",
            },
            {
                "Parameter": "Column sample fraction",
                "What it does": "Fraction of predictors sampled for each tree. Helps reduce overfitting.",
                "Suggested setting": "0.80-0.90",
            },
            {
                "Parameter": "Minimum child weight",
                "What it does": "Makes splits more conservative when increased.",
                "Suggested setting": "1-5",
            },
            {
                "Parameter": "Gamma",
                "What it does": "Minimum improvement needed to make a split. Higher values reduce overfitting.",
                "Suggested setting": "0 initially; try 0.1-1 if overfitting",
            },
            {
                "Parameter": "L1 regularisation alpha",
                "What it does": "Can shrink some variables toward zero.",
                "Suggested setting": "0 initially",
            },
            {
                "Parameter": "L2 regularisation lambda",
                "What it does": "Penalises large model weights and reduces overfitting.",
                "Suggested setting": "1",
            },
            {
                "Parameter": "Random seed",
                "What it does": "Makes the random split and model reproducible.",
                "Suggested setting": "42",
            },
            {
                "Parameter": "Validation set percentage",
                "What it does": "Percentage of patients held back to test the model after training.",
                "Suggested setting": "20%",
            },
        ]

    else:
        rows = []

    return pd.DataFrame(rows)


# ============================================================
# CUSTOM OPTION HELPERS
# ============================================================

def get_custom_options(key):
    """
    Return user-added options stored in session state.
    These persist during the current app session.
    """
    if key not in st.session_state:
        st.session_state[key] = []
    return st.session_state[key]


def add_custom_option(key, new_value):
    """
    Add a custom option to session state if it is not already present.
    """
    if key not in st.session_state:
        st.session_state[key] = []

    new_value = str(new_value).strip()

    if new_value != "" and new_value not in st.session_state[key]:
        st.session_state[key].append(new_value)


def multiselect_with_add_option(label, base_options, selected_key, custom_key, add_label, help_text=None):
    """
    A multiselect that allows the user to add new options directly.
    Added options are remembered in session state.
    """
    custom_options = get_custom_options(custom_key)
    all_options = list(dict.fromkeys(base_options + custom_options))

    existing_selected = st.session_state.get(selected_key, [])
    default_selected = [value for value in existing_selected if value in all_options]

    selected_values = st.multiselect(
        label,
        options=all_options,
        default=default_selected,
        key=selected_key,
        help=help_text,
    )

    add_col1, add_col2 = st.columns([3, 1])

    with add_col1:
        new_option = st.text_input(
            add_label,
            value="",
            key=f"{custom_key}_new_input",
            placeholder="Type a new option and click Add"
        )

    with add_col2:
        st.write("")
        st.write("")
        if st.button("Add", key=f"{custom_key}_add_button", use_container_width=True):
            cleaned = new_option.strip()
            add_custom_option(custom_key, cleaned)

            if cleaned != "":
                current_selected = st.session_state.get(selected_key, [])
                if cleaned not in current_selected:
                    st.session_state[selected_key] = current_selected + [cleaned]

            st.rerun()

    return st.session_state.get(selected_key, selected_values)


# ============================================================
# VOXEL PATIENT DATA HELPERS
# ============================================================

def make_voxel_patient_data_template():
    """
    Example Excel structure for voxel-based analysis patient metadata.
    First column must be Patient_ID and must match imaging file/folder IDs.
    """
    return pd.DataFrame({
        "Patient_ID": ["P001", "P002", "P003"],
        "Treatment_Group": ["Proton", "Photon", "Proton"],
        "Age": [45, 52, 38],
        "Sex": ["Female", "Male", "Female"],
        "Tumour_Site": ["Brain / CNS", "Brain / CNS", "Brain / CNS"],
        "Diagnosis": ["Glioma", "Meningioma", "Glioma"],
        "Baseline_Score": [28, 30, 27],
        "Followup_Score": [26, 29, 24],
        "Outcome_Decline": [1, 0, 1],
        "Outcome_Time_Months": [12, 12, 12],
        "Dose_Mean_Brain": [18.5, 22.1, 16.7],
        "Dose_Max_Brainstem": [42.0, 48.2, 39.5],
        "Image_File_ID": ["P001_T1.nii.gz", "P002_T1.nii.gz", "P003_T1.nii.gz"],
        "Mask_File_ID": ["P001_mask.nii.gz", "P002_mask.nii.gz", "P003_mask.nii.gz"],
    })


def validate_voxel_patient_data(df, patient_id_col):
    """
    Return validation messages for uploaded voxel patient metadata.
    """
    messages = []
    warnings = []

    if df is None or df.empty:
        warnings.append("The uploaded file is empty.")
        return messages, warnings

    if patient_id_col not in df.columns:
        warnings.append("The selected patient ID column was not found.")
        return messages, warnings

    missing_ids = df[patient_id_col].isna().sum()
    duplicate_ids = df[patient_id_col].duplicated().sum()

    if missing_ids > 0:
        warnings.append(f"{missing_ids} row(s) have missing patient IDs.")

    if duplicate_ids > 0:
        warnings.append(f"{duplicate_ids} duplicate patient ID row(s) found. Patient IDs should usually be unique.")

    if missing_ids == 0 and duplicate_ids == 0:
        messages.append("Patient ID column looks valid: no missing or duplicate IDs detected.")

    return messages, warnings



def check_voxel_excel_data_quality(df, patient_id_col=None):
    """
    Check uploaded patient metadata for common data-quality issues.

    Adds Excel-style row numbers:
    - row 1 = header
    - first patient/data row = Excel row 2

    Flags:
    - missing values
    - common missing-value text labels
    - duplicate patient IDs
    - possible numeric values entered with units, e.g. 20cc, 45Gy
    - mixed numeric/text columns
    - columns with only one unique value
    """
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    missing_labels = {
        "n/a", "na", "not available", "not applicable", "missing",
        "unknown", "unk", "none", "null", "-", "--", "."
    }

    issue_rows = []
    missing_rows = []

    total_rows = df.shape[0]

    def excel_rows_from_mask(mask, max_rows=20):
        """
        Convert a boolean mask/index to Excel-style row numbers.
        DataFrame index 0 -> Excel row 2 because row 1 is the header.
        """
        row_numbers = (df.index[mask] + 2).tolist()
        shown = row_numbers[:max_rows]
        if len(row_numbers) > max_rows:
            return ", ".join(map(str, shown)) + f" ... (+{len(row_numbers) - max_rows} more)"
        return ", ".join(map(str, shown))

    for col in df.columns:
        series = df[col]
        series_as_text = series.astype(str).str.strip()
        lower_text = series_as_text.str.lower()

        blank_mask = series.isna() | (series_as_text == "")
        missing_label_mask = lower_text.isin(missing_labels)
        missing_like_mask = blank_mask | missing_label_mask

        non_missing = series.dropna()
        true_missing_count = int(blank_mask.sum())
        missing_label_count = int(missing_label_mask.sum())
        total_missing_like = int(missing_like_mask.sum())

        missing_rows.append({
            "Column": col,
            "Missing blank count": true_missing_count,
            "Missing text-label count": missing_label_count,
            "Total missing / unavailable": total_missing_like,
            "Total rows": int(total_rows),
            "Missing %": round((total_missing_like / total_rows) * 100, 1) if total_rows > 0 else 0,
            "Rows with missing / unavailable": excel_rows_from_mask(missing_like_mask) if total_missing_like > 0 else "",
        })

        if total_missing_like > 0:
            issue_rows.append({
                "Column": col,
                "Issue": "Missing/unavailable values",
                "Rows": excel_rows_from_mask(missing_like_mask),
                "Details": f"{total_missing_like} value(s) are blank or coded as missing/unavailable.",
                "Suggestion": "Prefer blank cells for missing data, or use one consistent label such as N/A.",
                "Severity": "Check",
            })

        # Detect values that combine numbers and units, e.g. 20cc, 54Gy, 12months
        text_non_missing_mask = (series_as_text != "") & (~lower_text.isin(missing_labels))
        text_non_missing = series_as_text[text_non_missing_mask]

        unit_like_local_mask = text_non_missing.str.contains(
            r"^\s*-?\d+(\.\d+)?\s*[A-Za-zµ]+",
            regex=True,
            na=False
        )

        unit_like_count = int(unit_like_local_mask.sum())

        if unit_like_count > 0:
            unit_issue_index = text_non_missing[unit_like_local_mask].index
            full_unit_mask = df.index.isin(unit_issue_index)
            example_values = text_non_missing[unit_like_local_mask].head(5).tolist()

            issue_rows.append({
                "Column": col,
                "Issue": "Number appears to include units",
                "Rows": excel_rows_from_mask(full_unit_mask),
                "Details": f"{unit_like_count} value(s) look like numbers with units. Examples: {', '.join(map(str, example_values))}",
                "Suggestion": "Enter numbers only in the cell, e.g. 20 not 20cc. Put the unit in the column name, e.g. Volume_cc.",
                "Severity": "Important",
            })

        # Detect mixed numeric and text
        numeric_converted = pd.to_numeric(text_non_missing, errors="coerce")
        numeric_mask_local = numeric_converted.notna()
        text_mask_local = numeric_converted.isna()

        numeric_count = int(numeric_mask_local.sum())
        text_count = int(text_mask_local.sum())

        if numeric_count > 0 and text_count > 0:
            mixed_index = text_non_missing[text_mask_local].index
            full_mixed_mask = df.index.isin(mixed_index)
            example_text = text_non_missing[text_mask_local].head(5).tolist()

            issue_rows.append({
                "Column": col,
                "Issue": "Mixed numeric and text values",
                "Rows": excel_rows_from_mask(full_mixed_mask),
                "Details": f"{numeric_count} numeric-like value(s) and {text_count} text-like value(s) detected. Text examples: {', '.join(map(str, example_text))}",
                "Suggestion": "Keep each column as one data type. For numeric columns, use numbers only.",
                "Severity": "Check",
            })

        # Constant column
        unique_count = int(non_missing.nunique(dropna=True))
        if total_rows > 1 and unique_count == 1:
            non_missing_mask = ~series.isna()
            issue_rows.append({
                "Column": col,
                "Issue": "Only one unique value",
                "Rows": excel_rows_from_mask(non_missing_mask),
                "Details": "This column has the same value for all non-missing rows.",
                "Suggestion": "This may not be useful as a covariate/outcome unless it is expected.",
                "Severity": "Low",
            })

    if patient_id_col is not None and patient_id_col in df.columns:
        duplicated_mask = df[patient_id_col].duplicated(keep=False)
        duplicate_count = int(duplicated_mask.sum())

        if duplicate_count > 0:
            examples = df.loc[duplicated_mask, patient_id_col].astype(str).head(5).tolist()
            issue_rows.append({
                "Column": patient_id_col,
                "Issue": "Duplicate patient IDs",
                "Rows": excel_rows_from_mask(duplicated_mask),
                "Details": f"{duplicate_count} row(s) have duplicated patient IDs. Examples: {', '.join(examples)}",
                "Suggestion": "Patient IDs should usually be unique and must match imaging folders/files.",
                "Severity": "Important",
            })

    issue_df = pd.DataFrame(issue_rows)
    missing_df = pd.DataFrame(missing_rows)

    # Put row numbers near the front of the table.
    if not issue_df.empty:
        preferred_cols = ["Column", "Issue", "Rows", "Severity", "Details", "Suggestion"]
        issue_df = issue_df[[col for col in preferred_cols if col in issue_df.columns]]

    return issue_df, missing_df



# ============================================================


def infer_patient_id_from_path_or_filename(relative_path, filename):
    """
    Infer patient ID from folder path first, then filename.

    Preferred folder structures:
      Images/Pt001/CT/...
      Images/Pt001/MR/...
      Images/Pt001/Dose/...
      Images/Pt001/RTSTRUCT/...
    """
    rel = str(relative_path).replace("\\", "/")
    parts = [p.strip() for p in rel.split("/") if p.strip() != ""]

    role_words = {
        "ct", "mr", "mri", "pet", "dose", "rtdose",
        "rtstruct", "rt_struct", "struct", "structure",
        "mask", "masks", "seg", "segmentation", "dicom", "nifti",
        "images", "image", "planning", "scan", "scans"
    }

    for part in parts[:-1]:
        clean = part.replace("-", "_").replace(" ", "_")
        if clean.lower() in role_words:
            continue
        if any(char.isdigit() for char in clean):
            return clean

    return infer_patient_id_from_filename(filename)


def classify_image_file_role_from_path(relative_path, filename):
    """
    Classify file role using folder path plus filename.
    This helps if files are organised like Pt001/CT/file.dcm.
    """
    combined = f"{relative_path} {filename}".lower().replace("\\", "/")
    normalised = (
        combined
        .replace(".", "")
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )

    filename_lower = str(filename).lower()
    filename_stem = Path(str(filename)).stem.lower()

    if (
        "rtstruct" in normalised
        or "rtstructure" in normalised
        or "structureset" in normalised
        or "rtss" in normalised
        or filename_lower.startswith("rs.")
        or filename_lower.startswith("rs_")
        or filename_lower.startswith("rs-")
        or filename_stem == "rs"
        or "/rs/" in combined
        or "/rtstruct" in combined
        or "/rt.struct" in combined
        or "/structure" in combined
        or "/struct" in combined
    ):
        return "RTSTRUCT / structure"

    if (
        "rtdose" in normalised
        or "/dose/" in combined
        or "dose_" in combined
        or "_dose" in combined
        or filename_lower.startswith("rd.")
        or filename_lower.startswith("rtdose")
    ):
        return "Dose"

    if "/ct/" in combined or "ct_" in combined or "_ct" in combined or filename_lower.startswith("ct"):
        return "CT"

    if "/mr/" in combined or "/mri/" in combined or "mri" in combined or "mr_" in combined or "_mr" in combined or filename_lower.startswith("mr"):
        return "MRI"

    if "pet" in combined:
        return "PET"

    if "mask" in combined or "seg" in combined:
        return "Mask"

    return classify_image_file_role(filename)


def add_patient_grouping_columns(file_df):
    """
    Add inferred patient ID and file role columns.
    """
    if file_df is None or file_df.empty:
        return pd.DataFrame()

    df = file_df.copy()

    df["Inferred patient ID"] = df.apply(
        lambda row: infer_patient_id_from_path_or_filename(
            row.get("Relative path", row.get("Filename", "")),
            row.get("Filename", "")
        ),
        axis=1
    )

    df["File role"] = df.apply(
        lambda row: classify_image_file_role_from_path(
            row.get("Relative path", row.get("Filename", "")),
            row.get("Filename", "")
        ),
        axis=1
    )

    return df


def make_voxel_patient_file_summary(file_df):
    """
    Create one row per patient ID with file counts grouped by role.
    """
    df = add_patient_grouping_columns(file_df)

    if df.empty:
        return pd.DataFrame()

    roles = ["CT", "MRI", "PET", "Dose", "RTSTRUCT / structure", "Mask", "Main / unknown"]
    rows = []

    for patient_id, group in df.groupby("Inferred patient ID", dropna=False):
        patient_id = str(patient_id).strip()
        row = {
            "Patient ID": patient_id,
            "Total files": int(group.shape[0]),
        }

        for role in roles:
            role_group = group[group["File role"] == role]
            row[f"{role} count"] = int(role_group.shape[0])
            row[f"{role} files"] = "; ".join(role_group["Relative path"].astype(str).head(5).tolist())

        rows.append(row)

    summary_df = pd.DataFrame(rows)

    if not summary_df.empty:
        summary_df = summary_df.sort_values("Patient ID")

    return summary_df


def scan_voxel_image_directory(directory_path, selected_format):
    """
    Scan a local directory recursively for DICOM or NIfTI files.

    selected_format:
      - "DICOM"
      - "NIfTI"

    Returns a dataframe with file metadata.
    """
    records = []

    directory_path = str(directory_path).strip()

    if directory_path == "":
        return pd.DataFrame(records), "No directory path was entered."

    root = Path(directory_path)

    if not root.exists():
        return pd.DataFrame(records), f"Directory does not exist: {directory_path}"

    if not root.is_dir():
        return pd.DataFrame(records), f"This is not a directory: {directory_path}"

    if selected_format == "DICOM":
        allowed = [".dcm", ".dicom"]
    else:
        allowed = [".nii", ".nii.gz"]

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue

        name_lower = file_path.name.lower()

        if selected_format == "NIfTI":
            is_match = name_lower.endswith(".nii") or name_lower.endswith(".nii.gz")
            extension = ".nii.gz" if name_lower.endswith(".nii.gz") else file_path.suffix.lower()
        else:
            is_match = file_path.suffix.lower() in allowed
            extension = file_path.suffix.lower()

        if not is_match:
            continue

        try:
            size_mb = round(file_path.stat().st_size / (1024 * 1024), 2)
        except Exception:
            size_mb = None

        records.append({
            "File format": selected_format,
            "Filename": file_path.name,
            "Relative path": str(file_path.relative_to(root)),
            "Full path": str(file_path),
            "Extension": extension,
            "Size MB": size_mb,
        })

    file_df = pd.DataFrame(records)
    file_df = add_patient_grouping_columns(file_df)
    return file_df, ""


def get_uploaded_voxel_image_records(uploaded_files, selected_format):
    """
    Convert uploaded Streamlit files into a metadata dataframe.
    Used as a fallback if the user cannot scan a local directory.
    """
    records = []

    if uploaded_files is None:
        file_df = pd.DataFrame(records)
    file_df = add_patient_grouping_columns(file_df)
    return file_df

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        lower_name = filename.lower()

        if lower_name.endswith(".nii.gz"):
            extension = ".nii.gz"
        else:
            extension = Path(filename).suffix.lower()

        records.append({
            "File format": selected_format,
            "Filename": filename,
            "Relative path": filename,
            "Full path": "",
            "Extension": extension,
            "Size MB": round(len(uploaded_file.getvalue()) / (1024 * 1024), 2),
        })

    return pd.DataFrame(records)


# VOXEL IMAGE FILE HELPERS
# ============================================================

def get_uploaded_file_records(uploaded_files, file_group):
    """
    Convert Streamlit uploaded files into a simple metadata table.
    """
    records = []

    if uploaded_files is None:
        return pd.DataFrame(records)

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        lower_name = filename.lower()

        if lower_name.endswith(".nii.gz"):
            extension = ".nii.gz"
        else:
            extension = Path(filename).suffix.lower()

        records.append({
            "File group": file_group,
            "Filename": filename,
            "Extension": extension,
            "Size MB": round(len(uploaded_file.getvalue()) / (1024 * 1024), 2),
        })

    return pd.DataFrame(records)


def infer_patient_id_from_filename(filename):
    """
    Try to infer patient ID from filename using simple rules.

    Expected examples:
    - Pt001.dcm
    - CT_Pt001.dcm
    - MR_Pt001.nii.gz
    - Dose_Pt001.nii.gz
    - RTSTRUCT_Pt001.dcm
    """
    stem = filename
    if stem.lower().endswith(".nii.gz"):
        stem = stem[:-7]
    else:
        stem = Path(stem).stem

    tokens = stem.replace("-", "_").replace(" ", "_").split("_")

    prefixes = {"ct", "mr", "mri", "dose", "rtstruct", "rt_struct", "mask", "pet", "image", "main"}

    for token in tokens:
        clean = token.strip()
        if clean.lower() in prefixes:
            continue
        if any(char.isdigit() for char in clean):
            return clean

    return tokens[-1].strip() if len(tokens) > 0 else ""


def classify_image_file_role(filename):
    """
    Classify file role based on filename.
    """
    name = str(filename).lower()
    stem = Path(str(filename)).stem.lower()

    normalised = (
        name
        .replace(".", "")
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )

    if (
        "rtstruct" in normalised
        or "rtstructure" in normalised
        or "structureset" in normalised
        or "rtss" in normalised
        or name.startswith("rs.")
        or name.startswith("rs_")
        or name.startswith("rs-")
        or stem == "rs"
    ):
        return "RTSTRUCT / structure"

    if "rtdose" in normalised or "dose" in name or name.startswith("rd."):
        return "Dose"

    if "ct" in name:
        return "CT"

    if "mri" in name or "_mr" in name or name.startswith("mr"):
        return "MRI"

    if "pet" in name:
        return "PET"

    if "mask" in name or "seg" in name:
        return "Mask"

    return "Main / unknown"


def check_voxel_image_files_quality(file_df, patient_id_list=None):
    """
    Check image files for naming and matching issues.
    """
    issue_rows = []

    if file_df is None or file_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    records = add_patient_grouping_columns(file_df)

    allowed_extensions = [".dcm", ".dicom", ".nii", ".nii.gz"]

    for idx, row in records.iterrows():
        filename = row["Filename"]
        extension = row["Extension"]
        inferred_id = row["Inferred patient ID"]
        role = row["File role"]

        if extension not in allowed_extensions:
            issue_rows.append({
                "Filename": filename,
                "Issue": "Unexpected file format",
                "Details": f"Extension '{extension}' is not recognised as DICOM or NIfTI.",
                "Suggestion": "Use DICOM (.dcm/.dicom) or NIfTI (.nii/.nii.gz).",
                "Severity": "Important",
            })

        if inferred_id == "":
            issue_rows.append({
                "Filename": filename,
                "Issue": "Could not infer patient ID",
                "Details": "The filename does not appear to contain a patient ID.",
                "Suggestion": "Include patient ID in filename, e.g. CT_Pt001.dcm or MR_Pt001.nii.gz.",
                "Severity": "Important",
            })

        if role == "Main / unknown":
            issue_rows.append({
                "Filename": filename,
                "Issue": "File role unclear",
                "Details": "The filename does not clearly say CT, MR, Dose, RTSTRUCT, PET, or Mask.",
                "Suggestion": "Use labels such as CT_Pt001, MR_Pt001, Dose_Pt001, RTSTRUCT_Pt001.",
                "Severity": "Check",
            })

    if patient_id_list is not None and len(patient_id_list) > 0:
        patient_id_set = set([str(x).strip() for x in patient_id_list if str(x).strip() != ""])
        file_id_set = set(records["Inferred patient ID"].dropna().astype(str).str.strip())

        missing_from_files = sorted(list(patient_id_set - file_id_set))
        unexpected_file_ids = sorted(list(file_id_set - patient_id_set))

        if len(missing_from_files) > 0:
            issue_rows.append({
                "Filename": "Patient metadata",
                "Issue": "Patients missing imaging files",
                "Details": f"{len(missing_from_files)} patient ID(s) from the Excel sheet were not found in uploaded filenames. Examples: {', '.join(missing_from_files[:10])}",
                "Suggestion": "Check that imaging filenames contain the same patient IDs as the Excel sheet.",
                "Severity": "Important",
            })

        if len(unexpected_file_ids) > 0:
            issue_rows.append({
                "Filename": "Uploaded image files",
                "Issue": "Image files without matching patient metadata",
                "Details": f"{len(unexpected_file_ids)} inferred file ID(s) were not found in the Excel patient IDs. Examples: {', '.join(unexpected_file_ids[:10])}",
                "Suggestion": "Check patient ID spelling and naming convention.",
                "Severity": "Check",
            })

    # Check expected role coverage per patient
    for patient_id, group in records.groupby("Inferred patient ID"):
        if patient_id == "":
            continue
        roles = set(group["File role"].tolist())

        if "CT" not in roles and "MRI" not in roles and "Main / unknown" not in roles:
            issue_rows.append({
                "Filename": patient_id,
                "Issue": "No main image detected for patient",
                "Details": "No CT or MRI file was detected for this patient.",
                "Suggestion": "Include a main file labelled with patient ID, e.g. CT_Pt001 or MR_Pt001.",
                "Severity": "Important",
            })

        if "Dose" not in roles:
            issue_rows.append({
                "Filename": patient_id,
                "Issue": "No dose file detected",
                "Details": "No dose file was detected for this patient.",
                "Suggestion": "If dose is required, include Dose_Pt001 or RTDOSE_Pt001.",
                "Severity": "Check",
            })

        if "RTSTRUCT / structure" not in roles and "Mask" not in roles:
            issue_rows.append({
                "Filename": patient_id,
                "Issue": "No structure/mask file detected",
                "Details": "No RTSTRUCT or mask file was detected for this patient.",
                "Suggestion": "Include RTSTRUCT_Pt001 or mask files if structures are required.",
                "Severity": "Check",
            })

    issue_df = pd.DataFrame(issue_rows)

    if not issue_df.empty:
        issue_df = issue_df[["Filename", "Issue", "Severity", "Details", "Suggestion"]]

    return records, issue_df


# ============================================================
# VOXEL IMAGE VIEWER HELPERS
# ============================================================

def normalise_image_slice(slice_2d):
    """
    Convert an image slice to a display-friendly 0-1 range.
    """
    arr = np.asarray(slice_2d, dtype=float)
    arr = np.nan_to_num(arr)

    if arr.size == 0:
        return arr

    p1, p99 = np.percentile(arr, [1, 99])
    if p99 <= p1:
        p1, p99 = np.min(arr), np.max(arr)

    if p99 <= p1:
        return np.zeros_like(arr)

    arr = np.clip(arr, p1, p99)
    arr = (arr - p1) / (p99 - p1)
    return arr


def load_nifti_volume_from_path(file_path):
    """
    Load a NIfTI file from a local path.
    """
    if not NIBABEL_AVAILABLE:
        raise ImportError("nibabel is not installed. Install using: py -m pip install nibabel")

    img = nib.load(str(file_path))
    data = img.get_fdata()

    if data.ndim == 4:
        data = data[..., 0]

    return np.asarray(data)


def load_dicom_series_from_paths(file_paths):
    """
    Load a DICOM series from multiple local DICOM files and return a 3D volume.
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom is not installed. Install using: py -m pip install pydicom")

    slices = []

    for path in file_paths:
        try:
            ds = pydicom.dcmread(str(path), force=True)
            if not hasattr(ds, "pixel_array"):
                continue

            instance_number = getattr(ds, "InstanceNumber", len(slices))
            z_position = None

            try:
                z_position = float(ds.ImagePositionPatient[2])
            except Exception:
                z_position = None

            arr = ds.pixel_array.astype(float)

            slope = float(getattr(ds, "RescaleSlope", 1))
            intercept = float(getattr(ds, "RescaleIntercept", 0))
            arr = arr * slope + intercept

            slices.append({
                "instance_number": instance_number,
                "z_position": z_position,
                "array": arr,
                "path": str(path),
            })
        except Exception:
            continue

    if len(slices) == 0:
        raise ValueError("No readable DICOM slices were found.")

    if all(item["z_position"] is not None for item in slices):
        slices = sorted(slices, key=lambda x: x["z_position"])
    else:
        slices = sorted(slices, key=lambda x: x["instance_number"])

    volume = np.stack([item["array"] for item in slices], axis=-1)

    return volume


def get_patient_image_candidates(file_df, patient_id):
    """
    Return files for one patient grouped into image candidates and mask candidates.
    """
    if file_df is None or file_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = add_patient_grouping_columns(file_df)

    patient_df = df[df["Inferred patient ID"].astype(str) == str(patient_id)].copy()

    image_roles = ["CT", "MRI", "PET", "Main / unknown"]
    mask_roles = ["Mask", "RTSTRUCT / structure"]

    image_df = patient_df[patient_df["File role"].isin(image_roles)].copy()
    mask_df = patient_df[patient_df["File role"].isin(mask_roles)].copy()

    return image_df, mask_df


def render_axial_viewer(volume, mask_items=None, title="Axial viewer", key_prefix="viewer"):
    """
    Basic slicer-like axial viewer.

    - Axial slice only
    - Main image in grayscale
    - Multiple masks overlaid automatically
    - Slider below image
    - Small slice number
    """
    if volume is None:
        st.warning("No image volume available.")
        return

    vol = np.asarray(volume)

    if vol.ndim != 3:
        st.warning(f"Expected a 3D image volume, but got shape: {vol.shape}")
        return

    if mask_items is None:
        mask_items = []

    max_slice = vol.shape[2] - 1
    if max_slice < 0:
        st.warning("Image volume has no slices.")
        return

    default_slice = max_slice // 2

    # Keep orientation controls available but not visually dominant.
    with st.expander("Display orientation controls", expanded=False):
        rotate_degrees = st.selectbox(
            "Rotate display",
            options=[0, 90, 180, 270],
            index=0,
            key=f"{key_prefix}_rotate_display",
            help="Display-only rotation. It does not modify the file."
        )

        flip_lr = st.checkbox(
            "Flip left-right",
            value=False,
            key=f"{key_prefix}_flip_lr"
        )

        flip_ud = st.checkbox(
            "Flip up-down",
            value=False,
            key=f"{key_prefix}_flip_ud"
        )

    show_overlays = True
    mask_alpha = 0.35

    if len(mask_items) > 0:
        show_overlays = st.checkbox(
            "Show all mask overlays",
            value=True,
            key=f"{key_prefix}_show_all_masks"
        )

        mask_alpha = st.slider(
            "Mask opacity",
            min_value=0.05,
            max_value=0.95,
            value=0.35,
            step=0.05,
            key=f"{key_prefix}_mask_alpha"
        )

    slice_key = f"{key_prefix}_slice_slider"
    if slice_key not in st.session_state:
        st.session_state[slice_key] = default_slice

    slice_index = int(st.session_state.get(slice_key, default_slice))
    slice_index = max(0, min(slice_index, max_slice))

    image_slice = normalise_image_slice(vol[:, :, slice_index])

    def apply_display_orientation(arr):
        out = np.asarray(arr)

        if rotate_degrees == 90:
            out = np.rot90(out, k=3)
        elif rotate_degrees == 180:
            out = np.rot90(out, k=2)
        elif rotate_degrees == 270:
            out = np.rot90(out, k=1)

        if flip_lr:
            out = np.fliplr(out)

        if flip_ud:
            out = np.flipud(out)

        return out

    display_image = apply_display_orientation(image_slice)

    # Basic 3D-Slicer-like black view.
    st.markdown(
        """
        <div style="
            background-color:#000000;
            border-radius:8px;
            padding:8px;
            margin-top:4px;
            margin-bottom:4px;">
        """,
        unsafe_allow_html=True,
    )

    fig, ax = plt.subplots(figsize=(2.2, 2.2), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(display_image, cmap="gray")
    ax.axis("off")

    if show_overlays and len(mask_items) > 0:
        overlay_colormaps = ["autumn", "winter", "spring", "cool", "Wistia", "summer"]

        for idx, mask_item in enumerate(mask_items):
            mask = np.asarray(mask_item.get("volume"))

            if mask.shape != vol.shape:
                continue

            mask_slice = mask[:, :, slice_index]
            mask_slice = np.where(mask_slice > 0, 1, np.nan)
            mask_slice = apply_display_orientation(mask_slice)

            ax.imshow(
                mask_slice,
                alpha=mask_alpha,
                cmap=overlay_colormaps[idx % len(overlay_colormaps)],
                vmin=0,
                vmax=1,
            )

    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    st.markdown("</div>", unsafe_allow_html=True)

    st.caption(f"Slice {slice_index + 1} of {max_slice + 1}")

    st.slider(
        "Slice",
        min_value=0,
        max_value=max_slice,
        value=slice_index,
        step=1,
        key=slice_key,
        label_visibility="collapsed"
    )


def load_selected_image_volume(selected_row, all_patient_files=None):
    """
    Load either a NIfTI volume or a DICOM series depending on selected row format.
    """
    selected_format = selected_row.get("File format", "")
    full_path = selected_row.get("Full path", "")
    role = selected_row.get("File role", "")

    if selected_format == "NIfTI":
        return load_nifti_volume_from_path(full_path)

    if selected_format == "DICOM":
        # For DICOM, load all files from the same patient and same role.
        if all_patient_files is None or all_patient_files.empty:
            return load_dicom_series_from_paths([full_path])

        patient_id = selected_row.get("Inferred patient ID", "")
        dicom_df = all_patient_files[
            (all_patient_files["File format"] == "DICOM")
            & (all_patient_files["Inferred patient ID"].astype(str) == str(patient_id))
            & (all_patient_files["File role"] == role)
        ].copy()

        paths = dicom_df["Full path"].dropna().astype(str).tolist()

        if len(paths) == 0:
            paths = [full_path]

        return load_dicom_series_from_paths(paths)

    raise ValueError("Unknown image format selected.")



def describe_mask_candidates(mask_df):
    """
    Create a readable table of mask/structure candidates found for the selected patient.
    """
    if mask_df is None or mask_df.empty:
        return pd.DataFrame(columns=["Contour / mask", "Role", "File"])

    rows = []

    for _, row in mask_df.iterrows():
        filename = str(row.get("Filename", ""))
        relative_path = str(row.get("Relative path", filename))
        role = str(row.get("File role", "Mask"))

        contour_name = filename
        lower_name = filename.lower()

        for prefix in ["mask_", "seg_", "rtstruct_", "rt_struct_", "structure_"]:
            if lower_name.startswith(prefix):
                contour_name = filename[len(prefix):]
                break

        if contour_name.lower().endswith(".nii.gz"):
            contour_name = contour_name[:-7]
        else:
            contour_name = Path(contour_name).stem

        rows.append({
            "Contour / mask": contour_name,
            "Role": role,
            "File": relative_path,
        })

    return pd.DataFrame(rows)


def try_load_first_compatible_mask(mask_df, patient_files, image_shape):
    """
    Automatically load the first NIfTI mask candidate that has the same shape as the image.
    RTSTRUCT DICOM is listed but not overlaid yet.
    """
    if mask_df is None or mask_df.empty:
        return None, None, "No mask/structure files found for this patient."

    load_errors = []

    for _, row in mask_df.iterrows():
        row_dict = row.to_dict()
        filename = str(row_dict.get("Filename", "")).lower()
        file_format = row_dict.get("File format", "")

        is_nifti = file_format == "NIfTI" or filename.endswith(".nii") or filename.endswith(".nii.gz")

        if not is_nifti:
            load_errors.append(
                f"{row_dict.get('Filename', '')}: RTSTRUCT/DICOM contour overlay needs conversion before overlay."
            )
            continue

        try:
            mask_volume = load_selected_image_volume(row_dict, patient_files)

            if np.asarray(mask_volume).shape == tuple(image_shape):
                return mask_volume, row_dict, ""

            load_errors.append(
                f"{row_dict.get('Filename', '')}: mask shape {np.asarray(mask_volume).shape} does not match image shape {image_shape}."
            )
        except Exception as error:
            load_errors.append(f"{row_dict.get('Filename', '')}: {error}")

    if len(load_errors) == 0:
        return None, None, "No compatible mask found."

    return None, None, "No compatible overlay mask loaded. " + " | ".join(load_errors[:3])



def choose_primary_image_for_patient(patient_files):
    """
    Choose the most likely main anatomical image for one patient.

    Priority:
    CT > MRI > PET > Main / unknown
    """
    if patient_files is None or patient_files.empty:
        return None

    role_priority = ["CT", "MRI", "PET", "Main / unknown"]

    for role in role_priority:
        candidates = patient_files[patient_files["File role"] == role].copy()
        if not candidates.empty:
            # Prefer NIfTI single volumes when available, otherwise DICOM series.
            nifti_candidates = candidates[candidates["File format"] == "NIfTI"]
            if not nifti_candidates.empty:
                return nifti_candidates.iloc[0].to_dict()

            return candidates.iloc[0].to_dict()

    return None


def find_mask_candidates_for_patient(patient_files):
    """
    Find mask/structure candidates for the selected patient.

    NIfTI masks can be overlaid directly.
    DICOM RTSTRUCT is listed but needs conversion before overlay.
    """
    if patient_files is None or patient_files.empty:
        return pd.DataFrame()

    mask_roles = ["Mask", "RTSTRUCT / structure"]
    mask_df = patient_files[patient_files["File role"].isin(mask_roles)].copy()

    # Also catch common mask/seg filenames that may have been classified as unknown.
    extra_mask_df = patient_files[
        patient_files["Filename"].astype(str).str.lower().str.contains("mask|seg|structure|rtstruct", regex=True, na=False)
    ].copy()

    mask_df = pd.concat([mask_df, extra_mask_df], ignore_index=True).drop_duplicates(subset=["Relative path", "Filename"])

    return mask_df


def load_patient_image_and_masks(patient_files):
    """
    Load a patient automatically:
    - choose primary CT/MR/PET image
    - load all compatible NIfTI masks in the same patient files
    - list all contour/mask files found
    """
    if patient_files is None or patient_files.empty:
        raise ValueError("No files found for this patient.")

    patient_files = add_patient_grouping_columns(patient_files)

    primary_row = choose_primary_image_for_patient(patient_files)

    if primary_row is None:
        raise ValueError("No main CT/MR/PET image was found for this patient.")

    image_volume = load_selected_image_volume(primary_row, patient_files)
    image_shape = np.asarray(image_volume).shape

    mask_df = find_mask_candidates_for_patient(patient_files)

    loaded_masks = []
    contour_rows = []

    for _, mask_row in mask_df.iterrows():
        row_dict = mask_row.to_dict()
        filename = str(row_dict.get("Filename", ""))
        relative_path = str(row_dict.get("Relative path", filename))
        file_format = row_dict.get("File format", "")
        lower_name = filename.lower()

        is_nifti = file_format == "NIfTI" or lower_name.endswith(".nii") or lower_name.endswith(".nii.gz")

        contour_name = filename
        for prefix in ["mask_", "seg_", "rtstruct_", "rt_struct_", "structure_"]:
            if lower_name.startswith(prefix):
                contour_name = filename[len(prefix):]
                break

        if contour_name.lower().endswith(".nii.gz"):
            contour_name = contour_name[:-7]
        else:
            contour_name = Path(contour_name).stem

        contour_status = "Found"
        overlay_status = "Not overlaid"

        if is_nifti:
            try:
                mask_volume = load_selected_image_volume(row_dict, patient_files)
                mask_shape = np.asarray(mask_volume).shape

                if mask_shape == image_shape:
                    loaded_masks.append({
                        "name": contour_name,
                        "file": relative_path,
                        "volume": mask_volume,
                    })
                    overlay_status = "Overlaid"
                else:
                    overlay_status = f"Shape mismatch: {mask_shape} vs image {image_shape}"

            except Exception as error:
                overlay_status = f"Could not load: {error}"
        else:
            overlay_status = "RTSTRUCT/DICOM conversion needed"

        contour_rows.append({
            "Contour / mask": contour_name,
            "File": relative_path,
            "Format": file_format,
            "Status": contour_status,
            "Overlay": overlay_status,
        })

    contour_table = pd.DataFrame(contour_rows)

    return primary_row, image_volume, loaded_masks, contour_table


def render_contour_side_panel(contour_table):
    """
    Show contours/masks found beside the viewer.
    """
    st.markdown("#### Contours / masks")

    if contour_table is None or contour_table.empty:
        st.info("No masks or structures were found in this patient folder.")
        return

    overlay_count = int((contour_table["Overlay"] == "Overlaid").sum()) if "Overlay" in contour_table.columns else 0
    total_count = contour_table.shape[0]

    st.metric("Contours found", total_count)
    st.metric("Overlaid", overlay_count)

    st.dataframe(contour_table, use_container_width=True, hide_index=True)

    if overlay_count == 0:
        st.warning(
            "No mask could be overlaid. NIfTI masks overlay directly when their shape matches the image. "
            "RTSTRUCT DICOM contours need conversion before overlay."
        )


def render_voxel_quality_image_viewer(file_df):
    """
    Basic slicer-like patient viewer.

    User selects only the patient.
    The app automatically loads:
    - main CT/MR/PET image
    - all compatible masks found in the same patient files
    """
    if file_df is None or file_df.empty:
        st.info("No image files are loaded yet.")
        return

    file_df = add_patient_grouping_columns(file_df)

    st.markdown("### Viewer box")

    patient_ids = sorted([
        str(x) for x in file_df["Inferred patient ID"].dropna().unique().tolist()
        if str(x).strip() != ""
    ])

    if len(patient_ids) == 0:
        st.warning("No patient IDs could be inferred from the uploaded/scanned files.")
        return

    selected_patient = st.selectbox(
        "Select patient",
        options=patient_ids,
        key="voxel_viewer_patient"
    )

    patient_files = file_df[file_df["Inferred patient ID"].astype(str) == str(selected_patient)].copy()

    st.caption(
        "The viewer automatically loads the main CT/MR/PET image and overlays compatible masks found in the same patient folder."
    )

    if st.button("Load patient in viewer", type="primary", use_container_width=True):
        try:
            primary_row, image_volume, loaded_masks, contour_table = load_patient_image_and_masks(patient_files)

            st.session_state.voxel_viewer_image_volume = image_volume
            st.session_state.voxel_viewer_mask_items = loaded_masks
            st.session_state.voxel_viewer_contour_table = contour_table
            st.session_state.voxel_viewer_primary_image = primary_row

            st.success(f"Loaded patient {selected_patient}. Main image: {primary_row.get('Relative path', primary_row.get('Filename', ''))}")

        except Exception as error:
            st.error(f"Could not load patient viewer: {error}")

    viewer_col, contour_col = st.columns([1.1, 1])

    with viewer_col:
        image_volume = st.session_state.get("voxel_viewer_image_volume", None)
        mask_items = st.session_state.get("voxel_viewer_mask_items", [])

        if image_volume is not None:
            render_axial_viewer(
                image_volume,
                mask_items=mask_items,
                title="Axial viewer",
                key_prefix="voxel_quality_viewer"
            )
        else:
            st.info("Select a patient and click **Load patient in viewer**.")

    with contour_col:
        contour_table = st.session_state.get("voxel_viewer_contour_table", pd.DataFrame())
        primary_row = st.session_state.get("voxel_viewer_primary_image", {})

        if primary_row:
            st.markdown("#### Main image")
            st.write(primary_row.get("Relative path", primary_row.get("Filename", "")))

        render_contour_side_panel(contour_table)




# ============================================================
# IMAGE FILENAME QUALITY CHECK HELPERS
# ============================================================

def expected_image_role_from_path_and_name(relative_path, filename):
    """
    Infer file role from folder path and filename.

    RTSTRUCT detection is deliberately broad because centres export these with
    many naming variants, for example:
    - RTSTRUCT_Pt001.dcm
    - RT.Struct.Pt001.dcm
    - RT Struct Pt001.dcm
    - RT-STRUCT-Pt001.dcm
    - RS.Pt001.dcm
    - StructureSet_Pt001.dcm
    """
    raw_combined = f"{relative_path} {filename}"
    combined = raw_combined.lower().replace("\\", "/")

    # Normalised version removes common separators so RT.Struct, RT Struct,
    # RT-STRUCT, RT_STRUCT all become rtstruct.
    normalised = (
        combined
        .replace(".", "")
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )

    filename_lower = str(filename).lower()
    filename_stem = Path(str(filename)).stem.lower()

    # RTSTRUCT / structure set
    if (
        "rtstruct" in normalised
        or "rtstructure" in normalised
        or "structureset" in normalised
        or "structureset" in normalised
        or "rtss" in normalised
        or filename_lower.startswith("rs.")
        or filename_lower.startswith("rs_")
        or filename_lower.startswith("rs-")
        or filename_stem == "rs"
        or "/rs/" in combined
        or "/rtstruct" in combined
        or "/rt.struct" in combined
        or "/structure" in combined
        or "/struct" in combined
    ):
        return "RTSTRUCT"

    # Dose
    if (
        "rtdose" in normalised
        or "/dose" in combined
        or "dose_" in combined
        or "_dose" in combined
        or filename_lower.startswith("rd.")
        or filename_lower.startswith("rtdose")
    ):
        return "Dose"

    # CT
    if (
        "/ct" in combined
        or "ct_" in combined
        or "_ct" in combined
        or filename_lower.startswith("ct")
    ):
        return "CT"

    # MR / MRI
    if (
        "/mr" in combined
        or "/mri" in combined
        or "mr_" in combined
        or "_mr" in combined
        or "mri" in combined
        or filename_lower.startswith("mr")
    ):
        return "MRI"

    # Mask / segmentation
    if (
        "mask" in combined
        or "/seg" in combined
        or "seg_" in combined
        or "contour" in combined
        or "roi" in combined
    ):
        return "Mask"

    if "pet" in combined:
        return "PET"

    return "Unknown"


def expected_filename_example(patient_id, role):
    """
    Suggested naming convention for a file role.
    """
    patient_id = str(patient_id).strip() if str(patient_id).strip() != "" else "Pt001"

    if role == "CT":
        return f"CT_{patient_id}.dcm"
    if role == "MRI":
        return f"MR_{patient_id}.nii.gz"
    if role == "Dose":
        return f"Dose_{patient_id}.dcm or Dose_{patient_id}.nii.gz"
    if role == "RTSTRUCT":
        return f"RTSTRUCT_{patient_id}.dcm"
    if role == "Mask":
        return f"Mask_<StructureName>_{patient_id}.nii.gz"

    return f"<Role>_{patient_id}.dcm"


def run_image_filename_quality_check(file_df, expected_format="DICOM", patient_id_list=None):
    """
    Quality check for directory/file naming.

    Checks:
    - selected format consistency
    - patient ID detectable from folder or filename
    - file role detectable: CT, MR, Dose, RTSTRUCT, Mask
    - image patient IDs match loaded Excel patient IDs when available
    - per-patient required components
    """
    if file_df is None or file_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = add_patient_grouping_columns(file_df.copy())

    issue_rows = []

    expected_ext = [".dcm", ".dicom"] if expected_format == "DICOM" else [".nii", ".nii.gz"]

    if patient_id_list is None:
        patient_id_list = []

    patient_id_list = [
        str(x).strip()
        for x in patient_id_list
        if str(x).strip() != ""
    ]

    patient_id_set = set(patient_id_list)

    df["QC role"] = df.apply(
        lambda row: expected_image_role_from_path_and_name(
            row.get("Relative path", row.get("Filename", "")),
            row.get("Filename", "")
        ),
        axis=1
    )

    for _, row in df.iterrows():
        filename = str(row.get("Filename", ""))
        relative_path = str(row.get("Relative path", filename))
        extension = str(row.get("Extension", "")).lower()
        patient_id = str(row.get("Inferred patient ID", "")).strip()
        role = str(row.get("QC role", "Unknown"))

        if extension not in expected_ext:
            issue_rows.append({
                "Patient ID": patient_id,
                "File": relative_path,
                "Issue": "Wrong format for selected workflow",
                "Detected": extension,
                "Suggestion": f"Use only {expected_format} files in this workflow.",
                "Rename example": "",
                "Severity": "Important",
            })

        if patient_id == "":
            issue_rows.append({
                "Patient ID": "",
                "File": relative_path,
                "Issue": "Patient ID not clear",
                "Detected": "No patient ID detected",
                "Suggestion": "Add patient ID to folder name or filename.",
                "Rename example": expected_filename_example("Pt001", role),
                "Severity": "Important",
            })

        if role == "Unknown":
            issue_rows.append({
                "Patient ID": patient_id,
                "File": relative_path,
                "Issue": "File role not clear",
                "Detected": "Unknown",
                "Suggestion": "Rename file or folder to include CT, MR, Dose, RTSTRUCT, or Mask.",
                "Rename example": expected_filename_example(patient_id, "CT"),
                "Severity": "Important",
            })

        if patient_id_set and patient_id and patient_id not in patient_id_set:
            issue_rows.append({
                "Patient ID": patient_id,
                "File": relative_path,
                "Issue": "Image patient ID not found in loaded Excel sheet",
                "Detected": patient_id,
                "Suggestion": "Check spelling and make the image patient ID match the Excel Patient ID column.",
                "Rename example": "",
                "Severity": "Important",
            })

    summary_rows = []

    for patient_id, group in df.groupby("Inferred patient ID", dropna=False):
        patient_id = str(patient_id).strip()
        if patient_id == "":
            patient_id = "Unknown patient ID"

        roles = set(group["QC role"].tolist())

        row = {
            "Patient ID": patient_id,
            "Total files": int(group.shape[0]),
            "CT present": "Yes" if "CT" in roles else "No",
            "MR present": "Yes" if "MRI" in roles else "No",
            "Dose present": "Yes" if "Dose" in roles else "No",
            "RTSTRUCT present": "Yes" if "RTSTRUCT" in roles else "No",
            "Mask present": "Yes" if "Mask" in roles else "No",
            "Unknown-labelled files": int((group["QC role"] == "Unknown").sum()),
        }
        summary_rows.append(row)

        # Per-patient recommended component checks
        if "CT" not in roles and "MRI" not in roles:
            issue_rows.append({
                "Patient ID": patient_id,
                "File": "Patient folder / files",
                "Issue": "No main CT or MR image detected",
                "Detected": ", ".join(sorted(roles)),
                "Suggestion": "Each patient should have at least one main image labelled CT or MR.",
                "Rename example": expected_filename_example(patient_id, "CT"),
                "Severity": "Important",
            })

        if "Dose" not in roles:
            issue_rows.append({
                "Patient ID": patient_id,
                "File": "Patient folder / files",
                "Issue": "No dose file detected",
                "Detected": ", ".join(sorted(roles)),
                "Suggestion": "If dose analysis is planned, include a dose file labelled Dose or RTDOSE.",
                "Rename example": expected_filename_example(patient_id, "Dose"),
                "Severity": "Check",
            })

        if "RTSTRUCT" not in roles and "Mask" not in roles:
            issue_rows.append({
                "Patient ID": patient_id,
                "File": "Patient folder / files",
                "Issue": "No RTSTRUCT or mask detected",
                "Detected": ", ".join(sorted(roles)),
                "Suggestion": "Include RTSTRUCT or mask files if contours are needed.",
                "Rename example": expected_filename_example(patient_id, "RTSTRUCT"),
                "Severity": "Check",
            })

    summary_df = pd.DataFrame(summary_rows)

    # Excel/image ID match check.
    image_ids = sorted([
        str(x).strip()
        for x in df["Inferred patient ID"].dropna().astype(str).tolist()
        if str(x).strip() != ""
    ])
    image_id_set = set(image_ids)

    excel_match_rows = []

    if len(patient_id_set) == 0:
        excel_match_rows.append({
            "Check": "Excel patient ID match",
            "Status": "Skipped",
            "Count": "",
            "Details": "No Excel patient data file/setup was loaded, so image patient IDs were not compared with Excel patient IDs.",
        })
    else:
        matched_ids = sorted(list(image_id_set & patient_id_set))
        image_not_in_excel = sorted(list(image_id_set - patient_id_set))
        excel_missing_images = sorted(list(patient_id_set - image_id_set))

        excel_match_rows.extend([
            {
                "Check": "Image IDs matching Excel",
                "Status": "Matched",
                "Count": len(matched_ids),
                "Details": ", ".join(matched_ids[:20]) + (f" ... (+{len(matched_ids)-20} more)" if len(matched_ids) > 20 else ""),
            },
            {
                "Check": "Image IDs not found in Excel",
                "Status": "Review" if len(image_not_in_excel) > 0 else "OK",
                "Count": len(image_not_in_excel),
                "Details": ", ".join(image_not_in_excel[:20]) + (f" ... (+{len(image_not_in_excel)-20} more)" if len(image_not_in_excel) > 20 else ""),
            },
            {
                "Check": "Excel IDs without image files",
                "Status": "Review" if len(excel_missing_images) > 0 else "OK",
                "Count": len(excel_missing_images),
                "Details": ", ".join(excel_missing_images[:20]) + (f" ... (+{len(excel_missing_images)-20} more)" if len(excel_missing_images) > 20 else ""),
            },
        ])

        for missing_id in image_not_in_excel:
            issue_rows.append({
                "Patient ID": missing_id,
                "File": "Image directory",
                "Issue": "Image patient ID does not match any loaded Excel patient ID",
                "Detected": missing_id,
                "Suggestion": "Check image folder/file name or check the Excel Patient ID column.",
                "Rename example": "",
                "Severity": "Important",
            })

        for excel_id in excel_missing_images:
            issue_rows.append({
                "Patient ID": excel_id,
                "File": "Excel patient data",
                "Issue": "Excel patient ID has no matching image files",
                "Detected": excel_id,
                "Suggestion": "Check whether this patient's image folder/files are missing or named differently.",
                "Rename example": "",
                "Severity": "Check",
            })

    issue_df = pd.DataFrame(issue_rows)
    excel_match_df = pd.DataFrame(excel_match_rows)

    if not issue_df.empty:
        preferred_cols = ["Patient ID", "File", "Issue", "Severity", "Detected", "Suggestion", "Rename example"]
        issue_df = issue_df[[col for col in preferred_cols if col in issue_df.columns]]

    return summary_df, issue_df, excel_match_df



# ============================================================
# SIMPLE VIEWER PREPARATION HELPERS
# ============================================================

def viewer_downsample_volume(volume, max_inplane=384, max_slices=160):
    """
    Downsample for visualisation only.
    """
    vol = np.asarray(volume)

    if vol.ndim != 3:
        return vol

    sx = max(1, int(np.ceil(vol.shape[0] / max_inplane)))
    sy = max(1, int(np.ceil(vol.shape[1] / max_inplane)))
    sz = max(1, int(np.ceil(vol.shape[2] / max_slices)))

    return vol[::sx, ::sy, ::sz]


def viewer_normalise_slice(slice_2d):
    arr = np.asarray(slice_2d, dtype=float)
    arr = np.nan_to_num(arr)

    if arr.size == 0:
        return arr

    p1, p99 = np.percentile(arr, [1, 99])

    if p99 <= p1:
        p1, p99 = np.min(arr), np.max(arr)

    if p99 <= p1:
        return np.zeros_like(arr)

    arr = np.clip(arr, p1, p99)
    arr = (arr - p1) / (p99 - p1)

    return arr



def viewer_reorient_nifti_to_patient_axial(img):
    """
    Reorient NIfTI image into a consistent patient-anatomical display space.

    Target display convention:
    - Array axes become approximately: Left-Right, Posterior-Anterior, Inferior-Superior
    - Axial viewer scrolls along Inferior-Superior axis
    - This uses the NIfTI affine orientation, not raw voxel order.

    Returns:
    - reoriented nibabel image
    - orientation info dictionary
    """
    if not NIBABEL_AVAILABLE:
        raise ImportError("nibabel is not installed. Install with: py -m pip install nibabel")

    # Current voxel-axis orientation from affine, e.g. ('L','P','S') etc.
    current_axcodes = nib.aff2axcodes(img.affine)

    try:
        qform_code = int(img.header["qform_code"])
        sform_code = int(img.header["sform_code"])
    except Exception:
        qform_code = None
        sform_code = None

    # Target RAS orientation is stable and commonly used in nibabel:
    # axis 0 = Right/Left patient direction
    # axis 1 = Anterior/Posterior patient direction
    # axis 2 = Superior/Inferior patient direction
    target_axcodes = ("R", "A", "S")

    try:
        current_ornt = nib.orientations.io_orientation(img.affine)
        target_ornt = nib.orientations.axcodes2ornt(target_axcodes)
        transform = nib.orientations.ornt_transform(current_ornt, target_ornt)
        reoriented_img = img.as_reoriented(transform)

        info = {
            "original_axcodes": current_axcodes,
            "display_axcodes": nib.aff2axcodes(reoriented_img.affine),
            "target_axcodes": target_axcodes,
            "qform_code": qform_code,
            "sform_code": sform_code,
        }

        return reoriented_img, info

    except Exception:
        # Safe fallback: use as_closest_canonical
        reoriented_img = nib.as_closest_canonical(img)
        info = {
            "original_axcodes": current_axcodes,
            "display_axcodes": nib.aff2axcodes(reoriented_img.affine),
            "target_axcodes": target_axcodes,
            "qform_code": qform_code,
            "sform_code": sform_code,
        }
        return reoriented_img, info



def viewer_load_nifti(path, return_info=False):
    """
    Load a NIfTI image for display using the affine coordinate system.

    The image is reoriented to RAS anatomical space so axial scrolling is along
    the superior-inferior direction rather than arbitrary stored voxel order.
    """
    if not NIBABEL_AVAILABLE:
        raise ImportError("nibabel is not installed. Install with: py -m pip install nibabel")

    img = nib.load(str(path))
    img, orientation_info = viewer_reorient_nifti_to_patient_axial(img)

    data = img.get_fdata()

    if data.ndim == 4:
        data = data[..., 0]

    try:
        orientation_info["voxel_spacing"] = get_nifti_voxel_spacing(img)
        orientation_info["shape"] = tuple(int(x) for x in data.shape[:3])
    except Exception:
        orientation_info["voxel_spacing"] = (1.0, 1.0, 1.0)

    data = viewer_downsample_volume(np.asarray(data))

    if return_info:
        return data, orientation_info

    return data




def get_nifti_voxel_spacing(img):
    """
    Return voxel spacing from a NIfTI image header/affine.

    Why this matters:
    CTs can look skewed or stretched if the viewer treats anisotropic voxels
    as square pixels. We use this spacing to set the matplotlib display aspect.
    """
    try:
        zooms = img.header.get_zooms()[:3]
        return tuple(float(z) for z in zooms)
    except Exception:
        try:
            # Fallback from affine column norms.
            return tuple(float(np.linalg.norm(img.affine[:3, i])) for i in range(3))
        except Exception:
            return (1.0, 1.0, 1.0)


def load_nifti_image_object(path):
    """
    Load a NIfTI image without forced flipping/rotation.

    The image is reoriented using the affine to RAS canonical anatomical space.
    Voxel spacing is preserved in the header/affine and later used by the viewer
    to avoid skewed CT/MR display.
    """
    if not NIBABEL_AVAILABLE:
        raise ImportError("nibabel is not installed. Install with: py -m pip install nibabel")

    img = nib.load(str(path))
    img, orientation_info = viewer_reorient_nifti_to_patient_axial(img)

    data = img.get_fdata()
    if data.ndim == 4:
        data = data[..., 0]
        img = nib.Nifti1Image(data, img.affine, img.header)

    spacing = get_nifti_voxel_spacing(img)

    orientation_info["voxel_spacing"] = spacing
    orientation_info["shape"] = tuple(int(x) for x in data.shape[:3])
    orientation_info["affine"] = img.affine.tolist()

    return img, np.asarray(data), orientation_info


def normalise_for_viewer(arr):
    """
    Robustly normalise a 2D image slice to 0-1 for display.
    """
    arr = np.asarray(arr, dtype=float)

    if arr.size == 0:
        return arr

    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=float)

    vals = arr[finite]
    lo, hi = np.percentile(vals, [1, 99])

    if hi <= lo:
        lo, hi = np.min(vals), np.max(vals)

    if hi <= lo:
        return np.zeros_like(arr, dtype=float)

    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0, 1)


def find_registered_patient_outputs(patient_id):
    """
    Find final registered outputs for one patient.

    The registration step writes files into:
    <registration_output_dir>/<patient_id>/

    This function looks for:
    - final_target_CT/MR...
    - registered_MR_to_CT...
    - registered_CT_to_MR...
    - aligned_Dose...
    - aligned_Mask / aligned_RTSTRUCT...
    """
    output_dir = st.session_state.get("voxel_registration_output_directory", "")
    if not output_dir:
        return {}

    patient_dir = Path(output_dir).expanduser() / safe_registration_filename(patient_id)

    if not patient_dir.exists():
        return {}

    files = list(patient_dir.glob("*.nii")) + list(patient_dir.glob("*.nii.gz"))

    found = {
        "patient_dir": str(patient_dir),
        "ct": [],
        "mr": [],
        "dose": [],
        "masks": [],
        "other": [],
    }

    for path in files:
        name = path.name.lower()

        if "final_target_ct" in name or "registered_ct" in name or "_ct_" in name:
            found["ct"].append(path)
        elif "final_target_mr" in name or "registered_mr" in name or "_mr_" in name or "mri" in name:
            found["mr"].append(path)
        elif "dose" in name:
            found["dose"].append(path)
        elif "mask" in name or "rtstruct" in name or "structure" in name or "seg" in name:
            found["masks"].append(path)
        else:
            found["other"].append(path)

    return found


def render_registered_ct_mr_blend_viewer(patient_id):
    """
    Registered CT/MR QA viewer.

    Required behaviour:
    - Load final registered CT and MR outputs.
    - Show them on top of each other.
    - Slider left = CT, right = MR.
    - Later allow masks and dose overlays.
    """
    outputs = find_registered_patient_outputs(patient_id)

    if not outputs:
        st.info("No registered output folder found for this patient yet.")
        return False

    st.caption(f"Registered patient folder: {outputs.get('patient_dir', '')}")

    ct_options = [str(p) for p in outputs.get("ct", [])]
    mr_options = [str(p) for p in outputs.get("mr", [])]

    if len(ct_options) == 0 and len(mr_options) == 0:
        st.warning("No registered CT or MR NIfTI outputs were found for this patient.")
        return False

    col_ct, col_mr = st.columns(2)

    with col_ct:
        selected_ct = st.selectbox(
            "Registered CT",
            options=["None"] + ct_options,
            key=f"registered_ct_select_{patient_id}",
        )

    with col_mr:
        selected_mr = st.selectbox(
            "Registered MR",
            options=["None"] + mr_options,
            key=f"registered_mr_select_{patient_id}",
        )

    ct_data = None
    mr_data = None
    ct_info = {}
    mr_info = {}

    try:
        if selected_ct != "None":
            ct_img, ct_data, ct_info = load_nifti_image_object(selected_ct)

        if selected_mr != "None":
            mr_img, mr_data, mr_info = load_nifti_image_object(selected_mr)

    except Exception as error:
        st.error(f"Could not load registered CT/MR output: {error}")
        return True

    if ct_data is None and mr_data is None:
        st.info("Select a registered CT or MR file.")
        return True

    reference_data = ct_data if ct_data is not None else mr_data
    reference_info = ct_info if ct_data is not None else mr_info

    if ct_data is not None and mr_data is not None and ct_data.shape != mr_data.shape:
        st.warning(
            f"Registered CT/MR shapes do not match: CT {ct_data.shape}, MR {mr_data.shape}. "
            "This suggests the registration/resampling output is not in the same voxel grid."
        )

    zmax = reference_data.shape[2] - 1
    default_slice = zmax // 2

    slice_index = st.slider(
        "Axial slice",
        min_value=0,
        max_value=zmax,
        value=default_slice,
        step=1,
        key=f"registered_ct_mr_slice_{patient_id}",
    )

    blend = st.slider(
        "CT ↔ MR blend",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        key=f"registered_ct_mr_blend_{patient_id}",
        help="Move left for CT. Move right for MR. Middle shows both blended.",
    )

    st.caption("Blend: 0 = CT only, 1 = MR only.")

    # Keep axial display simple and spacing-aware:
    # after RAS canonical reorientation, slice is x/y plane, z is axial direction.
    # Transpose for row/column display, but do NOT flip/rotate.
    if ct_data is not None:
        ct_slice = normalise_for_viewer(ct_data[:, :, slice_index].T)
    else:
        ct_slice = None

    if mr_data is not None:
        mr_slice = normalise_for_viewer(mr_data[:, :, slice_index].T)
    else:
        mr_slice = None

    if ct_slice is not None and mr_slice is not None and ct_slice.shape == mr_slice.shape:
        display_slice = ((1.0 - blend) * ct_slice) + (blend * mr_slice)
    elif ct_slice is not None:
        display_slice = ct_slice
        if selected_mr != "None":
            st.warning("MR is selected but cannot be blended because its grid/shape differs from CT.")
    else:
        display_slice = mr_slice

    spacing = reference_info.get("voxel_spacing", (1.0, 1.0, 1.0))
    try:
        # Display aspect for axial view: row dimension is y spacing, column dimension is x spacing.
        aspect = float(spacing[1]) / float(spacing[0]) if float(spacing[0]) != 0 else 1.0
    except Exception:
        aspect = 1.0

    fig, ax = plt.subplots(figsize=(4.0, 4.0), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(display_slice, cmap="gray", aspect=aspect)
    ax.axis("off")
    ax.set_title(f"Registered CT↔MR QA · slice {slice_index}", color="white", fontsize=9)

    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    st.markdown("### Registered image geometry")
    geom_rows = []

    if ct_data is not None:
        geom_rows.append({
            "Image": "CT",
            "Shape": str(tuple(ct_data.shape)),
            "Voxel spacing": str(ct_info.get("voxel_spacing", "")),
            "Orientation": f"{ct_info.get('original_axcodes')} → {ct_info.get('display_axcodes')}",
        })

    if mr_data is not None:
        geom_rows.append({
            "Image": "MR",
            "Shape": str(tuple(mr_data.shape)),
            "Voxel spacing": str(mr_info.get("voxel_spacing", "")),
            "Orientation": f"{mr_info.get('original_axcodes')} → {mr_info.get('display_axcodes')}",
        })

    st.dataframe(pd.DataFrame(geom_rows), use_container_width=True)

    # Optional overlays after CT/MR check.
    with st.expander("Load registered masks / dose overlays", expanded=False):
        mask_options = [str(p) for p in outputs.get("masks", [])]
        dose_options = [str(p) for p in outputs.get("dose", [])]

        selected_masks = st.multiselect(
            "Registered masks",
            options=mask_options,
            key=f"registered_masks_select_{patient_id}",
        )

        selected_dose = st.selectbox(
            "Registered dose",
            options=["None"] + dose_options,
            key=f"registered_dose_select_{patient_id}",
        )

        st.caption(
            "Mask/dose overlay drawing will use these registered outputs. "
            "First check CT↔MR alignment with the blend slider."
        )

    return True



def viewer_load_dicom_series(paths):
    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom is not installed. Install with: py -m pip install pydicom")

    slices = []

    for path in paths:
        try:
            ds = pydicom.dcmread(str(path), force=True)

            if not hasattr(ds, "pixel_array"):
                continue

            try:
                instance = int(getattr(ds, "InstanceNumber", len(slices)))
            except Exception:
                instance = len(slices)

            z_position = None
            try:
                z_position = float(ds.ImagePositionPatient[2])
            except Exception:
                z_position = None

            arr = ds.pixel_array.astype(float)
            slope = float(getattr(ds, "RescaleSlope", 1))
            intercept = float(getattr(ds, "RescaleIntercept", 0))
            arr = arr * slope + intercept

            slices.append({
                "instance": instance,
                "z": z_position,
                "array": arr,
                "path": str(path),
            })
        except Exception:
            continue

    if len(slices) == 0:
        raise ValueError("No readable DICOM image slices found.")

    if all(item["z"] is not None for item in slices):
        slices = sorted(slices, key=lambda item: item["z"])
    else:
        slices = sorted(slices, key=lambda item: item["instance"])

    volume = np.stack([item["array"] for item in slices], axis=-1)

    return viewer_downsample_volume(volume)


def viewer_patient_files(file_df, patient_id):
    """
    Get files assigned to selected patient.
    """
    df = add_patient_grouping_columns(file_df.copy())
    return df[df["Inferred patient ID"].astype(str) == str(patient_id)].copy()


def viewer_choose_main_image(patient_df):
    """
    Select main image: CT first, then MRI, then PET, then unknown.
    """
    for role in ["CT", "MRI", "PET", "Main / unknown"]:
        candidates = patient_df[patient_df["File role"] == role].copy()
        if not candidates.empty:
            return candidates.iloc[0].to_dict()
    return None


def viewer_load_main_image(row, patient_df):
    """
    Load selected main image for viewer.
    """
    fmt = row.get("File format", "")
    role = row.get("File role", "")
    full_path = row.get("Full path", "")

    if fmt == "NIfTI":
        return viewer_load_nifti(full_path)

    if fmt == "DICOM":
        series_df = patient_df[
            (patient_df["File format"] == "DICOM")
            & (patient_df["File role"] == role)
        ].copy()

        paths = series_df["Full path"].dropna().astype(str).tolist()

        if len(paths) == 0:
            paths = [full_path]

        return viewer_load_dicom_series(paths)

    raise ValueError("Unknown file format.")


def viewer_find_patient_related_files(patient_df):
    """
    Return structures/masks and dose files for selected patient.
    """
    if patient_df is None or patient_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    structures = patient_df[
        patient_df["File role"].isin(["Mask", "RTSTRUCT / structure"])
        | patient_df["Filename"].astype(str).str.lower().str.contains("mask|seg|structure|rtstruct|roi|contour", regex=True, na=False)
        | patient_df["Relative path"].astype(str).str.lower().str.contains("mask|seg|structure|rtstruct|roi|contour", regex=True, na=False)
    ].copy()

    dose = patient_df[
        (patient_df["File role"] == "Dose")
        | patient_df["Filename"].astype(str).str.lower().str.contains("dose|rtdose", regex=True, na=False)
        | patient_df["Relative path"].astype(str).str.lower().str.contains("dose|rtdose", regex=True, na=False)
    ].copy()

    return structures, dose


def viewer_try_load_nifti_masks(structures_df, image_shape):
    """
    Try to load compatible NIfTI masks for overlay.
    """
    masks = []
    contour_rows = []

    if structures_df is None or structures_df.empty:
        return masks, pd.DataFrame()

    for _, row in structures_df.iterrows():
        filename = str(row.get("Filename", ""))
        relative_path = str(row.get("Relative path", filename))
        fmt = row.get("File format", "")
        full_path = row.get("Full path", "")
        lower = filename.lower()

        is_nifti = fmt == "NIfTI" or lower.endswith(".nii") or lower.endswith(".nii.gz")
        status = "Listed only"

        if is_nifti:
            try:
                mask_vol = viewer_load_nifti(full_path)

                if np.asarray(mask_vol).shape == tuple(image_shape):
                    masks.append({
                        "name": Path(filename.replace(".nii.gz", "")).stem,
                        "file": relative_path,
                        "volume": mask_vol,
                    })
                    status = "Overlaid"
                else:
                    status = f"Shape mismatch: {np.asarray(mask_vol).shape} vs image {image_shape}"
            except Exception as error:
                status = f"Could not load: {error}"
        else:
            status = "RTSTRUCT/DICOM listed; overlay later after conversion"

        contour_rows.append({
            "Structure / mask": Path(filename.replace(".nii.gz", "")).stem,
            "File": relative_path,
            "Format": fmt,
            "Viewer status": status,
        })

    return masks, pd.DataFrame(contour_rows)



def viewer_get_role_safe(row):
    """
    Determine role using existing File role plus filename/path.
    """
    file_role = str(row.get("File role", ""))
    viewer_role = expected_image_role_from_path_and_name(
        row.get("Relative path", row.get("Filename", "")),
        row.get("Filename", "")
    )

    if viewer_role != "Unknown":
        return viewer_role

    if file_role == "RTSTRUCT / structure":
        return "RTSTRUCT"

    return file_role


def viewer_choose_main_image_auto(patient_df):
    """
    Automatically choose the main image for the selected patient.
    Priority: CT > MRI > PET > Main / unknown readable image.
    """
    if patient_df is None or patient_df.empty:
        return None

    df = patient_df.copy()
    df["Auto role"] = df.apply(viewer_get_role_safe, axis=1)

    for role in ["CT", "MRI", "PET"]:
        role_df = df[df["Auto role"] == role].copy()
        if not role_df.empty:
            return role_df.iloc[0].to_dict()

    # Fallback: readable DICOM/NIfTI that is not dose/RTSTRUCT/mask.
    fallback = df[
        df["File format"].isin(["DICOM", "NIfTI"])
        & (~df["Auto role"].isin(["Dose", "RTSTRUCT", "Mask", "RTSTRUCT / structure"]))
    ].copy()

    if not fallback.empty:
        return fallback.iloc[0].to_dict()

    return None


def viewer_find_all_overlays_auto(patient_df):
    """
    Find all mask/structure/dose files for a selected patient.
    """
    if patient_df is None or patient_df.empty:
        return pd.DataFrame()

    df = patient_df.copy()
    df["Auto role"] = df.apply(viewer_get_role_safe, axis=1)

    # Avoid problematic regex/backslash patterns. Use simple contains checks.
    name_lower = df["Filename"].astype(str).str.lower()
    path_lower = df["Relative path"].astype(str).str.lower()

    is_mask = (
        (df["Auto role"] == "Mask")
        | name_lower.str.contains("mask", regex=False, na=False)
        | name_lower.str.contains("seg", regex=False, na=False)
        | name_lower.str.contains("roi", regex=False, na=False)
        | name_lower.str.contains("contour", regex=False, na=False)
        | path_lower.str.contains("mask", regex=False, na=False)
        | path_lower.str.contains("seg", regex=False, na=False)
        | path_lower.str.contains("roi", regex=False, na=False)
        | path_lower.str.contains("contour", regex=False, na=False)
    )

    is_dose = (
        (df["Auto role"] == "Dose")
        | name_lower.str.contains("dose", regex=False, na=False)
        | name_lower.str.contains("rtdose", regex=False, na=False)
        | path_lower.str.contains("dose", regex=False, na=False)
        | path_lower.str.contains("rtdose", regex=False, na=False)
    )

    is_rtstruct = (
        (df["Auto role"] == "RTSTRUCT")
        | (df["File role"] == "RTSTRUCT / structure")
        | name_lower.str.contains("rtstruct", regex=False, na=False)
        | name_lower.str.contains("rt.struct", regex=False, na=False)
        | name_lower.str.contains("rt_struct", regex=False, na=False)
        | name_lower.str.contains("rt-struct", regex=False, na=False)
        | name_lower.str.contains("structure", regex=False, na=False)
        | name_lower.str.startswith("rs.")
        | path_lower.str.contains("rtstruct", regex=False, na=False)
        | path_lower.str.contains("structure", regex=False, na=False)
    )

    overlays = df[is_mask | is_dose | is_rtstruct].copy()

    if overlays.empty:
        return overlays

    overlays = overlays.drop_duplicates(subset=["Relative path", "Filename"])

    return overlays


def viewer_load_auto_overlays(overlays_df, base_shape):
    """
    Try loading compatible overlays.
    NIfTI masks overlay directly.
    NIfTI/DICOM dose attempts to overlay when readable and same shape.
    RTSTRUCT DICOM is listed only for now.
    """
    overlay_items = []
    status_rows = []

    if overlays_df is None or overlays_df.empty:
        return overlay_items, pd.DataFrame()

    for _, row in overlays_df.iterrows():
        row_dict = row.to_dict()
        filename = str(row_dict.get("Filename", ""))
        rel_path = row_dict.get("Relative path", filename)
        fmt = row_dict.get("File format", "")
        auto_role = row_dict.get("Auto role", viewer_get_role_safe(row_dict))
        lower_name = filename.lower()

        status = "Listed only"
        details = ""

        is_nifti = fmt == "NIfTI" or lower_name.endswith(".nii") or lower_name.endswith(".nii.gz")

        # NIfTI masks can be directly overlaid.
        if auto_role == "Mask" or any(word in lower_name for word in ["mask", "seg", "roi", "contour"]):
            if is_nifti:
                try:
                    vol = viewer_load_nifti(row_dict.get("Full path", ""))

                    if np.asarray(vol).shape == tuple(base_shape):
                        overlay_items.append({
                            "type": "Mask",
                            "name": Path(filename.replace(".nii.gz", "")).stem,
                            "file": rel_path,
                            "volume": vol,
                        })
                        status = "Overlaid"
                        details = f"Shape matches image: {np.asarray(vol).shape}"
                    else:
                        status = "Not overlaid"
                        details = f"Shape mismatch: {np.asarray(vol).shape} vs image {tuple(base_shape)}"
                except Exception as error:
                    status = "Not overlaid"
                    details = f"Could not load NIfTI mask: {error}"
            else:
                status = "Listed only"
                details = "Mask/structure is not NIfTI. Conversion needed before overlay."

        elif auto_role == "Dose" or "dose" in lower_name:
            # Try NIfTI dose first; DICOM dose viewing is not robust yet.
            if is_nifti:
                try:
                    vol = viewer_load_nifti(row_dict.get("Full path", ""))

                    if np.asarray(vol).shape == tuple(base_shape):
                        overlay_items.append({
                            "type": "Dose",
                            "name": Path(filename.replace(".nii.gz", "")).stem,
                            "file": rel_path,
                            "volume": vol,
                        })
                        status = "Overlaid"
                        details = f"Shape matches image: {np.asarray(vol).shape}"
                    else:
                        status = "Not overlaid"
                        details = f"Shape mismatch: {np.asarray(vol).shape} vs image {tuple(base_shape)}"
                except Exception as error:
                    status = "Not overlaid"
                    details = f"Could not load dose: {error}"
            else:
                status = "Listed only"
                details = "DICOM dose overlay will be added later."

        elif auto_role == "RTSTRUCT" or "rtstruct" in lower_name or "structure" in lower_name:
            status = "Listed only"
            details = "RTSTRUCT conversion to mask overlay will be added later."

        status_rows.append({
            "File": rel_path,
            "Detected type": auto_role,
            "Viewer status": status,
            "Details": details,
        })

    return overlay_items, pd.DataFrame(status_rows)




def viewer_get_overlay_candidates_only(patient_df):
    """
    Return only candidate overlays: RTSTRUCT, dose, mask/seg/ROI/contour.
    Do not include CT/MR/PET image slices.
    """
    if patient_df is None or patient_df.empty:
        return pd.DataFrame()

    df = patient_df.copy()

    if "Auto role" not in df.columns:
        df["Auto role"] = df.apply(viewer_get_role_safe, axis=1)

    name_lower = df["Filename"].astype(str).str.lower()
    path_lower = df["Relative path"].astype(str).str.lower()

    is_mask = (
        (df["Auto role"] == "Mask")
        | name_lower.str.contains("mask", regex=False, na=False)
        | name_lower.str.contains("seg", regex=False, na=False)
        | name_lower.str.contains("roi", regex=False, na=False)
        | name_lower.str.contains("contour", regex=False, na=False)
        | path_lower.str.contains("mask", regex=False, na=False)
        | path_lower.str.contains("seg", regex=False, na=False)
        | path_lower.str.contains("roi", regex=False, na=False)
        | path_lower.str.contains("contour", regex=False, na=False)
    )

    is_dose = (
        (df["Auto role"] == "Dose")
        | name_lower.str.contains("dose", regex=False, na=False)
        | name_lower.str.contains("rtdose", regex=False, na=False)
        | path_lower.str.contains("dose", regex=False, na=False)
        | path_lower.str.contains("rtdose", regex=False, na=False)
    )

    is_rtstruct = (
        (df["Auto role"] == "RTSTRUCT")
        | (df["File role"] == "RTSTRUCT / structure")
        | name_lower.str.contains("rtstruct", regex=False, na=False)
        | name_lower.str.contains("rt.struct", regex=False, na=False)
        | name_lower.str.contains("rt_struct", regex=False, na=False)
        | name_lower.str.contains("rt-struct", regex=False, na=False)
        | name_lower.str.contains("structure", regex=False, na=False)
        | name_lower.str.startswith("rs.")
        | path_lower.str.contains("rtstruct", regex=False, na=False)
        | path_lower.str.contains("structure", regex=False, na=False)
    )

    candidates = df[is_mask | is_dose | is_rtstruct].copy()

    if candidates.empty:
        return candidates

    def overlay_type(row):
        role = str(row.get("Auto role", ""))
        filename = str(row.get("Filename", "")).lower()
        rel = str(row.get("Relative path", "")).lower()

        if role == "Dose" or "dose" in filename or "rtdose" in filename or "dose" in rel:
            return "Dose"
        if role == "RTSTRUCT" or "rtstruct" in filename or "rt.struct" in filename or "structure" in filename or filename.startswith("rs."):
            return "RTSTRUCT / structure"
        return "Mask"

    candidates["Overlay type"] = candidates.apply(overlay_type, axis=1)
    candidates["Overlay label"] = candidates.apply(
        lambda row: f"{row['Overlay type']} | {row.get('Relative path', row.get('Filename', ''))}",
        axis=1
    )

    return candidates.drop_duplicates(subset=["Relative path", "Filename"])


def viewer_load_clicked_overlays(candidates_df, selected_labels, base_shape):
    """
    Load only overlays selected by the user.
    """
    overlay_items = []
    status_rows = []

    if candidates_df is None or candidates_df.empty or not selected_labels:
        return overlay_items, pd.DataFrame()

    selected_df = candidates_df[candidates_df["Overlay label"].isin(selected_labels)].copy()

    for _, row in selected_df.iterrows():
        row_dict = row.to_dict()
        overlay_type = row_dict.get("Overlay type", "")
        filename = str(row_dict.get("Filename", ""))
        rel_path = row_dict.get("Relative path", filename)
        fmt = row_dict.get("File format", "")
        lower_name = filename.lower()

        status = "Listed only"
        details = ""

        is_nifti = fmt == "NIfTI" or lower_name.endswith(".nii") or lower_name.endswith(".nii.gz")

        if overlay_type == "Mask":
            if is_nifti:
                try:
                    vol = viewer_load_nifti(row_dict.get("Full path", ""))

                    if np.asarray(vol).shape == tuple(base_shape):
                        overlay_items.append({
                            "type": "Mask",
                            "name": Path(filename.replace(".nii.gz", "")).stem,
                            "file": rel_path,
                            "volume": vol,
                        })
                        status = "Overlaid"
                        details = f"Shape matches CT: {np.asarray(vol).shape}"
                    else:
                        status = "Not overlaid"
                        details = f"Shape mismatch: {np.asarray(vol).shape} vs CT {tuple(base_shape)}"
                except Exception as error:
                    status = "Not overlaid"
                    details = f"Could not load mask: {error}"
            else:
                status = "Listed only"
                details = "Only NIfTI masks overlay directly for now."

        elif overlay_type == "Dose":
            if is_nifti:
                try:
                    vol = viewer_load_nifti(row_dict.get("Full path", ""))

                    if np.asarray(vol).shape == tuple(base_shape):
                        overlay_items.append({
                            "type": "Dose",
                            "name": Path(filename.replace(".nii.gz", "")).stem,
                            "file": rel_path,
                            "volume": vol,
                        })
                        status = "Overlaid"
                        details = f"Shape matches CT: {np.asarray(vol).shape}"
                    else:
                        status = "Not overlaid"
                        details = f"Shape mismatch: {np.asarray(vol).shape} vs CT {tuple(base_shape)}"
                except Exception as error:
                    status = "Not overlaid"
                    details = f"Could not load dose: {error}"
            else:
                status = "Listed only"
                details = "DICOM dose overlay will be added later."

        elif overlay_type == "RTSTRUCT / structure":
            status = "Listed only"
            details = "RTSTRUCT is detected. Contour conversion to mask overlay will be added later."

        status_rows.append({
            "Overlay type": overlay_type,
            "File": rel_path,
            "Viewer status": status,
            "Details": details,
        })

    return overlay_items, pd.DataFrame(status_rows)




def viewer_find_contour_candidates(patient_df):
    """
    Find contour/structure/mask files for the selected patient.
    Includes:
    - RTSTRUCT / RS DICOM files
    - NIfTI masks
    - segmentation / ROI / contour files
    """
    if patient_df is None or patient_df.empty:
        return pd.DataFrame()

    df = patient_df.copy()

    name_lower = df["Filename"].astype(str).str.lower()
    path_lower = df["Relative path"].astype(str).str.lower()

    is_rtstruct = (
        (df["File role"] == "RTSTRUCT / structure")
        | name_lower.str.contains("rtstruct", regex=False, na=False)
        | name_lower.str.contains("rt.struct", regex=False, na=False)
        | name_lower.str.contains("rt_struct", regex=False, na=False)
        | name_lower.str.contains("rt-struct", regex=False, na=False)
        | name_lower.str.contains("structure", regex=False, na=False)
        | name_lower.str.startswith("rs.")
        | name_lower.str.startswith("rs_")
        | name_lower.str.startswith("rs-")
        | path_lower.str.contains("rtstruct", regex=False, na=False)
        | path_lower.str.contains("structure", regex=False, na=False)
    )

    is_mask = (
        (df["File role"] == "Mask")
        | name_lower.str.contains("mask", regex=False, na=False)
        | name_lower.str.contains("seg", regex=False, na=False)
        | name_lower.str.contains("roi", regex=False, na=False)
        | name_lower.str.contains("contour", regex=False, na=False)
        | path_lower.str.contains("mask", regex=False, na=False)
        | path_lower.str.contains("seg", regex=False, na=False)
        | path_lower.str.contains("roi", regex=False, na=False)
        | path_lower.str.contains("contour", regex=False, na=False)
    )

    candidates = df[is_rtstruct | is_mask].copy()

    if candidates.empty:
        return candidates

    def contour_type(row):
        filename = str(row.get("Filename", "")).lower()
        file_role = str(row.get("File role", ""))

        if file_role == "RTSTRUCT / structure" or "rtstruct" in filename or "rt.struct" in filename or filename.startswith("rs."):
            return "RTSTRUCT"
        return "Mask"

    candidates["Contour type"] = candidates.apply(contour_type, axis=1)
    candidates["Display"] = candidates.apply(
        lambda row: f"{row['Contour type']} | {row.get('Relative path', row.get('Filename', ''))}",
        axis=1
    )

    return candidates.drop_duplicates(subset=["Relative path", "Filename"])


def viewer_extract_rtstruct_names(rtstruct_row):
    """
    Extract ROI names from an RTSTRUCT DICOM file if pydicom can read it.
    Returns a list of names.
    """
    if not PYDICOM_AVAILABLE:
        return []

    try:
        ds = pydicom.dcmread(str(rtstruct_row.get("Full path", "")), force=True)

        names = []

        if hasattr(ds, "StructureSetROISequence"):
            for roi in ds.StructureSetROISequence:
                name = str(getattr(roi, "ROIName", "")).strip()
                if name:
                    names.append(name)

        return names

    except Exception:
        return []


def viewer_make_contour_table(candidates_df):
    """
    Build a table showing RTSTRUCT/mask files and RTSTRUCT ROI names where possible.
    """
    rows = []

    if candidates_df is None or candidates_df.empty:
        return pd.DataFrame()

    for _, row in candidates_df.iterrows():
        row_dict = row.to_dict()
        contour_type = row_dict.get("Contour type", "")
        filename = row_dict.get("Filename", "")
        rel_path = row_dict.get("Relative path", filename)
        fmt = row_dict.get("File format", "")

        if contour_type == "RTSTRUCT":
            roi_names = viewer_extract_rtstruct_names(row_dict)
            if roi_names:
                detail = ", ".join(roi_names[:20]) + (f" ... (+{len(roi_names)-20} more)" if len(roi_names) > 20 else "")
                count = len(roi_names)
            else:
                detail = "RTSTRUCT file detected. ROI names could not be read or pydicom unavailable."
                count = ""
        else:
            detail = Path(str(filename).replace(".nii.gz", "")).stem
            count = 1

        rows.append({
            "Type": contour_type,
            "File": rel_path,
            "Format": fmt,
            "Structures / ROI names": detail,
            "Count": count,
        })

    return pd.DataFrame(rows)


def viewer_load_selected_contours_on_ct(candidates_df, selected_labels, ct_shape):
    """
    Try to load selected masks onto CT.

    Current behaviour:
    - NIfTI masks can overlay if shape matches CT.
    - RTSTRUCT DICOM files are detected/listed but not converted to voxel masks yet.
    """
    overlay_items = []
    status_rows = []

    if candidates_df is None or candidates_df.empty or not selected_labels:
        return overlay_items, pd.DataFrame()

    selected_df = candidates_df[candidates_df["Display"].isin(selected_labels)].copy()

    for _, row in selected_df.iterrows():
        row_dict = row.to_dict()
        contour_type = row_dict.get("Contour type", "")
        filename = str(row_dict.get("Filename", ""))
        rel_path = row_dict.get("Relative path", filename)
        fmt = row_dict.get("File format", "")
        lower = filename.lower()

        if contour_type == "Mask":
            is_nifti = fmt == "NIfTI" or lower.endswith(".nii") or lower.endswith(".nii.gz")

            if is_nifti:
                try:
                    mask_vol = viewer_load_nifti(row_dict.get("Full path", ""))

                    if np.asarray(mask_vol).shape == tuple(ct_shape):
                        overlay_items.append({
                            "name": Path(filename.replace(".nii.gz", "")).stem,
                            "file": rel_path,
                            "volume": mask_vol,
                        })
                        status = "Overlaid"
                        details = f"Mask shape matches CT: {np.asarray(mask_vol).shape}"
                    else:
                        status = "Not overlaid"
                        details = f"Shape mismatch: mask {np.asarray(mask_vol).shape} vs CT {tuple(ct_shape)}"

                except Exception as error:
                    status = "Not overlaid"
                    details = f"Could not load mask: {error}"
            else:
                status = "Listed only"
                details = "Only NIfTI masks can be overlaid directly for now."

        else:
            status = "Listed only"
            details = "RTSTRUCT detected. Conversion to voxel mask overlay will be added next."

        status_rows.append({
            "File": rel_path,
            "Type": contour_type,
            "Status": status,
            "Details": details,
        })

    return overlay_items, pd.DataFrame(status_rows)



# ============================================================
# SIMPLE RADIOTHERAPY CT + RTSTRUCT VIEWER HELPERS
# ============================================================

def rtviewer_is_rtstruct_file(file_path, filename=""):
    """
    Return True if a DICOM file appears to be an RTSTRUCT.
    """
    name = str(filename).lower()

    if (
        "rtstruct" in name
        or "rt.struct" in name
        or "rt_struct" in name
        or "rt-struct" in name
        or "structure" in name
        or name.startswith("rs.")
        or name.startswith("rs_")
        or name.startswith("rs-")
    ):
        return True

    if not PYDICOM_AVAILABLE:
        return False

    try:
        ds = pydicom.dcmread(str(file_path), stop_before_pixels=True, force=True)
        return str(getattr(ds, "Modality", "")).upper() == "RTSTRUCT"
    except Exception:
        return False


def rtviewer_find_rtstruct_file(patient_df):
    """
    Find the first RTSTRUCT DICOM file for the selected patient.
    """
    if patient_df is None or patient_df.empty:
        return None

    for _, row in patient_df.iterrows():
        filename = row.get("Filename", "")
        full_path = row.get("Full path", "")
        if rtviewer_is_rtstruct_file(full_path, filename):
            return row.to_dict()

    return None


def rtviewer_get_ct_files(patient_df):
    """
    Return likely CT DICOM files only.
    Excludes RTSTRUCT and dose.
    """
    if patient_df is None or patient_df.empty:
        return pd.DataFrame()

    pf = patient_df.copy()
    name_lower = pf["Filename"].astype(str).str.lower()
    path_lower = pf["Relative path"].astype(str).str.lower()

    not_ct = (
        name_lower.str.contains("dose", regex=False, na=False)
        | name_lower.str.contains("rtdose", regex=False, na=False)
        | name_lower.str.contains("rtstruct", regex=False, na=False)
        | name_lower.str.contains("rt.struct", regex=False, na=False)
        | name_lower.str.contains("rt_struct", regex=False, na=False)
        | name_lower.str.contains("structure", regex=False, na=False)
        | name_lower.str.startswith("rs.")
        | path_lower.str.contains("dose", regex=False, na=False)
        | path_lower.str.contains("rtstruct", regex=False, na=False)
        | path_lower.str.contains("structure", regex=False, na=False)
    )

    ct_like = (
        name_lower.str.contains("ct", regex=False, na=False)
        | path_lower.str.contains("ct", regex=False, na=False)
        | (pf["File role"].astype(str) == "CT")
    )

    ct_df = pf[(pf["File format"] == "DICOM") & ct_like & (~not_ct)].copy()

    if not ct_df.empty:
        return ct_df

    # Fallback: use readable DICOM image files, still excluding dose/RTSTRUCT.
    fallback = pf[(pf["File format"] == "DICOM") & (~not_ct)].copy()
    return fallback


def rtviewer_load_ct_series(ct_df):
    """
    Load CT DICOM series and metadata needed to project RTSTRUCT contours.
    Returns:
    - volume: rows x cols x slices
    - meta: orientation, spacing, slice positions and IPPs
    - used_files: dataframe of successfully read CT slices
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom is not installed. Install with: py -m pip install pydicom")

    if ct_df is None or ct_df.empty:
        raise ValueError("No CT DICOM files found for this patient.")

    slices = []
    used_rows = []

    for _, row in ct_df.iterrows():
        full_path = row.get("Full path", "")
        filename = row.get("Filename", "")

        try:
            ds = pydicom.dcmread(str(full_path), force=True)

            # Ignore non-image DICOM files.
            if not hasattr(ds, "pixel_array"):
                continue

            modality = str(getattr(ds, "Modality", "")).upper()
            if modality in ["RTSTRUCT", "RTDOSE", "RTPLAN"]:
                continue

            arr = ds.pixel_array.astype(float)

            slope = float(getattr(ds, "RescaleSlope", 1))
            intercept = float(getattr(ds, "RescaleIntercept", 0))
            arr = arr * slope + intercept

            orientation = np.array(getattr(ds, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]), dtype=float)
            row_cos = orientation[:3]
            col_cos = orientation[3:]
            normal = np.cross(row_cos, col_cos)

            ipp = np.array(getattr(ds, "ImagePositionPatient", [0, 0, len(slices)]), dtype=float)
            slice_position = float(np.dot(ipp, normal))

            try:
                instance_number = int(getattr(ds, "InstanceNumber", len(slices)))
            except Exception:
                instance_number = len(slices)

            pixel_spacing = np.array(getattr(ds, "PixelSpacing", [1, 1]), dtype=float)

            slices.append({
                "array": arr,
                "ds": ds,
                "path": str(full_path),
                "filename": filename,
                "instance": instance_number,
                "ipp": ipp,
                "slice_position": slice_position,
                "row_cos": row_cos,
                "col_cos": col_cos,
                "normal": normal,
                "pixel_spacing": pixel_spacing,
            })
            used_rows.append(row.to_dict())

        except Exception:
            continue

    if len(slices) == 0:
        raise ValueError("No readable CT image slices were found.")

    # Sort by patient-space slice location; fallback to instance number if needed.
    try:
        slices = sorted(slices, key=lambda item: item["slice_position"])
    except Exception:
        slices = sorted(slices, key=lambda item: item["instance"])

    volume = np.stack([item["array"] for item in slices], axis=-1)

    meta = {
        "slices": slices,
        "row_cos": slices[0]["row_cos"],
        "col_cos": slices[0]["col_cos"],
        "normal": slices[0]["normal"],
        "pixel_spacing": slices[0]["pixel_spacing"],
        "slice_positions": np.array([item["slice_position"] for item in slices], dtype=float),
        "ipps": [item["ipp"] for item in slices],
        "shape": volume.shape,
    }

    used_files = pd.DataFrame(used_rows)

    return volume, meta, used_files


def rtviewer_load_rtstruct_contours(rtstruct_path, ct_meta):
    """
    Load all RTSTRUCT contours and convert contour points into CT pixel coordinates.
    Returns dict: slice_index -> list of contour dictionaries.
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom is not installed. Install with: py -m pip install pydicom")

    ds = pydicom.dcmread(str(rtstruct_path), force=True)

    roi_names = {}
    if hasattr(ds, "StructureSetROISequence"):
        for roi in ds.StructureSetROISequence:
            number = int(getattr(roi, "ROINumber", -1))
            name = str(getattr(roi, "ROIName", f"ROI {number}"))
            roi_names[number] = name

    contours_by_slice = {}

    if not hasattr(ds, "ROIContourSequence"):
        return contours_by_slice, roi_names

    slice_positions = ct_meta["slice_positions"]

    for roi_contour in ds.ROIContourSequence:
        roi_number = int(getattr(roi_contour, "ReferencedROINumber", -1))
        roi_name = roi_names.get(roi_number, f"ROI {roi_number}")

        if not hasattr(roi_contour, "ContourSequence"):
            continue

        for contour in roi_contour.ContourSequence:
            if not hasattr(contour, "ContourData"):
                continue

            coords = np.array(contour.ContourData, dtype=float).reshape(-1, 3)

            if coords.shape[0] < 2:
                continue

            # Find nearest CT slice using patient coordinate along CT normal.
            contour_position = float(np.mean(np.dot(coords, ct_meta["normal"])))
            slice_index = int(np.argmin(np.abs(slice_positions - contour_position)))

            ipp = ct_meta["ipps"][slice_index]
            row_cos = ct_meta["row_cos"]
            col_cos = ct_meta["col_cos"]
            spacing = ct_meta["pixel_spacing"]

            delta = coords - ipp

            # DICOM mapping:
            # column coordinate follows row direction / PixelSpacing[1]
            # row coordinate follows column direction / PixelSpacing[0]
            x_cols = np.dot(delta, row_cos) / spacing[1]
            y_rows = np.dot(delta, col_cos) / spacing[0]

            contours_by_slice.setdefault(slice_index, []).append({
                "name": roi_name,
                "roi_number": roi_number,
                "x": x_cols,
                "y": y_rows,
            })

    return contours_by_slice, roi_names



def rtviewer_get_nifti_image_files(patient_df, image_type="Auto"):
    """
    Return likely NIfTI image files for selected patient.

    image_type:
    - Auto
    - CT
    - MR
    - PET
    """
    if patient_df is None or patient_df.empty:
        return pd.DataFrame()

    pf = patient_df.copy()
    name_lower = pf["Filename"].astype(str).str.lower()
    path_lower = pf["Relative path"].astype(str).str.lower()

    is_nifti = pf["File format"].astype(str).eq("NIfTI") | name_lower.str.endswith(".nii") | name_lower.str.endswith(".nii.gz")

    not_main_image = (
        name_lower.str.contains("mask", regex=False, na=False)
        | name_lower.str.contains("seg", regex=False, na=False)
        | name_lower.str.contains("roi", regex=False, na=False)
        | name_lower.str.contains("contour", regex=False, na=False)
        | name_lower.str.contains("dose", regex=False, na=False)
        | name_lower.str.contains("rtdose", regex=False, na=False)
        | name_lower.str.contains("rtstruct", regex=False, na=False)
        | name_lower.str.contains("structure", regex=False, na=False)
        | path_lower.str.contains("mask", regex=False, na=False)
        | path_lower.str.contains("seg", regex=False, na=False)
        | path_lower.str.contains("dose", regex=False, na=False)
        | path_lower.str.contains("structure", regex=False, na=False)
    )

    image_type = str(image_type)

    if image_type == "CT":
        type_match = (
            name_lower.str.contains("ct", regex=False, na=False)
            | path_lower.str.contains("ct", regex=False, na=False)
        )
        preferred = pf[is_nifti & type_match & (~not_main_image)].copy()
        if not preferred.empty:
            return preferred

    elif image_type == "MR":
        type_match = (
            name_lower.str.contains("mr", regex=False, na=False)
            | name_lower.str.contains("mri", regex=False, na=False)
            | path_lower.str.contains("mr", regex=False, na=False)
            | path_lower.str.contains("mri", regex=False, na=False)
        )
        preferred = pf[is_nifti & type_match & (~not_main_image)].copy()
        if not preferred.empty:
            return preferred

    elif image_type == "PET":
        type_match = (
            name_lower.str.contains("pet", regex=False, na=False)
            | path_lower.str.contains("pet", regex=False, na=False)
        )
        preferred = pf[is_nifti & type_match & (~not_main_image)].copy()
        if not preferred.empty:
            return preferred

    # Auto mode, or fallback if CT/MR/PET label was not found.
    image_like = (
        name_lower.str.contains("ct", regex=False, na=False)
        | name_lower.str.contains("mr", regex=False, na=False)
        | name_lower.str.contains("mri", regex=False, na=False)
        | name_lower.str.contains("pet", regex=False, na=False)
        | path_lower.str.contains("ct", regex=False, na=False)
        | path_lower.str.contains("mr", regex=False, na=False)
        | path_lower.str.contains("mri", regex=False, na=False)
        | path_lower.str.contains("pet", regex=False, na=False)
    )

    preferred = pf[is_nifti & image_like & (~not_main_image)].copy()

    if not preferred.empty:
        return preferred

    # Last fallback: any NIfTI that is not mask/dose/structure.
    return pf[is_nifti & (~not_main_image)].copy()


def rtviewer_load_nifti_image(nifti_df):
    """
    Load first NIfTI image for viewer.
    """
    if not NIBABEL_AVAILABLE:
        raise ImportError("nibabel is not installed. Install with: py -m pip install nibabel")

    if nifti_df is None or nifti_df.empty:
        raise ValueError("No NIfTI image file found for this patient.")

    row = nifti_df.iloc[0].to_dict()
    volume, orientation_info = viewer_load_nifti(row.get("Full path", ""), return_info=True)
    row["NIfTI orientation"] = orientation_info

    return volume, row


def rtviewer_get_nifti_mask_files(patient_df):
    """
    Return likely NIfTI mask files for selected patient.
    """
    if patient_df is None or patient_df.empty:
        return pd.DataFrame()

    pf = patient_df.copy()
    name_lower = pf["Filename"].astype(str).str.lower()
    path_lower = pf["Relative path"].astype(str).str.lower()

    is_nifti = pf["File format"].astype(str).eq("NIfTI") | name_lower.str.endswith(".nii") | name_lower.str.endswith(".nii.gz")

    is_mask = (
        name_lower.str.contains("mask", regex=False, na=False)
        | name_lower.str.contains("seg", regex=False, na=False)
        | name_lower.str.contains("roi", regex=False, na=False)
        | name_lower.str.contains("contour", regex=False, na=False)
        | path_lower.str.contains("mask", regex=False, na=False)
        | path_lower.str.contains("seg", regex=False, na=False)
        | path_lower.str.contains("roi", regex=False, na=False)
        | path_lower.str.contains("contour", regex=False, na=False)
    )

    return pf[is_nifti & is_mask].copy()



def fit_mask_to_image_shape(mask, target_shape, max_difference=3):
    """
    Fit a mask to image shape by centre cropping/padding when the mismatch is small.

    This is useful for common NIfTI export differences such as:
    mask (342,342,88) vs image (341,341,88).

    If the mismatch is larger than max_difference in any dimension, return None.
    """
    mask = np.asarray(mask)
    target_shape = tuple(target_shape)

    if mask.shape == target_shape:
        return mask

    if mask.ndim != 3 or len(target_shape) != 3:
        return None

    diffs = [abs(mask.shape[i] - target_shape[i]) for i in range(3)]

    if any(diff > max_difference for diff in diffs):
        return None

    fitted = mask

    # Centre crop dimensions that are too large.
    slices = []
    for dim in range(3):
        current = fitted.shape[dim]
        target = target_shape[dim]

        if current > target:
            start = (current - target) // 2
            end = start + target
            slices.append(slice(start, end))
        else:
            slices.append(slice(0, current))

    fitted = fitted[tuple(slices)]

    # Centre pad dimensions that are too small.
    pad_width = []
    for dim in range(3):
        current = fitted.shape[dim]
        target = target_shape[dim]

        if current < target:
            before = (target - current) // 2
            after = target - current - before
            pad_width.append((before, after))
        else:
            pad_width.append((0, 0))

    if any(before > 0 or after > 0 for before, after in pad_width):
        fitted = np.pad(fitted, pad_width, mode="constant", constant_values=0)

    if fitted.shape != target_shape:
        return None

    return fitted



def rtviewer_load_compatible_nifti_masks(mask_df, image_shape):
    """
    Load NIfTI masks for overlay on NIfTI image.

    Individual masks are loaded as separate overlay items.
    Small shape differences are corrected by centre crop/pad when safe.
    """
    overlays = []
    status_rows = []

    if mask_df is None or mask_df.empty:
        return overlays, pd.DataFrame()

    for _, row in mask_df.iterrows():
        row_dict = row.to_dict()
        filename = str(row_dict.get("Filename", ""))
        rel_path = row_dict.get("Relative path", filename)
        mask_name = Path(filename.replace(".nii.gz", "")).stem

        try:
            mask = viewer_load_nifti(row_dict.get("Full path", ""))
            original_shape = np.asarray(mask).shape

            fitted_mask = fit_mask_to_image_shape(mask, image_shape, max_difference=3)

            if fitted_mask is not None:
                overlays.append({
                    "name": mask_name,
                    "file": rel_path,
                    "volume": fitted_mask,
                    "visible_key": f"nifti_mask_visible_{len(overlays)}",
                })

                if original_shape == tuple(image_shape):
                    status = "Overlaid"
                    details = f"Shape matches image: {original_shape}"
                else:
                    status = "Overlaid after shape fit"
                    details = f"Small mismatch corrected: mask {original_shape} → image {tuple(image_shape)}"
            else:
                status = "Not overlaid"
                details = f"Shape mismatch too large: mask {original_shape} vs image {tuple(image_shape)}"

        except Exception as error:
            status = "Not overlaid"
            details = f"Could not load mask: {error}"

        status_rows.append({
            "OAR / Mask": mask_name,
            "Status": status,
            "Details": details,
        })

    return overlays, pd.DataFrame(status_rows)


def rtviewer_get_dicom_image_files(patient_df, image_type="CT"):
    """
    Return likely DICOM image files for selected patient and selected image type.
    Excludes RTSTRUCT and dose.
    """
    if patient_df is None or patient_df.empty:
        return pd.DataFrame()

    pf = patient_df.copy()
    name_lower = pf["Filename"].astype(str).str.lower()
    path_lower = pf["Relative path"].astype(str).str.lower()

    not_image = (
        name_lower.str.contains("dose", regex=False, na=False)
        | name_lower.str.contains("rtdose", regex=False, na=False)
        | name_lower.str.contains("rtstruct", regex=False, na=False)
        | name_lower.str.contains("rt.struct", regex=False, na=False)
        | name_lower.str.contains("rt_struct", regex=False, na=False)
        | name_lower.str.contains("structure", regex=False, na=False)
        | name_lower.str.startswith("rs.")
        | path_lower.str.contains("dose", regex=False, na=False)
        | path_lower.str.contains("rtstruct", regex=False, na=False)
        | path_lower.str.contains("structure", regex=False, na=False)
    )

    image_type = str(image_type)

    if image_type == "CT":
        type_match = (
            name_lower.str.contains("ct", regex=False, na=False)
            | path_lower.str.contains("ct", regex=False, na=False)
            | (pf["File role"].astype(str) == "CT")
        )
    elif image_type == "MR":
        type_match = (
            name_lower.str.contains("mr", regex=False, na=False)
            | name_lower.str.contains("mri", regex=False, na=False)
            | path_lower.str.contains("mr", regex=False, na=False)
            | path_lower.str.contains("mri", regex=False, na=False)
            | (pf["File role"].astype(str) == "MRI")
        )
    elif image_type == "PET":
        type_match = (
            name_lower.str.contains("pet", regex=False, na=False)
            | path_lower.str.contains("pet", regex=False, na=False)
            | (pf["File role"].astype(str) == "PET")
        )
    else:
        type_match = pd.Series([True] * len(pf), index=pf.index)

    image_df = pf[(pf["File format"] == "DICOM") & type_match & (~not_image)].copy()

    if not image_df.empty:
        return image_df

    # Fallback: any readable DICOM image files excluding RT/dose.
    return pf[(pf["File format"] == "DICOM") & (~not_image)].copy()




# ============================================================
# VBA IMAGE REGISTRATION HELPERS
# ============================================================
# These functions are required by:
# - render_voxel_batch_preprocessing_panel()
# - render_prepare_viewer_panel()
#
# The design follows a physical spatial-reference approach:
# fixed image geometry is preserved; moving images are resampled into
# the fixed image grid using spacing, origin and direction.


def vbv_classify_file(row):
    """Classify a file as CT, MR, Dose, Mask, RTSTRUCT or Other."""
    filename = str(row.get("Filename", "")).lower()
    rel_path = str(row.get("Relative path", "")).lower()
    file_role = str(row.get("File role", "")).lower()
    text = f"{filename} {rel_path} {file_role}"

    if "rtstruct" in text or "rt.struct" in text or "rt_struct" in text or filename.startswith("rs.") or "structure" in text:
        return "RTSTRUCT"
    if "dose" in text or "rtdose" in text or "plan" in text:
        return "Dose"
    if "mask" in text or "seg" in text or "roi" in text or "contour" in text:
        return "Mask"
    if "mri" in text or "mr_" in text or "mr-" in text or "/mr" in text or "\\mr" in text:
        return "MR"
    if "ct" in text or file_role == "ct":
        return "CT"
    return "Other"


def vbv_prepare_file_table(file_df):
    """Add inferred patient ID and VBA file role."""
    if file_df is None or file_df.empty:
        return pd.DataFrame()

    try:
        df = add_patient_grouping_columns(file_df.copy())
    except Exception:
        df = file_df.copy()

    if "Inferred patient ID" not in df.columns:
        if "Patient ID" in df.columns:
            df["Inferred patient ID"] = df["Patient ID"].astype(str)
        elif "Patient" in df.columns:
            df["Inferred patient ID"] = df["Patient"].astype(str)
        else:
            df["Inferred patient ID"] = "Unknown"

    df["VBA role"] = df.apply(vbv_classify_file, axis=1)
    return df


def vbv_is_nifti_row(row):
    """Return True if a row appears to be a NIfTI file."""
    filename = str(row.get("Filename", "")).lower()
    fmt = str(row.get("File format", "")).lower()
    return fmt == "nifti" or filename.endswith(".nii") or filename.endswith(".nii.gz")


def vbv_safe_name(text):
    """Make text safe for folder/file names."""
    return str(text).replace("\\", "_").replace("/", "_").replace(":", "_").replace(" ", "_")


def vbv_read_image(path):
    """
    Read image with SimpleITK.

    SimpleITK preserves physical image geometry:
    spacing, origin and direction.
    """
    if not SIMPLEITK_AVAILABLE:
        raise ImportError("SimpleITK is not installed. Install with: py -m pip install SimpleITK")
    return sitk.ReadImage(str(path))


def vbv_geometry(img):
    """Return image geometry for display/QC."""
    return {
        "Size": tuple(int(x) for x in img.GetSize()),
        "Spacing": tuple(float(x) for x in img.GetSpacing()),
        "Origin": tuple(float(x) for x in img.GetOrigin()),
        "Direction": tuple(float(x) for x in img.GetDirection()),
    }


def vbv_resample_to_fixed(moving, fixed, transform=None, interpolation=None, default_value=0.0):
    """
    Resample moving image into fixed image geometry.

    Resample the moving image into the fixed image reference geometry.
    """
    if interpolation is None:
        interpolation = sitk.sitkLinear
    if transform is None:
        transform = sitk.Transform(3, sitk.sitkIdentity)

    return sitk.Resample(
        moving,
        fixed,
        transform,
        interpolation,
        default_value,
        moving.GetPixelID(),
    )


def vbv_register_mr_to_ct(mr_img, ct_img, registration_type="Rigid"):
    """
    Register MR to CT using Mattes mutual information.

    CT is fixed. MR is moving.
    """
    fixed = sitk.Cast(ct_img, sitk.sitkFloat32)
    moving = sitk.Cast(mr_img, sitk.sitkFloat32)

    if registration_type == "Affine":
        transform_model = sitk.AffineTransform(3)
    else:
        transform_model = sitk.Euler3DTransform()

    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        transform_model,
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.10)
    registration.SetInterpolator(sitk.sitkLinear)

    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration.SetOptimizerScalesFromPhysicalShift()

    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration.Execute(fixed, moving)

    registered_mr = vbv_resample_to_fixed(
        moving=mr_img,
        fixed=ct_img,
        transform=final_transform,
        interpolation=sitk.sitkLinear,
        default_value=0.0,
    )

    return registered_mr, final_transform, float(registration.GetMetricValue())


def vbv_sitk_to_xyz_array(img):
    """
    Convert SimpleITK image to numpy array in x, y, z order.

    SimpleITK returns z, y, x, so we transpose for viewer use.
    """
    return np.transpose(sitk.GetArrayFromImage(img), (2, 1, 0))


def vbv_normalise_slice(arr):
    """Robustly normalise a 2D slice to 0-1 for display."""
    arr = np.asarray(arr, dtype=float)
    finite = np.isfinite(arr)

    if not np.any(finite):
        return np.zeros_like(arr, dtype=float)

    vals = arr[finite]
    lo, hi = np.percentile(vals, [1, 99])

    if hi <= lo:
        lo, hi = np.min(vals), np.max(vals)

    if hi <= lo:
        return np.zeros_like(arr, dtype=float)

    return np.clip((arr - lo) / (hi - lo), 0, 1)


def vbv_overlay_mask(mask_arr, slice_index):
    """Return transparent mask overlay for one axial slice."""
    mask_slice = mask_arr[:, :, slice_index].T
    return np.where(mask_slice > 0, 1, np.nan)


def vbv_find_outputs(patient_id):
    """Find final registered outputs for one patient."""
    output_dir = st.session_state.get("voxel_registration_output_directory", "")
    if not output_dir:
        return {}

    patient_dir = Path(output_dir).expanduser() / vbv_safe_name(patient_id)
    if not patient_dir.exists():
        return {}

    files = list(patient_dir.glob("*.nii")) + list(patient_dir.glob("*.nii.gz"))

    outputs = {
        "patient_dir": str(patient_dir),
        "fixed_ct": [],
        "fixed_mr": [],
        "registered_mr": [],
        "registered_ct": [],
        "dose": [],
        "masks": [],
    }

    for path in files:
        name = path.name.lower()
        if name.startswith("fixed_ct"):
            outputs["fixed_ct"].append(path)
        elif name.startswith("fixed_mr"):
            outputs["fixed_mr"].append(path)
        elif "registered_mr_to_ct" in name:
            outputs["registered_mr"].append(path)
        elif "registered_ct_to_mr" in name:
            outputs["registered_ct"].append(path)
        elif "dose" in name:
            outputs["dose"].append(path)
        elif "mask" in name or "rtstruct" in name or "structure" in name or "seg" in name:
            outputs["masks"].append(path)

    return outputs


def vbv_build_registration_plan(file_df):
    """Build a patient-level and file-level registration plan."""
    df = vbv_prepare_file_table(file_df)

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    patient_rows = []
    file_rows = []

    for patient_id, patient_files in df.groupby("Inferred patient ID", dropna=False):
        patient_id = str(patient_id)
        counts = patient_files["VBA role"].value_counts().to_dict()

        fixed = "CT" if counts.get("CT", 0) > 0 else "MR" if counts.get("MR", 0) > 0 else "None"

        warnings = []
        if fixed == "None":
            warnings.append("No CT/MR fixed image")
        if counts.get("Dose", 0) == 0:
            warnings.append("No dose")
        if counts.get("Mask", 0) == 0 and counts.get("RTSTRUCT", 0) == 0:
            warnings.append("No mask/RTSTRUCT")

        patient_rows.append({
            "Patient ID": patient_id,
            "Fixed/reference image": fixed,
            "CT": counts.get("CT", 0),
            "MR": counts.get("MR", 0),
            "Dose": counts.get("Dose", 0),
            "Masks/RTSTRUCT": counts.get("Mask", 0) + counts.get("RTSTRUCT", 0),
            "Status": "Ready" if fixed != "None" else "Cannot register",
            "Warnings": "; ".join(warnings) if warnings else "None",
        })

        for _, row in patient_files.iterrows():
            role = row.get("VBA role", "Other")
            if role == "CT":
                action = "Use as fixed image when available"
            elif role == "MR":
                action = "Register/resample to CT when CT exists"
            elif role == "Dose":
                action = "Resample to fixed CT/MR geometry using linear interpolation"
            elif role in ["Mask", "RTSTRUCT"]:
                action = "Resample to fixed CT/MR geometry using nearest-neighbour interpolation"
            else:
                action = "Review only"

            file_rows.append({
                "Patient ID": patient_id,
                "File": row.get("Relative path", row.get("Filename", "")),
                "Role": role,
                "Planned action": action,
            })

    return pd.DataFrame(patient_rows), pd.DataFrame(file_rows)


def vbv_patient_output_status(output_directory, patient_id):
    """Check whether a patient's registered output folder already contains registration outputs."""
    patient_dir = Path(output_directory).expanduser() / vbv_safe_name(patient_id)
    if not patient_dir.exists():
        return {
            "Output folder": str(patient_dir),
            "Output exists": False,
            "Output file count": 0,
            "Already processed": False,
            "Existing output files": [],
            "Existing output summary": "",
        }

    all_files = [p for p in patient_dir.rglob("*") if p.is_file()]

    def _is_output_file(path_obj):
        name = path_obj.name.lower()
        return (
            name.endswith(".nii")
            or name.endswith(".nii.gz")
            or name.endswith(".tfm")
            or name.endswith(".nrrd")
            or name.endswith(".mha")
            or name.endswith(".mhd")
            or name.endswith(".csv")
        )

    output_files = [p for p in all_files if _is_output_file(p)]

    def _looks_like_registered_output(path_obj):
        name = path_obj.name.lower()
        return (
            "registered" in name
            or "transform" in name
            or "resampled" in name
            or "aligned" in name
            or "dose_to_" in name
            or "mask_to_" in name
        )

    registered_output_files = [p for p in output_files if _looks_like_registered_output(p)]

    output_count = len(output_files)
    registered_count = len(registered_output_files)

    display_files = registered_output_files if registered_output_files else output_files
    display_names = [p.name for p in sorted(display_files)[:8]]
    more_count = max(0, len(display_files) - len(display_names))
    summary = ", ".join(display_names)
    if more_count > 0:
        summary = f"{summary} (+{more_count} more)"
    if summary == "":
        summary = "No saved outputs detected"

    return {
        "Output folder": str(patient_dir),
        "Output exists": output_count > 0,
        "Output file count": output_count,
        "Already processed": registered_count > 0,
        "Existing output files": [str(p) for p in display_files],
        "Existing output summary": summary,
    }

def vbv_build_batch_preflight_table(file_df, output_directory, require_dose=True, require_masks=True):
    """
    Build a clean, mutually exclusive batch preflight table.
    """
    df = vbv_prepare_file_table(file_df)

    if df is None or df.empty:
        return pd.DataFrame()

    rows = []

    for patient_id, patient_files in df.groupby("Inferred patient ID", dropna=False):
        patient_id = str(patient_id)
        counts = patient_files["VBA role"].value_counts().to_dict()

        ct_count = int(counts.get("CT", 0))
        mr_count = int(counts.get("MR", 0))
        dose_count = int(counts.get("Dose", 0))
        mask_count = int(counts.get("Mask", 0) + counts.get("RTSTRUCT", 0))

        output_status = vbv_patient_output_status(output_directory, patient_id)

        missing_reasons = []
        review_reasons = []
        suggested_fixes = []

        if ct_count == 0 and mr_count == 0:
            missing_reasons.append("No CT or MR image detected")
            suggested_fixes.append("Add at least one CT or MR NIfTI image")

        if require_dose and dose_count == 0:
            missing_reasons.append("Dose file required but not detected")
            suggested_fixes.append("Add a dose file or turn off dose alignment")

        if require_masks and mask_count == 0:
            missing_reasons.append("Masks/RTSTRUCT required but not detected")
            suggested_fixes.append("Add masks/RTSTRUCT files or turn off mask alignment")

        if ct_count > 1:
            review_reasons.append(f"{ct_count} CT files detected")
            suggested_fixes.append("Select one CT as the fixed/reference image or rename/archive duplicates")

        if mr_count > 1:
            review_reasons.append(f"{mr_count} MR files detected")
            suggested_fixes.append("Select which MR should be registered to CT using the MR image selection control")

        if dose_count > 1:
            review_reasons.append(f"{dose_count} dose files detected")
            suggested_fixes.append("Confirm which dose should be resampled")

        if mask_count > 25:
            review_reasons.append(f"{mask_count} mask/structure files detected")
            suggested_fixes.append("Consider processing selected masks first")

        if output_status.get("Already processed", False):
            batch_status = "Already processed"
            review_reason = "Saved registered/resampled outputs already exist"
            suggested_action = "Skip by default; use Force overwrite only if you want to rerun"
            recommended_action = "Skip by default"
        elif missing_reasons:
            batch_status = "Incomplete input"
            review_reason = "; ".join(missing_reasons)
            suggested_action = "; ".join(dict.fromkeys(suggested_fixes))
            recommended_action = "Cannot process until fixed"
        elif review_reasons:
            batch_status = "Needs review"
            review_reason = "; ".join(review_reasons)
            suggested_action = "; ".join(dict.fromkeys(suggested_fixes))
            recommended_action = "Review before processing"
        else:
            batch_status = "Ready"
            review_reason = "No issue detected"
            suggested_action = "Ready to process"
            recommended_action = "Process"

        rows.append({
            "Patient ID": patient_id,
            "Batch status": batch_status,
            "Review reason": review_reason,
            "Suggested fix": suggested_action,
            "Recommended action": recommended_action,
            "CT": ct_count,
            "MR": mr_count,
            "Dose": dose_count,
            "Masks/RTSTRUCT": mask_count,
            "Output exists": output_status.get("Output exists", False),
            "Output file count": output_status.get("Output file count", 0),
            "Saved output folder": output_status.get("Output folder", ""),
            "Saved outputs": output_status.get("Existing output summary", ""),
        })

    return pd.DataFrame(rows)


def vbv_run_registration_batch_optimized(
    file_df,
    output_directory,
    patient_ids,
    registration_type,
    align_dose=True,
    align_masks=True,
    skip_existing=True,
    force_overwrite=False,
    progress_bar=None,
    progress_text=None,
    status_placeholder=None,
    incremental_csv_path=None,
    selected_mr_by_patient=None,
):
    """
    Run registration patient-by-patient with progress and optional skip-existing.
    """
    results = []
    selected_mr_by_patient = selected_mr_by_patient or {}
    total_patients = max(1, len(patient_ids))
    processed_patients = 0
    skipped_patients = 0
    failed_patients = 0

    for index, patient_id in enumerate(patient_ids, start=1):
        if progress_bar is not None:
            progress_bar.progress(min(index / total_patients, 1.0))

        output_status = vbv_patient_output_status(output_directory, patient_id)
        output_folder = output_status.get("Output folder", "")
        existing_files_summary = output_status.get("Existing output summary", "")

        if progress_text is not None:
            remaining = total_patients - index
            progress_text.write(
                f"Processing {index}/{total_patients}: {patient_id} | "
                f"Processed: {processed_patients} | Skipped: {skipped_patients} | Failed: {failed_patients} | "
                f"Remaining: {remaining}"
            )

        if skip_existing and not force_overwrite and output_status.get("Already processed", False):
            skipped_patients += 1
            patient_result = pd.DataFrame([{
                "Patient ID": patient_id,
                "Step": "Batch registration",
                "Role": "Patient",
                "Status": "Skipped",
                "Details": f"Existing registered outputs were found and this patient was skipped. Saved outputs: {existing_files_summary}",
                "Output file": output_folder,
            }])

            results.append(patient_result)

            if status_placeholder is not None:
                status_placeholder.info(
                    f"Skipping {patient_id}: registered outputs already exist in {output_folder}"
                )

            if incremental_csv_path:
                merged = pd.concat(results, ignore_index=True) if len(results) > 0 else patient_result.copy()
                Path(incremental_csv_path).parent.mkdir(parents=True, exist_ok=True)
                merged.to_csv(incremental_csv_path, index=False)

            continue

        try:
            patient_result = vbv_register_patient_batch(
                file_df=file_df,
                output_directory=output_directory,
                selected_patient_ids=[patient_id],
                registration_type=registration_type,
                align_dose=align_dose,
                align_masks=align_masks,
                progress_bar=None,
                progress_text=None,
                selected_mr_by_patient=selected_mr_by_patient,
            )

            if patient_result is None or patient_result.empty:
                failed_patients += 1
                patient_result = pd.DataFrame([{
                    "Patient ID": patient_id,
                    "Step": "Batch registration",
                    "Role": "Patient",
                    "Status": "Failed",
                    "Details": "No outputs were returned by the registration function.",
                    "Output file": output_folder,
                }])
            elif "Status" in patient_result.columns and (patient_result["Status"] == "Failed").any():
                failed_patients += 1
            else:
                processed_patients += 1

            if "Output file" not in patient_result.columns:
                patient_result["Output file"] = output_folder
            else:
                patient_result["Output file"] = patient_result["Output file"].fillna(output_folder).replace("", output_folder)

            results.append(patient_result)

            if status_placeholder is not None:
                status_placeholder.info(
                    f"Completed {patient_id}. Saved outputs in: {output_folder}"
                )

        except Exception as error:
            failed_patients += 1
            patient_result = pd.DataFrame([{
                "Patient ID": patient_id,
                "Step": "Batch registration",
                "Role": "Patient",
                "Status": "Failed",
                "Details": f"Patient registration failed: {error}",
                "Output file": output_folder,
            }])
            results.append(patient_result)

            if status_placeholder is not None:
                status_placeholder.error(f"Failed {patient_id}: {error}")

        if incremental_csv_path:
            merged = pd.concat(results, ignore_index=True) if len(results) > 0 else pd.DataFrame()
            Path(incremental_csv_path).parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(incremental_csv_path, index=False)

    if len(results) == 0:
        final_df = pd.DataFrame(columns=["Patient ID", "Step", "Role", "Status", "Details", "Output file"])
    else:
        final_df = pd.concat(results, ignore_index=True)

    if progress_bar is not None:
        progress_bar.progress(1.0)

    if progress_text is not None:
        progress_text.write(
            f"Done. Total: {len(patient_ids)} | Processed: {processed_patients} | "
            f"Skipped: {skipped_patients} | Failed: {failed_patients}"
        )

    if status_placeholder is not None:
        status_placeholder.success(
            f"Batch registration finished. Processed: {processed_patients}, "
            f"Skipped: {skipped_patients}, Failed: {failed_patients}"
        )

    if incremental_csv_path:
        Path(incremental_csv_path).parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(incremental_csv_path, index=False)

    return final_df

def vbv_get_mr_options_by_patient(file_df, patient_ids=None):
    """Return MR file options for patients who have more than one MR file."""
    df = vbv_prepare_file_table(file_df)

    if patient_ids:
        patient_ids = [str(x) for x in patient_ids]
        df = df[df["Inferred patient ID"].astype(str).isin(patient_ids)].copy()

    mr_options = {}

    for patient_id, patient_files in df.groupby("Inferred patient ID", dropna=False):
        patient_id = str(patient_id)
        mr_rows = patient_files[patient_files["VBA role"] == "MR"].copy()

        if mr_rows.shape[0] <= 1:
            continue

        label_col = "Relative path" if "Relative path" in mr_rows.columns else "Filename"

        options = []
        for _, row in mr_rows.iterrows():
            label = str(row.get(label_col, row.get("Filename", "")))
            full_path = str(row.get("Full path", ""))
            options.append({
                "label": label,
                "full_path": full_path,
                "filename": str(row.get("Filename", "")),
                "relative_path": str(row.get("Relative path", "")),
            })

        mr_options[patient_id] = options

    return mr_options


def render_mr_selection_controls(file_df, patient_ids=None, key_prefix="mr_select"):
    """
    Show MR selection boxes when patients have multiple MR images.

    The selected full path is stored in st.session_state.vbv_selected_mr_by_patient.
    """
    mr_options = vbv_get_mr_options_by_patient(file_df, patient_ids=patient_ids)

    if "vbv_selected_mr_by_patient" not in st.session_state:
        st.session_state.vbv_selected_mr_by_patient = {}

    if not mr_options:
        return st.session_state.vbv_selected_mr_by_patient

    st.markdown("### MR image selection")
    st.info("Multiple MR images were detected. Select the MR image to register for each patient.")

    for patient_id, options in mr_options.items():
        option_labels = [item["label"] for item in options]
        option_paths = [item["full_path"] for item in options]

        current_path = st.session_state.vbv_selected_mr_by_patient.get(patient_id, "")
        default_index = 0
        if current_path in option_paths:
            default_index = option_paths.index(current_path)

        selected_label = st.selectbox(
            f"MR image for patient {patient_id}",
            options=option_labels,
            index=default_index,
            key=f"{key_prefix}_{vbv_safe_name(patient_id)}",
        )

        selected_index = option_labels.index(selected_label)
        st.session_state.vbv_selected_mr_by_patient[patient_id] = option_paths[selected_index]

    return st.session_state.vbv_selected_mr_by_patient



def vbv_run_registration(
    file_df,
    output_directory,
    selected_patient_ids,
    registration_type,
    align_dose=True,
    align_masks=True,
    progress_bar=None,
    progress_text=None,
    selected_mr_by_patient=None,
):
    """Run CT/MR registration and dose/mask alignment."""
    results = []
    selected_mr_by_patient = selected_mr_by_patient or {}
    df = vbv_prepare_file_table(file_df)

    if selected_patient_ids:
        selected_patient_ids = [str(x) for x in selected_patient_ids]
        df = df[df["Inferred patient ID"].astype(str).isin(selected_patient_ids)].copy()

    output_root = Path(output_directory).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    total = int(df.shape[0])
    completed = 0

    def update(patient="", file_label=""):
        if total <= 0:
            return
        if progress_bar is not None:
            progress_bar.progress(min(completed / total, 1.0))
        if progress_text is not None:
            progress_text.info(
                f"Registration running... {completed} of {total} file(s) reviewed. "
                f"Patient: {patient}. File: {file_label}"
            )

    def _fmt_tuple(values, decimals=3):
        try:
            formatted = []
            for value in values:
                if isinstance(value, (int, np.integer)):
                    formatted.append(str(int(value)))
                else:
                    formatted.append(f"{float(value):.{decimals}f}")
            return " × ".join(formatted)
        except Exception:
            return ""

    def _add_image_metadata(row):
        """
        Add image metadata columns to the registration results table.

        Metadata are read from the output image if available. This keeps the
        registration table readable and makes voxel size / image grid checks
        visible without forcing users to open the Details text.
        """
        metadata_defaults = {
            "Image size": "",
            "Voxel size / spacing": "",
            "Origin": "",
            "Direction": "",
            "Pixel type": "",
            "Dimension": "",
        }

        for key, value in metadata_defaults.items():
            row.setdefault(key, value)

        output_file = str(row.get("Output file", "") or "")

        if output_file == "":
            return row

        try:
            output_path = Path(output_file)
            if not output_path.exists():
                return row

            img = sitk.ReadImage(str(output_path))
            row["Image size"] = _fmt_tuple(img.GetSize(), decimals=0)
            row["Voxel size / spacing"] = _fmt_tuple(img.GetSpacing(), decimals=3)
            row["Origin"] = _fmt_tuple(img.GetOrigin(), decimals=3)
            row["Direction"] = _fmt_tuple(img.GetDirection(), decimals=3)
            row["Pixel type"] = vbv_pixel_type_text(img)
            row["Dimension"] = int(img.GetDimension())

        except Exception as metadata_error:
            row["Metadata status"] = f"Could not read metadata: {metadata_error}"

        return row

    def add(row):
        nonlocal completed
        row = _add_image_metadata(row)
        results.append(row)
        completed += 1
        update(row.get("Patient ID", ""), row.get("File", ""))

    for patient_id, patient_files in df.groupby("Inferred patient ID", dropna=False):
        patient_id = str(patient_id)
        patient_out = output_root / vbv_safe_name(patient_id)
        patient_out.mkdir(parents=True, exist_ok=True)

        ct_rows = patient_files[patient_files["VBA role"] == "CT"]
        mr_rows = patient_files[patient_files["VBA role"] == "MR"]
        dose_rows = patient_files[patient_files["VBA role"] == "Dose"]
        mask_rows = patient_files[patient_files["VBA role"].isin(["Mask", "RTSTRUCT"])]

        ct_row = ct_rows.iloc[0] if not ct_rows.empty else None

        mr_row = None
        if not mr_rows.empty:
            selected_mr_path = str(selected_mr_by_patient.get(patient_id, ""))
            if selected_mr_path:
                mr_match = mr_rows[mr_rows["Full path"].astype(str) == selected_mr_path]
                if not mr_match.empty:
                    mr_row = mr_match.iloc[0]
                else:
                    mr_row = mr_rows.iloc[0]
            else:
                mr_row = mr_rows.iloc[0]

        fixed_row = None
        fixed_role = None

        if ct_row is not None and vbv_is_nifti_row(ct_row):
            fixed_row = ct_row
            fixed_role = "CT"
        elif mr_row is not None and vbv_is_nifti_row(mr_row):
            fixed_row = mr_row
            fixed_role = "MR"

        if fixed_row is None:
            for _, row in patient_files.iterrows():
                add({
                    "Patient ID": patient_id,
                    "Step": "Select fixed image",
                    "Role": row.get("VBA role", "Other"),
                    "File": row.get("Relative path", row.get("Filename", "")),
                    "Status": "Planned only",
                    "Details": "No NIfTI CT/MR fixed image. DICOM needs conversion/SimpleITK DICOM-series reader.",
                    "Output file": "",
                })
            continue

        try:
            fixed_img = vbv_read_image(fixed_row.get("Full path", ""))
            fixed_out = patient_out / f"fixed_{fixed_role}.nii.gz"
            sitk.WriteImage(fixed_img, str(fixed_out))

            add({
                "Patient ID": patient_id,
                "Step": "Read fixed image",
                "Role": fixed_role,
                "File": fixed_row.get("Relative path", fixed_row.get("Filename", "")),
                "Status": "Processed",
                "Details": f"{fixed_role} fixed image read with geometry preserved: {vbv_geometry(fixed_img)}",
                "Output file": str(fixed_out),
            })

        except Exception as error:
            add({
                "Patient ID": patient_id,
                "Step": "Read fixed image",
                "Role": fixed_role,
                "File": fixed_row.get("Relative path", fixed_row.get("Filename", "")),
                "Status": "Failed",
                "Details": f"Could not read fixed image: {error}",
                "Output file": "",
            })
            continue

        if fixed_role == "CT" and mr_row is not None:
            if not vbv_is_nifti_row(mr_row):
                add({
                    "Patient ID": patient_id,
                    "Step": "Register MR to CT",
                    "Role": "MR",
                    "File": mr_row.get("Relative path", mr_row.get("Filename", "")),
                    "Status": "Planned only",
                    "Details": "MR is not NIfTI. Convert DICOM MR before registration.",
                    "Output file": "",
                })
            else:
                try:
                    mr_img = vbv_read_image(mr_row.get("Full path", ""))
                    registered_mr, transform, metric = vbv_register_mr_to_ct(
                        mr_img,
                        fixed_img,
                        registration_type=registration_type,
                    )

                    mr_out = patient_out / "registered_MR_to_CT.nii.gz"
                    tx_out = patient_out / "MR_to_CT_transform.tfm"

                    sitk.WriteImage(registered_mr, str(mr_out))
                    sitk.WriteTransform(transform, str(tx_out))

                    add({
                        "Patient ID": patient_id,
                        "Step": "Register MR to CT",
                        "Role": "MR",
                        "File": mr_row.get("Relative path", mr_row.get("Filename", "")),
                        "Status": "Processed",
                        "Details": f"MR registered to CT using {registration_type} mutual information. Metric={metric:.5f}",
                        "Output file": str(mr_out),
                    })

                except Exception as error:
                    add({
                        "Patient ID": patient_id,
                        "Step": "Register MR to CT",
                        "Role": "MR",
                        "File": mr_row.get("Relative path", mr_row.get("Filename", "")),
                        "Status": "Failed",
                        "Details": f"MR registration failed: {error}",
                        "Output file": "",
                    })

        if fixed_role == "MR" and ct_row is not None:
            if not vbv_is_nifti_row(ct_row):
                add({
                    "Patient ID": patient_id,
                    "Step": "Resample CT to MR",
                    "Role": "CT",
                    "File": ct_row.get("Relative path", ct_row.get("Filename", "")),
                    "Status": "Planned only",
                    "Details": "CT is not NIfTI. Convert DICOM CT before registration.",
                    "Output file": "",
                })
            else:
                try:
                    ct_img = vbv_read_image(ct_row.get("Full path", ""))
                    ct_fixed = vbv_resample_to_fixed(
                        moving=ct_img,
                        fixed=fixed_img,
                        interpolation=sitk.sitkLinear,
                    )
                    ct_out = patient_out / "registered_CT_to_MR.nii.gz"
                    sitk.WriteImage(ct_fixed, str(ct_out))

                    add({
                        "Patient ID": patient_id,
                        "Step": "Resample CT to MR",
                        "Role": "CT",
                        "File": ct_row.get("Relative path", ct_row.get("Filename", "")),
                        "Status": "Processed",
                        "Details": "CT resampled into MR fixed image grid using physical geometry.",
                        "Output file": str(ct_out),
                    })

                except Exception as error:
                    add({
                        "Patient ID": patient_id,
                        "Step": "Resample CT to MR",
                        "Role": "CT",
                        "File": ct_row.get("Relative path", ct_row.get("Filename", "")),
                        "Status": "Failed",
                        "Details": f"CT resampling failed: {error}",
                        "Output file": "",
                    })

        if align_dose:
            for _, dose_row in dose_rows.iterrows():
                if not vbv_is_nifti_row(dose_row):
                    add({
                        "Patient ID": patient_id,
                        "Step": "Align dose",
                        "Role": "Dose",
                        "File": dose_row.get("Relative path", dose_row.get("Filename", "")),
                        "Status": "Planned only",
                        "Details": "Dose is not NIfTI. Convert RTDOSE first.",
                        "Output file": "",
                    })
                    continue

                try:
                    dose_img = vbv_read_image(dose_row.get("Full path", ""))
                    dose_fixed = vbv_resample_to_fixed(
                        moving=dose_img,
                        fixed=fixed_img,
                        interpolation=sitk.sitkLinear,
                    )
                    dose_out = patient_out / f"aligned_dose_{vbv_safe_name(Path(str(dose_row.get('Full path', 'dose.nii.gz'))).name)}"
                    if not str(dose_out).endswith(".nii.gz"):
                        dose_out = Path(str(dose_out) + ".nii.gz")
                    sitk.WriteImage(dose_fixed, str(dose_out))

                    add({
                        "Patient ID": patient_id,
                        "Step": "Align dose",
                        "Role": "Dose",
                        "File": dose_row.get("Relative path", dose_row.get("Filename", "")),
                        "Status": "Processed",
                        "Details": f"Dose resampled to fixed {fixed_role} geometry using linear interpolation.",
                        "Output file": str(dose_out),
                    })

                except Exception as error:
                    add({
                        "Patient ID": patient_id,
                        "Step": "Align dose",
                        "Role": "Dose",
                        "File": dose_row.get("Relative path", dose_row.get("Filename", "")),
                        "Status": "Failed",
                        "Details": f"Dose alignment failed: {error}",
                        "Output file": "",
                    })

        if align_masks:
            for _, mask_row in mask_rows.iterrows():
                role = mask_row.get("VBA role", "Mask")

                if not vbv_is_nifti_row(mask_row):
                    add({
                        "Patient ID": patient_id,
                        "Step": "Align mask/structure",
                        "Role": role,
                        "File": mask_row.get("Relative path", mask_row.get("Filename", "")),
                        "Status": "Planned only",
                        "Details": "Mask/RTSTRUCT is not NIfTI. Convert RTSTRUCT to masks first.",
                        "Output file": "",
                    })
                    continue

                try:
                    mask_img = vbv_read_image(mask_row.get("Full path", ""))
                    mask_fixed = vbv_resample_to_fixed(
                        moving=mask_img,
                        fixed=fixed_img,
                        interpolation=sitk.sitkNearestNeighbor,
                    )
                    mask_out = patient_out / f"aligned_{vbv_safe_name(role)}_{vbv_safe_name(Path(str(mask_row.get('Full path', 'mask.nii.gz'))).name)}"
                    if not str(mask_out).endswith(".nii.gz"):
                        mask_out = Path(str(mask_out) + ".nii.gz")
                    sitk.WriteImage(mask_fixed, str(mask_out))

                    add({
                        "Patient ID": patient_id,
                        "Step": "Align mask/structure",
                        "Role": role,
                        "File": mask_row.get("Relative path", mask_row.get("Filename", "")),
                        "Status": "Processed",
                        "Details": f"Mask/structure resampled to fixed {fixed_role} geometry using nearest-neighbour interpolation.",
                        "Output file": str(mask_out),
                    })

                except Exception as error:
                    add({
                        "Patient ID": patient_id,
                        "Step": "Align mask/structure",
                        "Role": role,
                        "File": mask_row.get("Relative path", mask_row.get("Filename", "")),
                        "Status": "Failed",
                        "Details": f"Mask alignment failed: {error}",
                        "Output file": "",
                    })

    if progress_bar is not None:
        progress_bar.progress(1.0)

    if progress_text is not None:
        progress_text.success(f"Normalisation finished. {completed} of {total} file(s) reviewed.")

    return pd.DataFrame(results)

def render_voxel_batch_preprocessing_panel():
    """
    Step 5: Register CT/MR and align dose/masks.

    This is the function called by the VBA workflow when current_step == "preprocess".
    It uses the physical image-geometry helpers:
    - fixed CT/MR image keeps spacing/origin/direction
    - MR registers to CT when CT is available
    - dose resamples with linear interpolation
    - masks resample with nearest-neighbour interpolation
    """
    st.markdown("## 🛠️ Step 5: Register CT/MR and align dose/masks")

    st.info(
        "This step uses physical image geometry, "
        "fixed image spacing, origin and direction are preserved. "
        "MR is registered to CT where available, then dose and masks are resampled "
        "to the final fixed CT/MR grid."
    )

    if not SIMPLEITK_AVAILABLE:
        st.error("SimpleITK is required. Install it with: py -m pip install SimpleITK")
        return

    file_df = st.session_state.get("voxel_loaded_image_files_df", pd.DataFrame())

    if file_df is None or file_df.empty:
        st.warning("No image files have been loaded yet.")
        if st.button("← Go to Load directory", use_container_width=True):
            st.session_state.voxel_image_load_step = "load"
            st.rerun()
        return

    # Output folder where final registered data will be written.
    default_output_dir = str(
        Path(st.session_state.get("voxel_image_directory_path", "") or ".")
        / "BrainRT_registered"
    )

    output_directory = st.text_input(
        "Output folder for registered images",
        value=st.session_state.get("voxel_registration_output_directory", default_output_dir),
        key="voxel_registration_output_directory",
        help="The app will create one subfolder per patient and write registered CT/MR, dose and masks there.",
    )

    # Prepare file table and patient list.
    df = vbv_prepare_file_table(file_df)

    patient_ids = sorted([
        str(x) for x in df["Inferred patient ID"].dropna().unique().tolist()
        if str(x).strip() != ""
    ])

    selected_patients = st.multiselect(
        "Patients to process",
        options=patient_ids,
        default=patient_ids[:1] if patient_ids else [],
        key="vbv_registration_selected_patients",
        help="Start with one patient first, then check Visual QA before processing the cohort.",
    )

    selected_mr_by_patient = render_mr_selection_controls(
        file_df,
        patient_ids=selected_patients,
        key_prefix="test_registration_mr_select",
    )

    registration_type = st.selectbox(
        "MR to CT registration type",
        options=["Rigid", "Affine"],
        index=0,
        key="vbv_registration_type",
    )

    col1, col2 = st.columns(2)

    with col1:
        align_dose = st.checkbox(
            "Align dose to fixed CT/MR grid",
            value=True,
            key="vbv_align_dose",
        )

    with col2:
        align_masks = st.checkbox(
            "Align masks/structures to fixed CT/MR grid",
            value=True,
            key="vbv_align_masks",
        )

    st.caption(
        "Images and dose use linear interpolation. "
        "Masks and structures use nearest-neighbour interpolation. "
        "The fixed image grid is preserved as the output reference geometry."
    )

    if st.button("Prepare registration plan", type="primary", use_container_width=True):
        summary_df, file_plan_df = vbv_build_registration_plan(file_df)

        if selected_patients:
            selected_set = [str(x) for x in selected_patients]
            summary_df = summary_df[summary_df["Patient ID"].astype(str).isin(selected_set)].copy()
            file_plan_df = file_plan_df[file_plan_df["Patient ID"].astype(str).isin(selected_set)].copy()

        st.session_state.vbv_registration_plan_summary = summary_df
        st.session_state.vbv_registration_file_plan = file_plan_df
        st.session_state.vbv_registration_plan_ran = True

    if st.session_state.get("vbv_registration_plan_ran", False):
        st.markdown("### Registration plan")
        st.dataframe(
            st.session_state.get("vbv_registration_plan_summary", pd.DataFrame()),
            use_container_width=True,
        )

        with st.expander("File-level plan"):
            st.dataframe(
                st.session_state.get("vbv_registration_file_plan", pd.DataFrame()),
                use_container_width=True,
            )

    st.markdown("### Run test registration")

    st.warning(
        "Run one patient first. Then inspect the automatic CT↔MR overlay in Visual QA "
        "before processing the full cohort."
    )

    if st.button("Run test registration", type="primary", use_container_width=True):
        progress_bar = st.progress(0.0)
        progress_text = st.empty()

        with st.spinner("Running normalisation using physical image geometry..."):
            results_df = vbv_run_registration(
                file_df=file_df,
                output_directory=output_directory,
                selected_patient_ids=selected_patients,
                registration_type=registration_type,
                align_dose=align_dose,
                align_masks=align_masks,
                progress_bar=progress_bar,
                progress_text=progress_text,
                selected_mr_by_patient=selected_mr_by_patient,
            )

        st.session_state.vbv_registration_results = results_df
        st.session_state.vbv_registration_ran = True
        st.session_state.voxel_registration_setup = {
            "output_directory": output_directory,
            "selected_patients": selected_patients,
            "registration_type": registration_type,
            "align_dose": align_dose,
            "align_masks": align_masks,
            "selected_mr_by_patient": selected_mr_by_patient,
            "image_geometry_based": True,
        }

        # After registration completes, open the Visual QA viewer automatically.
        st.rerun()

    if st.session_state.get("vbv_registration_ran", False):
        results_df = st.session_state.get("vbv_registration_results", pd.DataFrame())

        st.markdown("### Registration results")

        if results_df is None or results_df.empty:
            st.warning("No registration results generated.")
        else:
            processed = int((results_df["Status"] == "Processed").sum()) if "Status" in results_df.columns else 0
            planned = int((results_df["Status"] == "Planned only").sum()) if "Status" in results_df.columns else 0
            failed = int((results_df["Status"] == "Failed").sum()) if "Status" in results_df.columns else 0

            m1, m2, m3 = st.columns(3)
            m1.metric("Processed", processed)
            m2.metric("Planned only", planned)
            m3.metric("Failed", failed)

            preferred_cols = [
                "Patient ID", "Step", "Role", "Status", "Image size", "Voxel size / spacing",
                "Origin", "Direction", "Pixel type", "Dimension", "File", "Output file", "Details"
            ]
            display_cols = [col for col in preferred_cols if col in results_df.columns]
            display_cols += [col for col in results_df.columns if col not in display_cols]
            st.dataframe(results_df[display_cols], use_container_width=True)

            st.download_button(
                "Download registration results CSV",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name="vbv_registration_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if st.button("Save registration setup", use_container_width=True):
            st.session_state.voxel_registration_setup = {
                "output_directory": output_directory,
                "selected_patients": selected_patients,
                "registration_type": registration_type,
                "align_dose": align_dose,
                "align_masks": align_masks,
                "selected_mr_by_patient": selected_mr_by_patient,
                "image_geometry_based": True,
            }
            st.success("Registration setup saved.")




@st.cache_data(show_spinner=False)
def vbv_cached_load_registered_arrays(ct_path, mr_path, dose_paths, mask_paths):
    """
    Load registered CT/MR/dose/mask files safely for Streamlit slider redraws.

    Important:
    - Do NOT store SimpleITK image objects in st.session_state.
    - Do NOT mutate slider session_state values manually.
    - Cache by file path instead.
    """
    if not SIMPLEITK_AVAILABLE:
        raise ImportError("SimpleITK is not installed. Install with: py -m pip install SimpleITK")

    def load_one(path):
        if path is None or str(path) == "None" or str(path).strip() == "":
            return None, None, None

        img = sitk.ReadImage(str(path))
        arr = np.transpose(sitk.GetArrayFromImage(img), (2, 1, 0))
        geom = {
            "Size": tuple(int(x) for x in img.GetSize()),
            "Spacing": tuple(float(x) for x in img.GetSpacing()),
            "Origin": tuple(float(x) for x in img.GetOrigin()),
            "Direction": tuple(float(x) for x in img.GetDirection()),
        }
        return arr, geom, str(path)

    ct_arr, ct_geom, ct_file = load_one(ct_path)
    mr_arr, mr_geom, mr_file = load_one(mr_path)

    reference_arr = ct_arr if ct_arr is not None else mr_arr
    reference_geom = ct_geom if ct_arr is not None else mr_geom

    if reference_arr is None:
        raise ValueError("No CT or MR array could be loaded.")

    dose_items = []
    for path in dose_paths:
        try:
            arr, geom, file_path = load_one(path)
            dose_items.append({
                "name": Path(str(path)).name,
                "path": str(path),
                "array": arr,
                "geometry": geom,
                "shape_ok": arr is not None and arr.shape == reference_arr.shape,
                "error": "",
            })
        except Exception as error:
            dose_items.append({
                "name": Path(str(path)).name,
                "path": str(path),
                "array": None,
                "geometry": None,
                "shape_ok": False,
                "error": str(error),
            })

    mask_items = []
    for path in mask_paths:
        try:
            arr, geom, file_path = load_one(path)
            mask_items.append({
                "name": Path(str(path)).name,
                "path": str(path),
                "array": arr,
                "geometry": geom,
                "shape_ok": arr is not None and arr.shape == reference_arr.shape,
                "error": "",
            })
        except Exception as error:
            mask_items.append({
                "name": Path(str(path)).name,
                "path": str(path),
                "array": None,
                "geometry": None,
                "shape_ok": False,
                "error": str(error),
            })

    return {
        "ct_arr": ct_arr,
        "ct_geom": ct_geom,
        "ct_file": ct_file,
        "mr_arr": mr_arr,
        "mr_geom": mr_geom,
        "mr_file": mr_file,
        "reference_shape": tuple(reference_arr.shape),
        "reference_geom": reference_geom,
        "dose_items": dose_items,
        "mask_items": mask_items,
    }


def vbv_make_display_slice(arr, slice_index):
    """Return display-ready axial slice without modifying widget state."""
    if arr is None:
        return None
    slice_index = max(0, min(int(slice_index), arr.shape[2] - 1))
    return vbv_normalise_slice(arr[:, :, slice_index].T)


def vbv_display_downsample_2d(slice_2d, max_dim=512):
    """Downsample a 2D slice for browser display only."""
    arr = np.asarray(slice_2d)

    if arr.ndim != 2:
        return arr

    h, w = arr.shape
    largest = max(h, w)

    if largest <= max_dim:
        return arr

    step = int(np.ceil(largest / max_dim))
    return arr[::step, ::step]


def vbv_safe_normalised_slice(arr, slice_index, max_dim=512):
    """Extract one axial slice, normalise and downsample for display only."""
    if arr is None:
        return None

    slice_index = max(0, min(int(slice_index), arr.shape[2] - 1))
    sl = vbv_normalise_slice(arr[:, :, slice_index].T)
    return vbv_display_downsample_2d(sl, max_dim=max_dim)


def vbv_safe_raw_slice(arr, slice_index, max_dim=512):
    """Extract one axial raw slice and downsample for display only."""
    if arr is None:
        return None

    slice_index = max(0, min(int(slice_index), arr.shape[2] - 1))
    sl = np.asarray(arr[:, :, slice_index].T)
    return vbv_display_downsample_2d(sl, max_dim=max_dim)


def vbv_crop_to_common_shape(*arrays):
    """Crop non-None 2D arrays to the smallest shared shape."""
    valid = [a for a in arrays if a is not None]
    if not valid:
        return arrays

    min_h = min(a.shape[0] for a in valid)
    min_w = min(a.shape[1] for a in valid)

    cropped = []
    for arr in arrays:
        if arr is None:
            cropped.append(None)
        else:
            cropped.append(arr[:min_h, :min_w])
    return cropped


def vbv_make_rgb(base_slice):
    """Convert a normalised 2D grayscale slice to RGB."""
    base = np.asarray(base_slice, dtype=float)
    base = np.clip(base, 0, 1)
    return np.dstack([base, base, base])


def vbv_overlay_dose_rgb(rgb, dose_slice, alpha=0.35):
    """Overlay dose using a simple red/yellow display without sending large objects."""
    if dose_slice is None:
        return rgb

    dose = np.asarray(dose_slice, dtype=float)
    dose = vbv_normalise_slice(dose)
    mask = dose > 0.02

    out = rgb.copy()
    heat = np.zeros_like(out)
    heat[..., 0] = dose
    heat[..., 1] = np.clip(dose * 0.55, 0, 1)
    heat[..., 2] = 0

    out[mask] = (1 - alpha) * out[mask] + alpha * heat[mask]
    return np.clip(out, 0, 1)


def vbv_overlay_masks_rgb(rgb, mask_slices, alpha=0.85):
    """
    Overlay masks as coloured boundaries on RGB.

    This avoids Matplotlib contour objects and keeps the viewer stable when the
    slice changes. It draws simple mask edges directly into the image.
    """
    if not mask_slices:
        return rgb

    out = rgb.copy()

    palette = [
        np.array([0.0, 1.0, 0.0]),  # green
        np.array([0.0, 0.8, 1.0]),  # cyan
        np.array([1.0, 0.0, 1.0]),  # magenta
        np.array([1.0, 1.0, 0.0]),  # yellow
        np.array([1.0, 0.4, 0.0]),  # orange
        np.array([0.6, 0.2, 1.0]),  # purple
        np.array([1.0, 0.0, 0.0]),  # red
        np.array([0.2, 1.0, 0.4]),  # light green
    ]

    for idx, (_, mask_slice) in enumerate(mask_slices):
        if mask_slice is None:
            continue

        mask = np.asarray(mask_slice) > 0
        if not np.any(mask):
            continue

        # Edge pixels: mask minus a simple eroded interior using neighbouring pixels.
        interior = mask.copy()
        interior[1:, :] &= mask[:-1, :]
        interior[:-1, :] &= mask[1:, :]
        interior[:, 1:] &= mask[:, :-1]
        interior[:, :-1] &= mask[:, 1:]
        edge = mask & (~interior)

        colour = palette[idx % len(palette)]
        out[edge] = (1 - alpha) * out[edge] + alpha * colour

    return np.clip(out, 0, 1)


def vbv_build_overlay_panel(base_slice, dose_slice=None, mask_slices=None, dose_alpha=0.35, contour_alpha=0.85):
    """Build one RGB panel with optional dose and mask boundary overlays."""
    rgb = vbv_make_rgb(base_slice)

    if dose_slice is not None:
        rgb = vbv_overlay_dose_rgb(rgb, dose_slice, alpha=dose_alpha)

    if mask_slices:
        rgb = vbv_overlay_masks_rgb(rgb, mask_slices, alpha=contour_alpha)

    return rgb


def render_prepare_viewer_panel(file_df):
    """
    Lightweight 3-panel Visual QA viewer.

    Layout:
    - CT
    - MR
    - CT/MR blend

    Optional overlays:
    - dose heat overlay
    - mask/contour boundary overlay

    It renders only one downsampled slice at a time, avoiding Plotly frames and
    large browser transfers.
    """
    st.markdown("## 🖥️ Visual QA: registered axial viewer")

    if not SIMPLEITK_AVAILABLE:
        st.error("SimpleITK is required. Install it with: py -m pip install SimpleITK")
        return

    if file_df is None or file_df.empty:
        st.warning("No image files loaded yet.")
        return

    df = vbv_prepare_file_table(file_df)

    patient_ids = sorted([
        str(x) for x in df["Inferred patient ID"].dropna().unique().tolist()
        if str(x).strip() != ""
    ])

    if not patient_ids:
        st.warning("No patient IDs detected.")
        return

    default_patient_index = 0
    previous_patient = st.session_state.get("vbv_qa_patient", "")
    if previous_patient in patient_ids:
        default_patient_index = patient_ids.index(previous_patient)

    selected_patient = st.selectbox(
        "Select patient",
        options=patient_ids,
        index=default_patient_index,
        key="vbv_qa_patient",
    )

    outputs = vbv_find_outputs(selected_patient)

    if not outputs:
        st.warning("No registered outputs found for this patient. Run registration first.")
        return

    ct_candidates = outputs.get("fixed_ct", []) + outputs.get("registered_ct", [])
    mr_candidates = outputs.get("registered_mr", []) + outputs.get("fixed_mr", [])
    dose_candidates = outputs.get("dose", [])
    mask_candidates = outputs.get("masks", [])

    ct_path = str(ct_candidates[0]) if ct_candidates else "None"
    mr_path = str(mr_candidates[0]) if mr_candidates else "None"

    if ct_path == "None" and mr_path == "None":
        st.error("No registered CT or MR was found for this patient.")
        return

    st.caption(f"Registered output folder: {outputs.get('patient_dir', '')}")

    try:
        ct_img = sitk.ReadImage(ct_path) if ct_path != "None" else None
        mr_img = sitk.ReadImage(mr_path) if mr_path != "None" else None

        ct_arr = np.transpose(sitk.GetArrayFromImage(ct_img), (2, 1, 0)) if ct_img is not None else None
        mr_arr = np.transpose(sitk.GetArrayFromImage(mr_img), (2, 1, 0)) if mr_img is not None else None

    except Exception as error:
        st.error(f"Could not read registered CT/MR: {error}")
        return

    reference_img = ct_img if ct_img is not None else mr_img
    reference_arr = ct_arr if ct_arr is not None else mr_arr

    if reference_img is None or reference_arr is None:
        st.error("No displayable registered image was loaded.")
        return

    if ct_arr is not None and mr_arr is not None and ct_arr.shape != mr_arr.shape:
        st.error(
            f"CT and MR are not in the same grid: CT={ct_arr.shape}, MR={mr_arr.shape}. "
            "Run registration again."
        )
        return

    with st.expander("Automatically loaded files", expanded=False):
        loaded_rows = []
        if ct_path != "None":
            loaded_rows.append({"Type": "CT", "File": ct_path})
        if mr_path != "None":
            loaded_rows.append({"Type": "MR", "File": mr_path})
        for p in dose_candidates[:10]:
            loaded_rows.append({"Type": "Dose", "File": str(p)})
        for p in mask_candidates[:25]:
            loaded_rows.append({"Type": "Mask / contour", "File": str(p)})

        if len(dose_candidates) > 10 or len(mask_candidates) > 25:
            loaded_rows.append({
                "Type": "Note",
                "File": f"Showing first 10 dose and first 25 masks only. Full counts: dose={len(dose_candidates)}, masks={len(mask_candidates)}"
            })

        st.dataframe(pd.DataFrame(loaded_rows), use_container_width=True, hide_index=True)

    n_slices = int(reference_arr.shape[2])
    if n_slices <= 0:
        st.error("No axial slices were available.")
        return

    controls_a, controls_b, controls_c = st.columns([1.2, 1.0, 1.0])

    with controls_a:
        slice_index = st.slider(
            "Axial slice",
            min_value=0,
            max_value=n_slices - 1,
            value=min(n_slices // 2, n_slices - 1),
            step=1,
            key=f"vbv_qa_slice_{selected_patient}",
        )

    with controls_b:
        display_max_dim = st.select_slider(
            "Display resolution",
            options=[256, 384, 512, 768],
            value=384,
            key=f"vbv_qa_resolution_{selected_patient}",
            help="Display-only downsampling. Original files are not modified."
        )

    with controls_c:
        blend_weight = st.slider(
            "MR blend weight",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.05,
            key=f"vbv_qa_blend_{selected_patient}",
            disabled=not (ct_arr is not None and mr_arr is not None),
            help="0 = CT only, 1 = MR only."
        )

    st.markdown("### Overlays")

    overlay_col1, overlay_col2 = st.columns(2)

    with overlay_col1:
        show_dose = st.checkbox(
            "Show dose overlay",
            value=bool(dose_candidates),
            disabled=not bool(dose_candidates),
            key=f"vbv_qa_show_dose_{selected_patient}",
        )

        dose_path = None
        dose_alpha = 0.35

        if dose_candidates:
            dose_options = [str(p) for p in dose_candidates[:10]]
            dose_path = st.selectbox(
                "Dose file",
                options=dose_options,
                key=f"vbv_qa_dose_file_{selected_patient}",
            )
            dose_alpha = st.slider(
                "Dose opacity",
                min_value=0.0,
                max_value=1.0,
                value=0.35,
                step=0.05,
                key=f"vbv_qa_dose_alpha_{selected_patient}",
            )

    with overlay_col2:
        show_contours = st.checkbox(
            "Show mask / contour overlay",
            value=bool(mask_candidates),
            disabled=not bool(mask_candidates),
            key=f"vbv_qa_show_contours_{selected_patient}",
        )

        selected_masks = []
        contour_alpha = 0.85

        if mask_candidates:
            mask_options = [str(p) for p in mask_candidates[:25]]
            default_masks = mask_options[: min(3, len(mask_options))]
            selected_masks = st.multiselect(
                "Mask / contour files",
                options=mask_options,
                default=default_masks,
                key=f"vbv_qa_mask_files_{selected_patient}",
                help="Select a small number of masks for readability."
            )
            contour_alpha = st.slider(
                "Contour opacity",
                min_value=0.0,
                max_value=1.0,
                value=0.85,
                step=0.05,
                key=f"vbv_qa_contour_alpha_{selected_patient}",
            )

    ct_slice = vbv_safe_normalised_slice(ct_arr, slice_index, max_dim=display_max_dim) if ct_arr is not None else None
    mr_slice = vbv_safe_normalised_slice(mr_arr, slice_index, max_dim=display_max_dim) if mr_arr is not None else None

    if ct_slice is not None and mr_slice is not None:
        ct_slice, mr_slice = vbv_crop_to_common_shape(ct_slice, mr_slice)
        blend_slice = (1 - blend_weight) * ct_slice + blend_weight * mr_slice
    else:
        blend_slice = ct_slice if ct_slice is not None else mr_slice

    dose_slice = None

    if show_dose and dose_path:
        try:
            dose_img = sitk.ReadImage(dose_path)
            dose_arr = np.transpose(sitk.GetArrayFromImage(dose_img), (2, 1, 0))

            if dose_arr.shape == reference_arr.shape:
                dose_slice = vbv_safe_raw_slice(dose_arr, slice_index, max_dim=display_max_dim)
                dose_slice = vbv_normalise_slice(dose_slice)
            else:
                st.warning(f"Dose grid does not match the base image grid: dose={dose_arr.shape}, reference={reference_arr.shape}")
        except Exception as error:
            st.warning(f"Could not load dose overlay: {error}")

    contour_slices = []

    if show_contours and selected_masks:
        for mask_path in selected_masks[:8]:
            try:
                mask_img = sitk.ReadImage(mask_path)
                mask_arr = np.transpose(sitk.GetArrayFromImage(mask_img), (2, 1, 0))

                if mask_arr.shape == reference_arr.shape:
                    mask_slice = vbv_safe_raw_slice(mask_arr, slice_index, max_dim=display_max_dim)
                    contour_slices.append((Path(mask_path).name, mask_slice))
                else:
                    st.warning(f"Mask grid does not match base image: {Path(mask_path).name}")
            except Exception as error:
                st.warning(f"Could not load contour {Path(mask_path).name}: {error}")

    # Crop dose and masks to match display image size.
    arrays_to_crop = [arr for arr in [ct_slice, mr_slice, blend_slice, dose_slice] if arr is not None]
    for _, mask_slice in contour_slices:
        arrays_to_crop.append(mask_slice)

    if arrays_to_crop:
        min_h = min(arr.shape[0] for arr in arrays_to_crop)
        min_w = min(arr.shape[1] for arr in arrays_to_crop)

        def crop(arr):
            return arr[:min_h, :min_w] if arr is not None else None

        ct_slice = crop(ct_slice)
        mr_slice = crop(mr_slice)
        blend_slice = crop(blend_slice)
        dose_slice = crop(dose_slice)
        contour_slices = [(name, crop(mask_slice)) for name, mask_slice in contour_slices]

    panels = []

    if ct_slice is not None:
        panels.append(("CT", ct_slice))

    if mr_slice is not None:
        panels.append(("MR", mr_slice))

    if blend_slice is not None:
        panels.append(("CT/MR blend", blend_slice))

    if not panels:
        st.error("No displayable panels were available.")
        return

    st.markdown("### Axial view")

    panel_cols = st.columns(3)

    for col, (title, base_slice) in zip(panel_cols, panels):
        with col:
            st.markdown(f"#### {title}")
            panel_rgb = vbv_build_overlay_panel(
                base_slice,
                dose_slice=dose_slice if show_dose else None,
                mask_slices=contour_slices if show_contours else None,
                dose_alpha=dose_alpha,
                contour_alpha=contour_alpha,
            )
            st.image(
                panel_rgb,
                clamp=True,
                use_container_width=True,
                caption=f"{title} | slice {slice_index + 1} of {n_slices}",
            )

    st.caption(
        "Three-panel single-slice viewer. Dose is shown as a heat overlay; masks are shown as coloured boundaries."
    )

    if contour_slices:
        st.markdown("#### Visible contour overlays")
        st.write(", ".join([name for name, _ in contour_slices]))

    geometry_rows = []
    if ct_img is not None:
        geometry_rows.append({"Image": "CT", **vbv_geometry(ct_img), "File": ct_path})
    if mr_img is not None:
        geometry_rows.append({"Image": "MR", **vbv_geometry(mr_img), "File": mr_path})

    with st.expander("Geometry check", expanded=False):
        st.dataframe(pd.DataFrame(geometry_rows), use_container_width=True, hide_index=True)



# ============================================================
# HOME PAGE
# ============================================================

# Show navigation after the Home page has been left.
if st.session_state.page != "home":
    render_step_sidebar()
    render_breadcrumb_path()


def run_clinical_excel_qc(df):
    """
    Run generic clinical Excel QC without assuming variable units.

    The checks avoid hard-coded clinical units such as cc/Gy/cm3. They focus on:
    file structure, missingness, duplicate IDs, duplicate rows, column names,
    variable types, constant variables, high-cardinality categoricals, mixed
    numeric/text values and statistical numeric outliers.
    """
    qc_rows = []

    def add_qc(area, check, status, finding, suggestion=""):
        qc_rows.append({
            "QC area": area,
            "Check": check,
            "Status": status,
            "Finding": finding,
            "Suggestion": suggestion,
        })

    if df is None or df.empty:
        add_qc(
            "File structure",
            "Dataset is not empty",
            "❌ Needs attention",
            "No usable rows or columns were found.",
            "Check that the uploaded Excel sheet contains a patient-level table.",
        )
        return pd.DataFrame(qc_rows), {}, pd.DataFrame(), pd.DataFrame()

    n_rows, n_cols = df.shape

    add_qc(
        "File structure",
        "Dataset loaded",
        "✅ Passed" if n_rows > 0 and n_cols > 0 else "❌ Needs attention",
        f"{n_rows} row(s), {n_cols} column(s) loaded.",
        "",
    )

    # Column name checks.
    raw_columns = list(df.columns)
    blank_cols = [str(c) for c in raw_columns if str(c).strip() == "" or str(c).lower().startswith("unnamed")]
    duplicate_cols = pd.Series([str(c) for c in raw_columns]).value_counts()
    duplicate_cols = duplicate_cols[duplicate_cols > 1]

    add_qc(
        "Column names",
        "Blank or unnamed columns",
        "✅ Passed" if len(blank_cols) == 0 else "⚠️ Warning",
        "No blank/unnamed columns found." if len(blank_cols) == 0 else f"{len(blank_cols)} blank or unnamed column(s) found.",
        "Rename or remove blank/unnamed columns before modelling." if len(blank_cols) else "",
    )

    add_qc(
        "Column names",
        "Duplicate column names",
        "✅ Passed" if duplicate_cols.empty else "❌ Needs attention",
        "No duplicate column names found." if duplicate_cols.empty else f"Duplicate names: {', '.join(duplicate_cols.index.tolist()[:10])}",
        "Duplicate column names can break variable selection. Rename duplicates." if not duplicate_cols.empty else "",
    )

    long_cols = [str(c) for c in raw_columns if len(str(c)) > 80]
    add_qc(
        "Column names",
        "Very long column names",
        "✅ Passed" if len(long_cols) == 0 else "⚠️ Warning",
        "No very long column names found." if len(long_cols) == 0 else f"{len(long_cols)} column name(s) exceed 80 characters.",
        "Consider shortening very long variable names for clearer plots and exports." if len(long_cols) else "",
    )

    # Missingness.
    total_cells = max(n_rows * n_cols, 1)
    overall_missing = float(df.isna().sum().sum()) / total_cells * 100
    high_missing = df.isna().mean().sort_values(ascending=False)
    high_missing_80 = high_missing[high_missing >= 0.80]
    high_missing_50 = high_missing[(high_missing >= 0.50) & (high_missing < 0.80)]

    add_qc(
        "Missing data",
        "Overall missingness",
        "✅ Passed" if overall_missing < 20 else "⚠️ Warning",
        f"Overall missingness is {overall_missing:.1f}%.",
        "Review missingness before modelling; high missingness may require imputation or variable exclusion." if overall_missing >= 20 else "",
    )

    add_qc(
        "Missing data",
        "Columns with ≥80% missing values",
        "✅ Passed" if high_missing_80.empty else "⚠️ Warning",
        "No columns have ≥80% missing values." if high_missing_80.empty else f"{len(high_missing_80)} column(s) have ≥80% missing values.",
        "Consider excluding or reviewing these variables." if not high_missing_80.empty else "",
    )

    add_qc(
        "Missing data",
        "Columns with 50–79% missing values",
        "✅ Passed" if high_missing_50.empty else "⚠️ Warning",
        "No columns have 50–79% missing values." if high_missing_50.empty else f"{len(high_missing_50)} column(s) have 50–79% missing values.",
        "Check whether these variables are required and whether missingness is expected." if not high_missing_50.empty else "",
    )

    # Duplicate rows.
    duplicate_rows = int(df.duplicated().sum())
    add_qc(
        "Records",
        "Duplicate full rows",
        "✅ Passed" if duplicate_rows == 0 else "⚠️ Warning",
        "No duplicate full rows found." if duplicate_rows == 0 else f"{duplicate_rows} duplicate full row(s) found.",
        "Review duplicates to avoid double-counting patients." if duplicate_rows else "",
    )

    # Patient ID heuristic.
    id_keywords = ["patient", "pt", "subject", "study_id", "study id", "id", "mrn", "record"]
    likely_id_cols = []
    for col in df.columns:
        lower = str(col).strip().lower()
        if any(k in lower for k in id_keywords):
            likely_id_cols.append(col)

    best_id_col = None
    if likely_id_cols:
        # Prefer columns with high uniqueness.
        scores = []
        for col in likely_id_cols:
            non_missing = df[col].dropna()
            unique_ratio = non_missing.nunique() / max(len(non_missing), 1)
            scores.append((unique_ratio, str(col), col))
        scores.sort(reverse=True)
        best_id_col = scores[0][2]

    if best_id_col is None:
        add_qc(
            "Patient ID",
            "Likely patient ID column",
            "⚠️ Warning",
            "No likely patient ID column was detected from the column names.",
            "This is not fatal, but a patient ID column is useful for merging clinical, imaging and outcome data.",
        )
    else:
        id_missing = int(df[best_id_col].isna().sum())
        id_duplicates = int(df[best_id_col].duplicated().sum() - df[best_id_col].isna().sum() + df[best_id_col].isna().duplicated().sum()) if False else int(df[best_id_col].dropna().duplicated().sum())
        status = "✅ Passed" if id_missing == 0 and id_duplicates == 0 else "⚠️ Warning"
        add_qc(
            "Patient ID",
            f"Likely patient ID column: {best_id_col}",
            status,
            f"Missing IDs: {id_missing}; duplicated non-missing IDs: {id_duplicates}.",
            "Resolve missing or duplicate patient IDs if this column will be used for linking." if status != "✅ Passed" else "",
        )

    # Variable types.
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    date_like_cols = []
    categorical_cols = []
    text_cols = []

    for col in df.columns:
        if col in numeric_cols:
            continue
        series = df[col].dropna()
        if series.empty:
            categorical_cols.append(col)
            continue
        parsed_dates = pd.to_datetime(series, errors="coerce")
        date_ratio = parsed_dates.notna().mean() if len(parsed_dates) else 0
        unique_ratio = series.astype(str).nunique() / max(len(series), 1)
        if date_ratio >= 0.80:
            date_like_cols.append(col)
        elif unique_ratio <= 0.20 or series.astype(str).nunique() <= 20:
            categorical_cols.append(col)
        else:
            text_cols.append(col)

    add_qc(
        "Variable types",
        "Detected variable types",
        "✅ Passed",
        f"Numeric: {len(numeric_cols)}; categorical: {len(categorical_cols)}; date-like: {len(date_like_cols)}; free-text/high-cardinality: {len(text_cols)}.",
        "",
    )

    # Constant variables.
    unique_counts = df.nunique(dropna=True)
    constant_cols = unique_counts[unique_counts <= 1].index.tolist()
    add_qc(
        "Modelling readiness",
        "Constant variables",
        "✅ Passed" if len(constant_cols) == 0 else "⚠️ Warning",
        "No constant variables found." if len(constant_cols) == 0 else f"{len(constant_cols)} variable(s) have only one unique non-missing value.",
        "Constant variables do not help regression or machine learning and can be excluded." if constant_cols else "",
    )

    # High cardinality categorical/text.
    high_cardinality_cols = []
    for col in df.columns:
        if col not in numeric_cols:
            nunique = df[col].dropna().astype(str).nunique()
            if nunique > 30:
                high_cardinality_cols.append((col, nunique))

    add_qc(
        "Categorical variables",
        "High-cardinality non-numeric variables",
        "✅ Passed" if len(high_cardinality_cols) == 0 else "⚠️ Warning",
        "No high-cardinality non-numeric variables found." if len(high_cardinality_cols) == 0 else f"{len(high_cardinality_cols)} non-numeric variable(s) have >30 unique values.",
        "These may need grouping, exclusion, or special handling before modelling." if high_cardinality_cols else "",
    )

    # Numeric-looking text and mixed values.
    numeric_like_text = []
    mixed_numeric_text = []
    for col in df.columns:
        if col in numeric_cols:
            continue
        series = df[col].dropna().astype(str).str.strip()
        if series.empty:
            continue
        converted = pd.to_numeric(series, errors="coerce")
        numeric_ratio = converted.notna().mean()
        if numeric_ratio >= 0.90:
            numeric_like_text.append(col)
        elif 0.10 <= numeric_ratio < 0.90:
            mixed_numeric_text.append(col)

    add_qc(
        "Numeric sanity",
        "Numeric-looking columns stored as text",
        "✅ Passed" if len(numeric_like_text) == 0 else "⚠️ Warning",
        "No numeric-looking text columns found." if len(numeric_like_text) == 0 else f"{len(numeric_like_text)} column(s) look numeric but are stored as text.",
        "Convert these to numeric if they are intended to be model predictors." if numeric_like_text else "",
    )

    add_qc(
        "Numeric sanity",
        "Mixed numeric/text columns",
        "✅ Passed" if len(mixed_numeric_text) == 0 else "⚠️ Warning",
        "No mixed numeric/text columns found." if len(mixed_numeric_text) == 0 else f"{len(mixed_numeric_text)} column(s) contain mixed numeric and non-numeric values.",
        "Check coding such as '<5', 'unknown', 'not done', or combined values." if mixed_numeric_text else "",
    )

    # Generic numeric checks, no unit assumptions.
    numeric_summary_rows = []
    outlier_cols = []
    negative_cols = []
    low_usable_numeric = []

    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        usable = int(s.notna().sum())
        missing_pct = float(s.isna().mean() * 100)
        nunique = int(s.nunique(dropna=True))

        if usable < max(5, int(0.10 * n_rows)):
            low_usable_numeric.append(col)

        if usable > 0 and (s.dropna() < 0).any():
            negative_cols.append(col)

        outlier_count = 0
        if usable >= 8:
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            if pd.notna(iqr) and iqr > 0:
                lower = q1 - 3 * iqr
                upper = q3 + 3 * iqr
                outlier_count = int(((s < lower) | (s > upper)).sum())
                if outlier_count > 0:
                    outlier_cols.append((col, outlier_count))

        numeric_summary_rows.append({
            "Variable": col,
            "Usable values": usable,
            "Missing %": round(missing_pct, 1),
            "Unique values": nunique,
            "Min": s.min(skipna=True),
            "Median": s.median(skipna=True),
            "Max": s.max(skipna=True),
            "Potential outliers": outlier_count,
            "Negative values present": bool(usable > 0 and (s.dropna() < 0).any()),
        })

    add_qc(
        "Numeric sanity",
        "Potential statistical outliers",
        "✅ Passed" if len(outlier_cols) == 0 else "⚠️ Warning",
        "No potential numeric outliers detected by a generic IQR rule." if len(outlier_cols) == 0 else f"{len(outlier_cols)} numeric column(s) have potential outliers.",
        "Review outliers in context. No unit-specific assumptions were applied." if outlier_cols else "",
    )

    add_qc(
        "Numeric sanity",
        "Negative values present",
        "✅ Passed" if len(negative_cols) == 0 else "⚠️ Warning",
        "No negative values found in numeric columns." if len(negative_cols) == 0 else f"{len(negative_cols)} numeric column(s) contain negative values.",
        "Negative values may be valid for some variables; review in context." if negative_cols else "",
    )

    add_qc(
        "Numeric sanity",
        "Low usable numeric sample size",
        "✅ Passed" if len(low_usable_numeric) == 0 else "⚠️ Warning",
        "All numeric columns have a reasonable number of usable values." if len(low_usable_numeric) == 0 else f"{len(low_usable_numeric)} numeric column(s) have very few usable values.",
        "Variables with very few observed values may not be useful for modelling." if low_usable_numeric else "",
    )

    # Outcome/treatment readiness using broad naming only.
    outcome_keywords = ["outcome", "decline", "toxicity", "response", "event", "progression", "survival", "death", "status", "follow", "followup", "follow-up", "score"]
    treatment_keywords = ["treatment", "proton", "photon", "arm", "group", "cohort", "therapy", "modality"]

    likely_outcome_cols = [c for c in df.columns if any(k in str(c).lower() for k in outcome_keywords)]
    likely_treatment_cols = [c for c in df.columns if any(k in str(c).lower() for k in treatment_keywords)]

    add_qc(
        "Modelling readiness",
        "Possible outcome/follow-up variables",
        "✅ Passed" if len(likely_outcome_cols) > 0 else "⚠️ Warning",
        f"{len(likely_outcome_cols)} possible outcome/follow-up variable(s) detected by name." if likely_outcome_cols else "No obvious outcome/follow-up variable detected by name.",
        "This is only a naming heuristic. You will select the actual variables in the next step." if not likely_outcome_cols else "",
    )

    add_qc(
        "Modelling readiness",
        "Possible treatment/group variables",
        "✅ Passed" if len(likely_treatment_cols) > 0 else "⚠️ Warning",
        f"{len(likely_treatment_cols)} possible treatment/group variable(s) detected by name." if likely_treatment_cols else "No obvious treatment/group variable detected by name.",
        "This is only a naming heuristic. You can select the treatment/group variable later." if not likely_treatment_cols else "",
    )

    # Build summary.
    qc_df = pd.DataFrame(qc_rows)
    status_counts = qc_df["Status"].value_counts().to_dict() if not qc_df.empty else {}
    summary = {
        "Rows": n_rows,
        "Columns": n_cols,
        "Overall missing %": round(overall_missing, 1),
        "Likely patient ID": str(best_id_col) if best_id_col is not None else "Not detected",
        "Numeric columns": len(numeric_cols),
        "Categorical columns": len(categorical_cols),
        "Date-like columns": len(date_like_cols),
        "Free-text/high-cardinality columns": len(text_cols),
        "Passed checks": int(status_counts.get("✅ Passed", 0)),
        "Warnings": int(status_counts.get("⚠️ Warning", 0)),
        "Needs attention": int(status_counts.get("❌ Needs attention", 0)),
    }

    # Tables for optional detailed views.
    missing_table = pd.DataFrame({
        "Variable": high_missing.index.astype(str),
        "Missing %": (high_missing.values * 100).round(1),
        "Non-missing count": [int(df[c].notna().sum()) for c in high_missing.index],
    })

    numeric_summary = pd.DataFrame(numeric_summary_rows)

    return qc_df, summary, missing_table, numeric_summary


def render_clinical_excel_qc(df):
    """Display clinical upload QC results."""
    qc_df, summary, missing_table, numeric_summary = run_clinical_excel_qc(df)

    st.subheader("Clinical data QC")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", summary.get("Rows", 0))
    m2.metric("Columns", summary.get("Columns", 0))
    m3.metric("Missing", f"{summary.get('Overall missing %', 0)}%")
    m4.metric("Patient ID", summary.get("Likely patient ID", "Not detected"))

    c1, c2, c3 = st.columns(3)
    c1.metric("Passed", summary.get("Passed checks", 0))
    c2.metric("Warnings", summary.get("Warnings", 0))
    c3.metric("Needs attention", summary.get("Needs attention", 0))

    st.dataframe(qc_df, use_container_width=True)

    with st.expander("Missingness details", expanded=False):
        st.dataframe(missing_table, use_container_width=True)

    with st.expander("Numeric variable summary", expanded=False):
        if numeric_summary.empty:
            st.info("No numeric columns detected.")
        else:
            st.dataframe(numeric_summary, use_container_width=True)

    saved_folder = st.session_state.get("clinical_upload_saved_folder", "")
    if saved_folder:
        st.caption(f"QC report saved automatically in: {saved_folder}")





def save_step5_calculated_model(
    method,
    pipeline,
    predictors,
    outcome_variable,
    metrics,
    model_settings,
    model_results,
):
    """
    Single source of truth after Step 5 model calculation.

    The standalone risk calculator reads ONLY from:
        st.session_state.step5_calculated_model

    This avoids using stale Step 6 / Step 7 / trained_pipeline routing state.
    """
    model_payload = {
        "method": method,
        "pipeline": pipeline,
        "predictors": list(predictors),
        "outcome_variable": outcome_variable,
        "metrics": metrics,
        "model_settings": model_settings,
        "model_results": model_results,
    }

    st.session_state.step5_calculated_model = model_payload
    st.session_state.step5_calculated_model_name = method

    # Keep old keys populated only for backwards compatibility elsewhere.
    st.session_state.trained_pipeline = pipeline
    st.session_state.trained_predictors = list(predictors)
    st.session_state.trained_model_name = method
    st.session_state.trained_outcome_variable = outcome_variable

    if "trained_models" not in st.session_state:
        st.session_state.trained_models = {}
    st.session_state.trained_models[method] = model_payload

    return model_payload


def render_step5_risk_calculator_only():
    """
    Dedicated generated risk-calculator page.

    This page is opened after Step 5 model calculation and shows only:
    - predictive variables from the stored model
    - patient-specific input fields
    - risk calculation button
    - predicted risk
    - after risk calculation: select another algorithm or export model
    """
    model_payload = st.session_state.get("step5_calculated_model", None)

    if model_payload is None:
        st.error("No calculated model is available. Please calculate the model first.")
        return

    df = st.session_state.get("df", None)
    if df is None:
        st.error("No clinical dataset is loaded.")
        return

    pipeline = model_payload.get("pipeline", None)
    predictors = model_payload.get("predictors", [])
    method = model_payload.get("method", "Model")
    outcome_variable = model_payload.get("outcome_variable", "selected outcome")

    if pipeline is None:
        st.error("The fitted model pipeline was not stored. Please calculate the model again.")
        return

    if len(predictors) == 0:
        st.error("No predictive variables were stored for this model. Please calculate the model again.")
        return

    st.header("Risk calculator")
    st.caption(f"Generated from the calculated {method} model")

    st.markdown("### Predictive variables")
    predictor_table = pd.DataFrame({"Predictive variable": predictors})
    st.dataframe(predictor_table, use_container_width=True, hide_index=True)

    st.markdown("### Enter patient-specific values")

    patient_data = {}

    for predictor in predictors:
        if predictor not in df.columns:
            st.warning(f"Predictor '{predictor}' is missing from the loaded dataset and cannot be entered.")
            continue

        if is_numeric_column(df, predictor):
            values = pd.to_numeric(df[predictor], errors="coerce")
            median_value = values.median()
            default_value = float(median_value) if not pd.isna(median_value) else 0.0

            patient_data[predictor] = st.number_input(
                predictor,
                value=default_value,
                key=f"generated_calc_input_{method}_{predictor}",
            )
        else:
            options = (
                df[predictor]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )
            options = sorted([x for x in options if x != ""])

            if len(options) == 0:
                options = [""]

            patient_data[predictor] = st.selectbox(
                predictor,
                options=options,
                key=f"generated_calc_input_{method}_{predictor}",
            )

    st.divider()

    if st.button("Calculate risk", type="primary", use_container_width=True, key=f"calculate_generated_risk_{method}"):
        if len(patient_data) == 0:
            st.error("No patient values were entered.")
            return

        new_patient_df = pd.DataFrame([patient_data])

        try:
            risk = pipeline.predict_proba(new_patient_df)[0, 1]

            st.session_state.last_risk_prediction = {
                "model": method,
                "outcome": outcome_variable,
                "risk": float(risk),
                "patient_values": patient_data,
            }

        except Exception as error:
            st.error(f"Risk calculation failed: {error}")

    last_prediction = st.session_state.get("last_risk_prediction", None)

    if last_prediction is not None:
        st.metric("Predicted risk", f"{last_prediction.get('risk', 0) * 100:.1f}%")

        risk_value = float(last_prediction.get("risk", 0))
        if risk_value < 0.20:
            st.success("Low predicted risk")
        elif risk_value < 0.50:
            st.warning("Moderate predicted risk")
        else:
            st.error("High predicted risk")

        st.divider()
        st.markdown("### Next options")

        col_alg, col_export = st.columns(2)

        with col_alg:
            if st.button(
                "Select another machine learning algorithm",
                use_container_width=True,
                key=f"{method}_select_another_algorithm_after_risk"
            ):
                st.session_state.page = "clinical_model_selection"
                st.rerun()

        with col_export:
            if st.button(
                "Export model",
                type="primary",
                use_container_width=True,
                key=f"{method}_export_model_after_risk"
            ):
                model_payload = st.session_state.get("step5_calculated_model", {})
                export_method = model_payload.get("method", method)
                export_outcome = model_payload.get("outcome_variable", "selected outcome")
                suggested_name = f"{export_method} - {export_outcome}"

                st.session_state.export_model_source_signature = f"{export_method}::{export_outcome}"
                st.session_state.export_model_name_input = suggested_name
                st.session_state.page = "model_export_page"
                st.rerun()

def render_model_export_page():
    """
    Export page for the Step 5 calculated model.

    User enters model name; page automatically shows variables, parameters, and results.
    Provides:
    - button back to risk calculator
    - download export package
    - export to Established Model library for searchable use
    """
    model_payload = st.session_state.get("step5_calculated_model", None)

    if model_payload is None:
        st.error("No calculated model is available. Please calculate a model first.")
        return

    method = model_payload.get("method", "Model")
    outcome_variable = model_payload.get("outcome_variable", "selected outcome")
    predictors = model_payload.get("predictors", [])
    metrics = model_payload.get("metrics", {})
    model_settings = model_payload.get("model_settings", {})
    model_results = model_payload.get("model_results", pd.DataFrame())

    st.header("Export model")
    st.caption("Prepare the calculated model for download or for the searchable Established Model library.")

    suggested_model_name = clinical_default_established_model_name(method, outcome_variable)
    source_signature = f"{method}::{outcome_variable}::{st.session_state.get('clinical_project_folder', '')}"

    # Reset the editable export name when a different model/outcome is being exported.
    if st.session_state.get("export_model_source_signature", "") != source_signature:
        st.session_state.export_model_source_signature = source_signature
        st.session_state.export_model_name_input = suggested_model_name

    st.markdown("### Rename model before export")
    model_name = st.text_input(
        "Model name to save/export",
        key="export_model_name_input",
        help="Edit this name before downloading or exporting to the Established Model library."
    ).strip()

    if model_name == "":
        st.error("Please enter a model name before export.")
        st.stop()

    existing_models = st.session_state.get("established_calculators", {})
    duplicate_action = "No duplicate"
    final_model_name = model_name
    if model_name in existing_models:
        st.warning("A model with this name already exists in Established Model.")
        duplicate_action = st.radio(
            "Duplicate model name action",
            options=["Rename before export", "Overwrite existing model"],
            horizontal=True,
            key="export_duplicate_action",
        )
        if duplicate_action == "Rename before export":
            final_model_name = st.text_input(
                "New model name",
                value=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}",
                key="export_duplicate_new_name",
            ).strip()
            if final_model_name in existing_models:
                st.error("The new model name still exists. Please choose a different name or select overwrite.")
                st.stop()
    else:
        final_model_name = model_name

    st.markdown("### Model summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Method", method)
    c2.metric("Outcome", outcome_variable)
    c3.metric("Predictors", len(predictors))

    st.markdown("### Variables used")
    st.dataframe(
        pd.DataFrame({"Predictive variable": predictors}),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Parameters used")
    if len(model_settings) > 0:
        settings_df = pd.DataFrame([
            {"Parameter": key, "Value": value}
            for key, value in model_settings.items()
        ])
        st.dataframe(settings_df, use_container_width=True, hide_index=True)
    else:
        st.info("No custom model parameters were stored.")

    st.markdown("### Results")
    if len(metrics) > 0:
        metrics_df = pd.DataFrame([
            {"Metric": key, "Value": value}
            for key, value in metrics.items()
        ])
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    if isinstance(model_results, pd.DataFrame) and not model_results.empty:
        st.markdown("### Model output table")
        st.dataframe(model_results, use_container_width=True)

    export_payload_for_library = dict(model_payload)
    export_payload_for_library["method"] = method
    export_payload_for_library["outcome_variable"] = outcome_variable
    export_payload_for_library["predictors"] = predictors
    export_payload_for_library["metrics"] = metrics
    export_payload_for_library["model_settings"] = model_settings
    export_payload_for_library["model_results"] = model_results
    export_payload_for_library["pipeline"] = model_payload.get("pipeline")
    export_info = clinical_build_established_export_info(final_model_name, export_payload_for_library)
    st.divider()

    if st.button("Open risk calculator", use_container_width=True, key="export_page_open_risk_calculator"):
        st.session_state.page = "generated_risk_calculator_page"
        st.rerun()

    st.download_button(
        f"Download model export file: {final_model_name}",
        data=pickle.dumps(build_calculator_export_package(export_info, final_model_name)),
        file_name=f"{make_safe_column_name(final_model_name)}_calculator_export.pkl",
        mime="application/octet-stream",
        use_container_width=True,
    )

    if st.button(
        f"Export to Established Model as: {final_model_name}",
        type="primary",
        use_container_width=True,
        key="export_to_established_model_library"
    ):
        if "established_calculators" not in st.session_state:
            st.session_state.established_calculators = {}

        # Store the model under the final export name and keep the name/date inside the payload too.
        export_info["model_label"] = final_model_name
        export_info["exported_at"] = clinical_export_timestamp()
        export_info["exported_date"] = export_info["exported_at"].split(" ")[0]
        st.session_state.established_calculators[final_model_name] = export_info
        save_established_calculators_persistent()
        # Make the Established Model page show the exported model immediately.
        st.session_state.established_search_has_run = True
        st.session_state.est_search_name = ""
        st.session_state.est_search_site = "All"
        st.session_state.est_search_outcome = "All"
        st.session_state.est_search_method = "All"
        st.session_state.est_result_sort_order = "Descending results"
        st.session_state.established_last_exported_model = final_model_name
        save_established_model_workflow_state_persistent()

        st.success(f"Exported '{final_model_name}' to Established Model.")
        st.session_state.page = "established_model"
        st.rerun()


# ============================================================
# VOXEL NORMALISATION HELPERS
# ============================================================

def vbv_normalisation_project_folder():
    """Return the project folder used for normalisation outputs."""
    project_folder = st.session_state.get("voxel_project_folder", "")
    if project_folder:
        out = Path(project_folder) / "03_Normalisation"
    else:
        out = Path("data") / "voxel_normalisation"
    out.mkdir(parents=True, exist_ok=True)
    return out


def vbv_role_interpolator(role, image_kind="Intensity"):
    """Choose interpolation safely based on data type.

    Masks/segmentations/structures must always use nearest-neighbour so label
    values stay discrete after resampling. This intentionally recognises several
    possible role names because the image QC page may save masks as Mask,
    RTSTRUCT / structure, Structure, ROI, Contour, OAR, or Label.
    """
    role_text = str(role).lower()
    kind_text = str(image_kind).lower()
    mask_terms = [
        "mask", "structure", "rtstruct", "rt structure", "rtss",
        "seg", "segmentation", "roi", "contour", "label", "oar"
    ]
    if any(term in role_text for term in mask_terms) or kind_text == "mask/segmentation":
        return sitk.sitkNearestNeighbor, "Nearest-neighbour"
    if kind_text == "b-spline intensity":
        return sitk.sitkBSpline, "B-spline"
    return sitk.sitkLinear, "Linear"


def vbv_format_tuple(values, decimals=3):
    try:
        out = []
        for v in values:
            if decimals == 0:
                out.append(str(int(round(float(v)))))
            else:
                out.append(f"{float(v):.{decimals}f}")
        return " × ".join(out)
    except Exception:
        return ""

def vbv_pixel_type_text(sitk_object):
    """Return pixel type text from a SimpleITK Image or ImageFileReader safely.

    Some SimpleITK versions do not expose GetPixelIDTypeAsString() on
    ImageFileReader after ReadImageInformation(). In that case, use the
    reader pixel ID enum and convert it with sitk.GetPixelIDValueAsString().
    """
    try:
        if hasattr(sitk_object, "GetPixelIDTypeAsString"):
            return str(sitk_object.GetPixelIDTypeAsString())
    except Exception:
        pass
    try:
        if hasattr(sitk_object, "GetPixelID") and hasattr(sitk, "GetPixelIDValueAsString"):
            return str(sitk.GetPixelIDValueAsString(sitk_object.GetPixelID()))
    except Exception:
        pass
    try:
        if hasattr(sitk_object, "GetPixelID"):
            return f"PixelID {sitk_object.GetPixelID()}"
    except Exception:
        pass
    return "Unknown"



def vbv_normalisation_items(file_df):
    """Build one normalisation item per NIfTI file or DICOM directory/series."""
    df = vbv_prepare_file_table(file_df)
    if df is None or df.empty:
        return []

    items = []
    dicom_groups = {}
    for idx, row in df.iterrows():
        full_path = str(row.get("Full path", ""))
        if not full_path:
            continue
        path = Path(full_path)
        role = row.get("VBA role", "Other")
        patient_id = str(row.get("Inferred patient ID", "Unknown"))
        rel = row.get("Relative path", row.get("Filename", path.name))
        fmt = str(row.get("File format", "")).lower()
        filename = path.name.lower()
        is_nifti = fmt == "nifti" or filename.endswith(".nii") or filename.endswith(".nii.gz")
        if is_nifti:
            items.append({
                "item_type": "NIfTI file",
                "patient_id": patient_id,
                "role": role,
                "path": str(path),
                "display_name": str(rel),
                "source_path": str(path),
                "source_filename": path.name,
                "source_count": 1,
                "file_table_row": int(idx),
            })
        else:
            key = (patient_id, str(role), str(path.parent))
            if key not in dicom_groups:
                dicom_groups[key] = {
                    "item_type": "DICOM directory",
                    "patient_id": patient_id,
                    "role": role,
                    "path": str(path.parent),
                    "display_name": str(path.parent),
                    "source_path": str(path.parent),
                    "source_filename": path.parent.name,
                    "source_count": 0,
                    "file_table_row": int(idx),
                }
            dicom_groups[key]["source_count"] += 1

    items.extend(dicom_groups.values())
    return items


def vbv_read_normalisation_item(item):
    """Read one normalisation item using SimpleITK."""
    path = Path(item.get("path", ""))
    if item.get("item_type") == "NIfTI file":
        return sitk.ReadImage(str(path))

    # DICOM directory: read the largest series in that folder where possible.
    try:
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(path))
        if series_ids:
            best_files = []
            for sid in series_ids:
                files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(path), sid)
                if len(files) > len(best_files):
                    best_files = files
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(best_files)
            return reader.Execute()
    except Exception:
        pass

    # Fallback: try reading the first DICOM file in the directory.
    files = sorted([x for x in path.iterdir() if x.is_file()])
    if not files:
        raise FileNotFoundError(f"No readable files found in {path}")
    return sitk.ReadImage(str(files[0]))


def vbv_image_metadata_row(item, read_image=True):
    """Return one metadata row for an image item."""
    row = {
        "Patient ID": item.get("patient_id", ""),
        "Role": item.get("role", ""),
        "Item type": item.get("item_type", ""),
        "Source": item.get("display_name", item.get("path", "")),
        "Source path": item.get("path", ""),
        "Source file count": item.get("source_count", 1),
        "Status": "Not read",
        "Dimension": "",
        "Size": "",
        "Voxel spacing": "",
        "Origin": "",
        "Direction": "",
        "Pixel type": "",
        "Notes": "",
    }
    if not read_image:
        return row
    try:
        img = vbv_read_normalisation_item(item)
        row.update({
            "Status": "Read",
            "Dimension": int(img.GetDimension()),
            "Size": vbv_format_tuple(img.GetSize(), 0),
            "Voxel spacing": vbv_format_tuple(img.GetSpacing(), 3),
            "Origin": vbv_format_tuple(img.GetOrigin(), 3),
            "Direction": vbv_format_tuple(img.GetDirection(), 3),
            "Pixel type": vbv_pixel_type_text(img),
            "Notes": "Geometry read successfully",
        })
    except Exception as error:
        row["Status"] = "Failed"
        row["Notes"] = str(error)
    return row


def vbv_collect_metadata_table(file_df, limit=None):
    """Collect metadata for normalisation items."""
    items = vbv_normalisation_items(file_df)
    if limit is not None:
        items = items[:int(limit)]
    rows = [vbv_image_metadata_row(item, read_image=True) for item in items]
    return pd.DataFrame(rows)



def vbv_image_metadata_row_fast(item):
    """Return a lightweight metadata row without loading full voxel data where possible."""
    row = {
        "Patient ID": item.get("patient_id", ""),
        "Role": item.get("role", ""),
        "Item type": item.get("item_type", ""),
        "Source": item.get("display_name", item.get("path", "")),
        "Source path": item.get("path", ""),
        "Source file count": item.get("source_count", 1),
        "Status": "Not read",
        "Dimension": "",
        "Size": "",
        "Voxel spacing": "",
        "Origin": "",
        "Direction": "",
        "Pixel type": "",
        "Read method": "",
        "Notes": "",
    }

    path = Path(item.get("path", ""))
    try:
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        # NIfTI and other single-file images: read header only, not pixel data.
        if path.is_file():
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(path))
            reader.ReadImageInformation()
            row.update({
                "Status": "Read",
                "Dimension": int(reader.GetDimension()),
                "Size": vbv_format_tuple(reader.GetSize(), 0),
                "Voxel spacing": vbv_format_tuple(reader.GetSpacing(), 3),
                "Origin": vbv_format_tuple(reader.GetOrigin(), 3),
                "Direction": vbv_format_tuple(reader.GetDirection(), 3),
                "Pixel type": vbv_pixel_type_text(reader),
                "Read method": "Header only",
                "Notes": "Header/geometry read without loading image volume",
            })
            return row

        # DICOM folder: read tags from representative slices only. This is much faster than loading the series.
        dicom_files = []
        for candidate in path.rglob("*"):
            if candidate.is_file():
                dicom_files.append(candidate)
        if not dicom_files:
            raise FileNotFoundError(f"No files found in directory: {path}")

        series_files = []
        try:
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(path)) or []
            if series_ids:
                for sid in series_ids:
                    files = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(path), sid))
                    if len(files) > len(series_files):
                        series_files = files
        except Exception:
            series_files = []

        if not series_files:
            series_files = [str(x) for x in sorted(dicom_files)]

        first_file = series_files[0]
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(first_file))
        reader.ReadImageInformation()

        x_size = reader.GetSize()[0] if reader.GetDimension() >= 1 else ""
        y_size = reader.GetSize()[1] if reader.GetDimension() >= 2 else ""
        x_spacing = reader.GetSpacing()[0] if reader.GetDimension() >= 1 else ""
        y_spacing = reader.GetSpacing()[1] if reader.GetDimension() >= 2 else ""
        z_size = len(series_files)
        z_spacing = ""

        # Prefer DICOM tags for slice spacing if pydicom is available.
        if PYDICOM_AVAILABLE:
            try:
                ds0 = pydicom.dcmread(first_file, stop_before_pixels=True, force=True)
                rows = getattr(ds0, "Rows", y_size)
                cols = getattr(ds0, "Columns", x_size)
                x_size, y_size = int(cols), int(rows)
                px = getattr(ds0, "PixelSpacing", None)
                if px is not None and len(px) >= 2:
                    y_spacing = float(px[0])
                    x_spacing = float(px[1])
                z_spacing = float(getattr(ds0, "SpacingBetweenSlices", getattr(ds0, "SliceThickness", 0)) or 0)

                if len(series_files) > 1:
                    try:
                        ds1 = pydicom.dcmread(series_files[1], stop_before_pixels=True, force=True)
                        pos0 = getattr(ds0, "ImagePositionPatient", None)
                        pos1 = getattr(ds1, "ImagePositionPatient", None)
                        if pos0 is not None and pos1 is not None:
                            z_spacing = float(np.linalg.norm(np.array(pos1, dtype=float) - np.array(pos0, dtype=float)))
                    except Exception:
                        pass
            except Exception:
                pass

        spacing = [x_spacing, y_spacing]
        size = [x_size, y_size]
        if z_size:
            size.append(z_size)
            spacing.append(z_spacing if z_spacing != "" else "unknown")

        row.update({
            "Status": "Read",
            "Dimension": 3 if z_size and z_size > 1 else int(reader.GetDimension()),
            "Size": " × ".join(str(x) for x in size),
            "Voxel spacing": " × ".join(str(round(float(x), 3)) if isinstance(x, (int, float, np.floating)) else str(x) for x in spacing),
            "Origin": vbv_format_tuple(reader.GetOrigin(), 3),
            "Direction": vbv_format_tuple(reader.GetDirection(), 3),
            "Pixel type": vbv_pixel_type_text(reader),
            "Read method": "DICOM tags/header only",
            "Notes": f"Read representative DICOM metadata from {len(series_files)} file(s); volume pixels were not loaded",
        })
        return row

    except Exception as error:
        row["Status"] = "Failed"
        row["Read method"] = "Header/tag read failed"
        row["Notes"] = str(error)
        return row


def vbv_collect_metadata_table_fast(file_df, progress_bar=None, status_box=None, counter_box=None):
    """Collect lightweight metadata with optional Streamlit progress feedback."""
    items = vbv_normalisation_items(file_df)
    total = len(items)
    rows = []
    read_count = 0
    failed_count = 0

    for index, item in enumerate(items, start=1):
        if status_box is not None:
            status_box.write(f"Reading {index} of {total}: {item.get('display_name', item.get('path', ''))}")

        row = vbv_image_metadata_row_fast(item)
        rows.append(row)

        if str(row.get("Status", "")).lower() == "read":
            read_count += 1
        else:
            failed_count += 1

        if progress_bar is not None:
            progress_bar.progress(index / total if total else 1.0)
        if counter_box is not None:
            counter_box.info(f"Read: {read_count} / {total} | Failed: {failed_count} | Remaining: {max(total - index, 0)}")

    return pd.DataFrame(rows)


def vbv_resample_to_spacing(img, target_spacing, interpolator, default_value=0.0):
    """Reorient to LPS and resample to common voxel spacing."""
    oriented = sitk.DICOMOrient(img, "LPS")
    original_spacing = np.array(oriented.GetSpacing(), dtype=float)
    original_size = np.array(oriented.GetSize(), dtype=int)
    target_spacing = np.array(target_spacing[:oriented.GetDimension()], dtype=float)
    new_size = np.maximum(np.round(original_size * (original_spacing / target_spacing)).astype(int), 1)

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(float(x) for x in target_spacing))
    resampler.SetSize([int(x) for x in new_size])
    resampler.SetOutputDirection(oriented.GetDirection())
    resampler.SetOutputOrigin(oriented.GetOrigin())
    resampler.SetTransform(sitk.Transform(oriented.GetDimension(), sitk.sitkIdentity))
    resampler.SetDefaultPixelValue(float(default_value))
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputPixelType(oriented.GetPixelID())
    return resampler.Execute(oriented)


def vbv_run_normalisation(file_df, target_spacing, intensity_interpolation="Linear"):
    """Run orientation and spacing normalisation and save outputs to project folder."""
    if not SIMPLEITK_AVAILABLE:
        return pd.DataFrame([{"Status": "Failed", "Details": "SimpleITK is not installed."}])

    output_root = vbv_normalisation_project_folder()
    items = vbv_normalisation_items(file_df)
    results = []
    progress = st.progress(0.0)
    status = st.empty()

    for i, item in enumerate(items, start=1):
        patient_id = vbv_safe_name(item.get("patient_id", "Unknown"))
        role = vbv_safe_name(item.get("role", "Other"))
        patient_out = output_root / patient_id
        patient_out.mkdir(parents=True, exist_ok=True)

        result = {
            "Patient ID": item.get("patient_id", ""),
            "Role": item.get("role", ""),
            "Item type": item.get("item_type", ""),
            "Source": item.get("display_name", item.get("path", "")),
            "Source path": item.get("path", ""),
            "Target spacing": vbv_format_tuple(target_spacing, 3),
            "Interpolation": "",
            "Status": "",
            "Output file": "",
            "Output exists": False,
            "Original size": "",
            "Original spacing": "",
            "Normalised size": "",
            "Normalised spacing": "",
            "Details": "",
        }

        try:
            status.info(f"Normalising {i} of {len(items)}: {item.get('display_name', item.get('path', ''))}")
            img = vbv_read_normalisation_item(item)
            result["Original size"] = vbv_format_tuple(img.GetSize(), 0)
            result["Original spacing"] = vbv_format_tuple(img.GetSpacing(), 3)

            interpolator, interpolation_label = vbv_role_interpolator(item.get("role", ""), intensity_interpolation)
            result["Interpolation"] = interpolation_label
            normalised = vbv_resample_to_spacing(img, target_spacing, interpolator)

            result["Normalised size"] = vbv_format_tuple(normalised.GetSize(), 0)
            result["Normalised spacing"] = vbv_format_tuple(normalised.GetSpacing(), 3)

            source_name = vbv_safe_name(Path(str(item.get("source_filename", item.get("display_name", "image")))).name)
            if not source_name or source_name == ".":
                source_name = f"{role}_image"
            unique_id = str(item.get("file_table_row", i)).strip()
            safe_role = vbv_safe_name(role) or "Image"
            out_path = patient_out / f"normalised_{safe_role}_{unique_id}_{source_name}.nii.gz"
            sitk.WriteImage(normalised, str(out_path))

            if not out_path.exists():
                raise FileNotFoundError(f"Normalised output was not created: {out_path}")

            result["Status"] = "Processed"
            result["Output file"] = str(out_path)
            result["Output exists"] = True
            result["Details"] = "Reoriented to LPS, resampled to common voxel spacing, and saved inside the project folder."

        except Exception as error:
            result["Status"] = "Failed"
            result["Details"] = str(error)

        results.append(result)
        progress.progress(i / max(len(items), 1))

    results_df = pd.DataFrame(results)
    output_root.mkdir(parents=True, exist_ok=True)
    results_csv = output_root / "normalisation_results.csv"
    results_df.to_csv(results_csv, index=False)
    st.session_state.vbv_normalisation_results = results_df
    st.session_state.vbv_normalisation_results_csv = str(results_csv)
    st.session_state.vbv_normalisation_ran = True
    status.success("Normalisation complete.")
    return results_df



def vbv_load_saved_normalisation_results_if_available():
    """Restore normalisation results from the active project folder if available."""
    if st.session_state.get("vbv_normalisation_results", pd.DataFrame()) is not None:
        existing = st.session_state.get("vbv_normalisation_results", pd.DataFrame())
        if isinstance(existing, pd.DataFrame) and not existing.empty:
            return existing

    normalisation_folder = vbv_normalisation_project_folder()
    results_csv = normalisation_folder / "normalisation_results.csv"
    if results_csv.exists():
        try:
            results_df = pd.read_csv(results_csv)
            st.session_state.vbv_normalisation_results = results_df
            st.session_state.vbv_normalisation_results_csv = str(results_csv)
            if "Status" in results_df.columns and (results_df["Status"].astype(str) == "Processed").any():
                st.session_state.vbv_normalisation_ran = True
            return results_df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()



def vbv_preview_text_from_row(row):
    """Build a flexible text string for modality detection in the preview viewer."""
    parts = [
        row.get("Role", ""),
        row.get("File role", ""),
        row.get("VBA role", ""),
        row.get("QC role", ""),
        row.get("Auto role", ""),
        row.get("Item type", ""),
        row.get("Source", ""),
        row.get("Source path", ""),
        row.get("Full path", ""),
        row.get("Relative path", ""),
        row.get("Filename", ""),
        row.get("Output file", ""),
        row.get("Interpolation", ""),
    ]
    return " ".join(str(x) for x in parts if str(x).strip()).lower()


def vbv_is_excluded_preview_label(text):
    """Exclude dose, RTSTRUCT and mask-like outputs from the CT/MR preview."""
    text = str(text).lower()
    excluded_terms = [
        "mask", "seg", "segmentation", "structure", "rtstruct", "rtss",
        "label", "contour", "dose", "rtdose", "rd.",
    ]
    return any(term in text for term in excluded_terms)


def vbv_is_ct_role(role, path="", extra_text=""):
    text = f"{role} {path} {extra_text}".lower()
    if vbv_is_excluded_preview_label(text):
        return False
    compact_role = str(role).strip().lower()
    if compact_role in ["ct", "computed tomography"]:
        return True
    ct_terms = [" ct ", "ct_", "_ct", "-ct", "ct-", "computed tomography", "ctscan", "ct scan"]
    padded = f" {text} "
    return any(term in padded for term in ct_terms) or Path(str(path)).name.lower().startswith(("ct", "normalised_ct", "normalized_ct"))


def vbv_is_mr_role(role, path="", extra_text=""):
    text = f"{role} {path} {extra_text}".lower()
    if vbv_is_excluded_preview_label(text):
        return False
    compact_role = str(role).strip().lower()
    if compact_role in ["mr", "mri", "magnetic resonance"]:
        return True
    mr_terms = [" mr ", " mri ", "mr_", "_mr", "-mr", "mr-", "mri", "t1", "t2", "flair", "mpr", "bravo", "adc", "dwi", "dti"]
    padded = f" {text} "
    return any(term in padded for term in mr_terms) or Path(str(path)).name.lower().startswith(("mr", "mri", "normalised_mr", "normalized_mr"))


def vbv_is_generic_intensity_image(row):
    """Return True for processed intensity images when CT/MR labels were not assigned."""
    text = vbv_preview_text_from_row(row)
    if vbv_is_excluded_preview_label(text):
        return False
    interpolation = str(row.get("Interpolation", "")).lower()
    # Masks are normalised with nearest-neighbour. Intensity images use linear/B-spline.
    if "nearest" in interpolation:
        return False
    return True


def vbv_find_existing_normalised_output(output_file):
    """Resolve a saved normalisation output path robustly.

    Older CSVs may contain Windows paths with whitespace/quotes, or the project may
    have been reopened after a Streamlit rerun. This helper first checks the exact
    saved path, then searches the active normalisation folder for the same filename.
    """
    text = str(output_file).strip().strip('"').strip("'")
    if not text:
        return ""

    candidate = Path(text).expanduser()
    if candidate.exists() and candidate.is_file():
        return str(candidate)

    normalisation_folder = vbv_normalisation_project_folder()
    if normalisation_folder.exists():
        name = candidate.name
        if name:
            matches = list(normalisation_folder.rglob(name))
            for match in matches:
                if match.is_file():
                    return str(match)

        # Final fallback: match by the tail of a .nii or .nii.gz file name.
        tail = name.lower()
        if tail:
            for match in normalisation_folder.rglob("*.nii*"):
                if match.name.lower() == tail and match.is_file():
                    return str(match)

    return ""


def vbv_load_saved_image_role_lookup():
    """Load saved image-index/QC role information for preview classification."""
    lookup = {}
    frames = []

    file_df = st.session_state.get("voxel_loaded_image_files_df", pd.DataFrame())
    if isinstance(file_df, pd.DataFrame) and not file_df.empty:
        frames.append(file_df)

    project_folder = st.session_state.get("voxel_project_folder", "")
    if project_folder:
        index_path = Path(project_folder) / "02_Load_Images_Masks" / "image_directory_file_index.csv"
        if index_path.exists():
            try:
                frames.append(pd.read_csv(index_path))
            except Exception:
                pass

    for frame in frames:
        for _, r in frame.iterrows():
            role = (
                r.get("VBA role", "")
                or r.get("File role", "")
                or r.get("QC role", "")
                or r.get("Auto role", "")
                or ""
            )
            if not str(role).strip():
                continue
            keys = [
                r.get("Full path", ""),
                r.get("Relative path", ""),
                r.get("Filename", ""),
            ]
            for key in keys:
                key_text = str(key).strip().lower()
                if key_text:
                    lookup[key_text] = str(role).strip()
                    lookup[Path(key_text).name.lower()] = str(role).strip()
    return lookup


def vbv_lookup_role_for_normalised_row(row, role_lookup):
    """Recover CT/MR role from the saved image-directory index when needed."""
    current_role = str(row.get("Role", "")).strip()
    if current_role and current_role.lower() not in ["other", "image", "main / unknown", "unknown"]:
        return current_role

    source_candidates = [
        row.get("Source path", ""),
        row.get("Source", ""),
        row.get("Output file", ""),
    ]
    for source in source_candidates:
        source_text = str(source).strip().lower()
        if not source_text:
            continue
        for key in [source_text, Path(source_text).name.lower()]:
            if key in role_lookup:
                return role_lookup[key]

    return current_role



def vbv_search_normalised_outputs_for_row(row):
    """Find normalised NIfTI outputs saved in the project even if the CSV path is stale/blank."""
    normalisation_folder = vbv_normalisation_project_folder()
    if not normalisation_folder.exists():
        return []

    patient_id = str(row.get("Patient ID", "")).strip()
    role = str(row.get("Role", "")).strip()
    source = str(row.get("Source", "")).strip()
    source_path = str(row.get("Source path", "")).strip()

    safe_patient = vbv_safe_name(patient_id) if patient_id else ""
    safe_role = vbv_safe_name(role) if role else ""
    source_names = [Path(source).name.lower(), Path(source_path).name.lower()]
    source_names = [x for x in source_names if x and x not in [".", ""]]

    candidate_roots = []
    if safe_patient:
        patient_root = normalisation_folder / safe_patient
        if patient_root.exists():
            candidate_roots.append(patient_root)
    candidate_roots.append(normalisation_folder)

    matches = []
    seen = set()
    for root in candidate_roots:
        for path in root.rglob("*.nii*"):
            if not path.is_file():
                continue
            name = path.name.lower()
            parent = path.parent.name.lower()
            score = 0
            if safe_patient and safe_patient.lower() in parent:
                score += 4
            if safe_patient and safe_patient.lower() in name:
                score += 2
            if safe_role and safe_role.lower() in name:
                score += 3
            if any(sn and sn in name for sn in source_names):
                score += 4
            # Include plausible outputs even when names are generic, but keep scoring.
            if name.startswith(("normalised_", "normalized_")):
                score += 1
            if str(path) not in seen and score > 0:
                matches.append((score, path))
                seen.add(str(path))
    matches.sort(key=lambda x: x[0], reverse=True)
    return [str(path) for _, path in matches]


def vbv_scan_saved_normalised_outputs(role_lookup=None):
    """Build a preview table directly from all saved normalised NIfTI files.

    This intentionally includes masks, structures and dose files. Earlier versions
    excluded mask-like filenames from the fallback scan, which meant masks could
    be saved successfully but never appear in the preview dropdown.
    """
    normalisation_folder = vbv_normalisation_project_folder()
    if not normalisation_folder.exists():
        return pd.DataFrame()
    if role_lookup is None:
        role_lookup = vbv_load_saved_image_role_lookup()

    rows = []
    for path in normalisation_folder.rglob("*.nii*"):
        if not path.is_file():
            continue

        text = f"{path.name} {path.parent.name} {path}"
        role = ""
        lower_name = path.name.lower()

        for key, value in role_lookup.items():
            key_name = Path(str(key)).name.lower()
            if key_name and key_name in lower_name:
                role = value
                break

        row_for_classification = {
            "Role": role,
            "Source": "Saved normalised output",
            "Source path": "",
            "Output file": str(path),
            "Interpolation": "Nearest-neighbour" if any(term in lower_name for term in ["mask", "seg", "structure", "roi", "contour", "label", "oar", "rtstruct", "rtss"]) else "",
        }
        image_type, clean_role = vbv_classify_normalised_output(row_for_classification, str(path), role)

        rows.append({
            "Patient ID": path.parent.name,
            "Image type": image_type,
            "Role": clean_role,
            "Source": "Saved normalised output",
            "Source path": "",
            "Output file": str(path),
            "Normalised spacing": "",
            "Normalised size": "",
            "Interpolation": row_for_classification.get("Interpolation", ""),
        })
    return pd.DataFrame(rows)

def vbv_classify_normalised_output(row, output_file, recovered_role=""):
    """Classify a normalised output for preview: CT/MR, dose, mask, or other image.

    Classification uses the explicit role saved during image QC first, then falls
    back to filenames. The returned category controls display and interpolation QC.
    """
    role = str(recovered_role or row.get("Role", "") or "").strip()
    text = (vbv_preview_text_from_row(row) + " " + str(output_file) + " " + role).lower()
    interpolation = str(row.get("Interpolation", "")).lower()

    mask_terms = [
        "mask", "seg", "segmentation", "structure", "structure set",
        "rtstruct", "rtss", "roi", "contour", "label", "oar"
    ]
    dose_terms = ["dose", "rtdose", "rt dose", "rd."]

    if any(term in text for term in dose_terms):
        return "Dose", role or "Dose"

    if any(term in text for term in mask_terms) or "nearest" in interpolation:
        return "Mask / segmentation", role or "Mask"

    if vbv_is_ct_role(role, output_file, text):
        return "CT", role or "CT"

    if vbv_is_mr_role(role, output_file, text):
        return "MR", role or "MR"

    # RTSTRUCT itself should not be previewed as a voxel image unless it was already
    # converted to a voxel mask. If it exists here as a NIfTI, keep it under masks.
    if "rtstruct" in text or "rtss" in text:
        return "Mask / segmentation", role or "RTSTRUCT / mask"

    return "Other processed image", role or "Image"


def vbv_processed_ct_mr_table(results_df):
    """Return a one-to-one preview table for all normalised outputs.

    Despite the historical function name, this now includes CT/MR, dose, masks,
    and other processed intensity images. It uses the exact output file written
    for each normalisation row, so CT/MR/dose/mask rows cannot be mixed.
    """
    role_lookup = vbv_load_saved_image_role_lookup()
    rows = []
    debug_rows = []
    normalisation_folder = vbv_normalisation_project_folder().resolve()

    if results_df is not None and not results_df.empty:
        for row_number, row in results_df.iterrows():
            status = str(row.get("Status", "")).strip().lower()
            raw_out_file = str(row.get("Output file", "")).strip()
            if status != "processed":
                debug_rows.append({"Reason": "Row was not processed", "Output file": raw_out_file, "Status": status})
                continue

            out_file = vbv_find_existing_normalised_output(raw_out_file)
            if not out_file:
                debug_rows.append({
                    "Reason": "Exact normalised output file could not be found",
                    "Output file in CSV": raw_out_file,
                    "Patient ID": row.get("Patient ID", ""),
                    "Role": row.get("Role", ""),
                    "Source": row.get("Source", ""),
                })
                continue

            try:
                Path(out_file).resolve().relative_to(normalisation_folder)
            except Exception:
                debug_rows.append({
                    "Reason": "Output exists but is outside the normalisation folder; blocked to avoid raw-data preview",
                    "Output file": out_file,
                    "Patient ID": row.get("Patient ID", ""),
                    "Role": row.get("Role", ""),
                })
                continue

            role = vbv_lookup_role_for_normalised_row(row, role_lookup)
            category, clean_role = vbv_classify_normalised_output(row, out_file, role)

            rows.append({
                "Patient ID": str(row.get("Patient ID", "Unknown")),
                "Image type": category,
                "Role": clean_role,
                "Source": row.get("Source", ""),
                "Source path": row.get("Source path", ""),
                "Output file": out_file,
                "Normalised spacing": row.get("Normalised spacing", ""),
                "Normalised size": row.get("Normalised size", ""),
                "Interpolation": row.get("Interpolation", ""),
                "Normalisation row": row_number,
            })

    preview_df = pd.DataFrame(rows)

    # Always merge a direct scan of saved normalised outputs. This makes the
    # preview robust when masks/dose were saved on disk but omitted or poorly
    # labelled in normalisation_results.csv. Exact CSV rows are kept first; the
    # scan only adds missing output files.
    fallback_df = vbv_scan_saved_normalised_outputs(role_lookup)
    if fallback_df is not None and not fallback_df.empty:
        if preview_df.empty:
            preview_df = fallback_df
            debug_rows.append({"Reason": "Used direct scan of saved normalised outputs", "Output file": "", "Status": "Recovered"})
        else:
            preview_df = pd.concat([preview_df, fallback_df], ignore_index=True, sort=False)
            debug_rows.append({"Reason": "Merged direct scan of saved normalised outputs", "Output file": "", "Status": "Recovered"})

    if not preview_df.empty:
        preview_df = preview_df.drop_duplicates(subset=["Output file"], keep="first")

    st.session_state.vbv_preview_debug_rows = pd.DataFrame(debug_rows)
    return preview_df

def vbv_read_preview_array(image_path):
    """Read an image and return a display-safe numpy array with z, y, x ordering."""
    img = sitk.ReadImage(str(image_path))
    arr = sitk.GetArrayFromImage(img)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    if arr.ndim > 3:
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
    return arr, img


def vbv_window_for_display(slice_array):
    """Clip image slice to robust intensity percentiles for viewer display."""
    values = np.asarray(slice_array, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        return values
    return np.clip(values, low, high)



def vbv_downsample_slice_for_preview(slice_array, max_pixels=384):
    """Downsample a 2D slice for Streamlit preview only.

    This does not alter the saved normalised image; it only reduces the display
    array so the viewer stays responsive for large CT/MR volumes.
    """
    arr = np.asarray(slice_array)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        return arr
    height, width = arr.shape
    largest_dim = max(height, width)
    if largest_dim <= max_pixels:
        return arr
    step = int(np.ceil(largest_dim / max_pixels))
    return arr[::step, ::step]


def vbv_representative_slice_indices(number_of_slices):
    """Return up to three fixed representative axial slice indices."""
    if number_of_slices <= 0:
        return []
    if number_of_slices == 1:
        return [0]
    candidates = [
        int(round((number_of_slices - 1) * 0.25)),
        int(round((number_of_slices - 1) * 0.50)),
        int(round((number_of_slices - 1) * 0.75)),
    ]
    unique = []
    for idx in candidates:
        idx = max(0, min(int(idx), number_of_slices - 1))
        if idx not in unique:
            unique.append(idx)
    return unique

def vbv_render_processed_image_viewer(preview_table):
    """Compact single-file viewer for normalised outputs.

    The user selects a patient and then directly selects one saved normalised
    output file. There is no separate data-type filter, because that was causing
    confusing labels such as CT/MR while the file itself was MR. The selected row
    is previewed using its exact Output file only.
    """
    if preview_table is None or preview_table.empty:
        st.info("No processed normalised outputs were found for preview.")
        debug_df = st.session_state.get("vbv_preview_debug_rows", pd.DataFrame())
        if isinstance(debug_df, pd.DataFrame) and not debug_df.empty:
            with st.expander("Preview diagnostic details"):
                st.write("The normalisation results were found, but no readable normalised output could be matched for preview.")
                st.dataframe(debug_df, use_container_width=True)
        return

    preview_table = preview_table.copy()
    preview_table["Patient ID"] = preview_table["Patient ID"].astype(str)
    preview_table["Image type"] = preview_table["Image type"].astype(str)

    patients = sorted(preview_table["Patient ID"].dropna().unique().tolist())
    if not patients:
        st.info("No patient IDs are available for preview.")
        return

    st.caption("Preview source: saved normalised output only. Raw source images are not displayed here.")

    selected_patient = st.selectbox(
        "Select patient",
        patients,
        key="vbv_preview_patient_single",
    )

    patient_rows = preview_table[preview_table["Patient ID"].astype(str) == str(selected_patient)].copy()
    if patient_rows.empty:
        st.info("No processed images found for the selected patient.")
        return

    type_counts = patient_rows["Image type"].value_counts().to_dict()
    if type_counts:
        st.caption("Available normalised files: " + "; ".join(f"{k}: {v}" for k, v in type_counts.items()))

    # The viewer now lists the actual saved normalised files directly.
    # This avoids mixing CT/MR/mask/dose labels through an additional grouping filter.
    patient_rows["_output_name"] = patient_rows["Output file"].apply(lambda x: Path(str(x)).name)
    patient_rows["_type_order"] = patient_rows["Image type"].map({
        "CT": 0,
        "MR": 1,
        "Dose": 2,
        "Mask / segmentation": 3,
        "Other processed image": 4,
        "Image": 5,
    }).fillna(9)
    patient_rows = patient_rows.sort_values(["_type_order", "Role", "_output_name"])
    image_options = patient_rows.to_dict("records")

    def _image_label(row):
        image_type = str(row.get("Image type", "Image")).strip() or "Image"
        role = str(row.get("Role", "")).strip()
        output_name = Path(str(row.get("Output file", ""))).name
        source_name = Path(str(row.get("Source", ""))).name if str(row.get("Source", "")).strip() else ""
        if role and role.lower() != image_type.lower():
            prefix = f"{image_type} / {role}"
        else:
            prefix = image_type
        if source_name and source_name != output_name:
            return f"{prefix} | {output_name}  ←  {source_name}"
        return f"{prefix} | {output_name}"

    selected_image_index = st.selectbox(
        "Select normalised file",
        list(range(len(image_options))),
        format_func=lambda i: _image_label(image_options[int(i)]),
        key=f"vbv_preview_selected_normalised_file_{selected_patient}",
    )
    selected_row = image_options[int(selected_image_index)]
    image_path = selected_row.get("Output file", "")

    normalisation_folder = vbv_normalisation_project_folder().resolve()
    try:
        image_path_obj = Path(str(image_path)).resolve()
    except Exception:
        image_path_obj = Path(str(image_path))

    # Safety check: do not preview the raw source directory by accident.
    try:
        image_path_obj.relative_to(normalisation_folder)
    except Exception:
        st.error("Preview blocked: the selected file is not inside the project normalisation folder.")
        st.code(str(image_path_obj))
        st.caption("Run normalisation again so the processed output is saved inside 03_Normalisation.")
        return

    if not image_path_obj.exists():
        st.error("The selected normalised file could not be found on disk.")
        st.code(str(image_path_obj))
        return

    try:
        arr, img = vbv_read_preview_array(image_path_obj)
        number_of_slices = int(arr.shape[0])
        if number_of_slices <= 0:
            st.warning("This image has no readable slices for preview.")
            return

        default_slice = max(0, number_of_slices // 2)
        slider_key = f"vbv_preview_slice_exact_{selected_patient}_{Path(str(image_path_obj)).stem}_{number_of_slices}"
        slice_index = st.slider(
            "Slice",
            min_value=0,
            max_value=number_of_slices - 1,
            value=default_slice,
            key=slider_key,
        )

        raw_slice = np.asarray(arr[int(slice_index), :, :])
        image_type_text = str(selected_row.get("Image type", "")).lower()
        role_text = str(selected_row.get("Role", "")).lower()
        is_mask = "mask" in image_type_text or "seg" in image_type_text or "mask" in role_text or "seg" in role_text
        is_dose = "dose" in image_type_text or "dose" in role_text

        if is_mask:
            display_slice = raw_slice
            unique_values = np.unique(arr)
            if unique_values.size > 20:
                shown_unique = ", ".join(str(x) for x in unique_values[:20]) + f" ... ({unique_values.size} values)"
            else:
                shown_unique = ", ".join(str(x) for x in unique_values)
        else:
            display_slice = vbv_window_for_display(raw_slice)
            shown_unique = ""

        display_slice = vbv_downsample_slice_for_preview(display_slice, max_pixels=384)

        left, right = st.columns([0.46, 0.54])
        with left:
            fig, ax = plt.subplots(figsize=(3.2, 3.2))
            ax.imshow(display_slice, cmap="gray")
            ax.axis("off")
            ax.set_title(f"Slice {slice_index + 1}/{number_of_slices}", fontsize=9)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with right:
            st.markdown("**Normalised file details**")
            st.write(f"Patient: {selected_patient}")
            st.write(f"Selected file type: {selected_row.get('Image type', 'Image')}")
            st.write(f"Assigned role: {selected_row.get('Role', '')}")
            st.write(f"Size: {vbv_format_tuple(img.GetSize(), 0)}")
            st.write(f"Spacing: {vbv_format_tuple(img.GetSpacing(), 3)} mm")
            interpolation_text = str(selected_row.get("Interpolation", "")).strip()
            if interpolation_text:
                st.write(f"Interpolation: {interpolation_text}")
            if is_mask:
                st.write(f"Label values: {shown_unique}")
                st.caption("Masks/segmentations should preserve discrete labels. They must use nearest-neighbour interpolation.")
            elif is_dose:
                st.caption("Dose is treated as continuous data and can be resampled using linear/B-spline interpolation.")
            st.write(f"Preview file: {Path(str(image_path_obj)).name}")
            source_path_text = str(selected_row.get("Source path", "")).strip()
            if source_path_text:
                st.caption(f"Original source: {Path(source_path_text).name}")
            st.caption("The display slice is downsampled only for viewing. The saved normalised output is unchanged.")

    except Exception as error:
        st.error(f"Could not preview image: {error}")



# ============================================================
# VOXEL BATCH REGISTRATION HELPERS - normalised outputs only
# ============================================================

def vbv_batch_registration_project_folder():
    """Return the project folder used for batch registration outputs."""
    project_folder = st.session_state.get("voxel_project_folder", "")
    if project_folder:
        out = Path(project_folder) / "05_Batch_Registration"
    else:
        out = Path("data") / "voxel_batch_registration"
    out.mkdir(parents=True, exist_ok=True)
    return out


def vbv_load_normalised_outputs_for_batch():
    """Load the saved normalised outputs and return a clean table for batch registration."""
    results_df = vbv_load_saved_normalisation_results_if_available()
    preview_df = vbv_processed_ct_mr_table(results_df)
    if preview_df is None or preview_df.empty:
        return pd.DataFrame()

    df = preview_df.copy()
    df["Patient ID"] = df["Patient ID"].astype(str)
    df["Output file"] = df["Output file"].astype(str)
    normalisation_folder = vbv_normalisation_project_folder().resolve()

    keep_rows = []
    for _, row in df.iterrows():
        path_text = str(row.get("Output file", "")).strip()
        if not path_text:
            continue
        try:
            p = Path(path_text).resolve()
            p.relative_to(normalisation_folder)
        except Exception:
            continue
        if p.exists() and p.suffix.lower() in [".nii", ".gz"]:
            keep_rows.append(row)

    if not keep_rows:
        return pd.DataFrame()

    out = pd.DataFrame(keep_rows).drop_duplicates(subset=["Output file"], keep="first")
    out["File name"] = out["Output file"].apply(lambda x: Path(str(x)).name)
    out["Image type"] = out["Image type"].fillna("Image").astype(str)
    out["Role"] = out["Role"].fillna("").astype(str)
    return out


def vbv_batch_type_key(row):
    """Return a simplified type key used for filtering and interpolation."""
    text = " ".join([
        str(row.get("Image type", "")),
        str(row.get("Role", "")),
        str(row.get("File name", "")),
        str(row.get("Output file", "")),
    ]).lower()
    if any(term in text for term in ["mask", "seg", "segmentation", "structure", "rtstruct", "roi", "contour", "label", "oar"]):
        return "Mask / segmentation"
    if "dose" in text or "rtdose" in text:
        return "Dose"
    if any(term in text for term in ["ct", "computed"]):
        return "CT"
    if any(term in text for term in ["mr", "mri", "t1", "t2", "flair", "mpr", "bravo", "dwi", "adc"]):
        return "MR"
    return "Other image"


def vbv_batch_file_label(row):
    type_key = row.get("Batch type", vbv_batch_type_key(row))
    role = str(row.get("Role", "")).strip()
    file_name = Path(str(row.get("Output file", ""))).name
    if role and role.lower() not in str(type_key).lower():
        return f"{type_key} / {role} | {file_name}"
    return f"{type_key} | {file_name}"


def vbv_batch_find_default_reference_index(ref_rows):
    """Prefer reference CT, then MR, then the first available image."""
    if ref_rows is None or ref_rows.empty:
        return 0
    rows = ref_rows.reset_index(drop=True)
    for preferred in ["CT", "MR", "Other image"]:
        matches = rows.index[rows["Batch type"].astype(str) == preferred].tolist()
        if matches:
            return int(matches[0])
    return 0


def vbv_batch_default_transform_row(patient_rows):
    """Prefer CT then MR as the moving anatomy used to estimate the transform."""
    rows = patient_rows.reset_index(drop=True)
    for preferred in ["CT", "MR", "Other image"]:
        matches = rows.index[rows["Batch type"].astype(str) == preferred].tolist()
        if matches:
            return rows.iloc[int(matches[0])].to_dict()
    return rows.iloc[0].to_dict() if not rows.empty else {}


def vbv_registration_interpolator_for_batch(type_key, role=""):
    text = f"{type_key} {role}".lower()
    if any(term in text for term in ["mask", "seg", "structure", "rtstruct", "roi", "contour", "label", "oar"]):
        return sitk.sitkNearestNeighbor, "Nearest-neighbour"
    return sitk.sitkLinear, "Linear"


def vbv_initial_transform_for_method(fixed, moving, method):
    dim = fixed.GetDimension()
    if str(method).lower().startswith("affine"):
        model = sitk.AffineTransform(dim)
    else:
        model = sitk.Euler3DTransform() if dim == 3 else sitk.Euler2DTransform()
    return sitk.CenteredTransformInitializer(
        fixed,
        moving,
        model,
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )


def vbv_run_single_registration(fixed_img, moving_img, method="Rigid only", stage_callback=None):
    """Estimate registration transform from moving image into fixed-image space.

    Supported workflow options:
    - Rigid only
    - Rigid → Affine
    - Rigid → Affine → B-spline

    The B-spline option is intentionally treated as an advanced deformable step after
    rigid and affine alignment. The same final transform is then applied to CT/MR,
    dose and masks; interpolation remains data-type specific during resampling.
    """
    fixed = sitk.Cast(fixed_img, sitk.sitkFloat32)
    moving = sitk.Cast(moving_img, sitk.sitkFloat32)
    dim = fixed.GetDimension()

    def _stage(message):
        if stage_callback is not None:
            try:
                stage_callback(message)
            except Exception:
                pass

    def _execute(initial_transform, iterations=80, sampling_percentage=0.08):
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=40)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(float(sampling_percentage))
        registration.SetInterpolator(sitk.sitkLinear)
        registration.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=int(iterations),
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
        )
        registration.SetOptimizerScalesFromPhysicalShift()
        registration.SetShrinkFactorsPerLevel([4, 2, 1])
        registration.SetSmoothingSigmasPerLevel([2, 1, 0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        registration.SetInitialTransform(initial_transform, inPlace=False)
        final_transform = registration.Execute(fixed, moving)
        return final_transform, float(registration.GetMetricValue())

    def _rigid_then_affine():
        rigid_initial = vbv_initial_transform_for_method(fixed, moving, "Rigid")
        rigid_transform, rigid_metric = _execute(rigid_initial, iterations=60)

        affine_initial = sitk.AffineTransform(dim)
        if hasattr(rigid_transform, "GetMatrix"):
            try:
                affine_initial.SetMatrix(rigid_transform.GetMatrix())
                affine_initial.SetCenter(rigid_transform.GetCenter())
                affine_initial.SetTranslation(rigid_transform.GetTranslation())
            except Exception:
                pass
        affine_transform, affine_metric = _execute(affine_initial, iterations=80)
        return rigid_transform, rigid_metric, affine_transform, affine_metric

    method_text = str(method).strip()
    if method_text in ["Rigid", "Rigid only"]:
        _stage("Stage 1/1: running rigid registration")
        initial_transform = vbv_initial_transform_for_method(fixed, moving, "Rigid")
        final_transform, metric = _execute(initial_transform, iterations=100)
        return final_transform, metric, f"Rigid metric={metric:.5f}"

    if method_text in ["Rigid → Affine", "Affine"]:
        _stage("Stage 1/2: running rigid registration")
        _stage("Stage 2/2: running affine registration")
        _, rigid_metric, affine_transform, affine_metric = _rigid_then_affine()
        return affine_transform, affine_metric, f"Rigid metric={rigid_metric:.5f}; affine metric={affine_metric:.5f}"

    if method_text == "Rigid → Affine → B-spline":
        _stage("Stage 1/3: running rigid registration")
        _stage("Stage 2/3: running affine registration")
        _, rigid_metric, affine_transform, affine_metric = _rigid_then_affine()

        _stage("Stage 3/3: running B-spline deformable registration. This can take several minutes.")
        # Resample moving image with the affine transform first, then estimate a local B-spline correction.
        moving_affine = sitk.Resample(
            moving,
            fixed,
            affine_transform,
            sitk.sitkLinear,
            0.0,
            moving.GetPixelID(),
        )

        # Conservative control-point grid to reduce the risk of unstable or very slow local warping.
        # Larger values make B-spline more flexible but also increase memory/time and over-warping risk.
        mesh_size = [max(1, min(6, int(s / 60))) for s in fixed.GetSize()]
        bspline_initial = sitk.BSplineTransformInitializer(fixed, mesh_size, order=3)

        bspline_registration = sitk.ImageRegistrationMethod()
        bspline_registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=40)
        bspline_registration.SetMetricSamplingStrategy(bspline_registration.RANDOM)
        bspline_registration.SetMetricSamplingPercentage(0.03)
        bspline_registration.SetInterpolator(sitk.sitkLinear)
        bspline_registration.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=30,
            maximumNumberOfCorrections=5,
            maximumNumberOfFunctionEvaluations=300,
            costFunctionConvergenceFactor=1e7,
        )
        bspline_registration.SetShrinkFactorsPerLevel([4, 2, 1])
        bspline_registration.SetSmoothingSigmasPerLevel([2, 1, 0])
        bspline_registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        bspline_registration.SetInitialTransform(bspline_initial, inPlace=False)
        try:
            bspline_transform = bspline_registration.Execute(fixed, moving_affine)
            bspline_metric = float(bspline_registration.GetMetricValue())
        except Exception as bspline_error:
            _stage("B-spline failed; keeping the affine transform for this patient and recording the error.")
            return affine_transform, affine_metric, (
                f"Rigid metric={rigid_metric:.5f}; affine metric={affine_metric:.5f}; "
                f"B-spline failed, affine transform kept; error={bspline_error}"
            )

        composite = sitk.CompositeTransform(dim)
        composite.AddTransform(affine_transform)
        composite.AddTransform(bspline_transform)
        return composite, bspline_metric, (
            f"Rigid metric={rigid_metric:.5f}; affine metric={affine_metric:.5f}; "
            f"B-spline metric={bspline_metric:.5f}; advanced deformable step used; mesh={mesh_size}"
        )

    initial_transform = vbv_initial_transform_for_method(fixed, moving, "Rigid")
    final_transform, metric = _execute(initial_transform, iterations=100)
    return final_transform, metric, f"Rigid metric={metric:.5f}"


def vbv_resample_registered_to_fixed(moving_img, fixed_img, transform, type_key, role=""):
    interpolator, label = vbv_registration_interpolator_for_batch(type_key, role)
    registered = sitk.Resample(
        moving_img,
        fixed_img,
        transform,
        interpolator,
        0.0,
        moving_img.GetPixelID(),
    )
    return registered, label


def vbv_safe_registered_filename(patient_id, type_key, role, source_path):
    source_name = vbv_safe_name(Path(str(source_path)).name) or "image"
    safe_patient = vbv_safe_name(patient_id) or "Patient"
    safe_type = vbv_safe_name(type_key).replace("__", "_") or "Image"
    safe_role = vbv_safe_name(role) or safe_type
    return f"registered_{safe_patient}_{safe_type}_{safe_role}_{source_name}.nii.gz"


def vbv_run_batch_registration_from_normalised(
    normalised_df,
    fixed_row,
    moving_patient_ids,
    selected_type_keys,
    method,
    skip_existing=True,
    force_overwrite=False,
    progress_bar=None,
    progress_text=None,
):
    """Register selected normalised outputs to the selected fixed/reference image."""
    if not SIMPLEITK_AVAILABLE:
        return pd.DataFrame([{"Status": "Failed", "Registration details": "Missing dependency.", "Error message": "SimpleITK is not installed."}])

    output_root = vbv_batch_registration_project_folder()
    registered_root = output_root / "registered_images"
    transforms_root = output_root / "transforms"
    qc_root = output_root / "qc"
    for folder in [registered_root, transforms_root, qc_root]:
        folder.mkdir(parents=True, exist_ok=True)

    fixed_path = Path(str(fixed_row.get("Output file", ""))).resolve()
    fixed_patient = str(fixed_row.get("Patient ID", "Reference"))
    fixed_label = vbv_batch_file_label(fixed_row)
    fixed_img = sitk.ReadImage(str(fixed_path))

    results = []
    tasks = []
    for patient_id in moving_patient_ids:
        patient_rows = normalised_df[normalised_df["Patient ID"].astype(str) == str(patient_id)].copy()
        patient_rows = patient_rows[patient_rows["Batch type"].isin(selected_type_keys)].copy()
        if patient_rows.empty:
            results.append({
                "Patient ID": patient_id,
                "Moving image": "",
                "Reference image": fixed_label,
                "Image role": "",
                "Registration method": method,
                "Transform file": "",
                "Registered output file": "",
                "Status": "Skipped",
                "Error message": "No selected normalised files were found for this patient.",
            })
            continue
        tasks.append((patient_id, patient_rows))

    total = max(1, sum(len(rows) for _, rows in tasks))
    done = 0

    for patient_index, (patient_id, patient_rows) in enumerate(tasks, start=1):
        patient_folder = registered_root / vbv_safe_name(patient_id)
        patient_transform_folder = transforms_root / vbv_safe_name(patient_id)
        patient_folder.mkdir(parents=True, exist_ok=True)
        patient_transform_folder.mkdir(parents=True, exist_ok=True)

        transform_source_row = vbv_batch_default_transform_row(patient_rows)
        transform_source_path = Path(str(transform_source_row.get("Output file", ""))).resolve()
        transform_file = patient_transform_folder / f"{vbv_safe_name(patient_id)}_{vbv_safe_name(method)}_to_{vbv_safe_name(fixed_patient)}.tfm"

        try:
            if progress_text is not None:
                progress_text.write(f"Estimating transform for {patient_id} using {vbv_batch_file_label(transform_source_row)}")
            moving_for_transform = sitk.ReadImage(str(transform_source_path))
            def _registration_stage(message):
                if progress_text is not None:
                    progress_text.info(f"{patient_id}: {message}")

            transform, metric, metric_note = vbv_run_single_registration(
                fixed_img,
                moving_for_transform,
                method=method,
                stage_callback=_registration_stage,
            )
            sitk.WriteTransform(transform, str(transform_file))
        except Exception as error:
            for _, row in patient_rows.iterrows():
                results.append({
                    "Patient ID": patient_id,
                    "Moving image": vbv_batch_file_label(row),
                    "Reference image": fixed_label,
                    "Image role": row.get("Batch type", ""),
                    "Registration method": method,
                    "Transform file": str(transform_file),
                    "Registered output file": "",
                    "Status": "Failed",
                    "Registration details": "Transform estimation failed.",
                    "Error message": str(error),
                })
            done += len(patient_rows)
            if progress_bar is not None:
                progress_bar.progress(min(done / total, 1.0))
            continue

        for _, row in patient_rows.iterrows():
            done += 1
            type_key = str(row.get("Batch type", "Other image"))
            role = str(row.get("Role", ""))
            moving_path = Path(str(row.get("Output file", ""))).resolve()
            out_file = patient_folder / vbv_safe_registered_filename(patient_id, type_key, role, moving_path)

            if progress_text is not None:
                progress_text.write(f"Registering {done}/{total}: {patient_id} | {Path(str(moving_path)).name}")
            if progress_bar is not None:
                progress_bar.progress(min(done / total, 1.0))

            if out_file.exists() and skip_existing and not force_overwrite:
                results.append({
                    "Patient ID": patient_id,
                    "Moving image": vbv_batch_file_label(row),
                    "Reference image": fixed_label,
                    "Image role": type_key,
                    "Registration method": method,
                    "Transform file": str(transform_file),
                    "Registered output file": str(out_file),
                    "Status": "Skipped",
                    "Registration details": "Existing registered output kept.",
                    "Error message": "",
                })
                continue

            try:
                moving_img = sitk.ReadImage(str(moving_path))
                registered_img, interpolation_label = vbv_resample_registered_to_fixed(
                    moving_img=moving_img,
                    fixed_img=fixed_img,
                    transform=transform,
                    type_key=type_key,
                    role=role,
                )
                sitk.WriteImage(registered_img, str(out_file))
                results.append({
                    "Patient ID": patient_id,
                    "Moving image": vbv_batch_file_label(row),
                    "Reference image": fixed_label,
                    "Image role": type_key,
                    "Registration method": method,
                    "Interpolation": interpolation_label,
                    "Transform file": str(transform_file),
                    "Registered output file": str(out_file),
                    "Status": "Processed",
                    "Registration details": metric_note,
                    "Error message": "",
                })
            except Exception as error:
                results.append({
                    "Patient ID": patient_id,
                    "Moving image": vbv_batch_file_label(row),
                    "Reference image": fixed_label,
                    "Image role": type_key,
                    "Registration method": method,
                    "Transform file": str(transform_file),
                    "Registered output file": str(out_file),
                    "Status": "Failed",
                    "Registration details": "Registration/resampling failed.",
                    "Error message": str(error),
                })

    results_df = pd.DataFrame(results)
    summary_csv = output_root / "batch_registration_summary.csv"
    results_df.to_csv(summary_csv, index=False)
    st.session_state.vbv_batch_registration_results = results_df
    st.session_state.vbv_batch_registration_summary_csv = str(summary_csv)
    st.session_state.vbv_batch_registration_ran = True
    return results_df


def vbv_load_saved_batch_registration_results():
    """Load saved batch registration summary if available."""
    existing = st.session_state.get("vbv_batch_registration_results", pd.DataFrame())
    if isinstance(existing, pd.DataFrame) and not existing.empty:
        return existing
    summary_csv = vbv_batch_registration_project_folder() / "batch_registration_summary.csv"
    if summary_csv.exists():
        try:
            df = pd.read_csv(summary_csv)
            st.session_state.vbv_batch_registration_results = df
            st.session_state.vbv_batch_registration_summary_csv = str(summary_csv)
            st.session_state.vbv_batch_registration_ran = True
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def vbv_resolve_registered_output_path(path_text):
    """Resolve a registered output path robustly inside the batch-registration folder.

    The summary CSV may contain an absolute path from a previous run, a relative path,
    or a stale path after the project was moved. This function only accepts files
    inside the current project batch-registration folder and searches by filename
    when needed.
    """
    batch_root = vbv_batch_registration_project_folder().resolve()
    text = str(path_text).strip()
    if not text:
        return None

    candidates = []
    try:
        candidates.append(Path(text))
    except Exception:
        pass
    try:
        candidates.append(batch_root / text)
    except Exception:
        pass

    basename = Path(text).name
    if basename:
        try:
            candidates.extend(batch_root.rglob(basename))
        except Exception:
            pass

    for candidate in candidates:
        try:
            resolved = Path(candidate).resolve()
            resolved.relative_to(batch_root)
            if resolved.exists() and resolved.is_file() and (resolved.name.lower().endswith(".nii") or resolved.name.lower().endswith(".nii.gz")):
                return resolved
        except Exception:
            continue
    return None


def vbv_registered_outputs_from_folder():
    """Fallback table created by scanning the registered output folder directly."""
    batch_root = vbv_batch_registration_project_folder().resolve()
    registered_root = batch_root / "registered_images"
    if not registered_root.exists():
        return pd.DataFrame()
    rows = []
    for p in sorted(list(registered_root.rglob("*.nii")) + list(registered_root.rglob("*.nii.gz"))):
        try:
            resolved = p.resolve()
            resolved.relative_to(batch_root)
        except Exception:
            continue
        patient_id = resolved.parent.name if resolved.parent != registered_root else "Unknown"
        name_lower = resolved.name.lower()
        if any(term in name_lower for term in ["mask", "seg", "segmentation", "structure", "rtstruct", "roi", "contour", "label", "oar"]):
            role = "Mask / segmentation"
        elif "dose" in name_lower:
            role = "Dose"
        elif "ct" in name_lower:
            role = "CT"
        elif any(term in name_lower for term in ["mr", "mri", "t1", "t2", "flair", "mpr", "bravo", "dwi", "adc"]):
            role = "MR"
        else:
            role = "Registered image"
        rows.append({
            "Patient ID": patient_id,
            "Image role": role,
            "Registration method": "Saved output",
            "Registered output file": str(resolved),
            "Resolved registered output file": str(resolved),
            "Status": "Found in folder",
            "Registration details": "Recovered by scanning registered_images folder.",
            "Error message": "",
        })
    return pd.DataFrame(rows)


def vbv_prepare_registered_outputs_for_review(results_df):
    """Return only registered outputs that can actually be read for review."""
    rows = []
    if results_df is not None and not results_df.empty and "Registered output file" in results_df.columns:
        for _, row in results_df.copy().iterrows():
            resolved = vbv_resolve_registered_output_path(row.get("Registered output file", ""))
            if resolved is None:
                continue
            item = row.to_dict()
            item["Resolved registered output file"] = str(resolved)
            rows.append(item)

    df = pd.DataFrame(rows)

    # Fallback: if the summary table paths are stale or missing, scan the registered output folder.
    if df.empty:
        df = vbv_registered_outputs_from_folder()

    if df is None or df.empty:
        return pd.DataFrame()

    return df.drop_duplicates(subset=["Resolved registered output file"], keep="first")


def vbv_render_registered_output_viewer(results_df):
    """Overlay QC viewer for registered batch outputs.

    After registration, registered CT/MR/dose/masks should share the same fixed
    image grid. The review therefore shows one compact overlay viewer rather
    than opening each registered file separately.
    """
    if results_df is None or results_df.empty:
        return

    df = vbv_prepare_registered_outputs_for_review(results_df)
    if df.empty:
        st.info("No readable registered outputs were found for review.")
        st.caption("This usually means the summary table exists, but the registered output paths are empty, stale, or outside the project batch-registration folder.")
        with st.expander("Diagnostic: registered output folder"):
            batch_root = vbv_batch_registration_project_folder().resolve()
            st.code(str(batch_root), language=None)
            files = []
            try:
                files = [str(p) for p in batch_root.rglob("*.nii*")]
            except Exception:
                files = []
            if files:
                st.write("NIfTI files found:")
                st.dataframe(pd.DataFrame({"File": files}), use_container_width=True, hide_index=True)
            else:
                st.write("No .nii or .nii.gz files were found under the batch-registration folder.")
        return

    def _role_group(row):
        role_text = f"{row.get('Image role', '')} {row.get('Moving image', '')} {row.get('Registered output file', '')} {row.get('Resolved registered output file', '')}".lower()
        if any(term in role_text for term in ["mask", "seg", "segmentation", "structure", "rtstruct", "roi", "contour", "label", "oar"]):
            return "Mask / segmentation"
        if "dose" in role_text or "rtdose" in role_text:
            return "Dose"
        if "ct" in role_text:
            return "CT"
        if any(term in role_text for term in ["mr", "mri", "t1", "t2", "flair", "mpr", "bravo", "dwi", "adc"]):
            return "MR"
        return "Other"

    def _file_label(row):
        role = _role_group(row)
        if role in ["CT", "MR", "Other"]:
            return f"{role} anatomy"
        if role == "Dose":
            return "Dose"
        if role == "Mask / segmentation":
            return "Mask / structure"
        return role

    st.markdown("### Review registered anatomy")
    st.caption("Use this compact viewer to inspect a few registered patients before giving the overall registration QC decision.")

    patients = sorted(df["Patient ID"].astype(str).unique().tolist())
    selected_patient = st.selectbox("Select patient to review", patients, key="vbv_batch_overlay_review_patient")
    patient_rows = df[df["Patient ID"].astype(str) == str(selected_patient)].copy()
    if patient_rows.empty:
        st.info("No registered files were found for this patient.")
        return

    patient_rows["Viewer role"] = patient_rows.apply(_role_group, axis=1)
    batch_root = vbv_batch_registration_project_folder().resolve()

    # Prepare readable rows only, and block anything outside the batch folder.
    readable_rows = []
    for _, row in patient_rows.iterrows():
        try:
            image_path = Path(str(row.get("Resolved registered output file", row.get("Registered output file", "")))).resolve()
            image_path.relative_to(batch_root)
            if image_path.exists() and image_path.is_file():
                item = row.to_dict()
                item["Resolved registered output file"] = str(image_path)
                readable_rows.append(item)
        except Exception:
            continue

    if not readable_rows:
        st.warning("Registered rows exist for this patient, but no readable files inside the batch-registration folder could be opened.")
        return

    readable_df = pd.DataFrame(readable_rows)

    # Background should be anatomy: CT preferred, then MR, then any non-mask/non-dose image.
    background_df = readable_df[readable_df["Viewer role"].isin(["CT", "MR", "Other"])].copy()
    if background_df.empty:
        st.warning("No anatomical registered image was found to use as the background. At least one registered CT or MR is needed for overlay QC.")
        st.dataframe(readable_df[["Patient ID", "Viewer role", "Resolved registered output file", "Status"]], use_container_width=True, hide_index=True)
        return

    def _background_sort_key(row):
        role = row.get("Viewer role", "")
        if role == "CT":
            return 0
        if role == "MR":
            return 1
        return 2

    background_records = sorted(background_df.to_dict("records"), key=_background_sort_key)
    bg_index = st.selectbox(
        "Select background anatomy",
        list(range(len(background_records))),
        format_func=lambda i: _file_label(background_records[int(i)]),
        key=f"vbv_batch_overlay_background_{vbv_safe_name(selected_patient)}",
    )
    background_row = background_records[int(bg_index)]
    background_path = Path(str(background_row["Resolved registered output file"])).resolve()

    overlay_df = readable_df[readable_df["Resolved registered output file"].astype(str) != str(background_path)].copy()
    overlay_records = overlay_df.to_dict("records")
    default_overlay_indices = [
        i for i, row in enumerate(overlay_records)
        if row.get("Viewer role") in ["MR", "Dose", "Mask / segmentation"]
    ]
    selected_overlay_indices = st.multiselect(
        "Optional overlay type",
        list(range(len(overlay_records))),
        default=default_overlay_indices,
        format_func=lambda i: _file_label(overlay_records[int(i)]),
        key=f"vbv_batch_overlay_layers_{vbv_safe_name(selected_patient)}",
    )

    try:
        bg_arr, bg_img = vbv_read_preview_array(background_path)
        n_slices = int(bg_arr.shape[0])
        default_slice = max(0, n_slices // 2)
        slice_index = st.slider(
            "Slice",
            min_value=0,
            max_value=max(0, n_slices - 1),
            value=default_slice,
            key=f"vbv_batch_overlay_slice_{vbv_safe_name(selected_patient)}_{background_path.stem}_{n_slices}",
        )

        bg_slice = vbv_window_for_display(np.asarray(bg_arr[int(slice_index), :, :]))
        bg_slice = vbv_downsample_slice_for_preview(bg_slice, max_pixels=420)

        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        ax.imshow(bg_slice, cmap="gray")
        ax.axis("off")
        ax.set_title(f"Registered overlay QC: slice {slice_index + 1}/{n_slices}", fontsize=9)

        overlay_notes = []
        selected_overlay_rows = [overlay_records[int(i)] for i in selected_overlay_indices]
        for overlay_row in selected_overlay_rows:
            overlay_path = Path(str(overlay_row.get("Resolved registered output file", ""))).resolve()
            role = overlay_row.get("Viewer role", "Other")
            try:
                overlay_arr, _ = vbv_read_preview_array(overlay_path)
                if overlay_arr.shape[0] <= int(slice_index):
                    overlay_notes.append(f"Skipped {overlay_path.name}: fewer slices than background.")
                    continue
                ov_slice = np.asarray(overlay_arr[int(slice_index), :, :])
                if ov_slice.shape != np.asarray(bg_arr[int(slice_index), :, :]).shape:
                    overlay_notes.append(f"Skipped {overlay_path.name}: slice shape {ov_slice.shape} does not match background {np.asarray(bg_arr[int(slice_index), :, :]).shape}.")
                    continue
                ov_slice = vbv_downsample_slice_for_preview(ov_slice, max_pixels=420)

                if role == "Mask / segmentation":
                    mask_values = np.asarray(ov_slice)
                    mask = np.ma.masked_where(mask_values <= 0, mask_values)
                    ax.imshow(mask, cmap="autumn", alpha=0.35, interpolation="nearest")
                    unique_values = np.unique(mask_values[np.isfinite(mask_values)])
                    unique_values = unique_values[:8]
                    overlay_notes.append(f"Mask: {overlay_path.name}; labels shown: {', '.join(map(lambda x: str(round(float(x), 3)), unique_values))}")
                elif role == "Dose":
                    dose_values = np.asarray(ov_slice, dtype=float)
                    finite = dose_values[np.isfinite(dose_values)]
                    if finite.size > 0 and np.nanmax(finite) > np.nanmin(finite):
                        low, high = np.percentile(finite, [5, 99])
                        dose_values = np.clip(dose_values, low, high)
                        dose_values = (dose_values - low) / (high - low) if high > low else dose_values
                        dose_values = np.ma.masked_where(dose_values <= 0.05, dose_values)
                        ax.imshow(dose_values, cmap="jet", alpha=0.35)
                        overlay_notes.append(f"Dose: {overlay_path.name}")
                else:
                    intensity = vbv_window_for_display(ov_slice)
                    ax.imshow(intensity, cmap="gray", alpha=0.35)
                    overlay_notes.append(f"Image: {overlay_path.name}")
            except Exception as overlay_error:
                overlay_notes.append(f"Skipped {overlay_path.name}: {overlay_error}")

        left, right = st.columns([0.48, 0.52])
        with left:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with right:
            st.markdown("**Image being reviewed**")
            st.write(f"Patient: {selected_patient}")
            st.write(f"Anatomy: {_file_label(background_row)}")
            st.write(f"Size: {vbv_format_tuple(bg_img.GetSize(), 0)}")
            st.write(f"Spacing: {vbv_format_tuple(bg_img.GetSpacing(), 3)} mm")
            st.caption("The viewer displays registered outputs only. File names are hidden here to keep the QC view readable.")

    except Exception as error:
        st.error(f"Could not render registered overlay viewer: {error}")




# ============================================================
# VOXEL WARP TO CCS HELPERS
# ============================================================

def vbv_warp_to_ccs_project_folder():
    """Return the project folder used for warp-to-reference/CCS manifest outputs."""
    project_folder = st.session_state.get("voxel_project_folder", "")
    if project_folder:
        out = Path(project_folder) / "07_Warp_To_Reference_Space"
    else:
        out = Path("data") / "voxel_warp_to_reference"
    out.mkdir(parents=True, exist_ok=True)
    return out


def vbv_warp_type_from_summary_row(row):
    """Classify a registered output row for the warp manifest."""
    text = " ".join([
        str(row.get("Image role", "")),
        str(row.get("Moving image", "")),
        str(row.get("Registered output file", "")),
        str(row.get("Interpolation", "")),
    ]).lower()
    if any(term in text for term in ["mask", "seg", "segmentation", "structure", "rtstruct", "roi", "contour", "label", "oar"]):
        return "Mask / segmentation"
    if "dose" in text or "rtdose" in text:
        return "Dose"
    if any(term in text for term in ["ct", "computed"]):
        return "CT"
    if any(term in text for term in ["mr", "mri", "t1", "t2", "flair", "mpr", "bravo", "dwi", "adc"]):
        return "MR"
    return "Other image"


def vbv_load_warp_candidates_from_batch():
    """Load readable registered outputs from batch registration for warp-to-CCS confirmation.

    This intentionally reuses the same robust resolver used by the batch-registration
    overlay viewer. If the summary CSV contains stale/relative paths, or if outputs
    were recovered by scanning the registered_images folder, they should still appear
    here.
    """
    batch_df = vbv_load_saved_batch_registration_results()

    # First try the saved summary table; if paths are stale, this helper falls back
    # to scanning <project>/05_Batch_Registration/registered_images.
    readable_df = vbv_prepare_registered_outputs_for_review(batch_df)

    # If no summary exists but the folder contains registered outputs, scan directly.
    if readable_df is None or readable_df.empty:
        readable_df = vbv_registered_outputs_from_folder()

    if readable_df is None or readable_df.empty:
        return pd.DataFrame()

    rows = []
    for _, row in readable_df.iterrows():
        row_dict = row.to_dict()
        resolved_text = row_dict.get("Resolved registered output file", row_dict.get("Registered output file", ""))
        resolved = vbv_resolve_registered_output_path(resolved_text)
        if resolved is None:
            try:
                candidate = Path(str(resolved_text)).resolve()
                batch_root = vbv_batch_registration_project_folder().resolve()
                candidate.relative_to(batch_root)
                if candidate.exists() and candidate.is_file() and (candidate.name.lower().endswith(".nii") or candidate.name.lower().endswith(".nii.gz")):
                    resolved = candidate
            except Exception:
                resolved = None
        if resolved is None or not Path(resolved).exists():
            continue
        row_dict["Resolved registered output file"] = str(resolved)
        row_dict["Warp data type"] = vbv_warp_type_from_summary_row(row_dict)
        row_dict["Warp status"] = "Ready"
        rows.append(row_dict)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).drop_duplicates(subset=["Resolved registered output file"], keep="first")


def vbv_existing_warp_manifest():
    manifest_path = vbv_warp_to_ccs_project_folder() / "warp_to_reference_manifest.csv"
    if manifest_path.exists():
        try:
            return pd.read_csv(manifest_path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def vbv_save_warp_manifest(warp_df, selected_types, reference_setup):
    """Save a manifest of registered outputs confirmed as warped into CCS/reference space."""
    out_root = vbv_warp_to_ccs_project_folder()
    manifest_path = out_root / "warp_to_reference_manifest.csv"
    note_path = out_root / "warp_to_reference_note.txt"

    if warp_df is None or warp_df.empty:
        return pd.DataFrame(), manifest_path

    selected_df = warp_df[warp_df["Warp data type"].isin(selected_types)].copy()
    selected_df["Reference patient"] = reference_setup.get("reference_patient", "")
    selected_df["Reference image"] = reference_setup.get("reference_image", "")
    selected_df["CCS strategy"] = reference_setup.get("strategy", reference_setup.get("ccs_strategy", ""))
    selected_df["Warp interpretation"] = "Output already resampled/warped into the saved reference image grid by batch registration."

    preferred_cols = [
        "Patient ID", "Warp data type", "Moving image", "Reference image", "Reference patient",
        "Registration method", "Interpolation", "Transform file", "Resolved registered output file",
        "Warp status", "Warp interpretation", "Status", "Registration details", "Error message"
    ]
    cols = [c for c in preferred_cols if c in selected_df.columns]
    cols += [c for c in selected_df.columns if c not in cols]
    selected_df = selected_df[cols]
    selected_df.to_csv(manifest_path, index=False)

    note_lines = [
        "Warp to reference space / CCS summary",
        "====================================",
        "",
        f"Reference patient: {reference_setup.get('reference_patient', '')}",
        f"Reference image: {reference_setup.get('reference_image', '')}",
        f"Selected data types: {', '.join(selected_types)}",
        f"Number of confirmed warped outputs: {selected_df.shape[0]}",
        "",
        "Interpretation:",
        "The files listed in warp_to_reference_manifest.csv are registered outputs in the selected common coordinate system.",
        "CT/MR/dose were resampled with continuous-image interpolation where appropriate.",
        "Masks/segmentations must use nearest-neighbour interpolation to preserve label values.",
    ]
    note_path.write_text("\n".join(note_lines), encoding="utf-8")
    st.session_state.vbv_warp_manifest = selected_df
    st.session_state.vbv_warp_manifest_csv = str(manifest_path)
    return selected_df, manifest_path


# ============================================================
# VOXEL DOSE NORMALISATION HELPERS
# ============================================================

def vbv_dose_normalisation_project_folder():
    """Return the project folder used for dose-normalisation outputs."""
    project_folder = st.session_state.get("voxel_project_folder", "")
    if project_folder:
        out = Path(project_folder) / "08_Dose_Normalisation"
    else:
        out = Path("data") / "voxel_dose_normalisation"
    out.mkdir(parents=True, exist_ok=True)
    return out


def vbv_is_dose_like_row(row):
    """Identify registered dose outputs using robust role/name matching."""
    text_parts = []
    for col in [
        "Image role", "Image type", "Batch type", "Role", "Data type",
        "Moving image", "Registered output file", "Output file", "File name"
    ]:
        if col in row:
            text_parts.append(str(row.get(col, "")))
    text = " ".join(text_parts).lower()
    return any(term in text for term in ["dose", "rtdose", "rt dose", "rd."])


def vbv_load_registered_dose_outputs():
    """Load registered dose outputs from the batch-registration summary."""
    batch_folder = vbv_batch_registration_project_folder()
    summary_path = batch_folder / "batch_registration_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(summary_path)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if "Status" in df.columns:
        df = df[df["Status"].astype(str).str.lower().isin(["processed", "skipped"])].copy()
    dose_rows = []
    registered_root = batch_folder / "registered_images"
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        if not vbv_is_dose_like_row(row_dict):
            continue
        output_path = str(row_dict.get("Registered output file", "")).strip()
        path = Path(output_path) if output_path else None
        if path is None or not path.exists():
            file_name = Path(output_path).name if output_path else ""
            candidates = []
            if file_name:
                candidates.extend(registered_root.rglob(file_name))
            if not candidates:
                patient_text = str(row_dict.get("Patient ID", "")).lower()
                candidates.extend([
                    c for c in registered_root.rglob("*.nii*")
                    if "dose" in c.name.lower() and (not patient_text or patient_text in str(c).lower())
                ])
            path = candidates[0] if candidates else path
        if path is not None and path.exists():
            row_dict["Registered dose file"] = str(path)
            row_dict["Dose file name"] = path.name
            dose_rows.append(row_dict)
    if not dose_rows:
        return pd.DataFrame()
    return pd.DataFrame(dose_rows).drop_duplicates(subset=["Registered dose file"], keep="first")


def vbv_existing_dose_normalisation_summary():
    summary_path = vbv_dose_normalisation_project_folder() / "dose_normalisation_summary.csv"
    if summary_path.exists():
        try:
            return pd.read_csv(summary_path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def vbv_run_dose_normalisation(dose_df, scaling_factor, scaling_label, progress_bar=None, progress_text=None):
    """Apply a dose scaling factor to registered dose NIfTI files and save project outputs."""
    out_root = vbv_dose_normalisation_project_folder()
    image_root = out_root / "dose_normalised_images"
    image_root.mkdir(parents=True, exist_ok=True)
    rows = []
    total = int(dose_df.shape[0]) if dose_df is not None else 0
    for idx, (_, row) in enumerate(dose_df.iterrows(), start=1):
        patient_id = str(row.get("Patient ID", "Unknown_patient")) or "Unknown_patient"
        source_path = Path(str(row.get("Registered dose file", "")))
        patient_folder = image_root / make_safe_column_name(patient_id)
        patient_folder.mkdir(parents=True, exist_ok=True)
        output_path = patient_folder / f"dose_normalised_{idx}_{source_path.name}"
        result = {
            "Patient ID": patient_id,
            "Registered dose file": str(source_path),
            "Dose-normalised output file": str(output_path),
            "Scaling": scaling_label,
            "Scaling factor": scaling_factor,
            "Status": "Failed",
            "Error message": "",
        }
        try:
            if progress_text is not None:
                progress_text.info(f"Dose normalisation {idx}/{total}: {source_path.name}")
            image = sitk.ReadImage(str(source_path))
            scaled = sitk.Cast(image, sitk.sitkFloat32) * float(scaling_factor)
            sitk.WriteImage(scaled, str(output_path))
            result.update({
                "Status": "Processed",
                "Size": " × ".join(str(v) for v in scaled.GetSize()),
                "Spacing mm": " × ".join(f"{float(v):.3f}" for v in scaled.GetSpacing()),
            })
        except Exception as error:
            result["Error message"] = str(error)
        rows.append(result)
        if progress_bar is not None and total > 0:
            progress_bar.progress(idx / total)
    results_df = pd.DataFrame(rows)
    summary_path = out_root / "dose_normalisation_summary.csv"
    results_df.to_csv(summary_path, index=False)
    st.session_state.vbv_dose_normalisation_summary = results_df
    st.session_state.vbv_dose_normalisation_summary_csv = str(summary_path)
    if progress_text is not None:
        progress_text.success("Dose normalisation complete.")
    return results_df



# ============================================================
# VOXEL REGISTRATION QC HELPERS
# ============================================================

def vbv_registration_qc_project_folder():
    """Return the project folder used for registration QC outputs."""
    project_folder = st.session_state.get("voxel_project_folder", "")
    if project_folder:
        out = Path(project_folder) / "06_Registration_QC"
    else:
        out = Path("data") / "voxel_registration_qc"
    out.mkdir(parents=True, exist_ok=True)
    return out


def vbv_existing_registration_qc_summary():
    qc_path = vbv_registration_qc_project_folder() / "registration_qc_summary.csv"
    if qc_path.exists():
        try:
            df = pd.read_csv(qc_path)
            if df is not None and not df.empty:
                return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def vbv_geometry_key_from_file(path_text):
    """Read geometry key for QC without loading a full NumPy array."""
    result = {
        "Readable": "No",
        "Size": "",
        "Spacing mm": "",
        "Origin": "",
        "Direction": "",
        "Geometry key": "",
        "Error": "",
    }
    if not SIMPLEITK_AVAILABLE:
        result["Error"] = "SimpleITK is not available."
        return result
    try:
        img = sitk.ReadImage(str(path_text))
        size = tuple(int(v) for v in img.GetSize())
        spacing = tuple(round(float(v), 5) for v in img.GetSpacing())
        origin = tuple(round(float(v), 5) for v in img.GetOrigin())
        direction = tuple(round(float(v), 5) for v in img.GetDirection())
        result.update({
            "Readable": "Yes",
            "Size": " × ".join(str(v) for v in size),
            "Spacing mm": " × ".join(f"{v:.3f}" for v in spacing),
            "Origin": " × ".join(f"{v:.3f}" for v in origin),
            "Direction": ", ".join(f"{v:.3f}" for v in direction),
            "Geometry key": f"size={size}; spacing={spacing}; origin={origin}; direction={direction}",
        })
    except Exception as error:
        result["Error"] = str(error)
    return result


def vbv_role_group_for_qc(row):
    role_text = f"{row.get('Image role', '')} {row.get('Moving image', '')} {row.get('Registered output file', '')} {row.get('Resolved registered output file', '')}".lower()
    if any(term in role_text for term in ["mask", "seg", "segmentation", "structure", "rtstruct", "roi", "contour", "label", "oar"]):
        return "Mask / segmentation"
    if "dose" in role_text or "rtdose" in role_text:
        return "Dose"
    if "ct" in role_text:
        return "CT"
    if any(term in role_text for term in ["mr", "mri", "t1", "t2", "flair", "mpr", "bravo", "dwi", "adc"]):
        return "MR"
    return "Other"


def vbv_build_registration_qc_table(manual_review_notes=None, visual_approval_map=None):
    """Build patient-level registration QC from registered outputs.

    This is a pragmatic QC summary for the app. It checks readable outputs,
    geometry consistency after registration, and records visual approval/manual review.
    Advanced Jacobian folding checks are reported as not calculated unless a deformable
    displacement-field workflow is added later.
    """
    results_df = vbv_load_saved_batch_registration_results()
    registered_df = vbv_prepare_registered_outputs_for_review(results_df)
    if registered_df is None or registered_df.empty:
        return pd.DataFrame()

    manual_review_notes = manual_review_notes or {}
    visual_approval_map = visual_approval_map or {}

    registered_df = registered_df.copy()
    registered_df["QC role"] = registered_df.apply(vbv_role_group_for_qc, axis=1)
    patients = sorted(registered_df["Patient ID"].astype(str).unique().tolist())
    rows = []

    for patient_id in patients:
        patient_rows = registered_df[registered_df["Patient ID"].astype(str) == str(patient_id)].copy()
        geom_rows = []
        for _, r in patient_rows.iterrows():
            p = str(r.get("Resolved registered output file", r.get("Registered output file", ""))).strip()
            g = vbv_geometry_key_from_file(p) if p else {"Readable": "No", "Geometry key": "", "Error": "No path"}
            item = r.to_dict()
            item.update(g)
            geom_rows.append(item)
        geom_df = pd.DataFrame(geom_rows)

        readable_count = int((geom_df["Readable"].astype(str) == "Yes").sum()) if "Readable" in geom_df.columns else 0
        failed_count = int(geom_df.shape[0] - readable_count)
        has_anatomy = bool(geom_df[geom_df["QC role"].isin(["CT", "MR", "Other"])].shape[0] > 0)
        has_dose = bool((geom_df["QC role"].astype(str) == "Dose").any()) if "QC role" in geom_df.columns else False
        has_mask = bool((geom_df["QC role"].astype(str) == "Mask / segmentation").any()) if "QC role" in geom_df.columns else False

        readable_geom = geom_df[geom_df["Readable"].astype(str) == "Yes"].copy() if "Readable" in geom_df.columns else pd.DataFrame()
        unique_geom = sorted(readable_geom["Geometry key"].dropna().astype(str).unique().tolist()) if "Geometry key" in readable_geom.columns else []
        unique_geom = [g for g in unique_geom if g]
        geometry_consistent = "Yes" if len(unique_geom) <= 1 and readable_count > 0 else "No"
        if readable_count == 0:
            geometry_consistent = "Not assessed"

        methods = "; ".join(sorted(patient_rows.get("Registration method", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()))
        uses_bspline = "b-spline" in methods.lower() or "bspline" in methods.lower()
        jacobian_status = "Not calculated - advanced DIR QC"
        if not uses_bspline:
            jacobian_status = "Not applicable for rigid/affine-only transform"

        approval_value = visual_approval_map.get(str(patient_id), "Not reviewed")
        note_value = manual_review_notes.get(str(patient_id), "")

        issues = []
        if failed_count > 0:
            issues.append(f"{failed_count} unreadable registered output(s)")
        if not has_anatomy:
            issues.append("No registered CT/MR anatomy found")
        if geometry_consistent == "No":
            issues.append("Registered output geometries differ")
        if approval_value != "Approved":
            issues.append("Visual QC not approved")

        qc_status = "Approved" if not issues else "Needs review"
        rows.append({
            "Patient ID": patient_id,
            "Registration QC status": qc_status,
            "Visual QC approval": approval_value,
            "Manual review notes": note_value,
            "Registered outputs checked": int(geom_df.shape[0]),
            "Readable registered outputs": readable_count,
            "Unreadable registered outputs": failed_count,
            "Registered anatomy present": "Yes" if has_anatomy else "No",
            "Registered dose present": "Yes" if has_dose else "No",
            "Registered mask present": "Yes" if has_mask else "No",
            "Registered geometry consistent": geometry_consistent,
            "Jacobian/folding check": jacobian_status,
            "Registration method(s)": methods,
            "QC issue": "; ".join(issues),
        })
    return pd.DataFrame(rows)


def vbv_save_registration_qc_summary(qc_df):
    out = vbv_registration_qc_project_folder()
    qc_path = out / "registration_qc_summary.csv"
    qc_df.to_csv(qc_path, index=False)
    note = [
        "Registration QC summary",
        "=======================",
        "",
        "This file records patient-level registration QC before warp-to-reference and final VBA readiness.",
        "Visual QC approval is recorded by the user after reviewing overlay images.",
        "Jacobian/folding checks are listed as an advanced DIR QC item and are not calculated unless a displacement-field workflow is added.",
        "",
        f"Patients checked: {qc_df.shape[0] if qc_df is not None else 0}",
        f"Approved: {int((qc_df['Registration QC status'] == 'Approved').sum()) if qc_df is not None and not qc_df.empty and 'Registration QC status' in qc_df.columns else 0}",
    ]
    (out / "registration_qc_note.txt").write_text("\n".join(note), encoding="utf-8")
    st.session_state.vbv_registration_qc_summary = qc_df
    st.session_state.vbv_registration_qc_summary_csv = str(qc_path)
    return qc_path

# ============================================================
# VOXEL VBA-READY DATASET / FINAL QC HELPERS
# ============================================================

def vbv_vba_ready_project_folder():
    """Return the project folder for final VBA-ready manifest outputs."""
    project_folder = st.session_state.get("voxel_project_folder", "")
    if project_folder:
        out = Path(project_folder) / "09_VBA_Ready_Dataset_Final_QC"
    else:
        out = Path("data") / "voxel_vba_ready_dataset"
    out.mkdir(parents=True, exist_ok=True)
    return out


def vbv_load_clean_patient_clinical_data():
    """Load the saved clean clinical dataset for the active project when available."""
    df = st.session_state.get("voxel_patient_data", None)
    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
        return df.copy()

    project_folder = st.session_state.get("voxel_project_folder", "")
    if not project_folder:
        return pd.DataFrame()

    patient_folder = Path(project_folder) / "01_Load_Patient_Clinical_Data"
    candidates = [
        patient_folder / "patient_clinical_data_clean_copy.csv",
        patient_folder / "patient_clinical_data_clean_copy.xlsx",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() in [".xlsx", ".xls"]:
                loaded = pd.read_excel(path)
            else:
                loaded = pd.read_csv(path)
            if loaded is not None and not loaded.empty:
                st.session_state.voxel_patient_data = loaded
                return loaded.copy()
        except Exception:
            continue
    return pd.DataFrame()


def vbv_infer_patient_id_column(df):
    """Find the most likely patient identifier column."""
    if df is None or df.empty:
        return ""
    saved_col = st.session_state.get("voxel_patient_id_column", "")
    if saved_col and saved_col in df.columns:
        return saved_col
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for key in ["patient id", "patient_id", "patientid", "pt_id", "ptid", "id", "subject", "subject_id"]:
        if key in lower_map:
            return lower_map[key]
    return df.columns[0]


def vbv_existing_vba_ready_manifest():
    manifest_path = vbv_vba_ready_project_folder() / "vba_ready_dataset_manifest.csv"
    if manifest_path.exists():
        try:
            return pd.read_csv(manifest_path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def vbv_load_dose_normalised_outputs_for_manifest():
    """Load dose-normalised outputs produced after dose normalisation."""
    summary = vbv_existing_dose_normalisation_summary()
    if summary is None or summary.empty:
        summary = st.session_state.get("vbv_dose_normalisation_summary", pd.DataFrame())
    if summary is None or summary.empty:
        return pd.DataFrame()

    rows = []
    for _, row in summary.iterrows():
        row_dict = row.to_dict()
        status = str(row_dict.get("Status", "")).strip().lower()
        if status and status != "processed":
            continue
        path_text = str(row_dict.get("Dose-normalised output file", "")).strip()
        if not path_text:
            continue
        p = Path(path_text)
        if not p.exists():
            # Recover by filename if the project was moved.
            candidate_name = p.name
            candidates = list(vbv_dose_normalisation_project_folder().rglob(candidate_name)) if candidate_name else []
            if candidates:
                p = candidates[0]
        if p.exists():
            row_dict["Resolved dose-normalised output file"] = str(p)
            rows.append(row_dict)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["Resolved dose-normalised output file"], keep="first")


def vbv_load_reference_space_outputs_for_manifest():
    """Load all confirmed warped/registered outputs in the selected reference space."""
    warp_df = vbv_existing_warp_manifest()
    if warp_df is None or warp_df.empty:
        warp_df = st.session_state.get("vbv_warp_manifest", pd.DataFrame())
    if warp_df is None or warp_df.empty:
        warp_df = vbv_load_warp_candidates_from_batch()
    if warp_df is None or warp_df.empty:
        return pd.DataFrame()

    rows = []
    for _, row in warp_df.iterrows():
        row_dict = row.to_dict()
        path_text = ""
        for col in ["Resolved registered output file", "Registered output file", "Output file"]:
            if str(row_dict.get(col, "")).strip():
                path_text = str(row_dict.get(col, "")).strip()
                break
        if not path_text:
            continue
        p = Path(path_text)
        if not p.exists():
            candidate_name = p.name
            candidates = list(vbv_batch_registration_project_folder().rglob(candidate_name)) if candidate_name else []
            if candidates:
                p = candidates[0]
        if p.exists():
            row_dict["Resolved reference-space output file"] = str(p)
            if "Warp data type" not in row_dict or not str(row_dict.get("Warp data type", "")).strip():
                row_dict["Warp data type"] = vbv_warp_type_from_summary_row(row_dict)
            rows.append(row_dict)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["Resolved reference-space output file"], keep="first")


def vbv_header_geometry(path_text):
    """Read image header geometry without loading arrays."""
    result = {
        "Readable": "No",
        "Size": "",
        "Spacing mm": "",
        "Origin": "",
        "Direction": "",
        "Geometry key": "",
        "Geometry error": "",
    }
    if not SIMPLEITK_AVAILABLE:
        result["Geometry error"] = "SimpleITK is not available."
        return result
    try:
        img = sitk.ReadImage(str(path_text))
        size = tuple(int(v) for v in img.GetSize())
        spacing = tuple(round(float(v), 5) for v in img.GetSpacing())
        origin = tuple(round(float(v), 5) for v in img.GetOrigin())
        direction = tuple(round(float(v), 5) for v in img.GetDirection())
        result.update({
            "Readable": "Yes",
            "Size": " × ".join(str(v) for v in size),
            "Spacing mm": " × ".join(f"{v:.3f}" for v in spacing),
            "Origin": " × ".join(f"{v:.3f}" for v in origin),
            "Direction": ", ".join(f"{v:.3f}" for v in direction),
            "Geometry key": f"size={size}; spacing={spacing}; origin={origin}; direction={direction}",
        })
    except Exception as error:
        result["Geometry error"] = str(error)
    return result


def vbv_first_value_by_patient(df, patient_id, type_terms, path_col):
    """Return first matching row/path for a patient and data-type terms."""
    if df is None or df.empty or "Patient ID" not in df.columns:
        return None, None
    patient_rows = df[df["Patient ID"].astype(str) == str(patient_id)].copy()
    if patient_rows.empty:
        return None, None
    type_text = pd.Series([""] * patient_rows.shape[0], index=patient_rows.index)
    for col in ["Warp data type", "Image role", "Image type", "Batch type", "Role", "Moving image", "Registered output file", "Resolved reference-space output file"]:
        if col in patient_rows.columns:
            type_text = type_text + " " + patient_rows[col].astype(str)
    type_text = type_text.str.lower()
    mask = type_text.apply(lambda x: any(term in x for term in type_terms))
    matches = patient_rows[mask]
    if matches.empty:
        return None, None
    row = matches.iloc[0].to_dict()
    p = str(row.get(path_col, "")).strip()
    return row, p


def vbv_build_vba_ready_manifest():
    """Create the patient-level final QC manifest for the VBA-ready dataset."""
    clinical_df = vbv_load_clean_patient_clinical_data()
    patient_id_col = vbv_infer_patient_id_column(clinical_df)
    clinical_ids = []
    if patient_id_col and patient_id_col in clinical_df.columns:
        clinical_ids = clinical_df[patient_id_col].dropna().astype(str).str.strip().tolist()
        clinical_ids = [x for x in clinical_ids if x]

    ref_outputs = vbv_load_reference_space_outputs_for_manifest()
    dose_outputs = vbv_load_dose_normalised_outputs_for_manifest()
    registration_qc_df = st.session_state.get("vbv_registration_qc_summary", pd.DataFrame())
    if registration_qc_df is None or registration_qc_df.empty:
        registration_qc_df = vbv_existing_registration_qc_summary()

    ref_ids = ref_outputs["Patient ID"].dropna().astype(str).str.strip().tolist() if "Patient ID" in ref_outputs.columns else []
    dose_ids = dose_outputs["Patient ID"].dropna().astype(str).str.strip().tolist() if "Patient ID" in dose_outputs.columns else []
    all_ids = sorted(set(clinical_ids + ref_ids + dose_ids))

    rows = []
    reference_setup = st.session_state.get("vbv_reference_setup", {}) or st.session_state.get("voxel_reference_setup", {}) or {}
    reference_patient = str(reference_setup.get("reference_patient", ""))
    reference_image = str(reference_setup.get("reference_image", ""))

    for patient_id in all_ids:
        clinical_present = patient_id in set(clinical_ids)
        registration_qc_status = "Not assessed"
        visual_qc_approval = "Not reviewed"
        if registration_qc_df is not None and not registration_qc_df.empty and "Patient ID" in registration_qc_df.columns:
            qc_match = registration_qc_df[registration_qc_df["Patient ID"].astype(str) == str(patient_id)]
            if not qc_match.empty:
                registration_qc_status = str(qc_match.iloc[0].get("Registration QC status", "Not assessed"))
                visual_qc_approval = str(qc_match.iloc[0].get("Visual QC approval", "Not reviewed"))
        ct_row, ct_path = vbv_first_value_by_patient(ref_outputs, patient_id, ["ct", "computed"], "Resolved reference-space output file")
        mr_row, mr_path = vbv_first_value_by_patient(ref_outputs, patient_id, ["mr", "mri", "t1", "t2", "flair", "mpr", "bravo", "dwi", "adc"], "Resolved reference-space output file")
        mask_row, mask_path = vbv_first_value_by_patient(ref_outputs, patient_id, ["mask", "seg", "structure", "rtstruct", "roi", "contour", "label", "oar"], "Resolved reference-space output file")
        dose_row, dose_path = vbv_first_value_by_patient(dose_outputs, patient_id, ["dose", "rtdose", "rt dose"], "Resolved dose-normalised output file")

        anatomy_path = ct_path or mr_path
        geometry = vbv_header_geometry(anatomy_path) if anatomy_path else {"Readable": "No", "Size": "", "Spacing mm": "", "Origin": "", "Direction": "", "Geometry key": "", "Geometry error": "No CT/MR anatomy output found."}
        dose_geometry = vbv_header_geometry(dose_path) if dose_path else {"Readable": "No", "Geometry key": "", "Geometry error": "No dose-normalised output found."}
        mask_geometry = vbv_header_geometry(mask_path) if mask_path else {"Readable": "No", "Geometry key": "", "Geometry error": "No mask/segmentation output found."}

        geometry_match = "Not assessed"
        if anatomy_path and dose_path and geometry.get("Readable") == "Yes" and dose_geometry.get("Readable") == "Yes":
            geometry_match = "Yes" if geometry.get("Geometry key") == dose_geometry.get("Geometry key") else "No"

        issues = []
        if not clinical_present:
            issues.append("Clinical row missing")
        if not anatomy_path:
            issues.append("Reference-space CT/MR missing")
        if not dose_path:
            issues.append("Dose-normalised file missing")
        if anatomy_path and geometry.get("Readable") != "Yes":
            issues.append("Anatomy file not readable")
        if dose_path and dose_geometry.get("Readable") != "Yes":
            issues.append("Dose file not readable")
        if geometry_match == "No":
            issues.append("Dose geometry differs from anatomy/reference-space image")
        if registration_qc_status != "Approved":
            issues.append("Registration QC not approved")

        ready = "Yes" if not issues else "No"
        rows.append({
            "Patient ID": patient_id,
            "Clinical data present": "Yes" if clinical_present else "No",
            "Reference-space CT present": "Yes" if ct_path else "No",
            "Reference-space MR present": "Yes" if mr_path else "No",
            "Dose in Gy present": "Yes" if dose_path else "No",
            "Mask present": "Yes" if mask_path else "No",
            "Geometry readable": geometry.get("Readable", "No"),
            "Dose geometry matches anatomy": geometry_match,
            "Registration QC status": registration_qc_status,
            "Visual QC approval": visual_qc_approval,
            "Ready for VBA": ready,
            "Issue": "; ".join(issues),
            "Reference patient": reference_patient,
            "Reference image": reference_image,
            "Anatomy file path": anatomy_path or "",
            "CT file path": ct_path or "",
            "MR file path": mr_path or "",
            "Dose file path": dose_path or "",
            "Mask file path": mask_path or "",
            "Anatomy size": geometry.get("Size", ""),
            "Anatomy spacing mm": geometry.get("Spacing mm", ""),
            "Dose size": dose_geometry.get("Size", ""),
            "Dose spacing mm": dose_geometry.get("Spacing mm", ""),
            "Mask size": mask_geometry.get("Size", ""),
            "Mask spacing mm": mask_geometry.get("Spacing mm", ""),
            "Geometry error": geometry.get("Geometry error", ""),
            "Dose geometry error": dose_geometry.get("Geometry error", ""),
            "Mask geometry error": mask_geometry.get("Geometry error", ""),
        })

    manifest = pd.DataFrame(rows)
    out_root = vbv_vba_ready_project_folder()
    manifest_path = out_root / "vba_ready_dataset_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    note_lines = [
        "VBA-ready dataset / Final QC summary",
        "====================================",
        "",
        f"Patients listed: {manifest.shape[0]}",
        f"Ready for VBA: {int((manifest['Ready for VBA'] == 'Yes').sum()) if not manifest.empty and 'Ready for VBA' in manifest.columns else 0}",
        f"Not ready: {int((manifest['Ready for VBA'] == 'No').sum()) if not manifest.empty and 'Ready for VBA' in manifest.columns else 0}",
        "",
        "Interpretation:",
        "Ready patients have linked clinical data, a readable reference-space anatomy image, and a readable dose-normalised file.",
        "Mask availability is recorded separately because some VBA analyses may use whole-brain masks while others may use structure-specific masks.",
        "Geometry mismatches should be reviewed before voxel-wise statistics.",
    ]
    (out_root / "vba_ready_dataset_note.txt").write_text("\n".join(note_lines), encoding="utf-8")
    st.session_state.vbv_vba_ready_manifest = manifest
    st.session_state.vbv_vba_ready_manifest_csv = str(manifest_path)
    return manifest, manifest_path


# ============================================================
# VOXEL STATISTICAL ANALYSIS HELPERS
# ============================================================

def vbv_statistical_analysis_project_folder():
    """Return project folder for VBA statistical-analysis outputs."""
    project_folder = st.session_state.get("voxel_project_folder", "")
    if project_folder:
        out = Path(project_folder) / "10_Statistical_Analysis"
    else:
        out = Path("data") / "voxel_statistical_analysis"
    out.mkdir(parents=True, exist_ok=True)
    return out


def vbv_existing_statistical_analysis_summary():
    path = vbv_statistical_analysis_project_folder() / "statistical_analysis_run_summary.csv"
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def vbv_load_ready_manifest_for_stats():
    manifest = st.session_state.get("vbv_vba_ready_manifest", pd.DataFrame())
    if manifest is None or manifest.empty:
        manifest = vbv_existing_vba_ready_manifest()
    if manifest is None or manifest.empty:
        return pd.DataFrame()
    manifest = manifest.copy()
    if "Ready for VBA" in manifest.columns:
        manifest = manifest[manifest["Ready for VBA"].astype(str).str.lower().isin(["yes", "true", "approved", "1"])].copy()
    if "Dose file path" in manifest.columns:
        manifest["Dose file path"] = manifest["Dose file path"].astype(str).str.strip()
        manifest = manifest[manifest["Dose file path"] != ""].copy()
    return manifest.reset_index(drop=True)


def vbv_merge_ready_manifest_with_clinical(manifest):
    clinical_df = vbv_load_clean_patient_clinical_data()
    patient_id_col = vbv_infer_patient_id_column(clinical_df)
    if clinical_df is not None and not clinical_df.empty and patient_id_col in clinical_df.columns:
        clinical_df = clinical_df.copy()
        clinical_df[patient_id_col] = clinical_df[patient_id_col].astype(str).str.strip()
        merged = manifest.merge(clinical_df, left_on="Patient ID", right_on=patient_id_col, how="left", suffixes=("", "_clinical"))
        return merged, clinical_df, patient_id_col
    return manifest.copy(), pd.DataFrame(), ""


def vbv_write_stat_image(array_zyx, reference_img, out_path):
    out_img = sitk.GetImageFromArray(np.asarray(array_zyx, dtype=np.float32))
    out_img.CopyInformation(reference_img)
    sitk.WriteImage(out_img, str(out_path))


def vbv_load_dose_stack_for_stats(analysis_df, progress_bar=None, status_box=None):
    arrays = []
    rows = []
    errors = []
    reference_img = None
    reference_shape = None
    total = max(int(analysis_df.shape[0]), 1)
    for idx, (_, row) in enumerate(analysis_df.iterrows(), start=1):
        patient_id = str(row.get("Patient ID", ""))
        path_text = str(row.get("Dose file path", "")).strip()
        if status_box is not None:
            status_box.info(f"Loading dose map {idx}/{total}: {patient_id}")
        try:
            p = Path(path_text)
            if not p.exists():
                raise FileNotFoundError(path_text)
            img = sitk.ReadImage(str(p))
            arr = sitk.GetArrayFromImage(img).astype(np.float32)
            if reference_img is None:
                reference_img = img
                reference_shape = arr.shape
            if arr.shape != reference_shape:
                raise ValueError(f"Dose shape mismatch. Expected {reference_shape}, got {arr.shape}")
            arrays.append(arr)
            rows.append(row.to_dict())
        except Exception as error:
            errors.append({"Patient ID": patient_id, "Dose file path": path_text, "Error": str(error)})
        if progress_bar is not None:
            progress_bar.progress(min(25, int(25 * idx / total)))
    if not arrays:
        return None, None, pd.DataFrame(), pd.DataFrame(errors)
    return np.stack(arrays, axis=0), reference_img, pd.DataFrame(rows), pd.DataFrame(errors)


def vbv_residualise_outcome(y, covariate_df):
    y_numeric = pd.to_numeric(y, errors="coerce")
    valid = y_numeric.notna()
    if covariate_df is None or covariate_df.empty:
        return y_numeric, valid, "No adjustment variables selected."
    x = covariate_df.copy()
    usable_cols = []
    for col in x.columns:
        numeric = pd.to_numeric(x[col], errors="coerce")
        if numeric.notna().sum() >= 3 and numeric.nunique(dropna=True) > 1:
            x[col] = numeric
            usable_cols.append(col)
    if not usable_cols:
        return y_numeric, valid, "Adjustment variables were selected, but none were usable numeric covariates."
    valid = valid & x[usable_cols].notna().all(axis=1)
    if valid.sum() < max(3, len(usable_cols) + 2):
        return y_numeric, valid, "Too few complete cases for covariate adjustment; unadjusted outcome used."
    X = x.loc[valid, usable_cols].astype(float).to_numpy()
    Y = y_numeric.loc[valid].astype(float).to_numpy()
    X = np.column_stack([np.ones(X.shape[0]), X])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    fitted = X @ beta
    residuals = pd.Series(index=y_numeric.index, dtype=float)
    residuals.loc[valid] = Y - fitted
    return residuals, valid, f"Outcome residualised for adjustment variables: {', '.join(usable_cols)}."


def vbv_corr_vectorized(x, y, method):
    # x: patients × voxels; y: patients
    if method.lower().startswith("spearman"):
        x = stats.rankdata(x, axis=0)
        y = stats.rankdata(y)
    x = x.astype(np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - np.nanmean(x, axis=0, keepdims=True)
    y = y - np.nanmean(y)
    numer = np.nansum(x * y[:, None], axis=0)
    denom = np.sqrt(np.nansum(x ** 2, axis=0) * np.nansum(y ** 2))
    r = np.divide(numer, denom, out=np.zeros_like(numer, dtype=float), where=denom > 0)
    r = np.clip(r, -0.999999, 0.999999)
    df = max(len(y) - 2, 1)
    t = r * np.sqrt(df / np.maximum(1.0 - r ** 2, 1e-12))
    p = 2 * stats.t.sf(np.abs(t), df=df)
    return r.astype(np.float32), p.astype(np.float32)


def vbv_bh_fdr_mask(p_map, alpha, valid_mask):
    p = p_map[valid_mask]
    p = p[np.isfinite(p)]
    if p.size == 0:
        return np.zeros_like(p_map, dtype=np.float32), np.nan
    order = np.argsort(p)
    p_sorted = p[order]
    m = float(p_sorted.size)
    thresholds = alpha * (np.arange(1, p_sorted.size + 1) / m)
    passed = p_sorted <= thresholds
    if not passed.any():
        return np.zeros_like(p_map, dtype=np.float32), np.nan
    cutoff = float(p_sorted[np.where(passed)[0].max()])
    return ((p_map <= cutoff) & valid_mask).astype(np.float32), cutoff


def vbv_run_statistical_analysis(
    analysis_df,
    vba_statistic,
    outcome_column,
    group_column,
    group_a,
    group_b,
    correction_method,
    alpha,
    n_permutations,
    filter_method,
    sd_threshold,
    mean_dose_threshold,
    adjustment_columns,
    progress_bar=None,
    status_box=None,
):
    if not SIMPLEITK_AVAILABLE:
        raise ImportError("SimpleITK is required for voxel-wise statistical analysis.")
    out_root = vbv_statistical_analysis_project_folder()
    stack, reference_img, loaded_df, error_df = vbv_load_dose_stack_for_stats(analysis_df, progress_bar, status_box)
    if stack is None or reference_img is None or loaded_df.empty:
        raise ValueError("No readable dose-normalised files were available for statistical analysis.")

    if status_box is not None:
        status_box.info("Calculating mean and standard deviation dose maps...")
    mean_map = np.mean(stack, axis=0).astype(np.float32)
    sd_map = np.std(stack, axis=0, ddof=1 if stack.shape[0] > 1 else 0).astype(np.float32)
    mean_path = out_root / "mean_dose_map_for_interpretation.nii.gz"
    sd_path = out_root / "sd_dose_map_for_interpretation.nii.gz"
    vbv_write_stat_image(mean_map, reference_img, mean_path)
    vbv_write_stat_image(sd_map, reference_img, sd_path)
    if progress_bar is not None:
        progress_bar.progress(35)

    valid_voxels = np.ones(stack.shape[1:], dtype=bool)
    filter_notes = []
    if "Dose SD" in filter_method:
        valid_voxels &= sd_map >= float(sd_threshold)
        filter_notes.append(f"Dose SD >= {sd_threshold} Gy")
    if "Mean dose" in filter_method:
        valid_voxels &= mean_map >= float(mean_dose_threshold)
        filter_notes.append(f"Mean dose >= {mean_dose_threshold} Gy")
    if not filter_notes:
        filter_notes.append("No voxel filter applied")

    outputs = [
        {"Output": "Mean dose map for interpretation", "File": str(mean_path)},
        {"Output": "SD dose map for interpretation", "File": str(sd_path)},
    ]
    analysis_note = ""
    stat_map = None
    p_map = None
    sig_mask = None
    threshold_value = ""

    if vba_statistic == "Mean and SD dose maps only":
        analysis_note = "Only interpretation maps were created. No inferential voxel-wise statistic was calculated."
    elif vba_statistic in ["Spearman correlation: dose vs continuous outcome/slope", "Pearson correlation: dose vs continuous outcome"]:
        if outcome_column not in loaded_df.columns:
            raise ValueError("Selected outcome column is not available in the ready dataset.")
        y_raw = loaded_df[outcome_column]
        cov_df = loaded_df[adjustment_columns] if adjustment_columns else pd.DataFrame()
        y_use, valid_patients, adjustment_note = vbv_residualise_outcome(y_raw, cov_df)
        y_numeric = pd.to_numeric(y_use, errors="coerce")
        valid_patients = valid_patients & y_numeric.notna()
        if valid_patients.sum() < 3:
            raise ValueError("At least three patients with usable outcome values are required.")
        stack_use = stack[valid_patients.values]
        y = y_numeric.loc[valid_patients].astype(float).to_numpy()
        stat_map = np.zeros(stack.shape[1:], dtype=np.float32)
        p_map = np.ones(stack.shape[1:], dtype=np.float32)
        z_dim = stack.shape[1]
        corr_method = "Spearman" if vba_statistic.startswith("Spearman") else "Pearson"
        for z in range(z_dim):
            x = stack_use[:, z, :, :].reshape(stack_use.shape[0], -1)
            r, p = vbv_corr_vectorized(x, y, corr_method)
            stat_map[z] = r.reshape(stack.shape[2], stack.shape[3])
            p_map[z] = p.reshape(stack.shape[2], stack.shape[3])
            if progress_bar is not None:
                progress_bar.progress(35 + int(45 * (z + 1) / z_dim))
        analysis_note = f"{corr_method} voxel-wise correlation completed. {adjustment_note}"

    elif vba_statistic == "Welch t-test: dose difference between two groups":
        if group_column not in loaded_df.columns:
            raise ValueError("Selected grouping column is not available in the ready dataset.")
        groups = loaded_df[group_column].astype(str)
        mask_a = groups == str(group_a)
        mask_b = groups == str(group_b)
        if mask_a.sum() < 2 or mask_b.sum() < 2:
            raise ValueError("At least two patients are required in each group for a voxel-wise t-test.")
        stat_map = np.zeros(stack.shape[1:], dtype=np.float32)
        p_map = np.ones(stack.shape[1:], dtype=np.float32)
        z_dim = stack.shape[1]
        for z in range(z_dim):
            t_val, p_val = stats.ttest_ind(stack[mask_a.values, z, :, :], stack[mask_b.values, z, :, :], axis=0, equal_var=False, nan_policy="omit")
            stat_map[z] = np.nan_to_num(t_val, nan=0.0, posinf=0.0, neginf=0.0)
            p_map[z] = np.nan_to_num(p_val, nan=1.0, posinf=1.0, neginf=1.0)
            if progress_bar is not None:
                progress_bar.progress(35 + int(45 * (z + 1) / z_dim))
        analysis_note = f"Welch t-test completed: {group_a} vs {group_b}."
    else:
        raise ValueError(f"Unsupported VBA statistic: {vba_statistic}")

    if stat_map is not None and p_map is not None:
        valid_mask = valid_voxels & np.isfinite(p_map)
        if correction_method == "Uncorrected p-value threshold":
            sig_mask = ((p_map <= float(alpha)) & valid_mask).astype(np.float32)
            threshold_value = f"p <= {alpha} uncorrected"
        elif correction_method == "Bonferroni":
            n_tests = max(int(valid_mask.sum()), 1)
            corrected_alpha = float(alpha) / n_tests
            sig_mask = ((p_map <= corrected_alpha) & valid_mask).astype(np.float32)
            threshold_value = f"p <= {corrected_alpha:.3e} Bonferroni"
        elif correction_method == "FDR Benjamini-Hochberg":
            sig_mask, cutoff = vbv_bh_fdr_mask(p_map, float(alpha), valid_mask)
            threshold_value = "No FDR-significant voxels" if pd.isna(cutoff) else f"p <= {cutoff:.3e} FDR"
        elif correction_method == "Permutation max-statistic threshold":
            # Conservative placeholder implementation: saves requested permutation settings and uses uncorrected mask for display.
            # Full max-statistic permutation is intentionally separated because it can take a long time on full 3D dose maps.
            sig_mask = ((p_map <= float(alpha)) & valid_mask).astype(np.float32)
            threshold_value = f"Permutation settings saved ({n_permutations} permutations requested); display mask uses p <= {alpha} until full permutation engine is run."
        else:
            sig_mask = ((p_map <= float(alpha)) & valid_mask).astype(np.float32)
            threshold_value = f"p <= {alpha}"

        safe_name = vba_statistic.lower().replace(" ", "_").replace(":", "").replace("/", "_").replace("→", "to").replace("-", "_")
        stat_path = out_root / f"{safe_name}_statistic_map.nii.gz"
        p_path = out_root / f"{safe_name}_p_value_map.nii.gz"
        sig_path = out_root / f"{safe_name}_significance_mask.nii.gz"
        vbv_write_stat_image(stat_map, reference_img, stat_path)
        vbv_write_stat_image(p_map, reference_img, p_path)
        vbv_write_stat_image(sig_mask, reference_img, sig_path)
        outputs.extend([
            {"Output": "Voxel-wise statistic map", "File": str(stat_path)},
            {"Output": "Voxel-wise p-value map", "File": str(p_path)},
            {"Output": "Corrected/display significance mask", "File": str(sig_path)},
        ])

    options = {
        "VBA statistic": vba_statistic,
        "Outcome column": outcome_column,
        "Group column": group_column,
        "Group A": group_a,
        "Group B": group_b,
        "Multiple-comparison correction": correction_method,
        "Alpha": alpha,
        "Permutations requested": n_permutations,
        "Filter": filter_method,
        "Dose SD threshold": sd_threshold,
        "Mean dose threshold": mean_dose_threshold,
        "Adjustment variables": "; ".join(adjustment_columns),
        "Filter notes": "; ".join(filter_notes),
        "Threshold/display rule": threshold_value,
        "Analysis note": analysis_note,
    }
    options_path = out_root / "statistical_analysis_options.csv"
    pd.DataFrame([options]).to_csv(options_path, index=False)
    outputs.append({"Output": "Statistical analysis options", "File": str(options_path)})
    outputs_df = pd.DataFrame(outputs)
    outputs_csv = out_root / "statistical_analysis_output_files.csv"
    outputs_df.to_csv(outputs_csv, index=False)

    summary_row = {
        "Run time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Patients analysed": int(loaded_df.shape[0]),
        "VBA statistic": vba_statistic,
        "Correction": correction_method,
        "Filter": filter_method,
        "Adjustment variables": "; ".join(adjustment_columns),
        "Output files CSV": str(outputs_csv),
        "Notes": analysis_note,
    }
    summary_path = out_root / "statistical_analysis_run_summary.csv"
    existing = vbv_existing_statistical_analysis_summary()
    summary_df = pd.concat([existing, pd.DataFrame([summary_row])], ignore_index=True)
    summary_df.to_csv(summary_path, index=False)
    if progress_bar is not None:
        progress_bar.progress(100)
    if status_box is not None:
        status_box.success("Statistical analysis complete.")
    return outputs_df, summary_df, error_df, summary_path



# ============================================================
# KNOWLEDGE BASE DOCUMENT MANAGER HELPERS
# ============================================================

def render_kb_document_manager(category_name, show_heading=True):
    """Render saved-document cards and add/edit form for a Knowledge Base category."""
    if "knowledge_base_documents" not in st.session_state:
        st.session_state.knowledge_base_documents = {
            "Outcome assessment tools": [],
            "Model development tools": [],
            "Model evaluation tools": [],
            "Clinical applications": [],
            "Other": [],
        }

    if category_name not in st.session_state.knowledge_base_documents:
        st.session_state.knowledge_base_documents[category_name] = []

    icon_map = {
        "Outcome assessment tools": "🧠",
        "Model development tools": "🛠️",
        "Model evaluation tools": "📏",
        "Clinical applications": "🏥",
        "Clinical applications - Brain": "🧠",
        "Clinical applications - Head and neck": "🗣️",
        "Clinical applications - Thorax and abdomen": "🫁",
        "Clinical applications - Pelvis": "🦴",
        "Clinical applications - Other": "📦",
        "Other": "📦",
    }

    if show_heading:
        st.markdown(f"## {icon_map.get(category_name, '📚')} {category_name}")

    safe_category = make_safe_column_name(category_name)
    edit_key = f"kb_edit_index_{safe_category}"
    documents = st.session_state.knowledge_base_documents.get(category_name, [])

    st.markdown("### Saved documents")

    if len(documents) == 0:
        st.info("No documents added yet.")
    else:
        for idx, doc in enumerate(documents):
            doc_name = doc.get("name", "")
            doc_summary = doc.get("summary", "")
            doc_link = doc.get("link", "")
            doc_filename = doc.get("filename", "")

            with st.container(border=True):
                title_col, edit_col, delete_col = st.columns([0.82, 0.09, 0.09], vertical_alignment="top")
                with title_col:
                    st.markdown(f"### {idx + 1}. {doc_name if doc_name else 'Untitled document'}")
                with edit_col:
                    if st.button("✏️", key=f"kb_edit_doc_{safe_category}_{idx}", help="Edit this document"):
                        st.session_state[edit_key] = idx
                        st.session_state[f"kb_doc_name_{safe_category}"] = doc.get("name", "")
                        st.session_state[f"kb_doc_summary_{safe_category}"] = doc.get("summary", "")
                        st.session_state[f"kb_doc_link_{safe_category}"] = doc.get("link", "")
                        st.rerun()
                with delete_col:
                    if st.button("🗑️", key=f"kb_delete_doc_{safe_category}_{idx}", help="Delete this document"):
                        st.session_state.knowledge_base_documents[category_name].pop(idx)
                        if st.session_state.get(edit_key) == idx:
                            st.session_state.pop(edit_key, None)
                        save_knowledge_base_documents_persistent()
                        st.success("Document deleted.")
                        st.rerun()

                st.markdown("**Summary**")
                st.write(doc_summary if doc_summary else "No summary added.")

                if doc_link:
                    st.markdown("**Link**")
                    st.markdown(f"[Open link]({doc_link})")
                    st.caption(doc_link)

                if doc_filename:
                    st.markdown("**Uploaded file**")
                    st.write(doc_filename)

    st.divider()

    editing_idx = st.session_state.get(edit_key, None)
    editing_doc = None
    if isinstance(editing_idx, int) and 0 <= editing_idx < len(documents):
        editing_doc = documents[editing_idx]

    form_title = "Edit document" if editing_doc is not None else "Add new document"
    with st.expander(form_title, expanded=True):
        if editing_doc is not None:
            st.info(f"Editing: {editing_doc.get('name', 'Untitled document')}")

        doc_name = st.text_input("Name of document", key=f"kb_doc_name_{safe_category}")
        doc_summary = st.text_area("Summary", height=140, key=f"kb_doc_summary_{safe_category}")
        doc_link = st.text_input("Link", placeholder="https://...", key=f"kb_doc_link_{safe_category}")
        uploaded_file = st.file_uploader(
            "Optional: upload or replace document file",
            type=["pdf", "docx", "doc", "txt", "xlsx", "xls", "csv"],
            key=f"kb_doc_file_{safe_category}"
        )

        if editing_doc is not None and editing_doc.get("filename", ""):
            st.caption(f"Current uploaded file: {editing_doc.get('filename', '')}")

        save_label = "Update document" if editing_doc is not None else "Save document"
        save_col, cancel_col = st.columns([0.7, 0.3])

        with save_col:
            if st.button(save_label, type="primary", use_container_width=True, key=f"kb_save_doc_{safe_category}"):
                if doc_name.strip() == "":
                    st.error("Please enter the name of the document.")
                else:
                    if editing_doc is not None:
                        new_doc = dict(editing_doc)
                        new_doc["name"] = doc_name.strip()
                        new_doc["summary"] = doc_summary.strip()
                        new_doc["link"] = doc_link.strip()
                    else:
                        new_doc = {
                            "name": doc_name.strip(),
                            "summary": doc_summary.strip(),
                            "link": doc_link.strip(),
                            "filename": "",
                            "file_bytes": None,
                        }

                    if uploaded_file is not None:
                        new_doc["filename"] = uploaded_file.name
                        new_doc["file_bytes"] = uploaded_file.read()

                    if editing_doc is not None:
                        st.session_state.knowledge_base_documents[category_name][editing_idx] = new_doc
                        st.session_state.pop(edit_key, None)
                        st.success(f"Updated document: {doc_name.strip()}")
                    else:
                        st.session_state.knowledge_base_documents[category_name].append(new_doc)
                        st.success(f"Saved document: {doc_name.strip()}")

                    for clear_key in [f"kb_doc_name_{safe_category}", f"kb_doc_summary_{safe_category}", f"kb_doc_link_{safe_category}"]:
                        st.session_state.pop(clear_key, None)

                    save_knowledge_base_documents_persistent()
                    st.rerun()

        with cancel_col:
            if editing_doc is not None:
                if st.button("Cancel edit", use_container_width=True, key=f"kb_cancel_edit_{safe_category}"):
                    st.session_state.pop(edit_key, None)
                    for clear_key in [f"kb_doc_name_{safe_category}", f"kb_doc_summary_{safe_category}", f"kb_doc_link_{safe_category}"]:
                        st.session_state.pop(clear_key, None)
                    st.rerun()



# ============================================================
# CLINICAL PROJECT HELPERS
# ============================================================

def clinical_project_folder_from_state():
    """Return active clinical project folder as a Path, or None if no project is active."""
    folder = st.session_state.get("clinical_project_folder", "")
    if not folder:
        return None
    try:
        return Path(folder)
    except Exception:
        return None


def clinical_project_subfolders():
    """Folder layout matching the clinical workflow navigation."""
    return [
        "00_Start_Open_Project",
        "01_Upload_Excel",
        "02_Variables_Outcomes",
        "03_Treatment_Groups",
        "04_Statistics",
        "05_Machine_Learning",
        "06_Model_Comparison",
        "07_Reports_Logs",
    ]


def create_clinical_project_structure(project_folder):
    project_folder = Path(project_folder)
    project_folder.mkdir(parents=True, exist_ok=True)
    for folder in clinical_project_subfolders():
        (project_folder / folder).mkdir(parents=True, exist_ok=True)
    return project_folder


def write_clinical_project_setup_note(project_folder, project_setup):
    project_folder = create_clinical_project_structure(project_folder)
    note_path = project_folder / "00_Start_Open_Project" / "clinical_project_setup_notes.txt"
    lines = [
        "Clinical model project setup",
        "============================",
        "",
        f"Project name: {project_setup.get('Project name', '')}",
        f"Clinical focus: {project_setup.get('Clinical focus', '')}",
        f"Outcome of interest: {project_setup.get('Outcome of interest', '')}",
        f"Short description: {project_setup.get('Short description', '')}",
        "",
        "Folder structure:",
    ]
    lines.extend([f"- {folder}" for folder in clinical_project_subfolders()])
    note_path.write_text("\n".join(lines), encoding="utf-8")
    return note_path


def restore_clinical_project_from_folder(project_folder):
    """Open an existing clinical project and reload saved pathway state when available."""
    project_folder = create_clinical_project_structure(project_folder)
    st.session_state.clinical_project_folder = str(project_folder)
    st.session_state.clinical_project_setup = {
        "Project name": project_folder.name,
        "Project folder": str(project_folder),
    }

    # Prefer the latest derived dataset if Step 2 was completed, otherwise use the clean upload copy.
    derived_csv = project_folder / "02_Variables_Outcomes" / "clinical_data_with_derived_outcomes.csv"
    clean_csv = project_folder / "01_Upload_Excel" / "clinical_data_clean_copy.csv"
    for data_path in [derived_csv, clean_csv]:
        if data_path.exists():
            try:
                loaded = pd.read_csv(data_path)
                if loaded is not None and not loaded.empty:
                    st.session_state.df = loaded
                    st.session_state.clinical_restored_dataset_path = str(data_path)
                    st.session_state.clinical_upload_saved_folder = str(project_folder / "01_Upload_Excel")
                    # Try to show the original uploaded Excel filename if it exists; otherwise show the restored CSV.
                    upload_folder = project_folder / "01_Upload_Excel"
                    excel_candidates = sorted(list(upload_folder.glob("*.xlsx")) + list(upload_folder.glob("*.xls")))
                    if excel_candidates:
                        st.session_state.clinical_uploaded_filename = excel_candidates[0].name
                    else:
                        st.session_state.clinical_uploaded_filename = data_path.name
                    break
            except Exception:
                pass

    variable_settings = clinical_read_json(project_folder / "02_Variables_Outcomes" / "variables_outcomes_settings.json", {})
    if variable_settings:
        st.session_state.input_variables = variable_settings.get("input_variables", [])
        st.session_state.outcome_mapping_rows = variable_settings.get("outcome_mapping_rows", [])
        st.session_state.baseline_variables = variable_settings.get("baseline_variables", [])
        st.session_state.followup_variables = variable_settings.get("followup_variables", [])
        st.session_state.decline_threshold = variable_settings.get("decline_threshold", st.session_state.get("decline_threshold", 2.0))
        st.session_state.generated_change_columns = variable_settings.get("generated_change_columns", [])
        st.session_state.generated_decline_columns = variable_settings.get("generated_decline_columns", [])
        st.session_state.primary_derived_outcome = variable_settings.get("primary_derived_outcome", "")
        st.session_state.selected_outcome_variable = variable_settings.get("selected_outcome_variable", st.session_state.primary_derived_outcome)
        summary_path = project_folder / "02_Variables_Outcomes" / "generated_outcome_mapping_table.csv"
        if summary_path.exists():
            try:
                st.session_state.generated_outcome_mapping_table = pd.read_csv(summary_path)
            except Exception:
                pass

    treatment_settings = clinical_read_json(project_folder / "03_Treatment_Groups" / "treatment_group_settings.json", {})
    if treatment_settings:
        st.session_state.treatment_variable = treatment_settings.get("treatment_variable", "")
        st.session_state.treatment_options = treatment_settings.get("treatment_options", [])

    # Reload commonly used statistics results when they exist.
    stats_folder = project_folder / "04_Statistics"
    for state_key, filename in [
        ("descriptive_results_df", "descriptive_statistics.csv"),
        ("inferential_results_df", "inferential_statistics.csv"),
    ]:
        path = stats_folder / filename
        if path.exists():
            try:
                st.session_state[state_key] = pd.read_csv(path)
            except Exception:
                pass

    tp_folder = stats_folder / "timepoint_analysis"
    for state_key, filename in [
        ("timepoint_between_results", "timepoint_between_treatment_change.csv"),
        ("timepoint_within_results", "timepoint_within_treatment_paired_tests.csv"),
        ("timepoint_analysis_df", "timepoint_analysis_dataset.csv"),
    ]:
        path = tp_folder / filename
        if path.exists():
            try:
                st.session_state[state_key] = pd.read_csv(path)
            except Exception:
                pass
    tp_settings = clinical_read_json(tp_folder / "timepoint_analysis_settings.json", {})
    if tp_settings:
        st.session_state.timepoint_selected_baseline_col = tp_settings.get("baseline_col", "")
        st.session_state.timepoint_selected_followup_cols = tp_settings.get("followup_cols", [])
        st.session_state.timepoint_change_cols = tp_settings.get("generated_change_cols", [])

    return project_folder


def save_clinical_upload_outputs(df, uploaded_filename="", uploaded_bytes=None):
    """Save uploaded/clean clinical data and QC outputs inside the active clinical project folder."""
    project_folder = clinical_project_folder_from_state()
    if project_folder is None:
        return None

    project_folder = create_clinical_project_structure(project_folder)
    out_folder = project_folder / "01_Upload_Excel"
    out_folder.mkdir(parents=True, exist_ok=True)

    if uploaded_filename and uploaded_bytes is not None:
        safe_name = Path(uploaded_filename).name
        try:
            (out_folder / safe_name).write_bytes(uploaded_bytes)
        except Exception:
            pass

    clean_csv = out_folder / "clinical_data_clean_copy.csv"
    df.to_csv(clean_csv, index=False)

    qc_df, summary, missing_table, numeric_summary = run_clinical_excel_qc(df)
    qc_df.to_csv(out_folder / "clinical_upload_qc_report.csv", index=False)
    missing_table.to_csv(out_folder / "clinical_missingness_details.csv", index=False)
    numeric_summary.to_csv(out_folder / "clinical_numeric_variable_summary.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_folder / "clinical_upload_summary.csv", index=False)

    st.session_state.clinical_upload_saved_folder = str(out_folder)
    return out_folder


def clinical_write_json(path, data):
    """Write JSON safely, overwriting the previous version."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass


def clinical_read_json(path, default=None):
    """Read JSON safely."""
    if default is None:
        default = {}
    try:
        if Path(path).exists():
            return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def save_clinical_variables_outputs(derived_df, input_variables, mapping_rows, selected_baselines, selected_followups, decline_threshold, generated_changes, generated_outcomes, generated_summary, primary_derived_outcome):
    """Save Step 2 variables/outcomes outputs inside the active clinical project. Overwrites previous files."""
    project_folder = clinical_project_folder_from_state()
    if project_folder is None:
        return None
    out_folder = create_clinical_project_structure(project_folder) / "02_Variables_Outcomes"
    out_folder.mkdir(parents=True, exist_ok=True)
    derived_df.to_csv(out_folder / "clinical_data_with_derived_outcomes.csv", index=False)
    pd.DataFrame(mapping_rows).to_csv(out_folder / "outcome_mapping_rows.csv", index=False)
    generated_summary.to_csv(out_folder / "generated_outcome_mapping_table.csv", index=False)
    settings = {
        "input_variables": list(input_variables),
        "outcome_mapping_rows": list(mapping_rows),
        "baseline_variables": list(selected_baselines),
        "followup_variables": list(selected_followups),
        "decline_threshold": float(decline_threshold),
        "generated_change_columns": list(generated_changes),
        "generated_decline_columns": list(generated_outcomes),
        "primary_derived_outcome": primary_derived_outcome,
        "selected_outcome_variable": primary_derived_outcome,
    }
    clinical_write_json(out_folder / "variables_outcomes_settings.json", settings)
    st.session_state.clinical_variables_saved_folder = str(out_folder)
    return out_folder


def save_clinical_treatment_outputs(df, treatment_variable, selected_treatment_values):
    """Save Step 3 treatment-group settings and counts. Overwrites previous files."""
    project_folder = clinical_project_folder_from_state()
    if project_folder is None:
        return None
    out_folder = create_clinical_project_structure(project_folder) / "03_Treatment_Groups"
    out_folder.mkdir(parents=True, exist_ok=True)
    settings = {
        "treatment_variable": treatment_variable,
        "treatment_options": list(selected_treatment_values),
    }
    clinical_write_json(out_folder / "treatment_group_settings.json", settings)
    if treatment_variable and treatment_variable in df.columns:
        counts = df[treatment_variable].dropna().astype(str).str.strip().value_counts().reset_index()
        counts.columns = ["Treatment option", "Count"]
        counts.to_csv(out_folder / "treatment_group_counts.csv", index=False)
    st.session_state.clinical_treatment_saved_folder = str(out_folder)
    return out_folder


def save_clinical_statistics_output(results_df, filename, settings=None):
    """Save Step 4 statistics outputs in the clinical project. Overwrites previous files."""
    project_folder = clinical_project_folder_from_state()
    if project_folder is None or results_df is None:
        return None
    out_folder = create_clinical_project_structure(project_folder) / "04_Statistics"
    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = out_folder / filename
    results_df.to_csv(out_path, index=False)
    if settings is not None:
        clinical_write_json(out_folder / (Path(filename).stem + "_settings.json"), settings)
    st.session_state.clinical_statistics_saved_folder = str(out_folder)
    return out_path


def save_clinical_timepoint_outputs(analysis_df, between_results, within_results, settings):
    """Save Step 4C timepoint outputs. Overwrites previous files."""
    project_folder = clinical_project_folder_from_state()
    if project_folder is None:
        return None
    out_folder = create_clinical_project_structure(project_folder) / "04_Statistics" / "timepoint_analysis"
    out_folder.mkdir(parents=True, exist_ok=True)
    analysis_df.to_csv(out_folder / "timepoint_analysis_dataset.csv", index=False)
    between_results.to_csv(out_folder / "timepoint_between_treatment_change.csv", index=False)
    within_results.to_csv(out_folder / "timepoint_within_treatment_paired_tests.csv", index=False)
    clinical_write_json(out_folder / "timepoint_analysis_settings.json", settings)
    st.session_state.clinical_timepoint_saved_folder = str(out_folder)
    return out_folder


def save_clinical_model_outputs(method, pipeline, predictors, outcome_variable, metrics, model_settings, model_results):
    """Save Step 5 machine-learning model artefacts inside the clinical project. Overwrites previous files."""
    project_folder = clinical_project_folder_from_state()
    if project_folder is None:
        return None
    out_folder = create_clinical_project_structure(project_folder) / "05_Machine_Learning" / make_safe_column_name(method)
    out_folder.mkdir(parents=True, exist_ok=True)
    try:
        with open(out_folder / "trained_pipeline.pkl", "wb") as f:
            pickle.dump(pipeline, f)
    except Exception:
        pass
    if model_results is not None:
        try:
            model_results.to_csv(out_folder / "model_coefficients_or_importance.csv", index=False)
        except Exception:
            pass
    performance_rows = metrics.get("Performance table", []) if isinstance(metrics, dict) else []
    if performance_rows:
        pd.DataFrame(performance_rows).to_csv(out_folder / "model_performance_table.csv", index=False)
    safe_metrics = {}
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            if k.lower().endswith("confusion matrix"):
                try:
                    safe_metrics[k] = np.asarray(v).tolist()
                except Exception:
                    safe_metrics[k] = str(v)
            elif k != "Performance table":
                safe_metrics[k] = v
    clinical_write_json(out_folder / "model_settings_and_metrics.json", {
        "method": method,
        "predictors": list(predictors),
        "outcome_variable": outcome_variable,
        "model_settings": model_settings,
        "metrics": safe_metrics,
    })
    st.session_state.clinical_machine_learning_saved_folder = str(out_folder)
    return out_folder


def clinical_model_comparison_folder():
    """Folder for Step 6 model comparison outputs."""
    project_folder = clinical_project_folder_from_state()
    if project_folder is None:
        return None
    out_folder = create_clinical_project_structure(project_folder) / "06_Model_Comparison"
    out_folder.mkdir(parents=True, exist_ok=True)
    return out_folder


def load_generated_clinical_models_for_comparison():
    """Load generated clinical risk models from session state and the active clinical project folder."""
    models = {}

    # Models created in the current Streamlit session.
    setup = st.session_state.get("clinical_project_setup", {}) or {}
    project_name = str(setup.get("Project name", "")).strip() or "Clinical model"
    for name, payload in st.session_state.get("trained_models", {}).items():
        if isinstance(payload, dict):
            method = payload.get("method", name) or name
            label = clinical_canonical_established_model_name(project_name, method)
            models[label] = payload

    current_payload = st.session_state.get("step5_calculated_model", None)
    if isinstance(current_payload, dict):
        method = current_payload.get("method", st.session_state.get("step5_calculated_model_name", "Current model"))
        label = clinical_canonical_established_model_name(project_name, method)
        models[label] = current_payload

    # Models saved in the clinical project folder.
    project_folder = clinical_project_folder_from_state()
    if project_folder is not None:
        ml_root = create_clinical_project_structure(project_folder) / "05_Machine_Learning"
        if ml_root.exists():
            for model_dir in sorted([x for x in ml_root.iterdir() if x.is_dir()]):
                settings_path = model_dir / "model_settings_and_metrics.json"
                pipe_path = model_dir / "trained_pipeline.pkl"
                results_path = model_dir / "model_coefficients_or_importance.csv"
                if not settings_path.exists():
                    continue
                info = clinical_read_json(settings_path, {})
                method = info.get("method", model_dir.name)
                payload = {
                    "method": method,
                    "predictors": info.get("predictors", []),
                    "outcome_variable": info.get("outcome_variable", ""),
                    "metrics": info.get("metrics", {}),
                    "model_settings": info.get("model_settings", {}),
                    "model_results": pd.DataFrame(),
                    "pipeline": None,
                    "source_folder": str(model_dir),
                }
                if pipe_path.exists():
                    try:
                        with open(pipe_path, "rb") as f:
                            payload["pipeline"] = pickle.load(f)
                    except Exception:
                        payload["pipeline"] = None
                if results_path.exists():
                    try:
                        payload["model_results"] = pd.read_csv(results_path)
                    except Exception:
                        payload["model_results"] = pd.DataFrame()
                label = clinical_canonical_established_model_name(project_name, method)
                models[label] = payload

    return models


def clinical_model_comparison_table(models):
    """Create a compact side-by-side comparison table for generated clinical models."""
    rows = []
    for label, payload in models.items():
        metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
        predictors = payload.get("predictors", []) if isinstance(payload, dict) else []
        rows.append({
            "Model": label,
            "Method": payload.get("method", label),
            "Outcome": payload.get("outcome_variable", ""),
            "Predictors": len(predictors),
            "Predictor list": "; ".join([str(x) for x in predictors]),
            "Training rows": metrics.get("Training rows", ""),
            "Validation rows": metrics.get("Validation rows", ""),
            "Training AUC": metrics.get("Training AUC", ""),
            "Validation AUC": metrics.get("Validation AUC", metrics.get("AUC", "")),
            "Training accuracy": metrics.get("Training Accuracy", ""),
            "Validation accuracy": metrics.get("Validation Accuracy", metrics.get("Accuracy", "")),
            "Validation sensitivity": metrics.get("Validation Sensitivity", metrics.get("Sensitivity", "")),
            "Validation specificity": metrics.get("Validation Specificity", metrics.get("Specificity", "")),
            "Validation F1": metrics.get("Validation F1 Score", metrics.get("F1 Score", metrics.get("F1", ""))),
            "Brier score": metrics.get("Validation Brier Score", metrics.get("Brier Score", metrics.get("Brier score", ""))),
            "Source folder": payload.get("source_folder", ""),
        })
    return pd.DataFrame(rows)



def clinical_model_comparison_matrix(comparison_df, include_export_row=False):
    """Return a Parameter x Model matrix for easier clinical model comparison."""
    if comparison_df is None or comparison_df.empty or "Model" not in comparison_df.columns:
        return pd.DataFrame()
    parameters = [
        "Method", "Outcome", "Predictors", "Training rows", "Validation rows",
        "Training AUC", "Validation AUC", "Training accuracy", "Validation accuracy",
        "Validation sensitivity", "Validation specificity", "Validation F1", "Brier score",
        "Predictor list", "Source folder",
    ]
    parameters = [p for p in parameters if p in comparison_df.columns]
    rows = []
    for parameter in parameters:
        row = {"Parameter": parameter}
        for _, model_row in comparison_df.iterrows():
            model_label = str(model_row.get("Model", "Model"))
            value = model_row.get(parameter, "")
            if isinstance(value, (float, np.floating)) and not pd.isna(value):
                row[model_label] = round(float(value), 3)
            else:
                row[model_label] = value
        rows.append(row)

    if include_export_row:
        export_row = {"Parameter": "Export to Established Model"}
        for _, model_row in comparison_df.iterrows():
            model_label = str(model_row.get("Model", "Model"))
            export_row[model_label] = "Use export button"
        rows.append(export_row)

    return pd.DataFrame(rows)


def clinical_model_comparison_highlighter(matrix_df):
    """Highlight best-performing model cells in orange for each performance criterion."""
    if matrix_df is None or matrix_df.empty or "Parameter" not in matrix_df.columns:
        return matrix_df
    higher_is_better = {
        "Training AUC", "Validation AUC", "Training accuracy", "Validation accuracy",
        "Validation sensitivity", "Validation specificity", "Validation F1",
    }
    lower_is_better = {"Brier score"}
    model_cols = [c for c in matrix_df.columns if c != "Parameter"]

    def style_row(row):
        styles = [""] * len(row)
        parameter = str(row.get("Parameter", ""))
        if parameter not in higher_is_better and parameter not in lower_is_better:
            return styles
        values = pd.to_numeric(row[model_cols], errors="coerce")
        if values.notna().sum() == 0:
            return styles
        best_value = values.max() if parameter in higher_is_better else values.min()
        for idx, col in enumerate(matrix_df.columns):
            if col == "Parameter":
                continue
            val = values.get(col, np.nan)
            if pd.notna(val) and np.isclose(float(val), float(best_value), equal_nan=False):
                styles[idx] = "background-color: #ffb347; color: black; font-weight: bold"
        return styles

    return matrix_df.style.apply(style_row, axis=1)

def render_clinical_model_comparison_interactive_matrix(matrix_df, models):
    """Render a table-like model comparison matrix with real Export buttons inside the final row."""
    if matrix_df is None or matrix_df.empty or "Parameter" not in matrix_df.columns:
        st.info("No model comparison table could be created.")
        return

    model_cols = [c for c in matrix_df.columns if c != "Parameter"]
    model_payloads = {str(label): payload for label, payload in models.items()}
    higher_is_better = {
        "Training AUC", "Validation AUC", "Training accuracy", "Validation accuracy",
        "Validation sensitivity", "Validation specificity", "Validation F1",
    }
    lower_is_better = {"Brier score"}
    widths = [1.45] + [1.0] * len(model_cols)

    st.markdown(
        """
        <style>
        .clinical-matrix-cell {
            border: 1px solid rgba(120,120,120,0.35);
            border-radius: 8px;
            padding: 0.55rem 0.65rem;
            min-height: 2.75rem;
            display: flex;
            align-items: center;
            overflow-wrap: anywhere;
            font-size: 0.92rem;
        }
        .clinical-matrix-header {
            font-weight: 700;
            background: rgba(120,120,120,0.13);
        }
        .clinical-matrix-best {
            background: #ffb347;
            color: black;
            font-weight: 700;
        }
        .clinical-matrix-param {
            font-weight: 650;
        }
        .clinical-matrix-export-label {
            background: rgba(255,179,71,0.20);
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def _format_value(value):
        if isinstance(value, (float, np.floating)) and not pd.isna(value):
            return f"{float(value):.3f}"
        if pd.isna(value):
            return ""
        return str(value)

    def _cell(text, css=""):
        safe_text = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(f'<div class="clinical-matrix-cell {css}">{safe_text}</div>', unsafe_allow_html=True)

    header_cols = st.columns(widths)
    with header_cols[0]:
        _cell("Parameter", "clinical-matrix-header")
    for i, model_col in enumerate(model_cols, start=1):
        with header_cols[i]:
            _cell(model_col, "clinical-matrix-header")

    for row_idx, row in matrix_df.iterrows():
        parameter = str(row.get("Parameter", ""))
        cols = st.columns(widths)
        with cols[0]:
            label_css = "clinical-matrix-param"
            if parameter == "Export to Established Model":
                label_css += " clinical-matrix-export-label"
            _cell(parameter, label_css)

        values = pd.to_numeric(row[model_cols], errors="coerce") if parameter in higher_is_better.union(lower_is_better) else pd.Series(dtype=float)
        best_value = None
        if parameter in higher_is_better and values.notna().sum() > 0:
            best_value = values.max()
        elif parameter in lower_is_better and values.notna().sum() > 0:
            best_value = values.min()

        for i, model_col in enumerate(model_cols, start=1):
            with cols[i]:
                if parameter == "Export to Established Model":
                    payload = model_payloads.get(str(model_col), {})
                    suggested_export_name = clinical_default_established_model_name(
                        payload.get("method", str(model_col)) if isinstance(payload, dict) else str(model_col),
                        payload.get("outcome_variable", "") if isinstance(payload, dict) else "",
                    )
                    if st.button("Export", use_container_width=True, key=f"clinical_step6_export_in_matrix_{row_idx}_{make_safe_column_name(model_col)}"):
                        ok, msg = clinical_export_model_to_established(suggested_export_name, payload, overwrite=False)
                        if ok:
                            st.success(msg)
                        else:
                            st.warning(msg)
                else:
                    css = ""
                    if best_value is not None:
                        val = values.get(model_col, np.nan)
                        if pd.notna(val) and np.isclose(float(val), float(best_value), equal_nan=False):
                            css = "clinical-matrix-best"
                    _cell(_format_value(row.get(model_col, "")), css)




    pending = st.session_state.get("clinical_pending_established_export", None)
    if isinstance(pending, dict) and pending.get("model_name"):
        st.warning(f"A model named '{pending.get('model_name')}' already exists in the Established Model library.")
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("Overwrite existing model", use_container_width=True, key="clinical_step6_overwrite_duplicate_export"):
                ok, msg = clinical_export_model_to_established(pending.get("model_name"), pending.get("payload", {}), overwrite=True)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        with c2:
            new_default = f"{pending.get('model_name')}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            new_name = st.text_input("Rename model", value=new_default, key="clinical_step6_duplicate_rename_name")
        with c3:
            st.write("")
            st.write("")
            if st.button("Save with new name", use_container_width=True, key="clinical_step6_rename_duplicate_export"):
                if not new_name.strip():
                    st.error("Please enter a new model name.")
                else:
                    ok, msg = clinical_export_model_to_established(new_name.strip(), pending.get("payload", {}), overwrite=False)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.warning(msg)

def clinical_safe_export_name_part(value):
    """Create a readable model-name component while keeping clinical wording clear."""
    text = str(value or "").strip()
    if text == "":
        return "Clinical model"
    text = re.sub(r"[\\/:*?\"<>|]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clinical_project_name_for_model_display(info=None):
    """Return the clinical project component used in every established-model name."""
    info = info or {}
    setup = st.session_state.get("clinical_project_setup", {}) or {}
    return (
        str(info.get("clinical_project_name", "")).strip()
        or str(setup.get("Project name", "")).strip()
        or str(info.get("Project name", "")).strip()
        or str(info.get("project_name", "")).strip()
        or "Clinical model"
    )


def clinical_method_for_model_display(info=None, method=""):
    """Return the method/model-type component used in every established-model name."""
    info = info or {}
    return (
        str(method or "").strip()
        or str(info.get("method", "")).strip()
        or str(info.get("model_type", "")).strip()
        or "Model"
    )


def clinical_canonical_established_model_name(project_name="", method=""):
    """Return the required canonical model name: <Clinical project>_<Model type>."""
    return f"{clinical_safe_export_name_part(project_name)}_{clinical_safe_export_name_part(method)}"


def clinical_default_established_model_name(method="", outcome_variable=""):
    """Create established-model name as <Clinical project>_<Model type>."""
    setup = st.session_state.get("clinical_project_setup", {}) or {}
    project_name = (
        str(setup.get("Project name", "")).strip()
        or str(st.session_state.get("clinical_project_name", "")).strip()
        or "Clinical model"
    )
    method_text = method or "Model"
    return clinical_canonical_established_model_name(project_name, method_text)


def established_display_model_name(library_key, info):
    """Return the required user-facing model name everywhere.

    Standard format: <Clinical project>_<Model type>
    Example: Proton photon neurocognitive clinical model_Regression
    """
    info = info or {}
    project_name = clinical_project_name_for_model_display(info)
    method = clinical_method_for_model_display(info, info.get("method", ""))
    return clinical_canonical_established_model_name(project_name, method)

def established_clean_predictor_name(predictor):
    """Return a clean user-facing predictor name for tables."""
    text = str(predictor or "").strip()
    if text == "":
        return ""
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def established_format_predictor_list(predictors, max_items=8):
    """Format predictors as a short readable table cell."""
    if predictors is None:
        return "Not stored"
    if isinstance(predictors, str):
        raw = [p.strip() for p in re.split(r"[;,|]", predictors) if p.strip()]
    else:
        try:
            raw = list(predictors)
        except Exception:
            raw = [predictors]
    clean = []
    for item in raw:
        name = established_clean_predictor_name(item)
        if name and name not in clean:
            clean.append(name)
    if not clean:
        return "Not stored"
    shown = clean[:max_items]
    suffix = f"; +{len(clean) - max_items} more" if len(clean) > max_items else ""
    return "; ".join(shown) + suffix


def established_predictor_count(predictors):
    """Count stored predictors safely."""
    if predictors is None:
        return 0
    if isinstance(predictors, str):
        return len([p for p in re.split(r"[;,|]", predictors) if p.strip()])
    try:
        return len(list(predictors))
    except Exception:
        return 1 if str(predictors).strip() else 0


def established_normalise_column_for_matching(text):
    """Normalise predictor/column names for exact and fuzzy matching."""
    text = str(text or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ").replace("/", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def established_best_validation_column_match(required_predictor, available_columns):
    """Find an exact or close validation-dataset column for a stored model predictor.

    Returns a dictionary with the proposed column, match status and score. The user
    can still override the proposed mapping in the validation page.
    """
    import difflib

    required_raw = str(required_predictor or "").strip()
    available = [str(c) for c in available_columns]
    if not required_raw or not available:
        return {"column": "", "status": "Not mapped", "score": 0.0, "reason": "No available columns"}

    # 1) Literal exact match.
    for col in available:
        if col == required_raw:
            return {"column": col, "status": "Exact match", "score": 1.0, "reason": "Same column name"}

    required_norm = established_normalise_column_for_matching(required_raw)

    # 2) Normalised exact match, e.g. age_at_RT vs Age at RT.
    for col in available:
        if established_normalise_column_for_matching(col) == required_norm:
            return {"column": col, "status": "Exact/normalised match", "score": 0.98, "reason": "Same name after removing spaces/underscores/case"}

    # 3) Fuzzy match with token overlap support.
    best_col = ""
    best_score = 0.0
    req_tokens = set(required_norm.split())
    for col in available:
        col_norm = established_normalise_column_for_matching(col)
        ratio = difflib.SequenceMatcher(None, required_norm, col_norm).ratio()
        col_tokens = set(col_norm.split())
        token_score = 0.0
        if req_tokens or col_tokens:
            token_score = len(req_tokens & col_tokens) / max(len(req_tokens | col_tokens), 1)
        contains_bonus = 0.08 if (required_norm and (required_norm in col_norm or col_norm in required_norm)) else 0.0
        score = max(ratio, token_score) + contains_bonus
        if score > best_score:
            best_score = score
            best_col = col

    if best_col and best_score >= 0.60:
        return {"column": best_col, "status": "Close match", "score": round(min(best_score, 0.97), 2), "reason": "Suggested from similar wording"}

    return {"column": "", "status": "Not mapped", "score": round(best_score, 2), "reason": "No reliable match found"}


def clinical_export_timestamp():
    """Return a stable export timestamp for established-model records."""
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def clinical_build_established_export_info(model_name, payload):
    """Build the Established Model payload from a clinical model payload."""
    metrics = payload.get("metrics", {}) or {}
    predictors = payload.get("predictors", []) or []
    outcome_variable = payload.get("outcome_variable", "")
    method = payload.get("method", "Model") or "Model"
    setup = st.session_state.get("clinical_project_setup", {}) or {}
    project_name = str(setup.get("Project name", "")).strip() or str(payload.get("clinical_project_name", "")).strip() or "Clinical model"
    model_name = clinical_canonical_established_model_name(project_name, method)
    export_date = clinical_export_timestamp()
    return {
        "model_label": model_name,
        "model_type": "Clinical",
        "sample_size": metrics.get("Training rows", ""),
        "method": method,
        "model_site": setup.get("Clinical focus", "Brain"),
        "outcome_group": setup.get("Outcome of interest", outcome_variable),
        "outcome_variable": outcome_variable,
        "pipeline": payload.get("pipeline"),
        "predictors": predictors,
        "metrics": metrics,
        "model_settings": payload.get("model_settings", {}),
        "model_results": payload.get("model_results", pd.DataFrame()),
        "model_description": setup.get("Short description", "Exported from clinical model development."),
        "clinical_project_name": setup.get("Project name", ""),
        "clinical_project_folder": st.session_state.get("clinical_project_folder", ""),
        "exported_at": export_date,
        "exported_date": export_date.split(" ")[0],
        "publication_not_available": True,
        "supporting_publication_title": "",
        "supporting_publication_reference": "",
        "supporting_publication_filename": "",
        "supporting_publication_bytes": None,
    }


def clinical_export_model_to_established(model_name, payload, overwrite=False):
    """Export one Step 6 generated clinical model to the Established Model library."""
    if not isinstance(payload, dict):
        return False, "Invalid model payload."
    # Enforce the same model name everywhere: <Clinical project>_<Model type>.
    method_for_name = payload.get("method", "Model") if isinstance(payload, dict) else "Model"
    model_name = clinical_default_established_model_name(method_for_name, payload.get("outcome_variable", "") if isinstance(payload, dict) else "")
    if "established_calculators" not in st.session_state:
        st.session_state.established_calculators = {}

    if model_name in st.session_state.established_calculators and not overwrite:
        st.session_state.clinical_pending_established_export = {
            "model_name": model_name,
            "payload": payload,
        }
        return False, f"A model named '{model_name}' already exists. Choose overwrite or rename."

    export_info = clinical_build_established_export_info(model_name, payload)
    st.session_state.established_calculators[model_name] = export_info
    save_established_calculators_persistent()
    st.session_state.established_last_exported_model = model_name
    st.session_state.clinical_pending_established_export = None
    save_established_model_workflow_state_persistent()
    return True, f"Exported '{model_name}' to Established Model."

def save_clinical_model_comparison_outputs(comparison_df, risk_df=None):
    """Save Step 6 comparison outputs inside the active clinical project. Overwrites previous files."""
    out_folder = clinical_model_comparison_folder()
    if out_folder is None:
        return None
    comparison_df.to_csv(out_folder / "clinical_model_comparison_table.csv", index=False)
    if risk_df is not None and isinstance(risk_df, pd.DataFrame) and not risk_df.empty:
        risk_df.to_csv(out_folder / "clinical_model_risk_comparison.csv", index=False)
    st.session_state.clinical_model_comparison_df = comparison_df
    st.session_state.clinical_model_comparison_saved_folder = str(out_folder)
    return out_folder



# ============================================================
# ESTABLISHED MODEL LOCAL VALIDATION HELPERS
# ============================================================

def established_validation_output_folder():
    """Folder for local validation results from established models."""
    out = PERSISTENT_STORAGE_DIR / "established_model_local_validation"
    out.mkdir(parents=True, exist_ok=True)
    return out


def established_prepare_binary_outcome(series):
    """Return numeric binary outcome and a note."""
    y = pd.to_numeric(series, errors="coerce")
    if y.dropna().nunique() <= 2 and y.dropna().shape[0] > 0:
        unique_vals = sorted(y.dropna().unique().tolist())
        if set(unique_vals).issubset({0, 1}):
            return y.astype(float), "Outcome interpreted as 0/1."
        if len(unique_vals) == 2:
            mapped = y.map({unique_vals[0]: 0, unique_vals[1]: 1}).astype(float)
            return mapped, f"Outcome values mapped: {unique_vals[0]}→0, {unique_vals[1]}→1."

    clean = series.astype(str).str.strip().str.lower()
    values = sorted([v for v in clean.dropna().unique().tolist() if v not in ["", "nan", "none"]])
    if len(values) == 2:
        pos = None
        for v in values:
            if any(term in v for term in ["yes", "true", "decline", "tox", "event", "dead", "progress", "positive"]):
                pos = v
                break
        if pos is None:
            pos = values[1]
        neg = values[0] if values[1] == pos else values[1]
        mapped = clean.map({neg: 0, pos: 1}).astype(float)
        return mapped, f"Outcome values mapped: {neg}→0, {pos}→1."

    return pd.to_numeric(series, errors="coerce"), "Could not confidently map outcome; numeric conversion used."


def established_classification_metrics(y_true, y_prob, threshold=0.5):
    """Calculate validation metrics for predicted probabilities."""
    y_true = pd.Series(y_true).astype(float)
    y_prob = pd.Series(y_prob).astype(float)
    valid = y_true.notna() & y_prob.notna()
    y_true = y_true[valid].astype(int)
    y_prob = y_prob[valid]
    if y_true.empty or y_true.nunique() < 2:
        return {"N validated": int(y_true.shape[0]), "AUC": np.nan, "Accuracy": np.nan, "Sensitivity": np.nan, "Specificity": np.nan, "F1 score": np.nan, "Brier score": np.nan, "Note": "Outcome has fewer than two classes after filtering."}
    y_pred = (y_prob >= float(threshold)).astype(int)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = np.nan
    try:
        brier = brier_score_loss(y_true, y_prob)
    except Exception:
        brier = np.nan
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
        specificity = tn / (tn + fp) if (tn + fp) else np.nan
    except Exception:
        sensitivity = np.nan
        specificity = np.nan
    return {"N validated": int(y_true.shape[0]), "AUC": auc, "Accuracy": accuracy_score(y_true, y_pred), "Sensitivity": sensitivity, "Specificity": specificity, "F1 score": f1_score(y_true, y_pred, zero_division=0), "Brier score": brier, "Note": ""}


def established_validation_display_matrix(results_df):
    """Return a Parameter x Model matrix for external-validation results using the user's data."""
    if results_df is None or results_df.empty:
        return pd.DataFrame()
    keep = ["Model", "Validation type", "Validation source", "Validation date", "Outcome", "Method", "N validated", "AUC", "Accuracy", "Sensitivity", "Specificity", "F1 score", "Brier score", "Missing predictors", "Note"]
    keep = [c for c in keep if c in results_df.columns]
    return results_df[keep].set_index("Model").T.reset_index().rename(columns={"index": "Parameter"})


def established_highlight_validation_best(row):
    """Highlight best validation metric per row."""
    styles = [""] * len(row)
    parameter = str(row.get("Parameter", ""))
    higher_is_better = ["AUC", "Accuracy", "Sensitivity", "Specificity", "F1 score"]
    lower_is_better = ["Brier score"]
    model_columns = [col for col in row.index if col != "Parameter"]
    if parameter not in higher_is_better + lower_is_better:
        return styles
    values = pd.to_numeric(row[model_columns], errors="coerce")
    if values.dropna().empty:
        return styles
    best_value = values.min() if parameter in lower_is_better else values.max()
    for idx, col in enumerate(row.index):
        if col in model_columns:
            try:
                if pd.notna(values[col]) and float(values[col]) == float(best_value):
                    styles[idx] = "background-color: orange; color: black; font-weight: bold;"
            except Exception:
                pass
    return styles


# ============================================================
# APP LANDING PAGE
# ============================================================
if st.session_state.page == "home":
    st.markdown("## Welcome to BrainRT Analytics")
    st.write("Choose how you want to use the platform.")

    st.info(
        "BrainRT Analytics provides a no-code platform for healthcare professionals to build, evaluate, "
        "and apply prediction models using established analytical tools. The platform supports model "
        "development from local clinical, imaging, dose, and voxel-based datasets, while also enabling "
        "users to compare results across centres, facilitate external validation, and identify models that "
        "best fit their clinical practice. By supporting model sharing without the need to share patient-level "
        "data, BrainRT Analytics aims to promote collaborative, transparent, and clinically relevant research "
        "in radiotherapy and oncology."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧪 Model Development")
        st.write(
            "Develop prediction models using clinical, imaging, dose, voxel-based, or combined data."
        )
        st.markdown(
            """
            Build and validate new models from your own datasets.
            """
        )

        if st.button("Open Model Development", use_container_width=True, type="primary"):
            go_to("model_development_home")

    with col2:
        st.markdown("### 📌 Established Model")
        st.write(
            "Use an already-developed/validated model to calculate risk for a new patient."
        )
        st.markdown(
            """
            Planned for:
            - Select established model
            - Enter patient variables
            - Calculate risk
            - Export patient-level report
            """
        )

        if st.button("Open Established Model", use_container_width=True):
            go_to("established_model")


# ============================================================
# MODEL DEVELOPMENT HOME
# ============================================================

# ============================================================
# MODEL DEVELOPMENT LANDING PAGE
# ============================================================
elif st.session_state.page == "model_development_home":
    st.header("Model Development")
    st.subheader("Select a module")

    st.markdown(
        """
        <style>
        .model-dev-card {
            border: 1px solid rgba(120, 120, 120, 0.25);
            border-radius: 18px;
            padding: 1.4rem 1.2rem;
            min-height: 220px;
            background: rgba(250, 250, 250, 0.04);
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin-bottom: 0.75rem;
        }
        .model-dev-card h3 {
            margin-top: 0;
            margin-bottom: 0.6rem;
            font-size: 1.35rem;
        }
        .model-dev-card p {
            font-size: 0.95rem;
            line-height: 1.35;
            color: #666;
        }
        .model-dev-icon {
            font-size: 2.2rem;
            margin-bottom: 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown(
            """
            <div class="model-dev-card">
                <div class="model-dev-icon">🧠</div>
                <h3>Clinical Module</h3>
                <p>
                Build and evaluate patient-level clinical models using clinical variables,
                treatment groups, outcomes, statistics and machine learning.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Clinical Module", use_container_width=True, type="primary", key="big_open_clinical_module"):
            go_to("clinical_start_project")

    with col2:
        st.markdown(
            """
            <div class="model-dev-card">
                <div class="model-dev-icon">🧬</div>
                <h3>Voxel Based Analysis</h3>
                <p>
                Prepare imaging, dose and mask data for voxel-level spatial analysis
                and cohort-level association modelling.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Voxel Based Analysis", use_container_width=True, type="primary", key="big_open_voxel_based_analysis"):
            go_to("voxel_analysis_home")

    with col3:
        st.markdown(
            """
            <div class="model-dev-card">
                <div class="model-dev-icon">📚</div>
                <h3>Knowledge Base</h3>
                <p>
                Store project notes, literature summaries, supporting documents,
                reporting checklists and model documentation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Knowledge Base", use_container_width=True, type="primary", key="big_open_knowledge_base"):
            go_to("knowledge_base")

    st.divider()

    if st.button("← Back to Home", use_container_width=True, key="model_dev_big_tabs_back_home"):
        go_to("home")


# ============================================================
# VOXEL-BASED ANALYSIS MODULE - LANDING PAGE
# ============================================================
elif st.session_state.page == "voxel_analysis_home":
    st.header("Voxel-based Analysis")
    st.subheader("Voxel-level radiotherapy analysis")

    st.write(
        "Voxel-based analysis evaluates imaging, dose or anatomical information at each voxel across a registered patient cohort. "
        "In radiotherapy research, this approach can be used to explore where spatial patterns of delivered dose, tumour location, "
        "normal tissue exposure or imaging biomarkers are associated with clinical outcomes."
    )

    st.info(
        "Use the workflow below in the same order as the navigation menu. Each step saves its own outputs inside the active VBA project folder, "
        "so the project can be reopened without repeating completed setup steps."
    )

    st.markdown("### Voxel-based analysis workflow")

    voxel_home_steps = [
        {
            "label": "🚀 Start / Open project",
            "page": "voxel_start_project",
            "button": "Start / Open project",
            "description": "Create a new VBA project or reopen an existing project folder.",
            "items": [
                "Project name and description",
                "Cancer site and outcome type",
                "Project folder structure",
                "Saved project notes",
            ],
        },
        {
            "label": "📁 Load patient clinical data",
            "page": "voxel_load_patient_data",
            "button": "Load patient clinical data",
            "description": "Upload and save patient-level clinical, treatment and outcome data.",
            "items": [
                "Excel/CSV upload",
                "Patient ID mapping",
                "Clinical/outcome QC",
                "Saved clean dataset",
            ],
        },
        {
            "label": "🖼️ Load images / masks",
            "page": "voxel_load_images",
            "button": "Load images / masks",
            "description": "Index the source image directory without copying large raw image files.",
            "items": [
                "CT/MR/dose/mask directory",
                "File role assignment",
                "Filename and patient matching QC",
                "Saved image index",
            ],
        },
        {
            "label": "🛠️ Normalisation",
            "page": "voxel_registration_alignment",
            "button": "Open normalisation",
            "description": "Prepare images before registration by standardising orientation and voxel spacing.",
            "items": [
                "Image metadata extraction",
                "Common orientation",
                "Common voxel spacing",
                "Normalised CT/MR/dose/mask preview",
            ],
        },
        {
            "label": "🧭 Reference image / CCS",
            "page": "voxel_reference_ccs",
            "button": "Select reference image / CCS",
            "description": "Choose the fixed reference patient/image before cohort registration.",
            "items": [
                "Reference patient/image",
                "Common coordinate system",
                "Transform targets",
                "Saved reference setup",
            ],
        },
        {
            "label": "⚙️ Batch registration",
            "page": "voxel_batch_registration",
            "button": "Open batch registration",
            "description": "Register selected normalised cohort images into the saved reference space.",
            "items": [
                "Rigid only",
                "Rigid → Affine",
                "Rigid → Affine → B-spline",
                "Safe interpolation for dose and masks",
            ],
        },
        {
            "label": "🔎 Registration QC",
            "page": "voxel_registration_qc",
            "button": "Open registration QC",
            "description": "Check registration quality before accepting warped outputs for VBA.",
            "items": [
                "Readable registered outputs",
                "Geometry consistency checks",
                "Visual approval flag",
                "Manual review notes",
            ],
        },
        {
            "label": "🌀 Warp to reference space",
            "page": "voxel_warp_to_ccs",
            "button": "Open warp step",
            "description": "Confirm the cohort outputs that have been warped into the selected CCS/reference space.",
            "items": [
                "CT/MR image outputs",
                "Dose files in reference space",
                "Masks with nearest-neighbour labels",
                "Warp manifest and overlay QC",
            ],
        },
        {
            "label": "🧮 Dose normalisation",
            "page": "voxel_dose_normalisation",
            "button": "Open dose normalisation",
            "description": "Standardise registered dose files before voxel-wise dose-outcome analysis.",
            "items": [
                "Registered dose outputs",
                "Dose unit / scaling option",
                "Saved dose-normalised NIfTI files",
                "Dose-normalisation summary CSV",
            ],
        },
        {
            "label": "✅ VBA-ready dataset / Final QC",
            "page": "voxel_vba_ready_dataset",
            "button": "Open final QC",
            "description": "Create the final manifest linking each patient to clinical data, dose, anatomy and masks in reference space.",
            "items": [
                "Clinical row linked",
                "Dose in Gy present",
                "Reference-space images/masks",
                "Ready / not ready flag",
            ],
        },
        {
            "label": "📊 Statistical analysis",
            "page": "voxel_statistical_analysis",
            "button": "Open statistical analysis",
            "description": "Run paper-aligned voxel-wise statistics using the VBA-ready dose maps and clinical outcomes.",
            "items": [
                "Spearman dose–outcome correlation",
                "Permutation / FDR / Bonferroni correction",
                "Voxel filters and masks",
                "Adjustment variables",
            ],
        },
    ]

    for row_start in range(0, len(voxel_home_steps), 3):
        cols = st.columns(3)
        for col, step in zip(cols, voxel_home_steps[row_start:row_start + 3]):
            with col:
                bullet_html = "".join([f"<li>{item}</li>" for item in step["items"]])
                card_html = (
                    '<div style="border:1px solid #444; border-radius:14px; padding:18px; min-height:250px;">'
                    f'<h2 style="margin-top:0;">{step["label"]}</h2>'
                    f'<p>{step["description"]}</p>'
                    f'<ul>{bullet_html}</ul>'
                    '</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)
                if st.button(step["button"], use_container_width=True, key=f"voxel_home_open_{step['page']}"):
                    go_to(step["page"])

    st.divider()

    back_col, clinical_col = st.columns(2)
    with back_col:
        if st.button("← Back to Model Development pathways", use_container_width=True, key="voxel_home_back_model_development"):
            go_to("model_development_home")
    with clinical_col:
        if st.button("Open Clinical Model instead", use_container_width=True, key="voxel_home_open_clinical"):
            go_to("clinical_start_project")



# ============================================================
# VOXEL START / OPEN PROJECT PAGE
# ============================================================

# ============================================================
# VOXEL-BASED ANALYSIS MODULE - START / OPEN PROJECT
# ============================================================
elif st.session_state.page == "voxel_start_project":
    st.header("Voxel-based Analysis")
    st.subheader("Start / Open project")

    st.write(
        "Create a new voxel-based analysis project, or open an existing project folder. "
        "All files generated by this VBA workflow will be saved inside the active project folder."
    )

    cancer_site_options = [
        "Brain / CNS",
        "Head and Neck",
        "Lung",
        "Breast",
        "Prostate",
        "Gastrointestinal",
        "Gynaecological",
        "Paediatric",
        "Sarcoma",
        "Haematological",
        "Metastatic disease",
        "Other",
    ]

    outcome_type_options = [
        "Toxicity",
        "Survival",
        "Both survival and toxicity",
        "Other",
    ]

    data_file_type_options = [
        "CT",
        "MRI",
        "PET",
        "All imaging",
        "Planning CT",
        "Dose file",
        "RTSTRUCT / structure masks",
        "NIfTI image",
        "NIfTI mask",
        "DICOM",
        "Clinical Excel/CSV",
        "Other",
    ]

    toxicity_metric_suggestions = [
        "Neurocognitive decline",
        "Endocrine toxicity",
        "Radiation necrosis",
        "Hearing loss",
        "Vision toxicity",
        "Brainstem toxicity",
        "Fatigue",
        "Quality of life decline",
        "CTCAE grade ≥ 2 toxicity",
        "CTCAE grade ≥ 3 toxicity",
    ]

    survival_metric_suggestions = [
        "Overall survival",
        "Progression-free survival",
        "Local control",
        "Disease-free survival",
        "Time to progression",
        "Event-free survival",
        "Treatment failure",
        "Recurrence",
    ]

    general_metric_suggestions = [
        "Binary outcome",
        "Continuous change score",
        "Time-to-event outcome",
        "Ordinal toxicity grade",
        "Patient-reported outcome",
        "Clinician-reported outcome",
        "Other",
    ]

    def voxel_safe_folder_name(name):
        safe = str(name).strip()
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", safe)
        safe = safe.strip("._-")
        return safe or "Voxel_Based_Analysis_Project"

    def get_default_voxel_subfolders(project_folder):
        """Create folders that match the VBA navigation workflow, in the same order."""
        project_folder = Path(project_folder)
        return {
            "00_start_open_project": project_folder / "00_Start_Open_Project",
            "01_load_patient_clinical_data": project_folder / "01_Load_Patient_Clinical_Data",
            "02_load_images_masks": project_folder / "02_Load_Images_Masks",
            "03_normalisation": project_folder / "03_Normalisation",
            "04_reference_image_ccs": project_folder / "04_Reference_Image_CCS",
            "05_batch_registration": project_folder / "05_Batch_Registration",
            "06_registration_qc": project_folder / "06_Registration_QC",
            "07_warp_to_reference_space": project_folder / "07_Warp_To_Reference_Space",
            "08_dose_normalisation": project_folder / "08_Dose_Normalisation",
            "09_vba_ready_dataset_final_qc": project_folder / "09_VBA_Ready_Dataset_Final_QC",
            "10_statistical_analysis": project_folder / "10_Statistical_Analysis",
            "12_reports_logs": project_folder / "12_Reports_Logs",
        }

    def write_voxel_project_note(project_folder, project_setup, subfolders):
        project_folder = Path(project_folder)
        setup_folder = subfolders.get("00_start_open_project", project_folder)
        setup_folder.mkdir(parents=True, exist_ok=True)
        note_path = setup_folder / "project_setup_notes.txt"
        created_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        note_text = f"""Voxel-based analysis project setup
===================================

Created/updated: {created_time}

Project name:
{project_setup.get("project_name", "")}

Description:
{project_setup.get("description", "") or "Not provided"}

Site / cancer type:
{", ".join(project_setup.get("cancer_sites", [])) if project_setup.get("cancer_sites", []) else "Not selected"}

Outcome type:
{", ".join(project_setup.get("outcome_types", [])) if project_setup.get("outcome_types", []) else "Not selected"}

Data file types:
{", ".join(project_setup.get("data_file_types", [])) if project_setup.get("data_file_types", []) else "Not selected"}

Outcome metrics:
{", ".join(project_setup.get("outcome_metrics", [])) if project_setup.get("outcome_metrics", []) else "Not selected"}

Project folder:
{project_folder}

Generated data should be saved in the workflow-matched folders:
- Start / Open project: {subfolders["00_start_open_project"]}
- Load patient clinical data: {subfolders["01_load_patient_clinical_data"]}
- Load images / masks: {subfolders["02_load_images_masks"]}
- Normalisation: {subfolders["03_normalisation"]}
- Reference image / CCS: {subfolders["04_reference_image_ccs"]}
- Batch registration: {subfolders["05_batch_registration"]}
- Registration QC: {subfolders["06_registration_qc"]}
- Warp to reference space: {subfolders["07_warp_to_reference_space"]}
- Dose normalisation: {subfolders["08_dose_normalisation"]}
- VBA-ready dataset / Final QC: {subfolders["09_vba_ready_dataset_final_qc"]}
- Statistical analysis: {subfolders["10_statistical_analysis"]}
- Reports and logs: {subfolders["12_reports_logs"]}
"""
        note_path.write_text(note_text, encoding="utf-8")
        return note_path

    def activate_voxel_project(project_folder, project_setup=None):
        project_folder = Path(project_folder).expanduser()
        project_folder.mkdir(parents=True, exist_ok=True)

        subfolders = get_default_voxel_subfolders(project_folder)
        for folder in subfolders.values():
            folder.mkdir(parents=True, exist_ok=True)

        if project_setup is None:
            project_setup = st.session_state.get("voxel_project_setup", {})

        if not project_setup:
            project_setup = {
                "project_name": project_folder.name,
                "description": "",
                "cancer_sites": [],
                "outcome_types": [],
                "data_file_types": [],
                "outcome_metrics": [],
            }

        note_path = write_voxel_project_note(project_folder, project_setup, subfolders)

        st.session_state.voxel_project_setup = project_setup
        st.session_state.voxel_project_folder = str(project_folder)
        st.session_state.voxel_project_subfolders = {
            key: str(value) for key, value in subfolders.items()
        }
        st.session_state.voxel_project_note_file = str(note_path)

        # Do not write to widget-owned keys here, such as voxel_project_name,
        # voxel_project_description, voxel_cancer_sites, voxel_outcome_types,
        # voxel_data_file_types, or voxel_outcome_metrics. Streamlit raises an
        # exception if those keys are modified after their widgets are created.
        # The saved project metadata is kept in voxel_project_setup instead.
        return note_path

    mode = st.radio(
        "What do you want to do?",
        ["Create new project", "Open existing project"],
        horizontal=True,
        key="voxel_project_start_open_mode",
    )

    default_project_root = str(Path("data") / "voxel_projects")

    if mode == "Create new project":
        st.markdown("### Create new project")
        if "voxel_project_root_folder_input" not in st.session_state:
            st.session_state.voxel_project_root_folder_input = st.session_state.get("voxel_project_root_folder", default_project_root)
        if "voxel_project_root_folder_selected" not in st.session_state:
            st.session_state.voxel_project_root_folder_selected = ""
        if "voxel_project_root_folder_selected_applied" not in st.session_state:
            st.session_state.voxel_project_root_folder_selected_applied = ""

        selected_root_from_browse = st.session_state.get("voxel_project_root_folder_selected", "")
        applied_root_from_browse = st.session_state.get("voxel_project_root_folder_selected_applied", "")
        if selected_root_from_browse and selected_root_from_browse != applied_root_from_browse:
            st.session_state.voxel_project_root_folder_input = selected_root_from_browse
            st.session_state.voxel_project_root_folder_selected_applied = selected_root_from_browse

        def browse_for_new_voxel_project_parent_folder():
            """Open a directory chooser for selecting where new VBA project folders are saved."""
            selected_folder = ""

            try:
                import subprocess

                powershell_script = """
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = 'Select folder where the new VBA project will be saved'
$dialog.ShowNewFolderButton = $true
$dialog.RootFolder = [System.Environment+SpecialFolder]::Desktop
$result = $dialog.ShowDialog()
if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
    Write-Output $dialog.SelectedPath
}
"""
                completed = subprocess.run(
                    ["powershell", "-NoProfile", "-STA", "-Command", powershell_script],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                selected_folder = completed.stdout.strip().splitlines()[-1].strip() if completed.stdout.strip() else ""
            except Exception:
                selected_folder = ""

            if not selected_folder:
                try:
                    import tkinter as tk
                    from tkinter import filedialog

                    root = tk.Tk()
                    root.withdraw()
                    root.update()
                    try:
                        root.attributes("-topmost", True)
                    except Exception:
                        pass

                    selected_folder = filedialog.askdirectory(
                        parent=root,
                        title="Select folder where the new VBA project will be saved",
                        mustexist=True,
                    )
                    root.destroy()
                except Exception as error:
                    st.warning("Folder browser could not be opened from this Streamlit session.")
                    st.caption(f"Folder browser error: {error}")
                    selected_folder = ""

            if selected_folder:
                selected_path = Path(selected_folder).expanduser()
                if selected_path.exists() and selected_path.is_dir():
                    st.session_state.voxel_project_root_folder_selected = str(selected_path)
                else:
                    st.error("The selected path is not a directory.")

        project_name = st.text_input(
            "Project name",
            value=st.session_state.get("voxel_project_name", ""),
            key="voxel_project_name",
            placeholder="Example: Proton vs photon voxel-based neurocognitive toxicity"
        )

        project_description = st.text_area(
            "Description",
            value=st.session_state.get("voxel_project_description", ""),
            key="voxel_project_description",
            placeholder="Briefly describe the purpose of this voxel-based analysis."
        )

        selected_sites = st.multiselect(
            "Site / cancer type",
            options=cancer_site_options,
            default=st.session_state.get("voxel_cancer_sites", []),
            key="voxel_cancer_sites",
            help="Select one or more sites. Select Other only if the site is not listed."
        )

        other_sites = []
        if "Other" in selected_sites:
            other_site_text = st.text_input(
                "Describe other site / cancer type",
                value=st.session_state.get("voxel_other_site_description", ""),
                key="voxel_other_site_description",
                placeholder="Example: skull base chordoma, pituitary adenoma, ocular melanoma"
            )
            other_sites = [item.strip() for item in other_site_text.replace("\n", ",").split(",") if item.strip()]

        outcome_types = st.multiselect(
            "Type of outcome",
            options=outcome_type_options,
            default=st.session_state.get("voxel_outcome_types", ["Toxicity"]),
            key="voxel_outcome_types",
            help="Select one or more outcome types. Select Other only if the outcome type is not listed."
        )

        other_outcome_types = []
        if "Other" in outcome_types:
            other_outcome_text = st.text_input(
                "Describe other outcome type",
                value=st.session_state.get("voxel_other_outcome_type_description", ""),
                key="voxel_other_outcome_type_description",
                placeholder="Example: functional outcome, imaging response, biomarker change"
            )
            other_outcome_types = [item.strip() for item in other_outcome_text.replace("\n", ",").split(",") if item.strip()]

        if len(outcome_types) == 0:
            outcome_types = ["Toxicity"]

        selected_file_types = st.multiselect(
            "Data file types",
            options=data_file_type_options,
            default=st.session_state.get("voxel_data_file_types", ["CT", "MRI", "Dose file", "RTSTRUCT / structure masks"]),
            key="voxel_data_file_types",
            help="Select all file types expected in the project. Select Other only if the file type is not listed."
        )

        other_file_types = []
        if "Other" in selected_file_types:
            other_file_type_text = st.text_input(
                "Describe other data file types",
                value=st.session_state.get("voxel_other_file_type_description", ""),
                key="voxel_other_file_type_description",
                placeholder="Example: MR spectroscopy, perfusion maps, DTI, CSV dose metrics"
            )
            other_file_types = [item.strip() for item in other_file_type_text.replace("\n", ",").split(",") if item.strip()]

        if len(selected_file_types) == 0:
            selected_file_types = ["CT", "MRI", "Dose file", "RTSTRUCT / structure masks"]

        st.markdown("### Outcome metrics")
        metric_suggestions = list(dict.fromkeys(toxicity_metric_suggestions + survival_metric_suggestions + general_metric_suggestions))

        selected_metrics = st.multiselect(
            "Outcome metrics",
            options=metric_suggestions,
            default=st.session_state.get("voxel_outcome_metrics", []),
            key="voxel_outcome_metrics",
            help="Select all relevant metrics. Select Other only if the metric is not listed."
        )

        other_metrics = []
        if "Other" in selected_metrics:
            other_metric_text = st.text_area(
                "Describe other outcome metrics",
                value=st.session_state.get("voxel_other_metric_description", ""),
                key="voxel_other_metric_description",
                placeholder="Example: hippocampal volume loss, processing speed change, clinician-defined toxicity endpoint"
            )
            other_metrics = [item.strip() for item in other_metric_text.replace("\n", ",").split(",") if item.strip()]

        all_sites = list(dict.fromkeys([item for item in selected_sites if item != "Other"] + other_sites))
        all_outcome_types = list(dict.fromkeys([item for item in outcome_types if item != "Other"] + other_outcome_types))
        all_file_types = list(dict.fromkeys([item for item in selected_file_types if item != "Other"] + other_file_types))
        all_metrics = list(dict.fromkeys([item for item in selected_metrics if item != "Other"] + other_metrics))

        st.markdown("---")
        st.markdown("### Save project")

        save_col, browse_col = st.columns([0.86, 0.14])
        with save_col:
            project_root_folder = st.text_input(
                "Save in",
                key="voxel_project_root_folder_input",
                placeholder="Folder where the new project will be saved",
            )
        with browse_col:
            st.write("")
            if st.button("Browse...", key="voxel_browse_new_project_parent_folder", use_container_width=True):
                browse_for_new_voxel_project_parent_folder()
                st.rerun()

        st.session_state.voxel_project_root_folder = project_root_folder

        clean_project_name = voxel_safe_folder_name(project_name)
        if clean_project_name:
            preview_project_folder = Path(project_root_folder).expanduser() / clean_project_name
            st.caption("The project folder will be created using the project name:")
            st.code(str(preview_project_folder))
        else:
            preview_project_folder = None
            st.caption("Enter a project name to generate the project folder path.")

        def save_new_voxel_project_setup():
            if str(project_root_folder).strip() == "":
                st.error("Please select or enter where the project should be saved.")
                return False

            if str(project_name).strip() == "":
                st.error("Please enter a project name before saving.")
                return False

            project_folder = Path(project_root_folder).expanduser() / voxel_safe_folder_name(project_name)

            project_setup = {
                "project_name": project_name,
                "description": project_description,
                "cancer_sites": all_sites,
                "outcome_types": all_outcome_types,
                "data_file_types": all_file_types,
                "outcome_metrics": all_metrics,
            }

            try:
                note_path = activate_voxel_project(project_folder, project_setup)
                st.session_state.voxel_project_saved = True
                st.success("Voxel-based analysis project saved and activated.")
                st.info(f"Project folder: {st.session_state.voxel_project_folder}")
                st.info(f"Setup notes saved to: {note_path}")
                return True
            except Exception as error:
                st.error(f"Could not create the project folder or note file: {error}")
                return False

        if st.button("Save project", type="primary", use_container_width=True):
            save_new_voxel_project_setup()

        if st.session_state.get("voxel_project_saved", False):
            restored_file_df = st.session_state.get("voxel_loaded_image_files_df", pd.DataFrame())
            if restored_file_df is not None and not restored_file_df.empty:
                st.success("Patient clinical data and image-directory metadata are loaded. You can continue to normalisation.")
                if st.button("Continue →", use_container_width=True):
                    go_to("voxel_registration_alignment")
            elif st.session_state.get("voxel_patient_data", None) is not None:
                st.success("Clean patient clinical data is loaded. Continue to images/masks.")
                if st.button("Continue →", use_container_width=True):
                    go_to("voxel_load_images")
            else:
                st.info("No clean patient clinical dataset was found. Continue to upload patient clinical data.")
                if st.button("Continue →", use_container_width=True):
                    go_to("voxel_load_patient_data")

    else:
        st.markdown("### Open existing project")

        # Keep the text-input widget key separate from the path updated by Browse.
        # Streamlit does not allow changing a widget key after the widget is instantiated
        # during the same run, so Browse stores into a separate key and the value is
        # applied before the text input is created on the next rerun.
        if "voxel_existing_project_folder_input" not in st.session_state:
            st.session_state.voxel_existing_project_folder_input = st.session_state.get("voxel_existing_project_folder_to_open", "")
        if "voxel_existing_project_folder_selected" not in st.session_state:
            st.session_state.voxel_existing_project_folder_selected = ""
        if "voxel_existing_project_folder_selected_applied" not in st.session_state:
            st.session_state.voxel_existing_project_folder_selected_applied = ""

        selected_from_browse = st.session_state.get("voxel_existing_project_folder_selected", "")
        applied_from_browse = st.session_state.get("voxel_existing_project_folder_selected_applied", "")
        if selected_from_browse and selected_from_browse != applied_from_browse:
            st.session_state.voxel_existing_project_folder_input = selected_from_browse
            st.session_state.voxel_existing_project_folder_selected_applied = selected_from_browse

        if "voxel_existing_project_files_preview" not in st.session_state:
            st.session_state.voxel_existing_project_files_preview = pd.DataFrame()
        if "voxel_existing_project_opened" not in st.session_state:
            st.session_state.voxel_existing_project_opened = False

        def preview_voxel_project_folder(project_folder):
            """Return a compact preview of the selected project folder contents."""
            try:
                folder = Path(project_folder).expanduser()
                if not folder.exists() or not folder.is_dir():
                    return pd.DataFrame()

                rows = []
                for item in sorted(folder.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
                    try:
                        size_text = "Folder" if item.is_dir() else f"{item.stat().st_size / 1024:.1f} KB"
                    except Exception:
                        size_text = ""
                    rows.append({
                        "Name": item.name,
                        "Type": "Folder" if item.is_dir() else "File",
                        "Size": size_text,
                        "Path": str(item),
                    })
                return pd.DataFrame(rows)
            except Exception:
                return pd.DataFrame()


        def restore_saved_voxel_patient_clinical_data(project_folder):
            """Restore saved patient clinical data when reopening an existing VBA project."""
            project_folder = Path(project_folder).expanduser()
            patient_data_folder = project_folder / "01_Load_Patient_Clinical_Data"

            if not patient_data_folder.exists() or not patient_data_folder.is_dir():
                return False, "No 01_Load_Patient_Clinical_Data folder was found in this project."

            saved_files = {"patient_data_folder": str(patient_data_folder)}

            clean_csv_path = patient_data_folder / "patient_clinical_data_clean_copy.csv"
            setup_summary_path = patient_data_folder / "patient_clinical_data_setup_summary.csv"
            setup_note_path = patient_data_folder / "patient_clinical_data_setup_note.txt"
            qc_issue_path = patient_data_folder / "voxel_patient_data_quality_issues.csv"
            qc_missing_path = patient_data_folder / "voxel_patient_missing_data_summary.csv"

            if clean_csv_path.exists():
                try:
                    restored_df = pd.read_csv(clean_csv_path)
                    st.session_state.voxel_patient_data = restored_df
                    st.session_state.voxel_patient_data_filename = clean_csv_path.name
                    st.session_state.voxel_patient_data_file_bytes = None
                    st.session_state.voxel_patient_data_source = "clean_project_copy"
                    saved_files["clean_csv_copy"] = str(clean_csv_path)
                except Exception as error:
                    return False, f"Clinical data file was found but could not be loaded: {error}"
            else:
                # Fall back to the first saved Excel/CSV file if the clean copy is missing.
                candidate_files = [
                    p for p in patient_data_folder.iterdir()
                    if p.is_file()
                    and p.suffix.lower() in [".csv", ".xlsx", ".xls"]
                    and not p.name.lower().startswith((
                        "patient_clinical_data_setup_summary",
                        "voxel_patient_data_quality_issues",
                        "voxel_patient_missing_data_summary",
                    ))
                ]
                if candidate_files:
                    data_path = sorted(candidate_files, key=lambda p: p.name.lower())[0]
                    try:
                        if data_path.suffix.lower() == ".csv":
                            restored_df = pd.read_csv(data_path)
                        else:
                            restored_df = pd.read_excel(data_path)
                        st.session_state.voxel_patient_data = restored_df
                        st.session_state.voxel_patient_data_filename = data_path.name
                        st.session_state.voxel_patient_data_file_bytes = None
                        st.session_state.voxel_patient_data_source = "fallback_project_file"
                        saved_files["original_file"] = str(data_path)
                    except Exception as error:
                        return False, f"Clinical data file was found but could not be loaded: {error}"
                else:
                    return False, "No saved patient clinical data file was found."

            if setup_summary_path.exists():
                try:
                    setup_summary_df = pd.read_csv(setup_summary_path)
                    saved_files["setup_summary"] = str(setup_summary_path)
                    setup = {
                        "filename": st.session_state.get("voxel_patient_data_filename", ""),
                        "patient_id_column": "",
                        "covariate_columns": [],
                        "baseline_columns": [],
                        "followup_outcome_columns": [],
                    }
                    if {"Item", "Value"}.issubset(setup_summary_df.columns):
                        lookup = dict(zip(setup_summary_df["Item"].astype(str), setup_summary_df["Value"].fillna("").astype(str)))
                        setup["patient_id_column"] = lookup.get("Patient ID column", "")
                        setup["covariate_columns"] = [x.strip() for x in lookup.get("Covariables", "").split(",") if x.strip()]
                        setup["baseline_columns"] = [x.strip() for x in lookup.get("Baseline variables", "").split(",") if x.strip()]
                        setup["followup_outcome_columns"] = [x.strip() for x in lookup.get("Follow-up outcome variables", "").split(",") if x.strip()]
                    st.session_state.voxel_patient_data_setup = setup
                except Exception:
                    pass

            if setup_note_path.exists():
                saved_files["setup_note"] = str(setup_note_path)

            if qc_issue_path.exists():
                try:
                    st.session_state.voxel_quality_issues = pd.read_csv(qc_issue_path)
                    saved_files["quality_issues"] = str(qc_issue_path)
                except Exception:
                    pass

            if qc_missing_path.exists():
                try:
                    st.session_state.voxel_missing_summary = pd.read_csv(qc_missing_path)
                    saved_files["missing_summary"] = str(qc_missing_path)
                except Exception:
                    pass

            st.session_state.voxel_patient_data_saved_files = saved_files
            st.session_state.voxel_patient_data_step = "variables"
            if clean_csv_path.exists():
                return True, "Clean patient clinical dataset was automatically restored."
            return True, "Patient clinical data was restored from a fallback project file."

        def restore_saved_voxel_image_directory_setup(project_folder):
            """Restore saved image-directory metadata when reopening an existing VBA project.

            This restores only the directory path, scan index and QC tables. It does not
            copy or reload image bytes into the project folder.
            """
            project_folder = Path(project_folder).expanduser()
            image_data_folder = project_folder / "02_Load_Images_Masks"

            if not image_data_folder.exists() or not image_data_folder.is_dir():
                return False, "No 02_Load_Images_Masks folder was found in this project."

            setup_summary_path = image_data_folder / "image_directory_setup_summary.csv"
            file_index_path = image_data_folder / "image_directory_file_index.csv"
            patient_summary_path = image_data_folder / "image_patient_file_summary.csv"
            qc_summary_path = image_data_folder / "voxel_filename_qc_summary.csv"
            qc_issues_path = image_data_folder / "voxel_filename_quality_issues.csv"
            excel_match_path = image_data_folder / "voxel_filename_excel_patient_match.csv"
            setup_note_path = image_data_folder / "image_directory_setup_note.txt"

            if not file_index_path.exists():
                return False, "No saved image-directory file index was found."

            try:
                file_df = pd.read_csv(file_index_path)
                st.session_state.voxel_loaded_image_files_df = file_df
                st.session_state.voxel_image_directory_saved_files = {
                    "image_data_folder": str(image_data_folder),
                    "file_index": str(file_index_path),
                }

                if setup_summary_path.exists():
                    setup_df = pd.read_csv(setup_summary_path)
                    st.session_state.voxel_image_directory_saved_files["setup_summary"] = str(setup_summary_path)
                    if {"Item", "Value"}.issubset(setup_df.columns):
                        lookup = dict(zip(setup_df["Item"].astype(str), setup_df["Value"].fillna("").astype(str)))
                        saved_format = lookup.get("Image format", "") or lookup.get("Format", "")
                        saved_directory = lookup.get("Source image directory", "") or lookup.get("Image directory", "")
                        if saved_format:
                            st.session_state.voxel_loaded_image_format = saved_format
                            st.session_state.voxel_selected_image_format = saved_format
                        if saved_directory:
                            st.session_state.voxel_image_directory_path = saved_directory
                            st.session_state.voxel_image_directory_source_path = saved_directory
                else:
                    st.session_state.voxel_loaded_image_format = st.session_state.get("voxel_loaded_image_format", "DICOM")

                if patient_summary_path.exists():
                    st.session_state.voxel_image_directory_saved_files["patient_summary"] = str(patient_summary_path)

                if qc_summary_path.exists():
                    st.session_state.voxel_filename_qc_summary = pd.read_csv(qc_summary_path)
                    st.session_state.voxel_filename_qc_ran = True
                    st.session_state.voxel_image_directory_saved_files["filename_qc_summary"] = str(qc_summary_path)

                if qc_issues_path.exists():
                    st.session_state.voxel_filename_qc_issues = pd.read_csv(qc_issues_path)
                    st.session_state.voxel_filename_qc_ran = True
                    st.session_state.voxel_image_directory_saved_files["filename_qc_issues"] = str(qc_issues_path)

                if excel_match_path.exists():
                    st.session_state.voxel_filename_excel_match = pd.read_csv(excel_match_path)
                    st.session_state.voxel_image_directory_saved_files["excel_patient_match"] = str(excel_match_path)

                if setup_note_path.exists():
                    st.session_state.voxel_image_directory_saved_files["setup_note"] = str(setup_note_path)

                st.session_state.voxel_image_load_step = "quality" if st.session_state.get("voxel_filename_qc_ran", False) else "load"
                return True, "Saved image-directory scan was automatically restored."
            except Exception as error:
                return False, f"Image-directory metadata was found but could not be restored: {error}"

        def browse_for_existing_voxel_project_folder():
            """Open a directory chooser for selecting an existing VBA project folder.

            Streamlit cannot browse local folders directly from the browser. Because this
            app is intended to run locally on Windows, first try the native Windows
            FolderBrowserDialog through PowerShell. If that is unavailable, fall back
            to Tkinter askdirectory(). The manual path box remains the fallback.
            """
            selected_folder = ""

            # Preferred local Windows folder picker.
            try:
                import subprocess

                powershell_script = """
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = 'Select existing VBA project folder'
$dialog.ShowNewFolderButton = $false
$dialog.RootFolder = [System.Environment+SpecialFolder]::Desktop
$result = $dialog.ShowDialog()
if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
    Write-Output $dialog.SelectedPath
}
"""
                completed = subprocess.run(
                    ["powershell", "-NoProfile", "-STA", "-Command", powershell_script],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                selected_folder = completed.stdout.strip().splitlines()[-1].strip() if completed.stdout.strip() else ""
            except Exception:
                selected_folder = ""

            # Fallback for local Python sessions where Tkinter is available.
            if not selected_folder:
                try:
                    import tkinter as tk
                    from tkinter import filedialog

                    root = tk.Tk()
                    root.withdraw()
                    root.update()
                    try:
                        root.attributes("-topmost", True)
                    except Exception:
                        pass

                    selected_folder = filedialog.askdirectory(
                        parent=root,
                        title="Select existing VBA project folder",
                        mustexist=True,
                    )
                    root.destroy()
                except Exception as error:
                    st.warning("Folder browser could not be opened from this Streamlit session.")
                    st.caption(f"Folder browser error: {error}")
                    selected_folder = ""

            if selected_folder:
                selected_path = Path(selected_folder).expanduser()
                if selected_path.exists() and selected_path.is_dir():
                    st.session_state.voxel_existing_project_folder_selected = str(selected_path)
                    st.session_state.voxel_existing_project_files_preview = pd.DataFrame()
                    st.session_state.voxel_existing_project_opened = False
                else:
                    st.error("The selected path is not a directory.")

        path_col, browse_col = st.columns([0.86, 0.14])
        with path_col:
            existing_project_folder = st.text_input(
                "Project folder",
                key="voxel_existing_project_folder_input",
                placeholder="Project folder path",
                label_visibility="collapsed",
            )
        with browse_col:
            st.write("")
            if st.button("Browse...", key="voxel_browse_existing_project_folder", use_container_width=True):
                browse_for_existing_voxel_project_folder()
                st.rerun()

        def open_existing_voxel_project():
            selected_folder = str(st.session_state.get("voxel_existing_project_folder_input", "")).strip()
            if selected_folder == "":
                return False

            project_folder = Path(selected_folder).expanduser()
            if not project_folder.exists() or not project_folder.is_dir():
                st.error("The selected project folder does not exist.")
                st.session_state.voxel_existing_project_opened = False
                st.session_state.voxel_existing_project_files_preview = pd.DataFrame()
                return False

            try:
                project_setup = {
                    "project_name": project_folder.name,
                    "description": "Opened existing voxel-based analysis project.",
                    "cancer_sites": [],
                    "outcome_types": [],
                    "data_file_types": [],
                    "outcome_metrics": [],
                }
                activate_voxel_project(project_folder, project_setup)
                restored_patient_data, restore_message = restore_saved_voxel_patient_clinical_data(project_folder)
                restored_image_data, image_restore_message = restore_saved_voxel_image_directory_setup(project_folder)
                st.session_state.voxel_existing_project_files_preview = preview_voxel_project_folder(project_folder)
                st.session_state.voxel_existing_project_opened = True
                st.success("Project opened.")
                if restored_patient_data:
                    st.success(restore_message)
                else:
                    st.info(restore_message)
                if restored_image_data:
                    st.success(image_restore_message)
                else:
                    st.info(image_restore_message)
                return True
            except Exception as error:
                st.error(f"Could not open this project folder: {error}")
                st.session_state.voxel_existing_project_opened = False
                return False

        if st.button("Open", type="primary", use_container_width=True):
            # Open the project and restore the clean clinical dataset if available,
            # but do not jump pages immediately. Showing the folder contents first
            # keeps the reopen workflow stable and lets the user continue manually.
            open_existing_voxel_project()

        files_preview = st.session_state.get("voxel_existing_project_files_preview", pd.DataFrame())
        if st.session_state.get("voxel_existing_project_opened", False):
            st.markdown("#### Files in selected folder")
            if files_preview is not None and not files_preview.empty:
                st.dataframe(files_preview, use_container_width=True, hide_index=True)
            else:
                st.info("This folder is empty.")

            note_file = Path(st.session_state.get("voxel_project_folder", st.session_state.get("voxel_existing_project_folder_input", ""))) / "00_Start_Open_Project" / "project_setup_notes.txt"
            if note_file.exists():
                with st.expander("Project notes"):
                    try:
                        st.text(note_file.read_text(encoding="utf-8"))
                    except Exception as error:
                        st.warning(f"Could not read project notes: {error}")

            if st.session_state.get("voxel_patient_data", None) is not None:
                st.success("Clean patient clinical data is loaded. You can continue to images/masks.")
                if st.button("Continue →", use_container_width=True):
                    go_to("voxel_load_images")
            else:
                st.info("No clean patient clinical dataset was found. Continue to upload patient clinical data.")
                if st.button("Continue →", use_container_width=True):
                    go_to("voxel_load_patient_data")
    st.divider()

    active_project_folder = st.session_state.get("voxel_project_folder", "")
    if active_project_folder:
        st.success("Active VBA project")
        st.code(active_project_folder)

        subfolders = st.session_state.get("voxel_project_subfolders", {})
        if subfolders:
            with st.expander("Project output folders"):
                for label, folder in subfolders.items():
                    st.write(f"**{label.replace('_', ' ').title()}**")
                    st.code(folder)

        # Always show a safe Continue button whenever a VBA project is active.
        # This prevents navigation from depending on the temporary Open button state.
        st.markdown("---")
        restored_file_df = st.session_state.get("voxel_loaded_image_files_df", pd.DataFrame())
        if restored_file_df is not None and not restored_file_df.empty:
            st.success("Patient clinical data and image-directory metadata are loaded. Continue to normalisation.")
            if st.button("Continue →", key="voxel_active_project_continue_to_registration", use_container_width=True):
                go_to("voxel_registration_alignment")
        elif st.session_state.get("voxel_patient_data", None) is not None:
            st.success("Clean patient clinical data is loaded. Continue to images/masks.")
            if st.button("Continue →", key="voxel_active_project_continue_to_images", use_container_width=True):
                go_to("voxel_load_images")
        else:
            st.info("No clean patient clinical dataset is loaded yet. Continue to load patient clinical data.")
            if st.button("Continue →", key="voxel_active_project_continue_to_patient_data", use_container_width=True):
                go_to("voxel_load_patient_data")

# ============================================================
# VOXEL LOAD PATIENT DATA PAGE
# ============================================================

# ============================================================
# VOXEL-BASED ANALYSIS MODULE - LOAD PATIENT CLINICAL DATA
# ============================================================
elif st.session_state.page == "voxel_load_patient_data":
    st.header("Voxel-based Analysis")
    st.subheader("Load patient clinical data file")

    st.write(
        "Upload the patient-level Excel file that links clinical/covariate data and outcome measures "
        "to the voxel-based analysis project."
    )
    active_project_folder = st.session_state.get("voxel_project_folder", "")
    if active_project_folder:
        patient_data_folder = Path(active_project_folder) / "01_Load_Patient_Clinical_Data"
        if patient_data_folder.exists():
            clean_csv_path = patient_data_folder / "patient_clinical_data_clean_copy.csv"

            # Automatically use the clean saved dataset. Do not ask the user to choose
            # between the original upload, QC tables, and setup files.
            if st.session_state.get("voxel_patient_data", None) is None and clean_csv_path.exists():
                try:
                    st.session_state.voxel_patient_data = pd.read_csv(clean_csv_path)
                    st.session_state.voxel_patient_data_filename = clean_csv_path.name
                    st.session_state.voxel_patient_data_file_bytes = None
                    st.session_state.voxel_patient_data_source = "clean_project_copy"
                    st.session_state.voxel_patient_data_step = "variables"
                    st.success("Clean patient clinical dataset loaded automatically from this project.")
                    go_to("voxel_load_images")
                except Exception as error:
                    st.error(f"Could not load the clean patient clinical dataset: {error}")

            if st.session_state.get("voxel_patient_data", None) is not None:
                st.success(f"Active patient clinical dataset: {st.session_state.get('voxel_patient_data_filename', 'patient_clinical_data_clean_copy.csv')}")

            existing_patient_files = []
            for file_path in sorted(patient_data_folder.iterdir()):
                if file_path.is_file():
                    existing_patient_files.append({
                        "File": file_path.name,
                        "Size KB": round(file_path.stat().st_size / 1024, 1),
                        "Path": str(file_path),
                    })
            if existing_patient_files:
                with st.expander("Saved patient clinical data files", expanded=False):
                    st.dataframe(pd.DataFrame(existing_patient_files), use_container_width=True, hide_index=True)


    if "voxel_patient_data_step" not in st.session_state:
        st.session_state.voxel_patient_data_step = "instructions"

    st.markdown("### Patient data workflow")
    st.caption("Click an icon to open that step.")

    icon1, icon2, icon3, icon4 = st.columns(4)

    def step_button(label, icon, subtitle, step_key):
        active = st.session_state.voxel_patient_data_step == step_key
        button_label = f"{icon}\n\n{label}\n\n{subtitle}"
        if st.button(
            button_label,
            key=f"patient_data_step_{step_key}",
            use_container_width=True,
            type="primary" if active else "secondary",
        ):
            st.session_state.voxel_patient_data_step = step_key
            st.rerun()

    with icon1:
        step_button("1. Instructions", "📘", "Read data structure", "instructions")
    with icon2:
        step_button("2. Load Excel", "📄", "Upload and preview", "upload")
    with icon3:
        step_button("3. Quality check", "🔎", "Find issues and row numbers", "quality")
    with icon4:
        step_button("4. Select variables", "✅", "Map columns", "variables")

    st.divider()

    current_step = st.session_state.voxel_patient_data_step

    # ========================================================
    # STEP 1: INSTRUCTIONS
    # ========================================================

    if current_step == "instructions":
        st.markdown("## 📘 Step 1: Instructions / data processing sheet")

        st.info(
            "The first column should be the patient ID. This ID must match the patient imaging folder name "
            "or the patient identifier used in the imaging filenames."
        )

        st.markdown(
            """
            **Required**

            - **Column 1: Patient ID**
              - Example: `Patient_ID`
              - Must be unique for each patient
              - Must match the patient imaging file/folder ID

            **Recommended columns**

            - Treatment group, for example `Proton` / `Photon`
            - Clinical covariates, for example age, sex, diagnosis, tumour site
            - Baseline variable if applicable
            - Follow-up outcome variables
            - Derived outcome, for example decline yes/no
            - Timepoint, for example 6 months, 12 months, 24 months
            - Dose metrics, for example mean dose, max dose, dose to structures
            - Image filename or image ID
            - Mask filename or mask ID
            """
        )

        st.markdown("### Tips for easier voxel-based analysis")

        st.markdown(
            """
            - Use **one row per patient** for the first version of the workflow.
            - Keep patient IDs simple and consistent, for example `P001`, `P002`, `P003`.
            - Patient ID must match the imaging folder/file ID.
            - Avoid spaces in column names. Use underscores instead, for example `Baseline_MMSE`.
            - Use consistent coding:
              - Treatment: `Proton`, `Photon`
              - Sex: `Female`, `Male`
              - Binary outcome: `0` / `1`
            - Keep imaging filenames consistent with patient IDs.
              - Example: `P001_T1.nii.gz`, `P001_mask.nii.gz`
            - Put each outcome in a separate column.
              - Example: `MMSE_Baseline`, `MMSE_12m`, `Decline_12m`
            - Do **not** put units inside numeric cells.
              - Use `20`, not `20cc`.
              - Put units in the column name, for example `Volume_cc` or `Dose_Gy`.
            - For missing data, preferably leave the cell blank.
              - If you need a label, use one consistent label such as `N/A` or `Not available`.
            - Avoid merged cells in Excel.
            - Avoid colour-coded data as the only source of information.
            """
        )

        st.markdown("### Example data processing sheet")

        template_df = make_voxel_patient_data_template()
        st.dataframe(template_df, use_container_width=True)

        st.download_button(
            "Download example template as CSV",
            data=template_df.to_csv(index=False).encode("utf-8"),
            file_name="voxel_patient_data_template.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button("Next: Load Excel sheet →", use_container_width=True):
            st.session_state.voxel_patient_data_step = "upload"
            st.rerun()

    # ========================================================
    # STEP 2: LOAD EXCEL SHEET
    # ========================================================

    elif current_step == "upload":
        st.markdown("## 📄 Step 2: Load Excel / CSV sheet")
        st.info("Upload the patient data sheet and check the preview before moving to the quality check.")

        uploaded_patient_file = st.file_uploader(
            "Upload patient metadata file",
            type=["xlsx", "xls", "csv"],
            key="voxel_patient_data_upload"
        )

        if uploaded_patient_file is not None:
            try:
                if uploaded_patient_file.name.lower().endswith(".csv"):
                    patient_df = pd.read_csv(uploaded_patient_file)
                else:
                    patient_df = pd.read_excel(uploaded_patient_file)

                st.session_state.voxel_patient_data = patient_df
                st.session_state.voxel_patient_data_filename = uploaded_patient_file.name
                st.session_state.voxel_patient_data_file_bytes = uploaded_patient_file.getvalue()

                st.success(f"Loaded patient clinical data file: {uploaded_patient_file.name}")

            except Exception as error:
                st.error(f"Could not read uploaded patient data file: {error}")

        patient_df = st.session_state.get("voxel_patient_data", None)

        if patient_df is None:
            st.info("Upload an Excel or CSV file to preview the patient data.")
        else:
            st.markdown("### File preview")
            st.write("Filename:", st.session_state.get("voxel_patient_data_filename", "Uploaded file"))

            c1, c2 = st.columns(2)
            c1.metric("Rows", patient_df.shape[0])
            c2.metric("Columns", patient_df.shape[1])

            st.dataframe(patient_df.head(30), use_container_width=True)

            if st.button("Next: Run data quality check →", use_container_width=True):
                st.session_state.voxel_patient_data_step = "quality"
                st.rerun()

    # ========================================================
    # STEP 3: DATA QUALITY CHECK
    # ========================================================

    elif current_step == "quality":
        st.markdown("## 🔎 Step 3: Run data quality check")
        st.info(
            "The quality-check table includes a Rows column showing where each problem occurs. "
            "Excel row 1 is the header, so patient data start at row 2."
        )

        patient_df = st.session_state.get("voxel_patient_data", None)

        if patient_df is None:
            st.info("Load an Excel/CSV file first.")
            if st.button("← Go to Load Excel sheet", use_container_width=True):
                st.session_state.voxel_patient_data_step = "upload"
                st.rerun()
        else:
            columns = list(patient_df.columns)

            if len(columns) == 0:
                st.error("No columns were detected in the uploaded file.")
            else:
                default_patient_id_index = 0
                for idx, col in enumerate(columns):
                    if str(col).lower() in ["patient_id", "patientid", "pt_id", "ptid", "id"]:
                        default_patient_id_index = idx
                        break

                patient_id_col_for_quality = st.selectbox(
                    "Patient ID column for quality check",
                    options=columns,
                    index=default_patient_id_index,
                    key="voxel_patient_id_column_quality"
                )

                if st.button("Run data quality check", type="primary", use_container_width=True):
                    st.session_state.voxel_quality_check_ran = True

                    messages, warnings = validate_voxel_patient_data(patient_df, patient_id_col_for_quality)
                    issue_df, missing_df = check_voxel_excel_data_quality(patient_df, patient_id_col_for_quality)

                    st.session_state.voxel_quality_messages = messages
                    st.session_state.voxel_quality_warnings = warnings
                    st.session_state.voxel_quality_issues = issue_df
                    st.session_state.voxel_missing_summary = missing_df

                if st.session_state.get("voxel_quality_check_ran", False):
                    messages = st.session_state.get("voxel_quality_messages", [])
                    warnings = st.session_state.get("voxel_quality_warnings", [])
                    issue_df = st.session_state.get("voxel_quality_issues", pd.DataFrame())
                    missing_df = st.session_state.get("voxel_missing_summary", pd.DataFrame())

                    for message in messages:
                        st.success(message)

                    for warning in warnings:
                        st.warning(warning)

                    st.markdown("### Quality issues")

                    if issue_df.empty:
                        st.success("No obvious data-quality issues were detected.")
                    else:
                        important_count = int((issue_df["Severity"] == "Important").sum()) if "Severity" in issue_df.columns else 0
                        check_count = int((issue_df["Severity"] == "Check").sum()) if "Severity" in issue_df.columns else 0
                        low_count = int((issue_df["Severity"] == "Low").sum()) if "Severity" in issue_df.columns else 0

                        q1, q2, q3 = st.columns(3)
                        q1.metric("Important issues", important_count)
                        q2.metric("Check issues", check_count)
                        q3.metric("Low priority", low_count)

                        def highlight_issue_rows(row):
                            severity = row.get("Severity", "")
                            if severity == "Important":
                                return ["background-color: #ffd6d6"] * len(row)
                            if severity == "Check":
                                return ["background-color: #fff2cc"] * len(row)
                            if severity == "Low":
                                return ["background-color: #e8f4ff"] * len(row)
                            return [""] * len(row)

                        st.dataframe(
                            issue_df.style.apply(highlight_issue_rows, axis=1),
                            use_container_width=True
                        )

                        st.caption("This CSV is saved automatically when you save the patient clinical data setup.")

                    st.markdown("### Missing data count")

                    if missing_df.empty:
                        st.info("No missing data summary available.")
                    else:
                        st.dataframe(missing_df, use_container_width=True)

                        st.caption("This CSV is saved automatically when you save the patient clinical data setup.")

                    if st.button("Next: Select variables →", use_container_width=True):
                        st.session_state.voxel_patient_data_step = "variables"
                        st.rerun()
                else:
                    st.info("Click **Run data quality check** to inspect missing data and potential formatting issues.")

    # ========================================================
    # STEP 4: SELECT VARIABLES
    # ========================================================

    elif current_step == "variables":
        st.markdown("## ✅ Step 4: Select patient ID, covariates, baseline and follow-up outcomes")
        st.info("After checking the data, select which columns will be used in the voxel-based analysis.")

        patient_df = st.session_state.get("voxel_patient_data", None)

        if patient_df is None:
            st.info("Load an Excel/CSV file first.")
            if st.button("← Go to Load Excel sheet", use_container_width=True):
                st.session_state.voxel_patient_data_step = "upload"
                st.rerun()
        else:
            columns = list(patient_df.columns)

            if len(columns) == 0:
                st.error("No columns were detected in the uploaded file.")
            else:
                default_patient_id_index = 0
                for idx, col in enumerate(columns):
                    if str(col).lower() in ["patient_id", "patientid", "pt_id", "ptid", "id"]:
                        default_patient_id_index = idx
                        break

                patient_id_col = st.selectbox(
                    "Select patient ID column",
                    options=columns,
                    index=default_patient_id_index,
                    key="voxel_patient_id_column"
                )

                covariate_cols = st.multiselect(
                    "Select covariables / covariates",
                    options=[col for col in columns if col != patient_id_col],
                    default=[
                        col for col in columns
                        if col != patient_id_col
                        and any(term in str(col).lower() for term in ["age", "sex", "diagnosis", "tumour", "tumor", "treatment", "dose"])
                    ],
                    key="voxel_covariate_columns"
                )

                baseline_cols = st.multiselect(
                    "Select baseline variables, if applicable",
                    options=[col for col in columns if col != patient_id_col],
                    default=[
                        col for col in columns
                        if any(term in str(col).lower() for term in ["baseline", "pre", "pretreatment", "pre_treatment"])
                    ],
                    key="voxel_baseline_columns"
                )

                followup_outcome_cols = st.multiselect(
                    "Select follow-up outcome variables",
                    options=[col for col in columns if col != patient_id_col],
                    default=[
                        col for col in columns
                        if any(term in str(col).lower() for term in ["followup", "follow_up", "post", "12m", "24m", "decline", "toxicity", "survival", "outcome"])
                    ],
                    key="voxel_followup_outcome_columns"
                )

                st.markdown("### Selection summary")

                summary = pd.DataFrame([
                    {"Item": "Rows / patients", "Value": patient_df.shape[0]},
                    {"Item": "Columns", "Value": patient_df.shape[1]},
                    {"Item": "Patient ID column", "Value": patient_id_col},
                    {"Item": "Covariables", "Value": ", ".join(map(str, covariate_cols))},
                    {"Item": "Baseline variables", "Value": ", ".join(map(str, baseline_cols))},
                    {"Item": "Follow-up outcome variables", "Value": ", ".join(map(str, followup_outcome_cols))},
                ])

                st.dataframe(summary, use_container_width=True)

                active_project_folder = st.session_state.get("voxel_project_folder", "")
                if active_project_folder:
                    patient_data_folder = Path(active_project_folder) / "01_Load_Patient_Clinical_Data"
                    st.caption("The clinical dataset, setup summary and QC files will be saved in:")
                    st.code(str(patient_data_folder))
                else:
                    st.warning("No active VBA project is open. Create or open a project first so the dataset and QC files can be saved there.")

                def save_voxel_patient_clinical_data_to_project(setup_dict, summary_df):
                    project_folder = st.session_state.get("voxel_project_folder", "")
                    if not project_folder:
                        raise ValueError("No active VBA project folder was found.")

                    patient_data_folder = Path(project_folder) / "01_Load_Patient_Clinical_Data"
                    patient_data_folder.mkdir(parents=True, exist_ok=True)

                    original_filename = st.session_state.get("voxel_patient_data_filename", "patient_clinical_data.xlsx")
                    safe_original_filename = Path(original_filename).name
                    original_path = patient_data_folder / safe_original_filename

                    uploaded_bytes = st.session_state.get("voxel_patient_data_file_bytes", None)
                    if uploaded_bytes is not None:
                        original_path.write_bytes(uploaded_bytes)
                    else:
                        if safe_original_filename.lower().endswith(".csv"):
                            patient_df.to_csv(original_path, index=False)
                        else:
                            patient_df.to_excel(original_path, index=False)

                    clean_dataset_path = patient_data_folder / "patient_clinical_data_clean_copy.csv"
                    patient_df.to_csv(clean_dataset_path, index=False)

                    setup_summary_path = patient_data_folder / "patient_clinical_data_setup_summary.csv"
                    summary_df.to_csv(setup_summary_path, index=False)

                    setup_note_path = patient_data_folder / "patient_clinical_data_setup_note.txt"
                    setup_note = f"""Voxel-based analysis patient clinical data setup
=================================================

Saved/updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Source file:
{original_filename}

Patient ID column:
{setup_dict.get('patient_id_column', '')}

Covariables:
{', '.join(map(str, setup_dict.get('covariate_columns', []))) or 'None selected'}

Baseline variables:
{', '.join(map(str, setup_dict.get('baseline_columns', []))) or 'None selected'}

Follow-up outcome variables:
{', '.join(map(str, setup_dict.get('followup_outcome_columns', []))) or 'None selected'}

Rows / patients: {patient_df.shape[0]}
Columns: {patient_df.shape[1]}

Saved files:
- Original uploaded file: {original_path.name}
- Clean CSV copy: {clean_dataset_path.name}
- Setup summary: {setup_summary_path.name}
"""
                    setup_note_path.write_text(setup_note, encoding="utf-8")

                    issue_df = st.session_state.get("voxel_quality_issues", pd.DataFrame())
                    missing_df = st.session_state.get("voxel_missing_summary", pd.DataFrame())

                    qc_issue_path = patient_data_folder / "voxel_patient_data_quality_issues.csv"
                    qc_missing_path = patient_data_folder / "voxel_patient_missing_data_summary.csv"

                    if issue_df is not None and not issue_df.empty:
                        issue_df.to_csv(qc_issue_path, index=False)
                    else:
                        pd.DataFrame([{"Status": "No quality issues saved or quality check not run."}]).to_csv(qc_issue_path, index=False)

                    if missing_df is not None and not missing_df.empty:
                        missing_df.to_csv(qc_missing_path, index=False)
                    else:
                        pd.DataFrame([{"Status": "No missing-data summary saved or quality check not run."}]).to_csv(qc_missing_path, index=False)

                    st.session_state.voxel_patient_data_saved_files = {
                        "patient_data_folder": str(patient_data_folder),
                        "original_file": str(original_path),
                        "clean_csv_copy": str(clean_dataset_path),
                        "setup_summary": str(setup_summary_path),
                        "setup_note": str(setup_note_path),
                        "quality_issues": str(qc_issue_path),
                        "missing_summary": str(qc_missing_path),
                    }
                    return st.session_state.voxel_patient_data_saved_files

                if st.button("Save patient clinical data setup", type="primary", use_container_width=True):
                    setup = {
                        "filename": st.session_state.get("voxel_patient_data_filename", ""),
                        "patient_id_column": patient_id_col,
                        "covariate_columns": covariate_cols,
                        "baseline_columns": baseline_cols,
                        "followup_outcome_columns": followup_outcome_cols,
                    }
                    st.session_state.voxel_patient_data_setup = setup
                    try:
                        saved_files = save_voxel_patient_clinical_data_to_project(setup, summary)
                        st.success("Patient clinical data setup saved to the active project folder.")
                        with st.expander("Saved project files", expanded=True):
                            for label, path in saved_files.items():
                                st.write(f"**{label.replace('_', ' ').title()}**")
                                st.code(path)
                    except Exception as error:
                        st.error(f"Patient clinical data setup was saved in the app, but files could not be written to the project folder: {error}")

                saved_files = st.session_state.get("voxel_patient_data_saved_files", {})
                if saved_files:
                    with st.expander("Saved patient clinical data files"):
                        for label, path in saved_files.items():
                            st.write(f"**{label.replace('_', ' ').title()}**")
                            st.code(path)

    st.divider()

    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back to Voxel-based Analysis dashboard", use_container_width=True):
            go_to("voxel_analysis_home")
    with col_next:
        if st.button("Next: Load images / masks", use_container_width=True):
            go_to("voxel_load_images")


# ============================================================
# VOXEL LOAD IMAGES PAGE
# ============================================================

# ============================================================
# VOXEL-BASED ANALYSIS MODULE - LOAD IMAGES / MASKS
# ============================================================
elif st.session_state.page == "voxel_load_images":
    st.header("Voxel-based Analysis")
    st.subheader("Load image directory and check filenames")

    st.write(
        "This step organises image files before the viewer is prepared. "
        "Load the directory, scan files, run filename QC, select the reference image, then prepare the viewer."
    )

    if "voxel_image_load_step" not in st.session_state:
        st.session_state.voxel_image_load_step = "instructions"

    if "voxel_selected_image_format" not in st.session_state:
        st.session_state.voxel_selected_image_format = "DICOM"

    if "voxel_image_directory_setup_saved" not in st.session_state:
        st.session_state.voxel_image_directory_setup_saved = False


    if "voxel_image_directory_path_selected" not in st.session_state:
        st.session_state.voxel_image_directory_path_selected = ""
    if "voxel_image_directory_path_selected_applied" not in st.session_state:
        st.session_state.voxel_image_directory_path_selected_applied = ""

    selected_image_dir_from_browse = st.session_state.get("voxel_image_directory_path_selected", "")
    applied_image_dir_from_browse = st.session_state.get("voxel_image_directory_path_selected_applied", "")
    if selected_image_dir_from_browse and selected_image_dir_from_browse != applied_image_dir_from_browse:
        st.session_state.voxel_image_directory_path = selected_image_dir_from_browse
        st.session_state.voxel_image_directory_path_selected_applied = selected_image_dir_from_browse

    def browse_for_voxel_image_directory():
        """Open a local directory picker for the source image directory."""
        selected_folder = ""
        try:
            import subprocess
            powershell_script = """
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = 'Select source image directory'
$dialog.ShowNewFolderButton = $false
$dialog.RootFolder = [System.Environment+SpecialFolder]::Desktop
$result = $dialog.ShowDialog()
if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
    Write-Output $dialog.SelectedPath
}
"""
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-STA", "-Command", powershell_script],
                capture_output=True,
                text=True,
                timeout=120,
            )
            selected_folder = completed.stdout.strip().splitlines()[-1].strip() if completed.stdout.strip() else ""
        except Exception:
            selected_folder = ""

        if not selected_folder:
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                root.update()
                try:
                    root.attributes("-topmost", True)
                except Exception:
                    pass
                selected_folder = filedialog.askdirectory(
                    parent=root,
                    title="Select source image directory",
                    mustexist=True,
                )
                root.destroy()
            except Exception as error:
                st.warning("Folder browser could not be opened from this Streamlit session.")
                st.caption(f"Folder browser error: {error}")
                selected_folder = ""

        if selected_folder:
            selected_path = Path(selected_folder).expanduser()
            if selected_path.exists() and selected_path.is_dir():
                st.session_state.voxel_image_directory_path_selected = str(selected_path)
            else:
                st.error("The selected path is not a directory.")

    def save_voxel_image_directory_metadata_to_project(directory_path, selected_format):
        """Save image-directory setup and generated metadata to the active project.

        The original images are not copied because they may be very large. Only paths,
        scan indexes, summaries and QC tables are stored.
        """
        project_folder = st.session_state.get("voxel_project_folder", "")
        if not project_folder:
            return False, "No active VBA project folder was found. Open or create a project first."

        image_data_folder = Path(project_folder) / "02_Load_Images_Masks"
        image_data_folder.mkdir(parents=True, exist_ok=True)

        file_df = st.session_state.get("voxel_loaded_image_files_df", pd.DataFrame())
        if file_df is None or file_df.empty:
            return False, "No scanned image files are available to save."

        patient_summary_df = make_voxel_patient_file_summary(file_df)

        file_index_path = image_data_folder / "image_directory_file_index.csv"
        patient_summary_path = image_data_folder / "image_patient_file_summary.csv"
        setup_summary_path = image_data_folder / "image_directory_setup_summary.csv"
        setup_note_path = image_data_folder / "image_directory_setup_note.txt"
        qc_summary_path = image_data_folder / "voxel_filename_qc_summary.csv"
        qc_issues_path = image_data_folder / "voxel_filename_quality_issues.csv"
        excel_match_path = image_data_folder / "voxel_filename_excel_patient_match.csv"

        file_df.to_csv(file_index_path, index=False)
        patient_summary_df.to_csv(patient_summary_path, index=False)

        setup_summary_df = pd.DataFrame([
            {"Item": "Saved/updated", "Value": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")},
            {"Item": "Source image directory", "Value": str(directory_path)},
            {"Item": "Image format", "Value": str(selected_format)},
            {"Item": "Files found", "Value": int(file_df.shape[0])},
            {"Item": "Patients detected", "Value": int(patient_summary_df.shape[0])},
            {"Item": "Images copied to project folder", "Value": "No - original image files remain in the source directory"},
        ])
        setup_summary_df.to_csv(setup_summary_path, index=False)

        setup_note = f"""Voxel-based analysis image directory setup
=================================================

Saved/updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Source image directory:
{directory_path}

Image format:
{selected_format}

Files found: {file_df.shape[0]}
Patients detected: {patient_summary_df.shape[0]}

Important:
The original CT, dose, RTSTRUCT and mask image files were not copied into the project folder.
Only the source directory path, file index, patient file summary and QC outputs were saved.

Saved metadata files:
- File index: {file_index_path.name}
- Patient file summary: {patient_summary_path.name}
- Setup summary: {setup_summary_path.name}
"""
        setup_note_path.write_text(setup_note, encoding="utf-8")

        saved_files = {
            "image_data_folder": str(image_data_folder),
            "file_index": str(file_index_path),
            "patient_summary": str(patient_summary_path),
            "setup_summary": str(setup_summary_path),
            "setup_note": str(setup_note_path),
        }

        if st.session_state.get("voxel_filename_qc_ran", False):
            qc_summary_df = st.session_state.get("voxel_filename_qc_summary", pd.DataFrame())
            qc_issues_df = st.session_state.get("voxel_filename_qc_issues", pd.DataFrame())
            excel_match_df = st.session_state.get("voxel_filename_excel_match", pd.DataFrame())

            if qc_summary_df is not None:
                qc_summary_df.to_csv(qc_summary_path, index=False)
                saved_files["filename_qc_summary"] = str(qc_summary_path)
            if qc_issues_df is not None:
                qc_issues_df.to_csv(qc_issues_path, index=False)
                saved_files["filename_qc_issues"] = str(qc_issues_path)
            if excel_match_df is not None:
                excel_match_df.to_csv(excel_match_path, index=False)
                saved_files["excel_patient_match"] = str(excel_match_path)

        st.session_state.voxel_image_directory_saved_files = saved_files
        st.session_state.voxel_image_directory_source_path = str(directory_path)
        st.session_state.voxel_image_directory_setup_saved = True
        return True, f"Image-directory setup saved in {image_data_folder}"

    st.markdown("### Image file workflow")
    st.caption("Click a step to open it.")

    c1, c2, c3, c4 = st.columns(4)

    def image_step_button(label, icon, subtitle, step_key):
        active = st.session_state.voxel_image_load_step == step_key
        if st.button(
            f"{icon}\n\n{label}\n\n{subtitle}",
            key=f"voxel_image_step_{step_key}",
            use_container_width=True,
            type="primary" if active else "secondary",
        ):
            st.session_state.voxel_image_load_step = step_key
            st.rerun()

    with c1:
        image_step_button("1. Instructions", "📘", "Naming rules", "instructions")
    with c2:
        image_step_button("2. Format", "🧭", "DICOM or NIfTI", "format")
    with c3:
        image_step_button("3. Load directory", "📁", "Scan files", "load")
    with c4:
        image_step_button("4. Filename QC", "🔎", "Flag bad labels", "quality")

    st.divider()

    current_step = st.session_state.voxel_image_load_step

    if current_step == "instructions":
        st.markdown("## 📘 Step 1: Instructions")

        st.info(
            "For this step, all loaded imaging files should be either DICOM or NIfTI. "
            "The quality check focuses on filenames and folder labels."
        )

        st.markdown(
            """
            **Recommended folder structure**

            ```text
            Images/
              Pt001/
                CT/
                Dose/
                RTSTRUCT/
                Mask/
              Pt002/
                CT/
                Dose/
                RTSTRUCT/
                Mask/
            ```

            **Recommended filename labels**

            ```text
            CT_Pt001.dcm
            MR_Pt001.nii.gz
            Dose_Pt001.dcm
            RTSTRUCT_Pt001.dcm
            Mask_Hippocampus_Pt001.nii.gz
            ```

            **RTSTRUCT naming accepted by the quality check**

            ```text
            RTSTRUCT_Pt001.dcm
            RT.Struct.Pt001.dcm
            RT Struct Pt001.dcm
            RT-STRUCT-Pt001.dcm
            RS.Pt001.dcm
            StructureSet_Pt001.dcm
            ```

            **What the quality check looks for**

            - Patient folder / patient ID
            - CT or MR main image
            - Mask files
            - Dose files
            - RTSTRUCT files
            - Files that are poorly labelled or unclear

            If labels are unclear, the app will flag them and suggest a clearer filename.
            """
        )

        if st.button("Next: Choose format →", use_container_width=True):
            st.session_state.voxel_image_load_step = "format"
            st.rerun()

    elif current_step == "format":
        st.markdown("## 🧭 Step 2: Choose image format")

        previous_format = st.session_state.get("voxel_selected_image_format", "DICOM")

        selected_format = st.radio(
            "Select image format for this directory",
            options=["DICOM", "NIfTI"],
            index=0 if previous_format == "DICOM" else 1,
            horizontal=True,
            key="voxel_selected_image_format_widget"
        )

        if selected_format != previous_format:
            st.session_state.voxel_selected_image_format = selected_format
            st.session_state.voxel_loaded_image_files_df = pd.DataFrame()
            st.session_state.voxel_loaded_image_format = selected_format
            st.session_state.voxel_filename_qc_ran = False
            st.session_state.voxel_filename_qc_summary = pd.DataFrame()
            st.session_state.voxel_filename_qc_issues = pd.DataFrame()
            st.session_state.voxel_filename_excel_match = pd.DataFrame()
            st.success(f"Image format changed to {selected_format}. Previous scan results were cleared.")
        else:
            st.session_state.voxel_selected_image_format = selected_format

        if selected_format == "DICOM":
            st.info("The directory scan will include `.dcm` and `.dicom` files.")

            st.button(
                "Convert DICOM to NIfTI",
                use_container_width=True,
                disabled=True,
                help="Work in progress: DICOM-to-NIfTI conversion will be added in a later version."
            )
            st.caption("DICOM-to-NIfTI conversion: work in progress.")

        else:
            st.info("The directory scan will include `.nii` and `.nii.gz` files.")

        if st.button("Next: Load directory →", use_container_width=True):
            st.session_state.voxel_image_load_step = "load"
            st.rerun()

    elif current_step == "load":
        st.markdown("## 📁 Step 3: Load directory")

        selected_format = st.session_state.get(
            "voxel_selected_image_format",
            st.session_state.get("voxel_selected_image_format_widget", "DICOM")
        )

        st.info(
            f"Selected format: **{selected_format}**. Paste the folder path containing patient image folders/files."
        )

        path_col, browse_col = st.columns([0.86, 0.14])
        with path_col:
            directory_path = st.text_input(
                "Image directory path",
                value=st.session_state.get("voxel_image_directory_path", ""),
                key="voxel_image_directory_path",
                placeholder=r"Example: C:\Users\sueme\Desktop\Voxel_Project\Images"
            )
        with browse_col:
            st.write("")
            if st.button("Browse...", key="voxel_browse_image_directory", use_container_width=True):
                browse_for_voxel_image_directory()
                st.rerun()

        active_project_folder = st.session_state.get("voxel_project_folder", "")
        if active_project_folder:
            st.caption("Only the directory path and generated metadata will be saved. The original image files will not be copied.")
            st.code(str(Path(active_project_folder) / "02_Load_Images_Masks"))
        else:
            st.warning("No active VBA project is open. Open or create a project first so image metadata and QC files can be saved.")

        if st.button("Scan directory", type="primary", use_container_width=True):
            file_df, error_message = scan_voxel_image_directory(directory_path, selected_format)

            if error_message:
                st.error(error_message)
            else:
                st.session_state.voxel_loaded_image_files_df = file_df
                st.session_state.voxel_loaded_image_format = selected_format
                st.session_state.voxel_image_directory_source_path = directory_path

                if file_df.empty:
                    st.warning(f"No {selected_format} files were found in this directory.")
                else:
                    st.session_state.voxel_loaded_image_format = selected_format
                    st.session_state.voxel_filename_qc_ran = False
                    st.session_state.voxel_filename_qc_summary = pd.DataFrame()
                    st.session_state.voxel_filename_qc_issues = pd.DataFrame()
                    st.session_state.voxel_filename_excel_match = pd.DataFrame()
                    st.session_state.voxel_image_directory_setup_saved = False
                    st.success(f"Found {file_df.shape[0]} {selected_format} file(s).")
                    st.info("Review the detected files, then click **Save image setup to project** to store this setup in the active project folder.")

        file_df = st.session_state.get("voxel_loaded_image_files_df", pd.DataFrame())

        if file_df is not None and not file_df.empty:
            st.markdown("### Loaded files grouped by patient")

            loaded_format = st.session_state.get("voxel_loaded_image_format", selected_format)
            if loaded_format != selected_format:
                st.warning(
                    f"Loaded files were scanned as {loaded_format}, but the currently selected format is {selected_format}. "
                    "Click Scan directory again to reload using the selected format."
                )

            patient_summary_df = make_voxel_patient_file_summary(file_df)

            a, b, c = st.columns(3)
            a.metric("Files found", file_df.shape[0])
            b.metric("Patients detected", patient_summary_df.shape[0])
            c.metric("Format", st.session_state.get("voxel_loaded_image_format", selected_format))

            st.dataframe(patient_summary_df, use_container_width=True)

            with st.expander("Show individual file list"):
                st.dataframe(file_df.head(500), use_container_width=True)

            save_col, qc_col = st.columns(2)
            with save_col:
                if st.button("Save image setup to project", type="primary", use_container_width=True, key="save_image_setup_after_scan"):
                    saved_ok, saved_message = save_voxel_image_directory_metadata_to_project(
                        st.session_state.get("voxel_image_directory_source_path", directory_path),
                        loaded_format,
                    )
                    if saved_ok:
                        st.success(saved_message)
                    else:
                        st.warning(saved_message)
            with qc_col:
                if st.button("Next: Run filename QC →", use_container_width=True):
                    st.session_state.voxel_image_load_step = "quality"
                    st.rerun()
        else:
            st.info("No image files loaded yet.")

    elif current_step == "quality":
        st.markdown("## 🔎 Step 4: Filename quality check")

        file_df = st.session_state.get("voxel_loaded_image_files_df", pd.DataFrame())
        selected_format = st.session_state.get("voxel_loaded_image_format", st.session_state.get("voxel_selected_image_format", "DICOM"))

        if file_df is None or file_df.empty:
            st.warning("No image files have been loaded yet.")
            if st.button("← Go to Load directory", use_container_width=True):
                st.session_state.voxel_image_load_step = "load"
                st.rerun()
        else:
            patient_ids = []
            patient_data_setup = st.session_state.get("voxel_patient_data_setup", {})
            patient_df = st.session_state.get("voxel_patient_data", None)
            patient_id_col = patient_data_setup.get("patient_id_column", "")

            if patient_df is not None and patient_id_col in patient_df.columns:
                patient_ids = patient_df[patient_id_col].dropna().astype(str).str.strip().tolist()

            if st.button("Run filename quality check", type="primary", use_container_width=True):
                summary_df, issue_df, excel_match_df = run_image_filename_quality_check(
                    file_df=file_df,
                    expected_format=selected_format,
                    patient_id_list=patient_ids
                )

                st.session_state.voxel_filename_qc_summary = summary_df
                st.session_state.voxel_filename_qc_issues = issue_df
                st.session_state.voxel_filename_excel_match = excel_match_df
                st.session_state.voxel_filename_qc_ran = True
                st.session_state.voxel_image_directory_setup_saved = False
                st.success("Filename quality check completed.")
                st.info("Click **Save image setup to project** to save the file index, patient summary and QC outputs in the active project folder.")

            if st.session_state.get("voxel_filename_qc_ran", False):
                summary_df = st.session_state.get("voxel_filename_qc_summary", pd.DataFrame())
                issue_df = st.session_state.get("voxel_filename_qc_issues", pd.DataFrame())

                st.markdown("### Patient file summary")
                st.dataframe(summary_df, use_container_width=True)

                st.markdown("### Excel patient ID match check")
                excel_match_df = st.session_state.get("voxel_filename_excel_match", pd.DataFrame())

                if excel_match_df is None or excel_match_df.empty:
                    st.info("No Excel patient ID match check was available.")
                else:
                    if "Skipped" in excel_match_df.get("Status", pd.Series(dtype=str)).astype(str).tolist():
                        st.warning(
                            "No loaded Excel patient data setup was found. "
                            "Image patient IDs were not compared with an Excel Patient ID column."
                        )
                    st.dataframe(excel_match_df, use_container_width=True)

                st.markdown("### Files to review / rename")

                if issue_df.empty:
                    st.success("No obvious filename or folder labelling issues were detected.")
                else:
                    important_count = int((issue_df["Severity"] == "Important").sum()) if "Severity" in issue_df.columns else 0
                    check_count = int((issue_df["Severity"] == "Check").sum()) if "Severity" in issue_df.columns else 0

                    q1, q2 = st.columns(2)
                    q1.metric("Important issues", important_count)
                    q2.metric("Check issues", check_count)

                    def highlight_filename_issues(row):
                        severity = row.get("Severity", "")
                        if severity == "Important":
                            return ["background-color: #ffd6d6"] * len(row)
                        if severity == "Check":
                            return ["background-color: #fff2cc"] * len(row)
                        return [""] * len(row)

                    st.dataframe(
                        issue_df.style.apply(highlight_filename_issues, axis=1),
                        use_container_width=True
                    )

                    st.download_button(
                        "Download filename QC issues as CSV",
                        data=issue_df.to_csv(index=False).encode("utf-8"),
                        file_name="voxel_filename_quality_issues.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                st.divider()
                if st.button("Save image setup to project", type="primary", use_container_width=True, key="save_image_setup_after_qc"):
                    saved_ok, saved_message = save_voxel_image_directory_metadata_to_project(
                        st.session_state.get("voxel_image_directory_source_path", st.session_state.get("voxel_image_directory_path", "")),
                        selected_format,
                    )
                    if saved_ok:
                        st.success(saved_message)
                    else:
                        st.warning(saved_message)

            else:
                st.info("Click **Run filename quality check** to flag unclear patient folders, CT, mask, dose and RTSTRUCT labels.")


    saved_image_files = st.session_state.get("voxel_image_directory_saved_files", {})
    image_setup_saved = st.session_state.get("voxel_image_directory_setup_saved", False)
    if st.session_state.get("voxel_loaded_image_files_df", pd.DataFrame()) is not None and not st.session_state.get("voxel_loaded_image_files_df", pd.DataFrame()).empty:
        if image_setup_saved:
            st.success("Image setup has been saved to the active project folder.")
        else:
            st.warning("Image files have been scanned, but the image setup has not yet been saved to the project folder.")

    if saved_image_files:
        with st.expander("Saved image-directory metadata in this project", expanded=False):
            st.caption("Original image files are not copied. These are metadata/QC files saved in the project folder.")
            for label, path in saved_image_files.items():
                st.write(f"**{label.replace('_', ' ').title()}**")
                st.code(path)



    st.divider()

    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back to Voxel-based Analysis dashboard", use_container_width=True):
            go_to("voxel_analysis_home")
    with col_next:
        if st.button("Next: Normalisation", use_container_width=True):
            file_df_for_next = st.session_state.get("voxel_loaded_image_files_df", pd.DataFrame())
            if file_df_for_next is not None and not file_df_for_next.empty and not st.session_state.get("voxel_image_directory_setup_saved", False):
                st.warning("Please save the image setup to the project before moving to normalisation.")
            else:
                st.session_state.voxel_registration_step = "preprocess"
                go_to("voxel_registration_alignment")


# ============================================================
# VOXEL REGISTRATION / ALIGNMENT PAGE
# ============================================================

# ============================================================
# VOXEL-BASED ANALYSIS MODULE - NORMALISATION
# ============================================================
elif st.session_state.page == "voxel_registration_alignment":
    st.header("Voxel-based Analysis")
    st.subheader("Normalisation")

    st.write(
        "This step checks image geometry and prepares images for voxel-based analysis by standardising "
        "orientation and voxel spacing. It does not register patients to a common reference anatomy yet."
    )

    st.info(
        "Use linear or B-spline interpolation for CT, MRI, PET and dose. "
        "Use nearest-neighbour interpolation for masks/segmentations to avoid invalid label values."
    )

    if not SIMPLEITK_AVAILABLE:
        st.error("SimpleITK is required for image metadata reading and normalisation. Install it with: py -m pip install SimpleITK")
    else:
        file_df = st.session_state.get("voxel_loaded_image_files_df", pd.DataFrame())

        if file_df is None or file_df.empty:
            st.warning("No image files are loaded. Go back to Load images / masks first.")
            if st.button("← Back to Load images / masks", use_container_width=True):
                go_to("voxel_load_images")
        else:
            items = vbv_normalisation_items(file_df)

            st.markdown("### 1. Read image metadata")
            normalisation_folder = vbv_normalisation_project_folder()
            metadata_csv_path = normalisation_folder / "image_metadata_before_normalisation.csv"

            st.write(
                "Read metadata for all loaded image entries, save one CSV in the active project folder, "
                "and show a preview. This uses lightweight header/tag reading where possible, so it should be faster "
                "than loading full image volumes."
            )

            st.metric("Image entries to read", len(items))

            # Reload saved metadata when reopening an existing project.
            if (
                ("vbv_image_metadata_df" not in st.session_state or st.session_state.get("vbv_image_metadata_df", pd.DataFrame()).empty)
                and metadata_csv_path.exists()
            ):
                try:
                    st.session_state.vbv_image_metadata_df = pd.read_csv(metadata_csv_path)
                    st.session_state.vbv_image_metadata_csv = str(metadata_csv_path)
                except Exception:
                    pass

            if st.button("Read all metadata and save CSV", type="primary", use_container_width=True, key="vbv_read_all_metadata_with_progress"):
                progress_bar = st.progress(0.0)
                counter_box = st.empty()
                status_box = st.empty()

                metadata_df = vbv_collect_metadata_table_fast(
                    file_df,
                    progress_bar=progress_bar,
                    status_box=status_box,
                    counter_box=counter_box,
                )

                st.session_state.vbv_image_metadata_df = metadata_df
                metadata_df.to_csv(metadata_csv_path, index=False)
                st.session_state.vbv_image_metadata_csv = str(metadata_csv_path)
                status_box.success("Metadata reading complete.")
                st.success(f"Metadata CSV saved: {metadata_csv_path}")

            metadata_df = st.session_state.get("vbv_image_metadata_df", pd.DataFrame())
            if metadata_df is not None and not metadata_df.empty:
                st.markdown("#### Metadata preview")
                st.caption(f"Saved file: {metadata_csv_path}")

                if "Status" in metadata_df.columns:
                    read_count = int((metadata_df["Status"].astype(str).str.lower() == "read").sum())
                    failed_count = int((metadata_df["Status"].astype(str).str.lower() == "failed").sum())
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Total", len(metadata_df))
                    col_b.metric("Read", read_count)
                    col_c.metric("Failed", failed_count)

                st.dataframe(metadata_df.head(50), use_container_width=True)

                if "Status" in metadata_df.columns:
                    status_counts = metadata_df["Status"].astype(str).value_counts().reset_index()
                    status_counts.columns = ["Read status", "Count"]
                    st.markdown("#### Read status summary")
                    st.dataframe(status_counts, use_container_width=True, hide_index=True)

                failed_df = metadata_df[metadata_df.get("Status", "").astype(str).str.lower().eq("failed")] if "Status" in metadata_df.columns else pd.DataFrame()
                if failed_df is not None and not failed_df.empty:
                    with st.expander("Show files where metadata could not be read"):
                        cols = [c for c in ["Patient ID", "Role", "Item type", "Source", "Source path", "Read method", "Notes"] if c in failed_df.columns]
                        st.dataframe(failed_df[cols], use_container_width=True)
            else:
                st.info("No saved metadata table found yet. Click 'Read all metadata and save CSV' to create it.")

            st.divider()
            st.markdown("### 2. Orientation and voxel-spacing normalisation")

            st.write(
                "Choose a common voxel spacing. The app reorients images to LPS and resamples each readable image "
                "to the selected spacing. Normalised outputs are saved in the active project folder."
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                sx = st.number_input("Target spacing X (mm)", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="vbv_norm_spacing_x")
            with c2:
                sy = st.number_input("Target spacing Y (mm)", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="vbv_norm_spacing_y")
            with c3:
                sz = st.number_input("Target spacing Z (mm)", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="vbv_norm_spacing_z")

            intensity_interpolation = st.selectbox(
                "Interpolation for CT / MRI / PET / dose",
                ["Linear", "B-spline intensity"],
                index=0,
                key="vbv_norm_intensity_interpolation",
            )

            st.caption("Masks and segmentations are always resampled using nearest-neighbour interpolation.")

            normalisation_folder = vbv_normalisation_project_folder()
            st.write("Normalised outputs will be saved in:")
            st.code(str(normalisation_folder))

            if st.button("Run normalisation", type="primary", use_container_width=True, key="vbv_run_normalisation"):
                results_df = vbv_run_normalisation(
                    file_df=file_df,
                    target_spacing=(float(sx), float(sy), float(sz)),
                    intensity_interpolation=intensity_interpolation,
                )
                st.session_state.voxel_current_step = "Normalisation complete"
                st.success("Normalisation results saved to the project folder.")

            results_df = vbv_load_saved_normalisation_results_if_available()
            if results_df is not None and not results_df.empty:
                st.markdown("### Normalisation results")
                st.dataframe(results_df, use_container_width=True)
                st.download_button(
                    "Download normalisation results CSV",
                    data=results_df.to_csv(index=False).encode("utf-8"),
                    file_name="normalisation_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                st.markdown("### 3. Preview normalised outputs")
                preview_table = vbv_processed_ct_mr_table(results_df)
                vbv_render_processed_image_viewer(preview_table)

            st.divider()
            col_back, col_next = st.columns(2)
            with col_back:
                if st.button("← Back to Load images / masks", use_container_width=True, key="vbv_norm_back_images"):
                    go_to("voxel_load_images")
            with col_next:
                if st.button("Next: Reference image / CCS →", use_container_width=True, key="vbv_norm_next_reference"):
                    if not st.session_state.get("vbv_normalisation_ran", False):
                        st.warning("Run normalisation first, or confirm that existing normalised outputs are already available.")
                    else:
                        go_to("voxel_reference_ccs")


# ============================================================
# VOXEL BATCH REGISTRATION PAGE
# ============================================================

# ============================================================
# VOXEL-BASED ANALYSIS MODULE - BATCH REGISTRATION
# ============================================================
elif st.session_state.page == "voxel_batch_registration":
    st.header("Voxel-based Analysis")
    st.subheader("Batch registration")

    st.write(
        "Register selected normalised images into one reference-patient space. "
        "Inputs come only from the normalisation folder; raw source images are not used here."
    )

    if not SIMPLEITK_AVAILABLE:
        st.error("SimpleITK is required for batch registration. Install it with: py -m pip install SimpleITK")
        st.stop()

    normalised_df = vbv_load_normalised_outputs_for_batch()

    if normalised_df is None or normalised_df.empty:
        st.error("No saved normalised outputs were found. Run Normalisation first.")
        if st.button("← Back to Normalisation", use_container_width=True):
            go_to("voxel_registration_alignment")
        st.stop()

    normalised_df = normalised_df.copy()
    normalised_df["Batch type"] = normalised_df.apply(vbv_batch_type_key, axis=1)

    reference_setup = st.session_state.get("voxel_reference_ccs_setup", {})
    if not reference_setup:
        st.warning("Select and save the reference image / CCS before batch registration.")
        if st.button("Go to Reference image / CCS", use_container_width=True, key="vbv_batch_go_reference_required"):
            go_to("voxel_reference_ccs")
        st.stop()

    with st.expander("Saved reference / CCS used for this batch", expanded=True):
        st.write("Reference patient:", reference_setup.get("reference_patient", ""))
        st.write("Reference image:", reference_setup.get("reference_image", ""))
        st.caption("Registration method, transform targets and interpolation are selected below in Batch registration.")

    st.markdown("### 1. Select registration inputs")
    st.caption("Input folder: normalised outputs only")
    st.code(str(vbv_normalisation_project_folder().resolve()), language=None)

    patient_ids = sorted(normalised_df["Patient ID"].astype(str).unique().tolist())
    if len(patient_ids) < 1:
        st.error("No patient IDs were found in the normalised outputs.")
        st.stop()

    saved_reference_patient = str(reference_setup.get("reference_patient", ""))
    saved_reference_index = patient_ids.index(saved_reference_patient) if saved_reference_patient in patient_ids else 0
    reference_patient = st.selectbox(
        "Fixed/reference patient",
        patient_ids,
        index=saved_reference_index,
        key="vbv_batch_reference_patient",
        help="Default comes from the saved Reference image / CCS step.",
    )

    ref_rows = normalised_df[normalised_df["Patient ID"].astype(str) == str(reference_patient)].copy()
    ref_rows = ref_rows[ref_rows["Batch type"].isin(["CT", "MR", "Other image"])].copy()
    if ref_rows.empty:
        st.error("The selected reference patient has no CT/MR/intensity image available as fixed reference.")
        st.stop()

    ref_rows = ref_rows.reset_index(drop=True)
    ref_default_index = vbv_batch_find_default_reference_index(ref_rows)
    saved_reference_image = str(reference_setup.get("reference_image", ""))
    if saved_reference_image:
        for _idx, _row in ref_rows.iterrows():
            _label = vbv_batch_file_label(_row.to_dict())
            _out = str(_row.get("Output file", _row.get("Output path", _row.get("Full path", ""))))
            if saved_reference_image in _label or saved_reference_image in _out or Path(saved_reference_image).name in _label:
                ref_default_index = int(_idx)
                break
    fixed_idx = st.selectbox(
        "Fixed/reference image",
        list(range(len(ref_rows))),
        index=ref_default_index,
        format_func=lambda i: vbv_batch_file_label(ref_rows.iloc[int(i)].to_dict()),
        key="vbv_batch_fixed_image",
    )
    fixed_row = ref_rows.iloc[int(fixed_idx)].to_dict()

    moving_patient_options = [p for p in patient_ids if str(p) != str(reference_patient)]
    default_moving = moving_patient_options
    moving_patients = st.multiselect(
        "Moving patients to register",
        options=moving_patient_options,
        default=default_moving,
        key="vbv_batch_moving_patients",
    )

    available_type_order = ["CT", "MR", "Dose", "Mask / segmentation", "Other image"]
    available_types = [t for t in available_type_order if t in normalised_df["Batch type"].unique().tolist()]
    default_types = [t for t in ["CT", "MR", "Dose", "Mask / segmentation"] if t in available_types]
    selected_types = st.multiselect(
        "Normalised image types to register/apply transform to",
        options=available_types,
        default=default_types,
        key="vbv_batch_selected_image_types",
        help="The transform is estimated from the patient's CT/MR/intensity image. If B-spline is selected, it runs after rigid and affine; the final transform is then applied to CT/MR/dose/masks with safe interpolation.",
    )

    st.markdown("### 2. Select registration method")
    method = st.radio(
        "Registration method",
        options=["Rigid only", "Rigid → Affine", "Rigid → Affine → B-spline"],
        index=1,
        horizontal=True,
        key="vbv_batch_registration_method_revamped",
    )

    st.markdown("### 3. Interpolation rules")
    st.dataframe(
        pd.DataFrame([
            {"Data type": "CT / MR / PET", "Interpolation": "Linear", "Reason": "Continuous image intensities"},
            {"Data type": "Dose", "Interpolation": "Linear", "Reason": "Dose is continuous and should not be nearest-neighbour sampled"},
            {"Data type": "Masks / segmentations", "Interpolation": "Nearest-neighbour", "Reason": "Preserves discrete label values even when the transform is B-spline"},
        ]),
        use_container_width=True,
        hide_index=True,
    )
    if method == "Rigid → Affine → B-spline":
        st.warning(
            "B-spline is an advanced deformable registration step. Use it only after checking that rigid/affine alignment is reasonable, "
            "and review overlays carefully for unrealistic local warping."
        )
        st.info(
            "When you run this option, the status box will show: Stage 1 rigid registration → Stage 2 affine registration → "
            "Stage 3 B-spline deformable registration. If B-spline fails for a patient, the app keeps the affine transform and records the B-spline error instead of stopping the whole batch."
        )
        st.info(
            "When you run this option, the status box will show: Stage 1 rigid registration → Stage 2 affine registration → "
            "Stage 3 B-spline deformable registration. If B-spline fails for a patient, the app keeps the affine transform and records the B-spline error instead of stopping the whole batch."
        )

    st.markdown("### 4. Batch registration plan")
    if moving_patients and selected_types:
        plan_rows = []
        for patient_id in moving_patients:
            p_rows = normalised_df[normalised_df["Patient ID"].astype(str) == str(patient_id)].copy()
            p_rows = p_rows[p_rows["Batch type"].isin(selected_types)].copy()
            transform_row = vbv_batch_default_transform_row(p_rows[p_rows["Batch type"].isin(["CT", "MR", "Other image"])])
            for _, row in p_rows.iterrows():
                plan_rows.append({
                    "Patient ID": patient_id,
                    "Moving file": vbv_batch_file_label(row),
                    "Type": row.get("Batch type", ""),
                    "Transform estimated from": vbv_batch_file_label(transform_row) if transform_row else "No CT/MR/intensity image available",
                    "Fixed reference": vbv_batch_file_label(fixed_row),
                })
        plan_df = pd.DataFrame(plan_rows)
        st.dataframe(plan_df, use_container_width=True, hide_index=True)
    else:
        plan_df = pd.DataFrame()
        st.warning("Select at least one moving patient and one image type to register.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Moving patients", len(moving_patients))
    c2.metric("Files queued", int(plan_df.shape[0]) if not plan_df.empty else 0)
    c3.metric("Method", method)

    output_root = vbv_batch_registration_project_folder()
    st.markdown("**Batch registration output folder**")
    st.code(str(output_root.resolve()), language=None)

    col_a, col_b = st.columns(2)
    with col_a:
        skip_existing = st.checkbox("Skip existing registered outputs", value=True, key="vbv_batch_skip_existing_revamped")
    with col_b:
        force_overwrite = st.checkbox("Overwrite existing outputs", value=False, key="vbv_batch_force_overwrite_revamped")
    if force_overwrite:
        skip_existing = False

    if st.button(
        "Run batch registration",
        type="primary",
        use_container_width=True,
        disabled=(len(moving_patients) == 0 or len(selected_types) == 0),
        key="vbv_run_revamped_batch_registration",
    ):
        progress_bar = st.progress(0.0)
        progress_text = st.empty()
        with st.spinner("Running batch registration from normalised outputs..."):
            results_df = vbv_run_batch_registration_from_normalised(
                normalised_df=normalised_df,
                fixed_row=fixed_row,
                moving_patient_ids=moving_patients,
                selected_type_keys=selected_types,
                method=method,
                skip_existing=skip_existing,
                force_overwrite=force_overwrite,
                progress_bar=progress_bar,
                progress_text=progress_text,
            )
        st.success("Batch registration complete.")
        st.session_state.voxel_current_step = "Batch registration complete"

    results_df = vbv_load_saved_batch_registration_results()
    if results_df is not None and not results_df.empty:
        st.markdown("### 5. Batch registration results")
        processed = int((results_df["Status"] == "Processed").sum()) if "Status" in results_df.columns else 0
        skipped = int((results_df["Status"] == "Skipped").sum()) if "Status" in results_df.columns else 0
        failed = int((results_df["Status"] == "Failed").sum()) if "Status" in results_df.columns else 0
        r1, r2, r3 = st.columns(3)
        r1.metric("Processed", processed)
        r2.metric("Skipped", skipped)
        r3.metric("Failed", failed)

        preferred_cols = [
            "Patient ID", "Moving image", "Reference image", "Image role", "Registration method",
            "Interpolation", "Transform file", "Registered output file", "Status", "Registration details", "Error message"
        ]
        display_cols = [c for c in preferred_cols if c in results_df.columns]
        display_cols += [c for c in results_df.columns if c not in display_cols]
        st.dataframe(results_df[display_cols], use_container_width=True, hide_index=True)
        with st.expander("What do the status and error columns mean?"):
            st.markdown(
                "**Processed** means a registered NIfTI output was written. "
                "**Skipped** usually means an existing registered output was kept, or no selected image type was available for that patient. "
                "**Failed** means transform estimation or resampling failed for that row. "
                "The **Registration details** column contains metric notes or skip reasons. "
                "The **Error message** column is now reserved for true failures."
            )

        summary_csv = st.session_state.get("vbv_batch_registration_summary_csv", str(output_root / "batch_registration_summary.csv"))
        st.caption(f"Saved summary CSV: {summary_csv}")
        st.download_button(
            "Download batch registration summary CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="batch_registration_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

        vbv_render_registered_output_viewer(results_df)

    st.divider()
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back to Normalisation", use_container_width=True, key="vbv_batch_back_to_normalisation"):
            go_to("voxel_registration_alignment")
    with col_next:
        if st.button("Next: Registration QC →", use_container_width=True, key="vbv_batch_next_registration_qc"):
            go_to("voxel_registration_qc")



# ============================================================
# VOXEL REGISTRATION QC PAGE
# ============================================================

# ============================================================
# VOXEL-BASED ANALYSIS MODULE - REGISTRATION QC
# ============================================================
elif st.session_state.page == "voxel_registration_qc":
    st.header("Voxel-based Analysis")
    st.subheader("Registration QC")

    st.write(
        "Review registration quality before accepting the cohort outputs for warp-to-reference, dose normalisation and final VBA readiness. "
        "This step records readable outputs, geometry consistency, visual overlay approval and manual review notes."
    )

    out_folder = vbv_registration_qc_project_folder()
    st.markdown("**Registration QC output folder**")
    st.code(str(out_folder.resolve()), language=None)

    results_df = vbv_load_saved_batch_registration_results()
    if results_df is None or results_df.empty:
        st.error("No batch-registration summary was found. Run Batch registration first.")
        if st.button("← Back to Batch registration", use_container_width=True, key="vbv_qc_back_no_batch"):
            go_to("voxel_batch_registration")
        st.stop()

    registered_df = vbv_prepare_registered_outputs_for_review(results_df)
    if registered_df is None or registered_df.empty:
        st.error("No readable registered outputs were found for QC. Review the Batch registration output folder first.")
        if st.button("← Back to Batch registration", use_container_width=True, key="vbv_qc_back_no_outputs"):
            go_to("voxel_batch_registration")
        st.stop()

    st.markdown("### 1. Automated QC summary")
    st.caption("This checks whether registered files are readable and whether their saved grid geometry is consistent after registration.")

    saved_qc = st.session_state.get("vbv_registration_qc_summary", pd.DataFrame())
    if saved_qc is None or saved_qc.empty:
        saved_qc = vbv_existing_registration_qc_summary()
        if saved_qc is not None and not saved_qc.empty:
            st.session_state.vbv_registration_qc_summary = saved_qc

    saved_approval = {}
    saved_notes = {}
    if saved_qc is not None and not saved_qc.empty and "Patient ID" in saved_qc.columns:
        for _, r in saved_qc.iterrows():
            pid = str(r.get("Patient ID", ""))
            if pid:
                saved_approval[pid] = str(r.get("Visual QC approval", "Not reviewed"))
                saved_notes[pid] = str(r.get("Manual review notes", ""))

    qc_preview = vbv_build_registration_qc_table(saved_notes, saved_approval)
    if qc_preview.empty:
        st.warning("Registration QC table could not be created from the registered outputs.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Patients checked", int(qc_preview.shape[0]))
        c2.metric("Approved", int((qc_preview["Registration QC status"].astype(str) == "Approved").sum()))
        c3.metric("Needs review", int((qc_preview["Registration QC status"].astype(str) != "Approved").sum()))
        st.dataframe(qc_preview, use_container_width=True, hide_index=True)

    st.markdown("### 2. Visual image review")
    st.caption("Review a few registered patients using anatomy as the background. File names and overlay-layer notes are hidden to keep this step simple.")
    vbv_render_registered_output_viewer(results_df)

    st.markdown("### 3. Overall registration QC decision")
    patients = sorted(registered_df["Patient ID"].astype(str).unique().tolist())

    saved_decisions = list(saved_approval.values()) if saved_approval else []
    if saved_decisions and all(x == "Approved" for x in saved_decisions):
        default_decision = "Approved"
    elif saved_decisions and any(x == "Needs manual review" for x in saved_decisions):
        default_decision = "Needs manual review"
    else:
        default_decision = "Not reviewed"

    decision_options = ["Not reviewed", "Approved", "Needs manual review"]
    overall_decision = st.radio(
        "Overall visual registration QC decision for this cohort",
        options=decision_options,
        index=decision_options.index(default_decision),
        horizontal=True,
        key="vbv_registration_qc_overall_decision",
    )
    overall_note = st.text_area(
        "Overall QC note, optional",
        value="" if not saved_notes else next(iter(saved_notes.values()), ""),
        key="vbv_registration_qc_overall_note",
        placeholder="Example: Checked representative axial/sagittal slices. Anatomy alignment acceptable for VBA.",
    )

    if st.button("Save overall registration QC decision", type="primary", use_container_width=True, key="vbv_save_registration_qc_summary"):
        approval_map = {str(patient_id): overall_decision for patient_id in patients}
        notes_map = {str(patient_id): overall_note for patient_id in patients}
        qc_df = vbv_build_registration_qc_table(notes_map, approval_map)
        qc_path = vbv_save_registration_qc_summary(qc_df)
        st.success(f"Registration QC decision saved: {qc_path}")

    final_qc = st.session_state.get("vbv_registration_qc_summary", saved_qc)
    if final_qc is not None and not final_qc.empty:
        st.markdown("### 4. Saved registration QC")
        simple_cols = [
            "Patient ID", "Registration QC status", "Visual QC approval", "Registered anatomy present",
            "Registered dose present", "Registered mask present", "Registered geometry consistent", "QC issue",
        ]
        simple_cols = [c for c in simple_cols if c in final_qc.columns]
        st.dataframe(final_qc[simple_cols], use_container_width=True, hide_index=True)
        st.caption(f"Saved QC CSV: {vbv_registration_qc_project_folder() / 'registration_qc_summary.csv'}")

    st.info(
        "This QC page records the practical registration gate needed before VBA: readable registered anatomy, dose/mask presence, geometry consistency and overall visual approval. "
        "Jacobian/folding checks and contour-agreement metrics such as 95% Hausdorff distance, average distance-to-agreement and centre-of-mass displacement are not calculated yet; those can be added later when displacement fields and final structure sets are available."
    )

    st.divider()
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back to Batch registration", use_container_width=True, key="vbv_registration_qc_back_batch"):
            go_to("voxel_batch_registration")
    with col_next:
        if st.button("Next: Warp to reference space →", use_container_width=True, key="vbv_registration_qc_next_warp"):
            go_to("voxel_warp_to_ccs")


# ============================================================
# VOXEL REFERENCE IMAGE / CCS PAGE
# ============================================================



# ============================================================
# VOXEL WARP TO REFERENCE / CCS PAGE
# ============================================================

# ============================================================
# VOXEL-BASED ANALYSIS MODULE - WARP TO REFERENCE SPACE
# ============================================================
elif st.session_state.page == "voxel_warp_to_ccs":
    st.header("Voxel-based Analysis")
    st.subheader("Warp to reference space / CCS")

    st.write(
        "This step confirms which registered outputs have been propagated into the selected common coordinate system. "
        "According to the VBA workflow, after the CCS is chosen, each patient's anatomy-derived transform is used to map the associated dose and masks onto that CCS before voxel-wise statistics."
    )

    reference_setup = st.session_state.get("voxel_reference_ccs_setup", {})
    if not reference_setup:
        st.warning("Select and save the Reference image / CCS before confirming warped outputs.")
        if st.button("Go to Reference image / CCS", use_container_width=True, key="vbv_warp_go_reference_required"):
            go_to("voxel_reference_ccs")
        st.stop()

    with st.expander("Saved reference / CCS", expanded=True):
        st.write("Reference patient:", reference_setup.get("reference_patient", ""))
        st.write("Reference image:", reference_setup.get("reference_image", ""))
        st.caption("This is the fixed space where all cohort outputs should now be located.")

    warp_folder = vbv_warp_to_ccs_project_folder()
    st.markdown("**Warp manifest output folder**")
    st.code(str(warp_folder.resolve()), language=None)

    st.markdown("### 1. Find registered outputs")
    warp_candidates = vbv_load_warp_candidates_from_batch()
    saved_manifest = st.session_state.get("vbv_warp_manifest", pd.DataFrame())
    if saved_manifest is None or saved_manifest.empty:
        saved_manifest = vbv_existing_warp_manifest()
        if saved_manifest is not None and not saved_manifest.empty:
            st.session_state.vbv_warp_manifest = saved_manifest

    if warp_candidates is None or warp_candidates.empty:
        st.error("No readable registered outputs were found. Run Batch registration first.")
        if st.button("← Back to Batch registration", use_container_width=True, key="vbv_warp_back_no_outputs"):
            go_to("voxel_batch_registration")
        st.stop()

    type_counts = warp_candidates["Warp data type"].value_counts().reset_index()
    type_counts.columns = ["Data type", "Count"]
    st.dataframe(type_counts, use_container_width=True, hide_index=True)

    available_types = [t for t in ["CT", "MR", "Dose", "Mask / segmentation", "Other image"] if t in warp_candidates["Warp data type"].unique().tolist()]
    default_types = [t for t in ["CT", "MR", "Dose", "Mask / segmentation"] if t in available_types]
    selected_types = st.multiselect(
        "Select outputs to include in the CCS/warp manifest",
        options=available_types,
        default=default_types,
        key="vbv_warp_selected_types",
    )

    st.markdown("### 2. Interpolation rules used for warped outputs")
    st.dataframe(
        pd.DataFrame([
            {"Data type": "CT / MR / PET", "Interpolation": "Linear", "Reason": "Continuous image intensities"},
            {"Data type": "Dose", "Interpolation": "Linear", "Reason": "Dose is continuous and should not be nearest-neighbour sampled"},
            {"Data type": "Masks / segmentations", "Interpolation": "Nearest-neighbour", "Reason": "Preserves discrete label values even when the transform is B-spline"},
        ]),
        use_container_width=True,
        hide_index=True,
    )
    if method == "Rigid → Affine → B-spline":
        st.warning(
            "B-spline is an advanced deformable registration step. Use it only after checking that rigid/affine alignment is reasonable, "
            "and review overlays carefully for unrealistic local warping."
        )
        st.info(
            "When you run this option, the status box will show: Stage 1 rigid registration → Stage 2 affine registration → "
            "Stage 3 B-spline deformable registration. If B-spline fails for a patient, the app keeps the affine transform and records the B-spline error instead of stopping the whole batch."
        )
        st.info(
            "When you run this option, the status box will show: Stage 1 rigid registration → Stage 2 affine registration → "
            "Stage 3 B-spline deformable registration. If B-spline fails for a patient, the app keeps the affine transform and records the B-spline error instead of stopping the whole batch."
        )

    st.markdown("### 3. Warp output manifest")
    filtered = warp_candidates[warp_candidates["Warp data type"].isin(selected_types)].copy() if selected_types else pd.DataFrame()
    if filtered.empty:
        st.warning("Select at least one output type to include.")
    else:
        preview_cols = [
            "Patient ID", "Warp data type", "Moving image", "Reference image", "Registration method",
            "Interpolation", "Transform file", "Resolved registered output file", "Status", "Registration details", "Error message"
        ]
        display_cols = [c for c in preview_cols if c in filtered.columns]
        display_cols += [c for c in filtered.columns if c not in display_cols]
        st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Patients", filtered["Patient ID"].nunique() if not filtered.empty and "Patient ID" in filtered.columns else 0)
    c2.metric("Warped outputs", int(filtered.shape[0]) if not filtered.empty else 0)
    c3.metric("Data types", len(selected_types))

    if st.button(
        "Save warp manifest to project",
        type="primary",
        use_container_width=True,
        disabled=filtered.empty,
        key="vbv_save_warp_manifest_button",
    ):
        manifest_df, manifest_path = vbv_save_warp_manifest(warp_candidates, selected_types, reference_setup)
        st.success(f"Warp manifest saved: {manifest_path}")
        saved_manifest = manifest_df

    if saved_manifest is not None and not saved_manifest.empty:
        st.markdown("### 4. Saved warp manifest")
        st.caption(st.session_state.get("vbv_warp_manifest_csv", str(vbv_warp_to_ccs_project_folder() / "warp_to_reference_manifest.csv")))
        st.dataframe(saved_manifest.head(100), use_container_width=True, hide_index=True)
        st.markdown("### 5. Overlay QC viewer")
        vbv_render_registered_output_viewer(saved_manifest.rename(columns={"Resolved registered output file": "Registered output file"}) if "Registered output file" not in saved_manifest.columns else saved_manifest)

    st.divider()
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back to Batch registration", use_container_width=True, key="vbv_warp_back_batch"):
            go_to("voxel_batch_registration")
    with col_next:
        if st.button("Next: Dose normalisation →", use_container_width=True, key="vbv_warp_next_dose"):
            if saved_manifest is None or saved_manifest.empty:
                st.warning("Save the warp manifest first so the project records which files are in the CCS/reference space.")
            else:
                go_to("voxel_dose_normalisation")


# ============================================================
# VOXEL DOSE NORMALISATION PAGE
# ============================================================

# ============================================================
# VOXEL-BASED ANALYSIS MODULE - DOSE NORMALISATION
# ============================================================
elif st.session_state.page == "voxel_dose_normalisation":
    st.header("Voxel-based Analysis")
    st.subheader("Dose normalisation")

    st.write(
        "This step standardises registered dose outputs before voxel-wise analysis. "
        "It uses the registered dose files from batch registration and saves new dose-normalised outputs inside the active project folder."
    )

    if not SIMPLEITK_AVAILABLE:
        st.error("SimpleITK is required for dose normalisation. Install it with: py -m pip install SimpleITK")
        st.stop()

    dose_folder = vbv_dose_normalisation_project_folder()
    st.markdown("**Dose normalisation output folder**")
    st.code(str(dose_folder.resolve()), language=None)

    registered_dose_df = vbv_load_registered_dose_outputs()
    saved_summary_df = st.session_state.get("vbv_dose_normalisation_summary", pd.DataFrame())
    if saved_summary_df is None or saved_summary_df.empty:
        saved_summary_df = vbv_existing_dose_normalisation_summary()
        if saved_summary_df is not None and not saved_summary_df.empty:
            st.session_state.vbv_dose_normalisation_summary = saved_summary_df

    st.markdown("### 1. Registered dose files")
    if registered_dose_df is None or registered_dose_df.empty:
        st.warning("No registered dose files were found. Complete Batch registration first and include dose files if available.")
        if st.button("← Back to Batch registration", use_container_width=True, key="vbv_dose_back_to_batch_no_dose"):
            go_to("voxel_batch_registration")
        st.stop()

    st.metric("Registered dose files found", int(registered_dose_df.shape[0]))
    dose_preview_cols = [c for c in ["Patient ID", "Dose file name", "Registered dose file", "Status", "Registration details"] if c in registered_dose_df.columns]
    st.dataframe(registered_dose_df[dose_preview_cols], use_container_width=True, hide_index=True)

    st.markdown("### 2. Select dose scaling")
    scaling_option = st.radio(
        "Dose scaling / unit conversion",
        options=[
            "Keep values as stored",
            "Convert cGy to Gy (multiply by 0.01)",
            "Custom scaling factor",
        ],
        index=0,
        key="vbv_dose_scaling_option",
    )
    if scaling_option == "Convert cGy to Gy (multiply by 0.01)":
        scaling_factor = 0.01
    elif scaling_option == "Custom scaling factor":
        scaling_factor = st.number_input(
            "Custom factor applied to dose values",
            min_value=0.0,
            value=1.0,
            step=0.01,
            format="%.6f",
            key="vbv_dose_custom_scaling_factor",
        )
    else:
        scaling_factor = 1.0
    st.caption(f"Selected scaling factor: {scaling_factor}")

    st.markdown("### 3. Save dose-normalised outputs")
    if st.button("Run and save dose normalisation", type="primary", use_container_width=True, key="vbv_run_dose_normalisation"):
        progress_bar = st.progress(0.0)
        progress_text = st.empty()
        with st.spinner("Saving dose-normalised outputs..."):
            saved_summary_df = vbv_run_dose_normalisation(
                registered_dose_df,
                scaling_factor=float(scaling_factor),
                scaling_label=scaling_option,
                progress_bar=progress_bar,
                progress_text=progress_text,
            )
        st.success("Dose-normalised outputs saved.")

    saved_summary_df = st.session_state.get("vbv_dose_normalisation_summary", saved_summary_df)
    if saved_summary_df is not None and not saved_summary_df.empty:
        st.markdown("### 4. Saved dose-normalisation summary")
        processed = int((saved_summary_df["Status"].astype(str) == "Processed").sum()) if "Status" in saved_summary_df.columns else 0
        failed = int((saved_summary_df["Status"].astype(str) == "Failed").sum()) if "Status" in saved_summary_df.columns else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Dose outputs", int(saved_summary_df.shape[0]))
        c2.metric("Processed", processed)
        c3.metric("Failed", failed)
        st.caption(f"Saved summary CSV: {dose_folder / 'dose_normalisation_summary.csv'}")
        st.dataframe(saved_summary_df, use_container_width=True, hide_index=True)
    else:
        st.info("No dose-normalised outputs have been saved yet.")

    st.divider()
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back to Batch registration", use_container_width=True, key="vbv_dose_back_to_batch"):
            go_to("voxel_batch_registration")
    with col_next:
        if st.button("Next: VBA-ready dataset / Final QC →", use_container_width=True, key="vbv_dose_next_vba_ready"):
            go_to("voxel_vba_ready_dataset")



# ============================================================
# VOXEL VBA-READY DATASET / FINAL QC PAGE
# ============================================================

# ============================================================
# VOXEL-BASED ANALYSIS MODULE - VBA READY DATASET / FINAL QC
# ============================================================
elif st.session_state.page == "voxel_vba_ready_dataset":
    st.header("Voxel-based Analysis")
    st.subheader("VBA-ready dataset / Final QC")

    st.write(
        "This step creates the final patient-level manifest before voxel-wise statistics. "
        "It links each patient to clinical data, reference-space anatomy, dose-normalised files and masks."
    )

    out_folder = vbv_vba_ready_project_folder()
    st.markdown("**VBA-ready dataset output folder**")
    st.code(str(out_folder.resolve()), language=None)

    st.markdown("### 1. Build final QC manifest")
    st.caption(
        "The manifest is saved automatically in the project folder. It does not copy the image files; "
        "it records the final file paths and readiness checks."
    )

    existing_manifest = st.session_state.get("vbv_vba_ready_manifest", pd.DataFrame())
    if existing_manifest is None or existing_manifest.empty:
        existing_manifest = vbv_existing_vba_ready_manifest()
        if existing_manifest is not None and not existing_manifest.empty:
            st.session_state.vbv_vba_ready_manifest = existing_manifest

    if st.button("Create / refresh VBA-ready manifest", type="primary", use_container_width=True, key="vbv_create_vba_ready_manifest"):
        with st.spinner("Checking clinical data, reference-space images, dose-normalised files and masks..."):
            existing_manifest, manifest_path = vbv_build_vba_ready_manifest()
        st.success(f"VBA-ready manifest saved: {manifest_path}")

    manifest_df = st.session_state.get("vbv_vba_ready_manifest", existing_manifest)

    if manifest_df is None or manifest_df.empty:
        st.info("No VBA-ready manifest has been created yet. Click the button above to build it.")
    else:
        st.markdown("### 2. Readiness summary")
        ready_count = int((manifest_df["Ready for VBA"].astype(str) == "Yes").sum()) if "Ready for VBA" in manifest_df.columns else 0
        not_ready_count = int((manifest_df["Ready for VBA"].astype(str) == "No").sum()) if "Ready for VBA" in manifest_df.columns else 0
        total_count = int(manifest_df.shape[0])
        c1, c2, c3 = st.columns(3)
        c1.metric("Patients checked", total_count)
        c2.metric("Ready", ready_count)
        c3.metric("Not ready", not_ready_count)

        manifest_path = vbv_vba_ready_project_folder() / "vba_ready_dataset_manifest.csv"
        st.caption(f"Saved manifest CSV: {manifest_path}")

        preferred_cols = [
            "Patient ID", "Clinical data present", "Reference-space CT present", "Reference-space MR present",
            "Dose in Gy present", "Mask present", "Dose geometry matches anatomy", "Registration QC status", "Visual QC approval", "Ready for VBA", "Issue",
        ]
        display_cols = [c for c in preferred_cols if c in manifest_df.columns]
        display_cols += [c for c in manifest_df.columns if c not in display_cols]
        st.dataframe(manifest_df[display_cols], use_container_width=True, hide_index=True)

        if "Ready for VBA" in manifest_df.columns and not_ready_count > 0:
            with st.expander("Show patients not ready for VBA"):
                st.dataframe(
                    manifest_df[manifest_df["Ready for VBA"].astype(str) != "Yes"][display_cols],
                    use_container_width=True,
                    hide_index=True,
                )

        if "Dose geometry matches anatomy" in manifest_df.columns:
            mismatch_df = manifest_df[manifest_df["Dose geometry matches anatomy"].astype(str) == "No"]
            if not mismatch_df.empty:
                st.warning("Some dose files do not have the same geometry as the selected anatomy/reference-space image. Review these before voxel-wise statistics.")
                with st.expander("Show geometry mismatches"):
                    geom_cols = [
                        "Patient ID", "Anatomy size", "Anatomy spacing mm", "Dose size", "Dose spacing mm",
                        "Anatomy file path", "Dose file path",
                    ]
                    geom_cols = [c for c in geom_cols if c in mismatch_df.columns]
                    st.dataframe(mismatch_df[geom_cols], use_container_width=True, hide_index=True)

    st.divider()
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back to Dose normalisation", use_container_width=True, key="vbv_ready_back_dose"):
            go_to("voxel_dose_normalisation")
    with col_next:
        if st.button("Next: Statistical analysis →", use_container_width=True, key="vbv_ready_next_statistics"):
            go_to("voxel_statistical_analysis")


# ============================================================
# VOXEL STATISTICAL ANALYSIS PAGE
# ============================================================

# ============================================================
# VOXEL-BASED ANALYSIS MODULE - STATISTICAL ANALYSIS
# ============================================================
elif st.session_state.page == "voxel_statistical_analysis":
    st.header("Voxel-based Analysis")
    st.subheader("Statistical analysis")

    st.write(
        "This final tab runs paper-aligned voxel-wise statistical analysis after the VBA-ready dataset has passed final QC. "
        "Select the VBA statistic, multiple-comparison correction, voxel filter, and any adjustment variables, then calculate."
    )

    out_folder = vbv_statistical_analysis_project_folder()
    st.markdown("**Statistical-analysis output folder**")
    st.code(str(out_folder.resolve()), language=None)

    ready_manifest = vbv_load_ready_manifest_for_stats()
    if ready_manifest.empty:
        st.warning("No ready-for-VBA dose maps were found. Create or refresh the VBA-ready dataset / Final QC manifest first.")
        if st.button("← Back to VBA-ready dataset / Final QC", use_container_width=True, key="stats_back_to_ready_empty"):
            go_to("voxel_vba_ready_dataset")
    else:
        analysis_df, clinical_df, patient_id_col = vbv_merge_ready_manifest_with_clinical(ready_manifest)

        st.markdown("### 1. Cohort")
        c1, c2 = st.columns(2)
        c1.metric("Ready patients with dose maps", int(analysis_df.shape[0]))
        c2.metric("Clinical columns available", int(clinical_df.shape[1]) if clinical_df is not None and not clinical_df.empty else 0)

        patient_options = analysis_df["Patient ID"].astype(str).tolist() if "Patient ID" in analysis_df.columns else []
        selected_patients = st.multiselect(
            "Patients to include",
            options=patient_options,
            default=patient_options,
            key="stats_selected_patients",
        )
        if selected_patients:
            analysis_df = analysis_df[analysis_df["Patient ID"].astype(str).isin(selected_patients)].copy()

        st.markdown("### 2. VBA statistic")
        vba_statistic = st.selectbox(
            "VBA statistic",
            options=[
                "Spearman correlation: dose vs continuous outcome/slope",
                "Pearson correlation: dose vs continuous outcome",
                "Welch t-test: dose difference between two groups",
                "Mean and SD dose maps only",
            ],
            index=0,
            key="stats_vba_statistic",
            help="The uploaded appendix used voxel-wise Spearman correlation between dose and neurocognitive slope, followed by permutation-style thresholding.",
        )

        available_cols = list(analysis_df.columns)
        excluded_cols = {
            "Patient ID", "Dose file path", "Anatomy file path", "CT file path", "MR file path", "Mask file path",
            "Issue", "Ready for VBA", "Reference image", "Reference patient",
        }
        numeric_candidates = []
        categorical_candidates = []
        for col in available_cols:
            if col in excluded_cols:
                continue
            numeric = pd.to_numeric(analysis_df[col], errors="coerce")
            if numeric.notna().sum() >= 3:
                numeric_candidates.append(col)
            unique_vals = analysis_df[col].dropna().astype(str).str.strip().unique().tolist()
            unique_vals = [v for v in unique_vals if v != ""]
            if 2 <= len(unique_vals) <= 10:
                categorical_candidates.append(col)

        outcome_column = ""
        group_column = ""
        group_a = ""
        group_b = ""
        adjustment_columns = []

        if vba_statistic in ["Spearman correlation: dose vs continuous outcome/slope", "Pearson correlation: dose vs continuous outcome"]:
            if numeric_candidates:
                outcome_column = st.selectbox(
                    "Continuous outcome / slope variable",
                    options=numeric_candidates,
                    key="stats_outcome_column",
                    help="For the paper-aligned neurocognitive workflow, select the neurocognitive slope or continuous outcome variable.",
                )
            else:
                st.warning("No numeric outcome/slope columns were found in the ready dataset.")
        elif vba_statistic == "Welch t-test: dose difference between two groups":
            if categorical_candidates:
                group_column = st.selectbox("Grouping variable", options=categorical_candidates, key="stats_group_column")
                group_values = analysis_df[group_column].dropna().astype(str).str.strip().unique().tolist()
                group_values = sorted([v for v in group_values if v != ""])
                if len(group_values) >= 2:
                    g1, g2 = st.columns(2)
                    with g1:
                        group_a = st.selectbox("Group A", options=group_values, key="stats_group_a")
                    with g2:
                        group_b = st.selectbox("Group B", options=group_values, index=1 if len(group_values) > 1 else 0, key="stats_group_b")
            else:
                st.warning("No suitable two-group/categorical columns were found.")

        st.markdown("### 3. Correction for multiple comparison")
        correction_method = st.selectbox(
            "Correction for multiple comparison",
            options=[
                "Permutation max-statistic threshold",
                "FDR Benjamini-Hochberg",
                "Bonferroni",
                "Uncorrected p-value threshold",
            ],
            index=0,
            key="stats_correction_method",
        )
        c_alpha, c_perm = st.columns(2)
        with c_alpha:
            alpha = st.number_input("Significance level / alpha", min_value=0.001, max_value=0.20, value=0.01, step=0.001, format="%.3f", key="stats_alpha")
        with c_perm:
            n_permutations = st.number_input("Permutations requested", min_value=10, max_value=5000, value=1000, step=10, key="stats_n_permutations")
        if correction_method == "Permutation max-statistic threshold":
            st.info("This option records the paper-aligned permutation max-statistic settings. The current run saves the maps and display mask; a full long-running permutation engine can be added as the next performance-focused step.")

        st.markdown("### 4. Filter")
        filter_method = st.selectbox(
            "Filter / voxel inclusion rule",
            options=[
                "No filter",
                "Dose SD filter",
                "Mean dose filter",
                "Dose SD + Mean dose filter",
            ],
            index=1,
            key="stats_filter_method",
            help="The paper highlights that voxels with very low dose variation have limited power, so a dose-variation filter is useful for interpretation.",
        )
        f1, f2 = st.columns(2)
        with f1:
            sd_threshold = st.number_input("Minimum dose SD threshold (Gy)", min_value=0.0, value=1.0, step=0.5, key="stats_sd_threshold")
        with f2:
            mean_dose_threshold = st.number_input("Minimum mean dose threshold (Gy)", min_value=0.0, value=0.0, step=0.5, key="stats_mean_threshold")

        st.markdown("### 5. Adjustment variables")
        if vba_statistic in ["Spearman correlation: dose vs continuous outcome/slope", "Pearson correlation: dose vs continuous outcome"]:
            adjustment_options = [c for c in numeric_candidates if c != outcome_column]
            adjustment_columns = st.multiselect(
                "Adjustment variables / covariates",
                options=adjustment_options,
                default=[],
                key="stats_adjustment_columns",
                help="For continuous dose–outcome analyses, the selected outcome is residualised against these numeric covariates before voxel-wise correlation.",
            )
        else:
            st.caption("Adjustment variables are only applied to continuous dose–outcome correlation in this version.")

        st.markdown("### 6. Calculate")
        st.caption("Outputs are saved automatically inside the active project folder.")
        if st.button("Calculate statistical analysis", type="primary", use_container_width=True, key="stats_calculate"):
            try:
                if analysis_df.empty:
                    raise ValueError("No patients selected for statistical analysis.")
                if vba_statistic in ["Spearman correlation: dose vs continuous outcome/slope", "Pearson correlation: dose vs continuous outcome"] and not outcome_column:
                    raise ValueError("Select a continuous outcome/slope variable.")
                if vba_statistic == "Welch t-test: dose difference between two groups":
                    if not group_column or str(group_a) == str(group_b):
                        raise ValueError("Select a grouping column and two different groups.")

                progress_bar = st.progress(0)
                status_box = st.empty()
                with st.spinner("Running statistical analysis..."):
                    outputs_df, summary_df, error_df, summary_path = vbv_run_statistical_analysis(
                        analysis_df=analysis_df,
                        vba_statistic=vba_statistic,
                        outcome_column=outcome_column,
                        group_column=group_column,
                        group_a=group_a,
                        group_b=group_b,
                        correction_method=correction_method,
                        alpha=alpha,
                        n_permutations=int(n_permutations),
                        filter_method=filter_method,
                        sd_threshold=float(sd_threshold),
                        mean_dose_threshold=float(mean_dose_threshold),
                        adjustment_columns=adjustment_columns,
                        progress_bar=progress_bar,
                        status_box=status_box,
                    )
                st.session_state.vbv_statistical_analysis_outputs = outputs_df
                st.session_state.vbv_statistical_analysis_summary = summary_df
                st.session_state.vbv_statistical_analysis_errors = error_df
                st.success(f"Statistical analysis saved. Run summary: {summary_path}")
            except Exception as error:
                st.error(f"Statistical analysis failed: {error}")

        outputs_df = st.session_state.get("vbv_statistical_analysis_outputs", pd.DataFrame())
        if outputs_df is not None and not outputs_df.empty:
            st.markdown("### Latest outputs")
            st.dataframe(outputs_df, use_container_width=True, hide_index=True)

        summary_df = st.session_state.get("vbv_statistical_analysis_summary", pd.DataFrame())
        if summary_df is None or summary_df.empty:
            summary_df = vbv_existing_statistical_analysis_summary()
        if summary_df is not None and not summary_df.empty:
            st.markdown("### Previous statistical-analysis runs")
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            st.caption(f"Saved run summary: {vbv_statistical_analysis_project_folder() / 'statistical_analysis_run_summary.csv'}")

        st.divider()
        col_back, col_notes, col_home = st.columns(3)
        with col_back:
            if st.button("← Back to VBA-ready dataset / Final QC", use_container_width=True, key="stats_back_to_ready"):
                go_to("voxel_vba_ready_dataset")
        with col_notes:
            if st.button("Back to voxel workflow home", use_container_width=True, key="stats_back_voxel_home"):
                go_to("voxel_analysis_home")
        with col_home:
            if st.button("Back to voxel workflow home", use_container_width=True, key="stats_back_home"):
                go_to("voxel_analysis_home")



# ============================================================
# VOXEL-BASED ANALYSIS MODULE - REFERENCE IMAGE / CCS
# ============================================================
elif st.session_state.page == "voxel_reference_ccs":
    st.header("Voxel-based Analysis")
    st.subheader("Reference image / common coordinate system")

    if st.session_state.get("vbv_batch_registration_skipped", False):
        st.info("Existing registered outputs are being used. Batch registration was skipped.")

    st.write(
        "Select the fixed reference anatomy / common coordinate system before batch registration. "
        "All other patients' CT/MR/dose/masks will be warped into this reference space."
    )

    normalised_reference_df = vbv_load_normalised_outputs_for_batch() if SIMPLEITK_AVAILABLE else pd.DataFrame()
    if normalised_reference_df is not None and not normalised_reference_df.empty:
        ref_df = normalised_reference_df.copy()
        if "Patient ID" in ref_df.columns:
            ref_df["Inferred patient ID"] = ref_df["Patient ID"].astype(str)
        if "Output file" in ref_df.columns and "Full path" not in ref_df.columns:
            ref_df["Full path"] = ref_df["Output file"].astype(str)
        if "Output file" in ref_df.columns and "Filename" not in ref_df.columns:
            ref_df["Filename"] = ref_df["Output file"].astype(str).apply(lambda x: Path(x).name)
        if "Relative path" not in ref_df.columns:
            ref_df["Relative path"] = ref_df.get("Filename", pd.Series(dtype=str)).astype(str)
        st.caption("Reference candidates are taken from saved normalised outputs.")
    else:
        file_df = st.session_state.get("voxel_loaded_image_files_df", pd.DataFrame())

        if file_df is None or file_df.empty:
            st.warning("No image files have been loaded yet.")
            if st.button("← Back to Load images / filename QC", use_container_width=True):
                go_to("voxel_load_images")
            st.stop()

        try:
            ref_df = add_patient_grouping_columns(file_df.copy())
        except Exception:
            ref_df = file_df.copy()
        st.caption("Reference candidates are taken from the loaded image index because no normalised outputs were found yet.")

    if "Inferred patient ID" not in ref_df.columns:
        if "Patient ID" in ref_df.columns:
            ref_df["Inferred patient ID"] = ref_df["Patient ID"].astype(str)
        elif "Patient" in ref_df.columns:
            ref_df["Inferred patient ID"] = ref_df["Patient"].astype(str)
        else:
            ref_df["Inferred patient ID"] = "Unknown"

    patient_ids = sorted([
        str(x) for x in ref_df["Inferred patient ID"].dropna().unique().tolist()
        if str(x).strip() != ""
    ])

    st.markdown("### CCS reference selection")

    ccs_strategy = st.radio(
        "Choose how the common coordinate system/reference will be selected",
        options=[
            "1. Select a reference patient",
            "2. Select a reference organ",
            "3. Select a reference atlas",
            "4. Population average",
            "5. Group-wise registration / study-specific template",
        ],
        index=0,
        key="voxel_reference_ccs_strategy",
        help="This defines the coordinate space where voxel-wise statistics will be run."
    )

    selected_reference_patient = ""
    selected_reference_image = ""
    selected_reference_organs = []
    reference_selection_table = pd.DataFrame()

    def _candidate_reference_images(patient_files):
        name_lower = patient_files.get("Filename", pd.Series(dtype=str)).astype(str).str.lower()
        path_lower = patient_files.get("Relative path", pd.Series(dtype=str)).astype(str).str.lower()

        is_candidate = (
            name_lower.str.contains("ct", regex=False, na=False)
            | name_lower.str.contains("mr", regex=False, na=False)
            | name_lower.str.contains("mri", regex=False, na=False)
            | path_lower.str.contains("ct", regex=False, na=False)
            | path_lower.str.contains("mr", regex=False, na=False)
            | path_lower.str.contains("mri", regex=False, na=False)
        )

        is_not_candidate = (
            name_lower.str.contains("dose", regex=False, na=False)
            | name_lower.str.contains("mask", regex=False, na=False)
            | name_lower.str.contains("seg", regex=False, na=False)
            | name_lower.str.contains("rtstruct", regex=False, na=False)
            | name_lower.str.contains("structure", regex=False, na=False)
        )

        candidate_df = patient_files[is_candidate & (~is_not_candidate)].copy()

        if candidate_df.empty:
            candidate_df = patient_files.copy()

        return candidate_df

    def _detect_mask_organ_name(path_text):
        text = str(path_text)
        name = Path(text).name
        lower_name = name.lower()

        for suffix in [".nii.gz", ".nii", ".dcm", ".dicom", ".nrrd", ".mha", ".mhd"]:
            if lower_name.endswith(suffix):
                name = name[: -len(suffix)]
                break

        tokens_to_remove = [
            "mask", "seg", "roi", "structure", "rtstruct", "contour",
            "ptv", "ctv", "gtv", "patient", "pt", "nii", "gz"
        ]

        parts = re.split(r"[_\-\s\.]+", name)
        cleaned_parts = []
        for part in parts:
            part_clean = str(part).strip()
            if part_clean == "":
                continue
            if part_clean.lower() in tokens_to_remove:
                continue
            if part_clean.lower().startswith("pt") and any(ch.isdigit() for ch in part_clean):
                continue
            cleaned_parts.append(part_clean)

        if len(cleaned_parts) == 0:
            return name

        return "_".join(cleaned_parts)

    def _build_organ_reference_table(mask_df, selected_organs):
        rows = []

        if mask_df.empty or len(selected_organs) == 0 or not SIMPLEITK_AVAILABLE:
            return pd.DataFrame()

        for _, row in mask_df.iterrows():
            file_label = str(row.get("Relative path", row.get("Filename", "")))
            organ_name = _detect_mask_organ_name(file_label)

            if organ_name not in selected_organs:
                continue

            patient_id = str(row.get("Inferred patient ID", ""))
            full_path = row.get("Full path", "")

            try:
                img = sitk.ReadImage(str(full_path))
                arr = sitk.GetArrayFromImage(img)
                spacing = img.GetSpacing()
                voxel_volume = float(spacing[0] * spacing[1] * spacing[2])
                voxel_count = int(np.count_nonzero(arr > 0))
                volume_mm3 = voxel_count * voxel_volume
                volume_cc = volume_mm3 / 1000.0

                rows.append({
                    "Patient ID": patient_id,
                    "Organ": organ_name,
                    "Mask file": file_label,
                    "Voxel count": voxel_count,
                    "Voxel size / spacing": " × ".join([f"{float(x):.3f}" for x in spacing]),
                    "Volume cc": volume_cc,
                    "Status": "Measured",
                })
            except Exception as error:
                rows.append({
                    "Patient ID": patient_id,
                    "Organ": organ_name,
                    "Mask file": file_label,
                    "Voxel count": np.nan,
                    "Voxel size / spacing": "",
                    "Volume cc": np.nan,
                    "Status": f"Could not measure: {error}",
                })

        table = pd.DataFrame(rows)

        if table.empty or "Volume cc" not in table.columns:
            return table

        measured = table.dropna(subset=["Volume cc"]).copy()

        if measured.empty:
            return table

        measured_count = int(measured.shape[0])
        median_volume = float(measured["Volume cc"].median())
        q1 = float(measured["Volume cc"].quantile(0.25))
        q3 = float(measured["Volume cc"].quantile(0.75))
        iqr = q3 - q1

        table["Measured masks used"] = measured_count
        table["Median volume cc"] = median_volume
        table["Q1 volume cc"] = q1
        table["Q3 volume cc"] = q3
        table["IQR volume cc"] = iqr
        table["Distance from median cc"] = (table["Volume cc"] - median_volume).abs()
        table["Reference candidate"] = "No"

        # Simple visual flag for the table:
        # outside the interquartile range, not statistical 1.5 × IQR outliers.
        table["Outside IQR"] = table["Volume cc"].apply(
            lambda x: "Yes" if pd.notna(x) and (x < q1 or x > q3) else "No"
        )

        non_outlier = table[
            (table["Status"].astype(str) == "Measured")
            & table["Distance from median cc"].notna()
        ].copy()

        if non_outlier.empty:
            non_outlier = table[
                (table["Status"].astype(str) == "Measured")
                & table["Distance from median cc"].notna()
            ].copy()

        if not non_outlier.empty:
            best_idx = non_outlier["Distance from median cc"].idxmin()
            table.loc[best_idx, "Reference candidate"] = "Yes"

        return table


    if ccs_strategy == "1. Select a reference patient":
        st.info("Select a patient from the loaded cohort to act as the reference anatomy.")

        if len(patient_ids) == 0:
            st.warning("No patient IDs were detected.")
        else:
            selected_reference_patient = st.selectbox(
                "Select reference patient",
                options=patient_ids,
                key="voxel_common_reference_patient"
            )

            patient_files = ref_df[ref_df["Inferred patient ID"].astype(str) == str(selected_reference_patient)].copy()
            reference_candidates = _candidate_reference_images(patient_files)

            label_col = "Relative path" if "Relative path" in reference_candidates.columns else "Filename"
            reference_options = reference_candidates[label_col].astype(str).tolist()

            if reference_options:
                selected_reference_image = st.selectbox(
                    "Select CT/MR reference image",
                    options=reference_options,
                    key="voxel_common_reference_image"
                )

            st.dataframe(reference_candidates, use_container_width=True)

    elif ccs_strategy == "2. Select a reference organ":
        st.info(
            "Select one or more organs from the masks. The app estimates organ volumes, calculates the median volume, "
            "flags significant outliers, and suggests the patient whose selected organ volume is closest to the median."
        )

        role_series = ref_df.get("File role", pd.Series(dtype=str)).astype(str).str.lower()
        vba_role_series = ref_df.get("VBA role", pd.Series(dtype=str)).astype(str).str.lower()
        filename_series = ref_df.get("Filename", pd.Series(dtype=str)).astype(str).str.lower()
        path_series = ref_df.get("Relative path", pd.Series(dtype=str)).astype(str).str.lower()

        mask_df = ref_df[
            role_series.str.contains("mask|structure|rtstruct|seg", regex=True, na=False)
            | vba_role_series.str.contains("mask|rtstruct", regex=True, na=False)
            | filename_series.str.contains("mask|structure|rtstruct|seg|roi|contour", regex=True, na=False)
            | path_series.str.contains("mask|structure|rtstruct|seg|roi|contour", regex=True, na=False)
        ].copy()

        if mask_df.empty:
            st.warning("No mask/structure files were detected.")
        else:
            mask_df["Detected organ"] = mask_df.apply(
                lambda row: _detect_mask_organ_name(row.get("Relative path", row.get("Filename", ""))),
                axis=1
            )

            organ_options = sorted([
                str(x) for x in mask_df["Detected organ"].dropna().unique().tolist()
                if str(x).strip() != ""
            ])

            selected_reference_organs = st.multiselect(
                "Select reference organ(s)",
                options=organ_options,
                default=organ_options[:1],
                key="voxel_reference_organs"
            )

            if len(selected_reference_organs) == 0:
                st.info("Select at least one organ to calculate reference anatomy.")
            else:
                reference_selection_table = _build_organ_reference_table(mask_df, selected_reference_organs)

                if reference_selection_table.empty:
                    st.warning("No measurable mask volumes were available for the selected organ(s).")
                else:
                    st.markdown("### Organ-volume reference selection")

                    measured_table = reference_selection_table[
                        reference_selection_table["Status"].astype(str) == "Measured"
                    ].copy()

                    candidate_rows = reference_selection_table[
                        reference_selection_table.get("Reference candidate", pd.Series(dtype=str)) == "Yes"
                    ]

                    median_volume = float(measured_table["Volume cc"].median()) if not measured_table.empty else np.nan
                    q1_volume = float(measured_table["Volume cc"].quantile(0.25)) if not measured_table.empty else np.nan
                    q3_volume = float(measured_table["Volume cc"].quantile(0.75)) if not measured_table.empty else np.nan
                    iqr_volume = q3_volume - q1_volume if pd.notna(q1_volume) and pd.notna(q3_volume) else np.nan
                    outside_iqr_count = int((reference_selection_table.get("Outside IQR", pd.Series(dtype=str)) == "Yes").sum())

                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Measured masks", int(measured_table.shape[0]))
                    k2.metric("Median volume", "Not available" if pd.isna(median_volume) else f"{median_volume:.2f} cc")
                    k3.metric("IQR", "Not available" if pd.isna(iqr_volume) else f"{iqr_volume:.2f} cc")
                    k4.metric("Outside IQR", outside_iqr_count)

                    if pd.notna(q1_volume) and pd.notna(q3_volume):
                        st.caption(f"IQR range: {q1_volume:.2f} to {q3_volume:.2f} cc")

                    if not candidate_rows.empty:
                        selected_reference_patient = str(candidate_rows.iloc[0].get("Patient ID", ""))
                        selected_reference_image = "Reference selected from median organ volume"
                        st.success(f"Suggested reference patient: {selected_reference_patient} — closest to the median volume.")

                    display_table = reference_selection_table.copy()

                    display_table["Imaging"] = display_table["Mask file"].astype(str).apply(
                        lambda x: "MR" if any(term in x.lower() for term in ["mr", "mri", "t1", "t2", "flair"]) else "CT"
                    )

                    simple_cols = ["Patient ID", "Mask file", "Imaging", "Volume cc", "Reference candidate", "Outside IQR"]
                    display_cols = [col for col in simple_cols if col in display_table.columns]
                    display_table = display_table[display_cols].copy()

                    if "Volume cc" in display_table.columns:
                        display_table["Volume cc"] = display_table["Volume cc"].apply(
                            lambda x: "" if pd.isna(x) else round(float(x), 2)
                        )

                    def _highlight_reference_table(row):
                        styles = [""] * len(row)

                        if row.get("Outside IQR", "No") == "Yes":
                            styles = ["background-color: #FEE2E2"] * len(row)

                        if row.get("Reference candidate", "No") == "Yes":
                            styles = ["background-color: #FEF3C7"] * len(row)

                        return styles

                    styled_table = display_table.style.apply(_highlight_reference_table, axis=1)

                    st.dataframe(styled_table, use_container_width=True, hide_index=True)

                    st.caption(
                        "Yellow = closest to the median organ volume. "
                        "Red = mask volume outside the IQR range."
                    )

                    selected_reference_patient = st.selectbox(
                        "Confirm or change suggested reference patient",
                        options=patient_ids,
                        index=patient_ids.index(selected_reference_patient) if selected_reference_patient in patient_ids else 0,
                        key="voxel_organ_based_reference_patient"
                    )

    elif ccs_strategy == "3. Select a reference atlas":
        st.info("Upload a reference atlas/template to use as the common coordinate system.")

        uploaded_template = st.file_uploader(
            "Upload reference atlas / template NIfTI",
            type=["nii", "nii.gz"],
            key="voxel_external_template_upload"
        )

        selected_reference_patient = "Reference atlas"
        selected_reference_image = uploaded_template.name if uploaded_template is not None else ""

        if uploaded_template is not None:
            st.success(f"Reference atlas uploaded: {uploaded_template.name}")

    elif ccs_strategy == "4. Population average":
        st.info("Population average reference generation is upcoming.")
        selected_reference_patient = "Population average"
        selected_reference_image = "Upcoming"

    else:
        st.info("Group-wise registration / study-specific template generation is planned as an advanced option.")
        selected_reference_patient = "Group-wise template"
        selected_reference_image = "To be generated by group-wise registration"

    st.markdown("### Reference setup only")
    st.info(
        "This step only defines the fixed common coordinate space. "
        "Registration method, transform targets and interpolation rules are selected in the Batch registration tab."
    )

    if st.button("Save reference image / CCS setup", type="primary", use_container_width=True):
        st.session_state.voxel_reference_ccs_setup = {
            "ccs_strategy": ccs_strategy,
            "reference_patient": selected_reference_patient,
            "reference_image": selected_reference_image,
            "reference_organs": selected_reference_organs,
            "organ_reference_table": reference_selection_table,
            "note": "Reference/CCS selected before batch registration. This page only saves the fixed common coordinate space.",
        }
        st.success("Reference image / CCS setup saved.")

    if st.session_state.get("voxel_reference_ccs_setup"):
        with st.expander("Current reference / CCS setup", expanded=True):
            setup_to_show = dict(st.session_state.voxel_reference_ccs_setup)
            if isinstance(setup_to_show.get("organ_reference_table"), pd.DataFrame):
                setup_to_show["organ_reference_table"] = "Stored as table"
            st.json(setup_to_show)

            organ_table = st.session_state.voxel_reference_ccs_setup.get("organ_reference_table", pd.DataFrame())
            if isinstance(organ_table, pd.DataFrame) and not organ_table.empty:
                st.markdown("#### Stored organ-volume reference table")
                st.dataframe(organ_table, use_container_width=True, hide_index=True)

    st.divider()

    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back to Normalisation", use_container_width=True):
            go_to("voxel_registration_alignment")

    with col_next:
        if st.button("Next: Batch registration →", use_container_width=True):
            if not st.session_state.get("voxel_reference_ccs_setup"):
                st.warning("Save the reference image / CCS setup before continuing.")
            else:
                go_to("voxel_batch_registration")



# ============================================================
# ESTABLISHED MODEL PAGE
# ============================================================


# ============================================================
# ESTABLISHED MODEL LANDING PAGE
# ============================================================

# ============================================================
# ESTABLISHED MODEL MODULE - LANDING PAGE
# ============================================================
elif st.session_state.page == "established_model":
    st.header("Established Model Library")

    st.markdown("### Established model workflow")
    cards = [
        ("🔎 Search", "established_model_search", "Find models by name, treatment site, outcome, method and result ranking."),
        ("📊 Compare", "established_model_compare", "Compare available models side-by-side using method, predictors and performance metrics."),
        ("🧮 Risk calculator", "established_model_risk_calculator", "Select an established model and calculate risk for an individual patient profile."),
        ("✅ Validate model using my data", "established_model_validate", "Upload a local validation dataset, map variables and validate a selected model without sharing patient-level data."),
        ("🌍 External validation results", "established_model_external_validation", "Record and review aggregate external validation results from other centres."),
        ("📎 Documentation", "established_model_documentation", "Store publication links, TRIPOD/PROBAST notes, implementation notes and supporting evidence."),
        ("🤝 Collaborate", "established_model_collaborate", "Record collaboration requests and centre-level contact notes for model sharing or validation."),
    ]
    for i in range(0, len(cards), 4):
        cols = st.columns(4)
        for col, (title, page, description) in zip(cols, cards[i:i + 4]):
            with col:
                st.markdown(
                    f"""
                    <div style='border:1px solid #444; border-radius:14px; padding:16px; min-height:190px;'>
                        <h3 style='margin-top:0;'>{title}</h3>
                        <p>{description}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(title.replace('🔎 ', '').replace('📊 ', '').replace('🧮 ', '').replace('✅ ', '').replace('🌍 ', '').replace('📎 ', '').replace('⭐ ', '').replace('🤝 ', ''), use_container_width=True, key=f"est_landing_{page}"):
                    go_to(page)

    st.divider()
    if st.button("Save Established Model library now", use_container_width=True, key="est_landing_save_everything"):
        save_established_model_everything_persistent()
        st.success("Established Model library, selections and validation state saved to app storage.")
    if st.button("← Back to Home", use_container_width=True, key="est_landing_back_home"):
        go_to("home")

# ============================================================
# ESTABLISHED MODEL MODULE - SEARCH
# ============================================================
elif st.session_state.page == "established_model_search":
    st.header("Established Model Library")
    st.subheader("Search")

    st.write(
        "Search the established model library and select one or more models. "
        "Selected models can then be opened together in the Compare tab."
    )

    exported = st.session_state.get("established_calculators", {})

    if len(exported) == 0:
        st.info("No established models are available yet. Export a model from the clinical model workflow first.")
        if st.button("← Back to Established Model landing", use_container_width=True, key="est_search_back_empty"):
            go_to("established_model")
        st.stop()

    registry_rows = []
    for name, info in exported.items():
        metrics = info.get("metrics", {})
        auc = metrics.get("Validation AUC", metrics.get("AUC", np.nan))
        accuracy = metrics.get("Validation Accuracy", metrics.get("Accuracy", np.nan))
        brier = metrics.get("Validation Brier Score", metrics.get("Brier Score", np.nan))

        if auc is not None and not pd.isna(auc):
            result_value = float(auc)
            result_summary = f"Validation AUC {result_value:.3f}"
        elif accuracy is not None and not pd.isna(accuracy):
            result_value = float(accuracy)
            result_summary = f"Validation accuracy {result_value:.3f}"
        else:
            result_value = np.nan
            result_summary = "Not available"

        display_name = established_display_model_name(name, info)
        predictors = info.get("predictors", [])
        registry_rows.append({
            "Name": display_name,
            "Library key": name,
            "Site": info.get("model_site", "Brain"),
            "Predictive outcome": info.get("outcome_group", info.get("outcome_variable", "")),
            "Method": info.get("method", "Unknown"),
            "Predictors": established_format_predictor_list(predictors),
            "No. predictors": established_predictor_count(predictors),
            "Exported date": info.get("exported_date", str(info.get("exported_at", "")).split(" ")[0] if info.get("exported_at", "") else ""),
            "Results": result_summary,
            "Result value": result_value,
            "Brier score": brier,
        })

    registry_df = pd.DataFrame(registry_rows)

    st.markdown("### Search filters")

    filter_tab1, filter_tab2, filter_tab3, filter_tab4, filter_tab5 = st.tabs([
        "Model name",
        "Site",
        "Predictive outcome",
        "Method",
        "Results order",
    ])

    with filter_tab1:
        name_filter = st.text_input(
            "Search model name",
            value=st.session_state.get("est_search_name", ""),
            placeholder="e.g. Neurocognitive decline_Regression",
            key="est_search_name",
        ).strip().lower()

    with filter_tab2:
        site_options = ["All"] + sorted(registry_df["Site"].dropna().astype(str).unique().tolist())
        site_filter = st.selectbox("Filter by site", options=site_options, key="est_search_site")

    with filter_tab3:
        outcome_options = ["All"] + sorted(registry_df["Predictive outcome"].dropna().astype(str).unique().tolist())
        outcome_filter = st.selectbox("Filter by predictive outcome", options=outcome_options, key="est_search_outcome")

    with filter_tab4:
        method_options = ["All"] + sorted(registry_df["Method"].dropna().astype(str).unique().tolist())
        method_filter = st.selectbox("Filter by method", options=method_options, key="est_search_method")

    with filter_tab5:
        result_sort = st.radio(
            "Order results by",
            options=["Descending results", "Ascending results"],
            horizontal=True,
            key="est_result_sort_order",
        )

    if st.button("Search", type="primary", use_container_width=True, key="established_model_search_button"):
        st.session_state.established_search_has_run = True
        save_established_model_workflow_state_persistent()

    if not st.session_state.get("established_search_has_run", False):
        st.info("Set filters above, then press Search to show matching models.")
        if st.button("← Back to Established Model landing", use_container_width=True, key="est_search_back_before_run"):
            go_to("established_model")
        st.stop()

    filtered_df = registry_df.copy()
    if name_filter:
        filtered_df = filtered_df[filtered_df["Name"].astype(str).str.lower().str.contains(name_filter, na=False)]
    if site_filter != "All":
        filtered_df = filtered_df[filtered_df["Site"].astype(str) == site_filter]
    if outcome_filter != "All":
        filtered_df = filtered_df[filtered_df["Predictive outcome"].astype(str) == outcome_filter]
    if method_filter != "All":
        filtered_df = filtered_df[filtered_df["Method"].astype(str) == method_filter]

    sort_ascending = result_sort == "Ascending results"
    filtered_df = filtered_df.sort_values(by="Result value", ascending=sort_ascending, na_position="last")

    st.markdown("### Search results")

    if filtered_df.empty:
        st.warning("No models match the selected filters.")
        if st.button("← Back to Established Model landing", use_container_width=True, key="est_search_back_no_results"):
            go_to("established_model")
        st.stop()

    display_df = filtered_df[["Name", "Site", "Predictive outcome", "Method", "Predictors", "No. predictors", "Exported date", "Results", "Library key"]].copy()
    table_df = display_df[["Name", "Site", "Predictive outcome", "Method", "Predictors", "No. predictors", "Exported date", "Results"]].copy()

    st.caption("Select one or more model rows, then click 'Move selected models to Compare'.")
    table_event = st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        key="established_model_search_results_multi_select",
    )

    selected_rows = table_event.selection.rows if table_event and table_event.selection else []
    selected_model_keys = display_df.iloc[selected_rows]["Library key"].astype(str).tolist() if selected_rows else []
    selected_models = display_df.iloc[selected_rows]["Name"].astype(str).tolist() if selected_rows else []

    if selected_models:
        st.success(f"Selected {len(selected_models)} model(s): {', '.join(selected_models)}")
    else:
        st.info("No models selected yet.")

    col_back, col_compare = st.columns(2)
    with col_back:
        if st.button("← Back to Established Model landing", use_container_width=True, key="est_search_back_landing"):
            go_to("established_model")
    with col_compare:
        if st.button("Move selected models to Compare →", type="primary", use_container_width=True, key="est_search_move_selected_to_compare"):
            if not selected_models:
                st.warning("Select at least one model before moving to Compare.")
            else:
                st.session_state.established_selected_models_for_compare = selected_model_keys
                save_established_model_workflow_state_persistent()
                go_to("established_model_compare")


# ============================================================
# KNOWLEDGE BASE MODULE
# ============================================================
elif st.session_state.page == "knowledge_base":
    st.header("Knowledge Base")
    st.write(
        "Choose a knowledge-base area. Each area contains a document list and an upload/add-document form."
    )

    if "knowledge_base_documents" not in st.session_state:
        st.session_state.knowledge_base_documents = {
            "Outcome assessment tools": [],
            "Model development tools": [],
            "Model evaluation tools": [],
            "Clinical applications": [],
            "Other": [],
        }

    for category in [
        "Outcome assessment tools",
        "Model development tools",
        "Model evaluation tools",
        "Clinical applications",
        "Other",
    ]:
        if category not in st.session_state.knowledge_base_documents:
            st.session_state.knowledge_base_documents[category] = []

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("## 🧠 Outcome assessment tools")
            st.write("Cognitive tests, toxicity scales, patient-reported outcomes, and outcome definitions.")
            if st.button("Open outcome assessment tools", use_container_width=True, type="primary"):
                go_to("kb_outcome_assessment_tools")

        with st.container(border=True):
            st.markdown("## 📏 Model evaluation tools")
            st.write("TRIPOD, calibration, discrimination, external validation, and reporting tools.")
            if st.button("Open model evaluation tools", use_container_width=True):
                go_to("kb_model_evaluation_tools")

    with col2:
        with st.container(border=True):
            st.markdown("## 🛠️ Model development tools")
            st.write("Feature selection, regression, random forests, XGBoost, model-building references.")
            if st.button("Open model development tools", use_container_width=True):
                go_to("kb_model_development_tools")

        with st.container(border=True):
            st.markdown("## 🏥 Clinical applications")
            st.write("Clinical use-cases, implementation examples, decision-support scenarios, and practice-focused notes.")
            if st.button("Open clinical applications", use_container_width=True):
                go_to("kb_clinical_applications")

        with st.container(border=True):
            st.markdown("## 📦 Other")
            st.write("General references, project notes, methods, and miscellaneous documentation.")
            if st.button("Open other", use_container_width=True):
                go_to("kb_other")

    st.divider()

    if st.button("← Back to Model Development", use_container_width=True, key="kb_back_model_dev"):
        go_to("model_development_home")


elif st.session_state.page in [
    "kb_outcome_assessment_tools",
    "kb_model_development_tools",
    "kb_model_evaluation_tools",
    "kb_clinical_applications",
    "kb_other",
]:
    page_to_category = {
        "kb_outcome_assessment_tools": "Outcome assessment tools",
        "kb_model_development_tools": "Model development tools",
        "kb_model_evaluation_tools": "Model evaluation tools",
        "kb_clinical_applications": "Clinical applications",
        "kb_other": "Other",
    }

    category_name = page_to_category[st.session_state.page]
    st.header(category_name)

    if category_name == "Clinical applications":
        st.markdown("## 🏥 Clinical applications")
        st.write("Organise clinical-application documents by treatment site.")

        site_tabs = st.tabs([
            "🧠 Brain",
            "🗣️ Head and neck",
            "🫁 Thorax and abdomen",
            "🦴 Pelvis",
            "📦 Other",
        ])

        site_categories = [
            "Clinical applications - Brain",
            "Clinical applications - Head and neck",
            "Clinical applications - Thorax and abdomen",
            "Clinical applications - Pelvis",
            "Clinical applications - Other",
        ]

        for tab, site_category in zip(site_tabs, site_categories):
            with tab:
                render_kb_document_manager(site_category, show_heading=True)
    else:
        render_kb_document_manager(category_name, show_heading=True)

    st.divider()

    if st.button("← Back to Knowledge Base", use_container_width=True):
        go_to("knowledge_base")



# ============================================================
# ESTABLISHED MODEL COMPARE PAGE
# ============================================================

# ============================================================
# ESTABLISHED MODEL MODULE - COMPARE
# ============================================================
elif st.session_state.page == "established_model_compare":
    st.header("Established Model Library")
    st.subheader("Compare")

    exported = st.session_state.get("established_calculators", {})
    if not exported:
        st.info("No established models are available yet.")
        if st.button("← Back to Established Model landing", use_container_width=True, key="est_compare_back_empty"):
            go_to("established_model")
        st.stop()

    selected_from_search = [
        model_name for model_name in st.session_state.get("established_selected_models_for_compare", [])
        if model_name in exported
    ]

    if not selected_from_search:
        st.info("No models have been selected for comparison. Go to Search, select one or more models, then move them to Compare.")
        if st.button("Open Search", type="primary", use_container_width=True, key="est_compare_open_search_no_selection"):
            go_to("established_model_search")
        st.stop()

    st.caption("Models selected from Search. You can adjust the selection here before comparing.")
    selected_models = st.multiselect(
        "Models to compare",
        options=sorted(exported.keys()),
        default=selected_from_search,
        format_func=lambda key: established_display_model_name(key, exported.get(key, {})),
        key="est_compare_selected_models_multiselect",
    )

    if not selected_models:
        st.warning("Select at least one model to compare.")
        st.stop()

    rows = []
    for name in selected_models:
        info = exported.get(name, {})
        metrics = info.get("metrics", {})
        display_name = established_display_model_name(name, info)
        rows.append({
            "Model": display_name,
            "Site": info.get("model_site", "Brain"),
            "Outcome": info.get("outcome_group", info.get("outcome_variable", "")),
            "Method": info.get("method", "Unknown"),
            "Exported date": info.get("exported_date", str(info.get("exported_at", "")).split(" ")[0] if info.get("exported_at", "") else ""),
            "Predictors": established_format_predictor_list(info.get("predictors", [])),
            "No. predictors": established_predictor_count(info.get("predictors", [])),
            "Training AUC": metrics.get("Training AUC", np.nan),
            "Validation AUC": metrics.get("Validation AUC", metrics.get("AUC", np.nan)),
            "Training accuracy": metrics.get("Training Accuracy", np.nan),
            "Validation accuracy": metrics.get("Validation Accuracy", metrics.get("Accuracy", np.nan)),
            "Validation sensitivity": metrics.get("Validation Sensitivity", metrics.get("Sensitivity", np.nan)),
            "Validation specificity": metrics.get("Validation Specificity", metrics.get("Specificity", np.nan)),
            "Validation F1": metrics.get("Validation F1 Score", metrics.get("F1 Score", np.nan)),
            "Brier score": metrics.get("Validation Brier Score", metrics.get("Brier Score", np.nan)),
        })

    compare_df = pd.DataFrame(rows)
    matrix_df = compare_df.set_index("Model").T.reset_index().rename(columns={"index": "Parameter"})

    def _highlight_best_in_row(row):
        styles = [""] * len(row)
        parameter = str(row.get("Parameter", ""))
        higher_is_better = ["AUC", "accuracy", "sensitivity", "specificity", "F1"]
        lower_is_better = ["Brier score"]
        model_columns = [col for col in row.index if col != "Parameter"]

        if not any(term in parameter for term in higher_is_better + lower_is_better):
            return styles

        values = pd.to_numeric(row[model_columns], errors="coerce")
        if values.dropna().empty:
            return styles

        if any(term in parameter for term in lower_is_better):
            best_value = values.min()
        else:
            best_value = values.max()

        for idx, col in enumerate(row.index):
            if col in model_columns:
                try:
                    if pd.notna(values[col]) and float(values[col]) == float(best_value):
                        styles[idx] = "background-color: orange; color: black; font-weight: bold;"
                except Exception:
                    pass
        return styles

    st.markdown("### Model comparison matrix")
    st.dataframe(matrix_df.style.apply(_highlight_best_in_row, axis=1), use_container_width=True, hide_index=True)

    with st.expander("Predictors by model"):
        predictor_rows = []
        for name in selected_models:
            info = exported.get(name, {})
            predictors = info.get("predictors", [])
            if isinstance(predictors, str):
                predictor_list = [p.strip() for p in re.split(r"[;,|]", predictors) if p.strip()]
            else:
                try:
                    predictor_list = list(predictors)
                except Exception:
                    predictor_list = []
            if not predictor_list:
                predictor_rows.append({"Model": established_display_model_name(name, info), "No.": "", "Predictor": "No predictors stored"})
            else:
                for idx, predictor in enumerate(predictor_list, start=1):
                    predictor_rows.append({
                        "Model": established_display_model_name(name, info),
                        "No.": idx,
                        "Predictor": established_clean_predictor_name(predictor),
                    })
        st.dataframe(pd.DataFrame(predictor_rows), use_container_width=True, hide_index=True)

    col_search, col_save, col_risk, col_back = st.columns(4)
    with col_search:
        if st.button("← Back to Search", use_container_width=True, key="est_compare_back_search"):
            go_to("established_model_search")
    with col_save:
        if st.button("Save comparison snapshot", use_container_width=True, key="est_compare_save_snapshot"):
            out = PERSISTENT_STORAGE_DIR / "established_model_compare_snapshot.csv"
            compare_df.to_csv(out, index=False)
            matrix_df.to_csv(PERSISTENT_STORAGE_DIR / "established_model_compare_matrix.csv", index=False)
            st.success(f"Comparison snapshot saved: {out}")
    with col_risk:
        if st.button("Next: Risk calculator →", type="primary", use_container_width=True, key="est_compare_next_risk"):
            st.session_state.established_models_for_risk_calculator = selected_models
            save_established_model_workflow_state_persistent()
            go_to("established_model_risk_calculator")
    with col_back:
        if st.button("Back to Established Model landing", use_container_width=True, key="est_compare_back_landing"):
            go_to("established_model")


# ============================================================
# ESTABLISHED MODEL RISK CALCULATOR PAGE
# ============================================================

# ============================================================
# ESTABLISHED MODEL MODULE - RISK CALCULATOR
# ============================================================
elif st.session_state.page == "established_model_risk_calculator":
    st.header("Established Model Library")
    st.subheader("Risk calculator")
    exported = st.session_state.get("established_calculators", {})
    if not exported:
        st.info("No established models are available yet.")
    else:
        compared_models = [
            model_key for model_key in st.session_state.get("established_models_for_risk_calculator", st.session_state.get("established_selected_models_for_compare", []))
            if model_key in exported
        ]
        model_options = compared_models if compared_models else sorted(exported.keys())
        if compared_models:
            st.info("Risk calculator is using the models selected in the Compare tab. Select one of the compared models below.")
        else:
            st.info("No compared models were carried forward. Select any available established model below.")

        selected_model = st.selectbox(
            "Select one of the compared models",
            options=model_options,
            format_func=lambda key: established_display_model_name(key, exported.get(key, {})),
            key="est_calc_model_select",
        )
        selected_info = exported[selected_model]
        pipeline = selected_info.get("pipeline", None)
        predictors = selected_info.get("predictors", [])
        df = st.session_state.get("df", None)
        outcome_label = str(selected_info.get("outcome_group", selected_info.get("outcome_variable", "Measured outcome"))).strip()
        if outcome_label == "":
            outcome_label = "Measured outcome"
        st.write(f"Outcome measured: {outcome_label}")
        st.caption(f"Model: {established_display_model_name(selected_model, selected_info)}")
        if pipeline is None:
            st.error("This model does not contain a fitted pipeline.")
        else:
            patient_data = {}
            for predictor in predictors:
                clean_label = established_clean_predictor_name(predictor) if 'established_clean_predictor_name' in globals() else str(predictor)
                if df is not None and predictor in df.columns and is_numeric_column(df, predictor):
                    default_value = pd.to_numeric(df[predictor], errors="coerce").median()
                    patient_data[predictor] = st.number_input(clean_label, value=float(default_value) if not pd.isna(default_value) else 0.0, key=f"est_calc_page_{selected_model}_{predictor}")
                elif df is not None and predictor in df.columns:
                    options = sorted([x for x in df[predictor].dropna().astype(str).str.strip().unique().tolist() if x != ""])
                    if not options:
                        options = [""]
                    patient_data[predictor] = st.selectbox(clean_label, options=options, key=f"est_calc_page_{selected_model}_{predictor}")
                else:
                    patient_data[predictor] = st.text_input(clean_label, key=f"est_calc_page_{selected_model}_{predictor}")
            if st.button("Calculate patient risk", type="primary", use_container_width=True, key="est_calc_page_calculate"):
                try:
                    risk = pipeline.predict_proba(pd.DataFrame([patient_data]))[0, 1]
                    result_row = {
                        "Model": established_display_model_name(selected_model, selected_info),
                        "Outcome measured": outcome_label,
                        "Predicted risk": risk,
                        "Predicted risk percent": risk * 100,
                        "Calculation date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    for key, value in patient_data.items():
                        result_row[f"Input: {key}"] = value
                    st.session_state.established_risk_calculator_last_result = pd.DataFrame([result_row])
                    st.session_state.established_risk_selected_model = selected_model
                    save_established_model_workflow_state_persistent()
                    st.metric(f"Predicted risk of {outcome_label}", f"{risk * 100:.1f}%")
                    st.success(f"Predicted risk of {outcome_label}: {risk * 100:.1f}%")
                except Exception as error:
                    st.error(f"Risk calculation failed: {error}")

            last_risk_result = st.session_state.get("established_risk_calculator_last_result", pd.DataFrame())
            if last_risk_result is not None and not last_risk_result.empty:
                st.markdown("### Download risk calculation")
                st.dataframe(last_risk_result, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download risk calculation as CSV",
                    data=last_risk_result.to_csv(index=False).encode("utf-8"),
                    file_name="established_model_risk_calculation.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="est_calc_download_csv",
                )

    st.divider()
    col_compare, col_next, col_back = st.columns(3)
    with col_compare:
        if st.button("← Back to Compare", use_container_width=True, key="est_calc_back_compare"):
            go_to("established_model_compare")
    with col_next:
        if st.button("Next: External validation results →", type="primary", use_container_width=True, key="est_calc_next_external_validation"):
            try:
                if "selected_model" in locals():
                    st.session_state.established_external_validation_selected_model = selected_model
            except Exception:
                pass
            save_established_model_workflow_state_persistent()
            go_to("established_model_external_validation")
    with col_back:
        if st.button("Back to Established Model landing", use_container_width=True, key="est_calc_back"):
            go_to("established_model")


# ============================================================
# ESTABLISHED MODEL VALIDATION PAGE
# ============================================================

# ============================================================
# ESTABLISHED MODEL MODULE - VALIDATE MODEL USING MY DATA
# ============================================================
elif st.session_state.page == "established_model_validate":
    st.header("Established Model Library")
    st.subheader("Validate model using my data")

    st.write(
        "Validate one selected established model using your own dataset. "
        "This page only calculates the external-validation performance. "
        "Publishing, rating and assessment are completed on the External validation results page."
    )

    exported = st.session_state.get("established_calculators", {})
    if not exported:
        st.info("No established models are available yet.")
        if st.button("← Back to Established Model landing", use_container_width=True, key="est_validate_back_empty"):
            go_to("established_model")
        st.stop()

    default_models = [m for m in st.session_state.get("established_selected_models_for_compare", []) if m in exported]
    if not default_models:
        default_models = [m for m in st.session_state.get("established_models_for_risk_calculator", []) if m in exported]
    if not default_models:
        last_model = st.session_state.get("established_risk_selected_model", "") or st.session_state.get("established_last_exported_model", "")
        default_models = [last_model] if last_model in exported else []

    available_model_keys = sorted(exported.keys())
    default_index = 0
    if default_models:
        try:
            default_index = available_model_keys.index(default_models[0])
        except Exception:
            default_index = 0

    selected_model = st.selectbox(
        "Model to validate",
        options=available_model_keys,
        index=default_index,
        format_func=lambda key: established_display_model_name(key, exported.get(key, {})),
        key="est_validate_single_model_select_clean",
    )
    st.session_state.established_validate_selected_model = selected_model
    selected_info = exported.get(selected_model, {})

    st.markdown("### 1. Required predictors")
    predictors = selected_info.get("predictors", [])
    if isinstance(predictors, str):
        predictor_list = [p.strip() for p in re.split(r"[;,|]", predictors) if p.strip()]
    else:
        try:
            predictor_list = list(predictors)
        except Exception:
            predictor_list = []

    if predictor_list:
        predictor_df = pd.DataFrame({
            "No.": list(range(1, len(predictor_list) + 1)),
            "Predictor": [established_clean_predictor_name(p) for p in predictor_list],
            "Required column name": predictor_list,
        })
        st.dataframe(predictor_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No predictors were stored for this model.")

    st.markdown("### 2. Validation dataset")
    data_source = st.radio(
        "Validation data source",
        options=["Use current clinical dataset", "Upload validation Excel/CSV"],
        horizontal=True,
        key="est_validate_data_source_clean",
    )

    validation_df = pd.DataFrame()
    validation_dataset_label = "Current clinical dataset"
    if data_source == "Use current clinical dataset":
        if st.session_state.get("df", None) is not None:
            validation_df = st.session_state.df.copy()
            st.success(f"Using current clinical dataset: {validation_df.shape[0]} rows × {validation_df.shape[1]} columns.")
        else:
            st.info("No current clinical dataset is loaded. Upload a validation Excel/CSV file instead.")
    else:
        uploaded_validation = st.file_uploader(
            "Upload validation Excel/CSV file",
            type=["xlsx", "xls", "csv"],
            key="est_validate_excel_upload_clean",
        )
        if uploaded_validation is not None:
            try:
                validation_dataset_label = uploaded_validation.name
                if uploaded_validation.name.lower().endswith(".csv"):
                    validation_df = pd.read_csv(uploaded_validation)
                else:
                    validation_df = pd.read_excel(uploaded_validation)
                st.success(f"Validation dataset loaded: {validation_df.shape[0]} rows × {validation_df.shape[1]} columns.")
            except Exception as error:
                st.error(f"Could not read validation dataset: {error}")

    if validation_df is None or validation_df.empty:
        st.stop()

    with st.expander("Preview validation dataset", expanded=False):
        st.dataframe(validation_df.head(20), use_container_width=True)

    st.markdown("### 3. Required predictor mapping")
    st.caption(
        "The app suggests exact or close matches from the validation dataset. "
        "Please review each row and change the selected column if needed before validation."
    )

    predictor_column_map = {}
    predictor_mapping_rows = []
    validation_columns = [str(c) for c in validation_df.columns]
    validation_column_options = ["Not mapped"] + validation_columns

    if predictor_list:
        for idx, required_predictor in enumerate(predictor_list, start=1):
            suggestion = established_best_validation_column_match(required_predictor, validation_columns)
            suggested_column = suggestion.get("column", "")
            default_choice = suggested_column if suggested_column in validation_columns else "Not mapped"
            default_index = validation_column_options.index(default_choice) if default_choice in validation_column_options else 0

            map_cols = st.columns([0.32, 0.34, 0.18, 0.16])
            with map_cols[0]:
                st.markdown(f"**{idx}. {established_clean_predictor_name(required_predictor)}**")
                st.caption(f"Model variable: `{required_predictor}`")
            with map_cols[1]:
                selected_column = st.selectbox(
                    "Dataset column",
                    options=validation_column_options,
                    index=default_index,
                    key=f"est_validate_predictor_map_{make_safe_column_name(selected_model)}_{make_safe_column_name(required_predictor)}",
                    label_visibility="collapsed",
                )
            with map_cols[2]:
                st.write(suggestion.get("status", ""))
            with map_cols[3]:
                st.write(f"{float(suggestion.get('score', 0.0)):.2f}")

            mapped_column = "" if selected_column == "Not mapped" else selected_column
            predictor_column_map[required_predictor] = mapped_column
            predictor_mapping_rows.append({
                "Required predictor": established_clean_predictor_name(required_predictor),
                "Model variable": required_predictor,
                "Suggested column": suggested_column or "",
                "Selected validation column": mapped_column or "Not mapped",
                "Match status": suggestion.get("status", ""),
                "Match score": float(suggestion.get("score", 0.0)),
            })

        predictor_mapping_df = pd.DataFrame(predictor_mapping_rows)
        with st.expander("Show predictor mapping table", expanded=False):
            st.dataframe(predictor_mapping_df, use_container_width=True, hide_index=True)
    else:
        predictor_mapping_df = pd.DataFrame()
        st.warning("No stored predictors were available to map for this model.")

    st.markdown("### 4. Outcome and threshold")
    likely_outcomes = [col for col in validation_df.columns if any(term in str(col).lower() for term in ["outcome", "decline", "toxicity", "survival", "event", "response", "progression"])]
    outcome_options = likely_outcomes + [c for c in validation_df.columns if c not in likely_outcomes]
    selected_outcome = st.selectbox(
        "Observed outcome column in validation dataset",
        options=outcome_options,
        key="est_validate_outcome_column_clean",
    )
    threshold = st.slider(
        "Risk threshold for classification metrics",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        key="est_validate_threshold_clean",
    )

    st.markdown("### 5. Calculate validation")
    if st.button("Validate selected model using my data", type="primary", use_container_width=True, key="est_validate_run_single_clean"):
        validation_date_label = datetime.now().strftime("%Y-%m-%d")
        y_true, y_note = established_prepare_binary_outcome(validation_df[selected_outcome])
        display_name = established_display_model_name(selected_model, selected_info)
        pipeline = selected_info.get("pipeline", None)
        missing = [p for p in predictor_list if not predictor_column_map.get(p, "")]

        base_row = {
            "Model key": selected_model,
            "Model name": display_name,
            "Validation type": "External validation using my data",
            "Validation source": validation_dataset_label,
            "Validation date": validation_date_label,
            "Outcome measured": selected_info.get("outcome_group", selected_info.get("outcome_variable", selected_outcome)),
            "Observed outcome column": selected_outcome,
            "Method": selected_info.get("method", selected_info.get("analytics_method", "")),
            "Risk threshold": float(threshold),
            "Missing predictors": "; ".join(missing),
            "Predictor mapping": "; ".join([f"{p} -> {predictor_column_map.get(p, 'Not mapped')}" for p in predictor_list]),
        }

        prediction_rows = []
        if pipeline is None:
            result_row = {**base_row, "N validated": 0, "AUC": np.nan, "Accuracy": np.nan, "Sensitivity": np.nan, "Specificity": np.nan, "F1 score": np.nan, "Brier score": np.nan, "Note": "No saved model pipeline is available for this model."}
        elif missing:
            result_row = {**base_row, "N validated": 0, "AUC": np.nan, "Accuracy": np.nan, "Sensitivity": np.nan, "Specificity": np.nan, "F1 score": np.nan, "Brier score": np.nan, "Note": "Missing required predictors in validation dataset."}
        else:
            try:
                X_val = pd.DataFrame(index=validation_df.index)
                for required_predictor in predictor_list:
                    mapped_column = predictor_column_map.get(required_predictor, "")
                    X_val[required_predictor] = validation_df[mapped_column]
                y_prob = pipeline.predict_proba(X_val)[:, 1]
                metrics = established_classification_metrics(y_true, y_prob, threshold=threshold)
                result_row = {
                    **base_row,
                    "N validated": metrics.get("N validated", 0),
                    "AUC": metrics.get("AUC", np.nan),
                    "Accuracy": metrics.get("Accuracy", np.nan),
                    "Sensitivity": metrics.get("Sensitivity", np.nan),
                    "Specificity": metrics.get("Specificity", np.nan),
                    "F1 score": metrics.get("F1 score", np.nan),
                    "Brier score": metrics.get("Brier score", np.nan),
                    "Note": "; ".join([n for n in [y_note, metrics.get("Note", "")] if n]),
                }
                patient_id_col = validation_df.columns[0]
                for idx, prob in enumerate(y_prob):
                    prediction_rows.append({
                        "Model name": display_name,
                        "Patient row": idx + 1,
                        "Patient ID": validation_df.iloc[idx].get(patient_id_col, idx + 1),
                        "Observed outcome": y_true.iloc[idx] if idx < len(y_true) else np.nan,
                        "Predicted risk": prob,
                    })
            except Exception as error:
                result_row = {**base_row, "N validated": 0, "AUC": np.nan, "Accuracy": np.nan, "Sensitivity": np.nan, "Specificity": np.nan, "F1 score": np.nan, "Brier score": np.nan, "Note": f"Validation failed: {error}"}

        validation_results_df = pd.DataFrame([result_row])
        prediction_df = pd.DataFrame(prediction_rows)
        out = established_validation_output_folder()
        validation_path = out / "external_validation_using_my_data_results.csv"
        prediction_path = out / "external_validation_using_my_data_patient_predictions.csv"
        validation_results_df.to_csv(validation_path, index=False)
        if not prediction_df.empty:
            prediction_df.to_csv(prediction_path, index=False)
        st.session_state.established_local_validation_results = validation_results_df
        st.session_state.established_local_validation_predictions = prediction_df
        st.session_state.established_local_validation_results_path = str(validation_path)
        st.session_state.established_local_validation_model_key = selected_model
        st.session_state.established_pending_external_validation_result = result_row
        st.session_state.established_pending_external_validation_predictions = prediction_df
        save_established_model_workflow_state_persistent()
        st.success(f"Validation calculated and saved: {validation_path}")

    results_df = st.session_state.get("established_local_validation_results", pd.DataFrame())
    if results_df is not None and not results_df.empty:
        st.markdown("### 5. Validation result")
        display_cols = [
            "Model name", "Validation type", "Validation source", "Validation date", "Outcome measured", "N validated",
            "AUC", "Accuracy", "Sensitivity", "Specificity", "F1 score", "Brier score", "Missing predictors", "Predictor mapping", "Note",
        ]
        display_cols = [c for c in display_cols if c in results_df.columns]
        st.dataframe(results_df[display_cols], use_container_width=True, hide_index=True)

        pred_df = st.session_state.get("established_local_validation_predictions", pd.DataFrame())
        if pred_df is not None and not pred_df.empty:
            with st.expander("Patient-level predictions from this validation run"):
                st.dataframe(pred_df, use_container_width=True, hide_index=True)

        st.info("Next, publish this validation result by adding institute/contact details, rating the model, and optionally uploading a TRIPOD/PROBAST/CHARMS assessment.")
        if st.button("Publish this validation result →", type="primary", use_container_width=True, key="est_validate_publish_result_clean"):
            st.session_state.established_external_validation_selected_model = st.session_state.get("established_local_validation_model_key", selected_model)
            save_established_model_workflow_state_persistent()
            go_to("established_model_external_validation")

    st.divider()
    col_back, col_ext = st.columns(2)
    with col_back:
        if st.button("← Back to Risk calculator", use_container_width=True, key="est_validate_back_risk_clean"):
            go_to("established_model_risk_calculator")
    with col_ext:
        if st.button("Open External validation results", use_container_width=True, key="est_validate_open_ext_clean"):
            go_to("established_model_external_validation")


# ============================================================
# ESTABLISHED MODEL EXTERNAL VALIDATION RESULTS PAGE
# ============================================================

# ============================================================
# ESTABLISHED MODEL MODULE - EXTERNAL VALIDATION RESULTS
# ============================================================
elif st.session_state.page == "established_model_external_validation":
    st.header("Established Model Library")
    st.subheader("External validation results")

    st.write(
        "This page shows published external-validation results from all institutes and models. "
        "After validating a model using your data, publish the aggregate result here with institute details, rating and optional assessment documentation."
    )

    exported = st.session_state.get("established_calculators", {})
    if not exported:
        st.info("No established models are available yet.")
        if st.button("← Back to Established Model landing", use_container_width=True, key="est_extval_back_empty_clean"):
            go_to("established_model")
        st.stop()

    def _model_display_name(model_key, model_info):
        try:
            return established_display_model_name(model_key, model_info)
        except Exception:
            return str(model_info.get("model_name", model_key))

    def _external_validation_registry(calculators):
        rows = []
        for model_key, model_info in calculators.items():
            display_name = _model_display_name(model_key, model_info)
            for result in model_info.get("external_validation_results", []) or []:
                rows.append({
                    "Institute": result.get("Institute", result.get("Centre", result.get("Validation source", ""))),
                    "Contact email": result.get("Contact email", ""),
                    "Date published": result.get("Date published", result.get("Published timestamp", result.get("Validation date", ""))),
                    "Model name": result.get("Model", display_name),
                    "Validation date": result.get("Validation date", ""),
                    "Outcome measured": result.get("Outcome measured", result.get("Outcome", "")),
                    "Sample size": result.get("Sample size", result.get("N validated", "")),
                    "AUC": result.get("AUC", np.nan),
                    "Accuracy": result.get("Accuracy", np.nan),
                    "Sensitivity": result.get("Sensitivity", np.nan),
                    "Specificity": result.get("Specificity", np.nan),
                    "F1 score": result.get("F1 score", np.nan),
                    "Brier score": result.get("Brier score", np.nan),
                    "Usability rating": result.get("Usability rating", ""),
                    "Clinical relevance rating": result.get("Clinical relevance rating", ""),
                    "Evidence quality rating": result.get("Evidence quality rating", ""),
                    "Implementation readiness rating": result.get("Implementation readiness rating", ""),
                    "Assessment tool": result.get("Assessment tool", ""),
                    "Assessment file": result.get("Assessment file", ""),
                    "Assessment summary": result.get("Assessment summary", ""),
                    "Notes": result.get("Notes", ""),
                })
        return pd.DataFrame(rows)

    st.markdown("### Published external-validation registry")
    registry_df = _external_validation_registry(exported)
    if registry_df.empty:
        st.info("No external validation results have been published yet.")
    else:
        preferred_cols = [
            "Institute", "Contact email", "Date published", "Model name", "Validation date", "Outcome measured", "Sample size",
            "AUC", "Accuracy", "Sensitivity", "Specificity", "F1 score", "Brier score",
            "Usability rating", "Clinical relevance rating", "Evidence quality rating", "Implementation readiness rating",
            "Assessment tool", "Assessment file", "Assessment summary", "Notes",
        ]
        display_cols = [c for c in preferred_cols if c in registry_df.columns]
        st.dataframe(registry_df[display_cols], use_container_width=True, hide_index=True)
        export_folder = PERSISTENT_STORAGE_DIR / "established_model_external_validation"
        export_folder.mkdir(parents=True, exist_ok=True)
        registry_path = export_folder / "external_validation_results_registry.csv"
        registry_df.to_csv(registry_path, index=False)
        st.caption(f"Registry saved locally: {registry_path}")

    pending = st.session_state.get("established_pending_external_validation_result", None)
    if pending:
        st.divider()
        st.markdown("### Publish latest validation result")
        st.caption("Add institute/contact details, review the validation result, rate the model, optionally upload an assessment, then publish to the registry table above.")

        pending_model_key = str(pending.get("Model key", st.session_state.get("established_external_validation_selected_model", "")))
        pending_model_info = exported.get(pending_model_key, {})
        pending_model_name = str(pending.get("Model name", _model_display_name(pending_model_key, pending_model_info)))

        c1, c2 = st.columns(2)
        with c1:
            institute = st.text_input("Institute / centre", key="est_publish_institute")
        with c2:
            contact_email = st.text_input("Contact email", key="est_publish_contact_email")

        st.markdown("#### Validation result to publish")
        pending_df = pd.DataFrame([pending])
        pending_cols = [
            "Model name", "Validation type", "Validation source", "Validation date", "Outcome measured", "N validated",
            "AUC", "Accuracy", "Sensitivity", "Specificity", "F1 score", "Brier score", "Missing predictors", "Predictor mapping", "Note",
        ]
        pending_cols = [c for c in pending_cols if c in pending_df.columns]
        st.dataframe(pending_df[pending_cols], use_container_width=True, hide_index=True)

        st.markdown("#### Rate this externally validated model")
        r1, r2, r3, r4 = st.columns(4)
        usability_rating = r1.slider("Usability", 1, 5, 3, key="est_publish_rating_usability")
        relevance_rating = r2.slider("Clinical relevance", 1, 5, 3, key="est_publish_rating_relevance")
        evidence_rating = r3.slider("Evidence quality", 1, 5, 3, key="est_publish_rating_evidence")
        readiness_rating = r4.slider("Implementation readiness", 1, 5, 3, key="est_publish_rating_readiness")
        rating_notes = st.text_area("Rating notes", key="est_publish_rating_notes")

        st.markdown("#### Standardised assessment")
        assessment_options = [
            "None",
            "TRIPOD / TRIPOD-AI reporting checklist",
            "PROBAST / PROBAST-AI risk-of-bias assessment",
            "CHARMS data extraction checklist",
            "Custom / other assessment tool",
        ]
        assessment_tool = st.selectbox("Assessment tool", options=assessment_options, key="est_publish_assessment_tool")
        assessment_upload = st.file_uploader(
            "Upload completed assessment document",
            type=["pdf", "docx", "xlsx", "xls", "csv", "txt"],
            key="est_publish_assessment_upload",
            help="Optional. Upload the completed TRIPOD, PROBAST, CHARMS or custom assessment document.",
        )
        assessment_summary = st.text_area("Assessment summary / notes", key="est_publish_assessment_summary", height=120)
        notes = st.text_area("Additional publication notes", key="est_publish_notes")

        if st.button("Publish external validation result", type="primary", use_container_width=True, key="est_publish_pending_validation"):
            if not institute.strip():
                st.error("Please enter the institute / centre before publishing.")
            elif not contact_email.strip():
                st.error("Please enter a contact email before publishing.")
            else:
                assessment_file_path = ""
                if assessment_upload is not None:
                    assessment_folder = PERSISTENT_STORAGE_DIR / "established_model_external_validation" / "assessment_files"
                    assessment_folder.mkdir(parents=True, exist_ok=True)
                    safe_model = re.sub(r"[^A-Za-z0-9_\-]+", "_", pending_model_name).strip("_") or "model"
                    safe_institute = re.sub(r"[^A-Za-z0-9_\-]+", "_", institute).strip("_") or "institute"
                    assessment_file_path = str(assessment_folder / f"{safe_model}_{safe_institute}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{assessment_upload.name}")
                    with open(assessment_file_path, "wb") as f:
                        f.write(assessment_upload.getbuffer())

                new_result = {
                    "Model": pending_model_name,
                    "Institute": institute.strip(),
                    "Centre": institute.strip(),
                    "Contact email": contact_email.strip(),
                    "Date published": datetime.now().strftime("%Y-%m-%d"),
                    "Published timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Validation type": pending.get("Validation type", "External validation using my data"),
                    "Validation source": pending.get("Validation source", ""),
                    "Validation date": pending.get("Validation date", datetime.now().strftime("%Y-%m-%d")),
                    "Outcome measured": pending.get("Outcome measured", pending.get("Outcome", "")),
                    "Method": pending.get("Method", ""),
                    "Sample size": int(float(pending.get("N validated", 0) or 0)),
                    "N validated": int(float(pending.get("N validated", 0) or 0)),
                    "AUC": pending.get("AUC", np.nan),
                    "Accuracy": pending.get("Accuracy", np.nan),
                    "Sensitivity": pending.get("Sensitivity", np.nan),
                    "Specificity": pending.get("Specificity", np.nan),
                    "F1 score": pending.get("F1 score", np.nan),
                    "Brier score": pending.get("Brier score", np.nan),
                    "Usability rating": int(usability_rating),
                    "Clinical relevance rating": int(relevance_rating),
                    "Evidence quality rating": int(evidence_rating),
                    "Implementation readiness rating": int(readiness_rating),
                    "Rating notes": rating_notes.strip(),
                    "Assessment tool": assessment_tool,
                    "Assessment file": assessment_file_path,
                    "Assessment summary": assessment_summary.strip(),
                    "Published": "Yes",
                    "Notes": notes.strip(),
                }

                model_info = exported.get(pending_model_key, pending_model_info)
                existing_results = model_info.get("external_validation_results", []) or []
                duplicate_index = None
                for idx, existing in enumerate(existing_results):
                    same_institute = str(existing.get("Institute", existing.get("Centre", ""))).strip().lower() == new_result["Institute"].lower()
                    same_date = str(existing.get("Validation date", "")).strip() == str(new_result["Validation date"])
                    same_model = str(existing.get("Model", "")).strip().lower() == str(new_result["Model"]).lower()
                    if same_institute and same_date and same_model:
                        duplicate_index = idx
                        break
                if duplicate_index is not None:
                    existing_results[duplicate_index] = new_result
                else:
                    existing_results.append(new_result)
                model_info["external_validation_results"] = existing_results
                exported[pending_model_key] = model_info
                st.session_state.established_calculators = exported
                save_established_calculators_persistent()
                st.session_state.established_pending_external_validation_result = None
                st.session_state.established_pending_external_validation_predictions = pd.DataFrame()
                save_established_model_workflow_state_persistent()
                st.success("External validation result published to the registry.")
                st.rerun()
    else:
        st.info("To publish a new result, first run 'Validate model using my data' and then click 'Publish this validation result'.")

    st.divider()
    col_back, col_docs = st.columns(2)
    with col_back:
        if st.button("← Back to Validate model using my data", use_container_width=True, key="est_extval_back_validate_clean"):
            go_to("established_model_validate")
    with col_docs:
        if st.button("Next: Documentation →", use_container_width=True, key="est_extval_next_docs_clean"):
            go_to("established_model_documentation")


# ============================================================
# ESTABLISHED MODEL DOCUMENTATION PAGE
# ============================================================

# ============================================================
# ESTABLISHED MODEL MODULE - DOCUMENTATION
# ============================================================
elif st.session_state.page == "established_model_documentation":
    st.header("Established Model Library")
    st.subheader("Documentation")
    exported = st.session_state.get("established_calculators", {})
    if not exported:
        st.info("No established models are available yet.")
    else:
        selected_model = st.selectbox("Select model", options=sorted(exported.keys()), key="est_docs_model_select")
        info = exported[selected_model]
        st.text_input("Publication title", value=info.get("supporting_publication_title", ""), key="est_docs_pub_title")
        st.text_area("Publication/reference", value=info.get("supporting_publication_reference", ""), key="est_docs_pub_ref")
        st.text_area("Documentation notes", value=info.get("documentation_notes", ""), height=180, key="est_docs_notes")
        if st.button("Save documentation", use_container_width=True, key="est_docs_save"):
            info["supporting_publication_title"] = st.session_state.est_docs_pub_title
            info["supporting_publication_reference"] = st.session_state.est_docs_pub_ref
            info["documentation_notes"] = st.session_state.est_docs_notes
            exported[selected_model] = info
            st.session_state.established_calculators = exported
            save_established_model_everything_persistent()
            st.success("Documentation saved.")
    st.divider()
    if st.button("← Back to Established Model landing", use_container_width=True, key="est_docs_back"):
        go_to("established_model")


# Ratings are now captured during external validation.

# ============================================================
# ESTABLISHED MODEL COLLABORATE PAGE
# ============================================================

# ============================================================
# ESTABLISHED MODEL MODULE - COLLABORATE
# ============================================================
elif st.session_state.page == "established_model_collaborate":
    st.header("Established Model Library")
    st.subheader("Collaborate")
    exported = st.session_state.get("established_calculators", {})
    if not exported:
        st.info("No established models are available yet.")
    else:
        selected_model = st.selectbox("Select model", options=sorted(exported.keys()), key="est_collab_model_select")
        info = exported[selected_model]
        requests = info.get("collaboration_requests", [])
        c1, c2 = st.columns(2)
        name = c1.text_input("Name", key="est_collab_name")
        email = c2.text_input("Email", key="est_collab_email")
        institution = st.text_input("Institution", key="est_collab_institution")
        message = st.text_area("Collaboration note", key="est_collab_message")
        if st.button("Save collaboration note", use_container_width=True, key="est_collab_save"):
            requests.append({"name": name, "email": email, "institution": institution, "message": message})
            info["collaboration_requests"] = requests
            exported[selected_model] = info
            st.session_state.established_calculators = exported
            save_established_calculators_persistent()
            st.success("Collaboration note saved.")
        if requests:
            st.markdown("### Saved collaboration notes")
            st.dataframe(pd.DataFrame(requests), use_container_width=True, hide_index=True)
    st.divider()
    if st.button("← Back to Established Model landing", use_container_width=True, key="est_collab_back"):
        go_to("established_model")

elif st.session_state.page == "supporting_documentation":
    st.header("Supporting Documentation")
    st.subheader("Load paper and TRIPOD assessment")

    st.write(
        "Use this page to attach supporting documentation to an established model."
    )

    calculators = st.session_state.get("established_calculators", {})

    if len(calculators) == 0:
        st.warning(
            "No established calculators are available yet. "
            "Export or load a calculator first, then return here to add documentation."
        )

        if st.button("← Back to Established Model", use_container_width=True):
            go_to("established_model")

        st.stop()

    selected_calculator = st.selectbox(
        "Select model/calculator to document",
        options=list(calculators.keys()),
        key="documentation_selected_calculator"
    )

    calculator_info = calculators[selected_calculator]

    st.markdown("### Selected model")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Type", calculator_info.get("model_type", "Clinical"))
    c2.metric("Method", calculator_info.get("method", "Unknown"))
    c3.metric("Site", calculator_info.get("model_site", "Brain"))
    c4.metric("Outcome", calculator_info.get("outcome_group", calculator_info.get("outcome_variable", "")))

    st.divider()

    doc_tab1, doc_tab2 = st.tabs(["Load paper", "TRIPOD assessment"])

    with doc_tab1:
        st.markdown("### Load supporting paper")

        publication_available = st.radio(
            "Is a supporting publication available?",
            options=["Yes", "No / not available"],
            index=0 if not calculator_info.get("publication_not_available", False) else 1,
            key="doc_publication_available"
        )

        if publication_available == "No / not available":
            calculator_info["publication_not_available"] = True
            calculator_info["supporting_publication_title"] = ""
            calculator_info["supporting_publication_reference"] = ""
            calculator_info["supporting_publication_filename"] = ""
            calculator_info["supporting_publication_bytes"] = None
            st.info("This model will be marked as: publication not available.")
        else:
            calculator_info["publication_not_available"] = False

            paper_title = st.text_input(
                "Publication title / citation",
                value=calculator_info.get("supporting_publication_title", ""),
                key="doc_publication_title"
            )

            paper_reference = st.text_input(
                "DOI / URL / reference",
                value=calculator_info.get("supporting_publication_reference", ""),
                key="doc_publication_reference"
            )

            paper_file = st.file_uploader(
                "Upload supporting paper",
                type=["pdf", "doc", "docx", "txt"],
                key="doc_supporting_paper"
            )

            calculator_info["supporting_publication_title"] = paper_title
            calculator_info["supporting_publication_reference"] = paper_reference

            if paper_file is not None:
                calculator_info["supporting_publication_filename"] = paper_file.name
                calculator_info["supporting_publication_bytes"] = paper_file.read()
                st.success(f"Paper uploaded: {paper_file.name}")

            existing_file = calculator_info.get("supporting_publication_filename", "")
            if existing_file:
                st.write("Current attached paper:", existing_file)
                existing_bytes = calculator_info.get("supporting_publication_bytes", None)
                if existing_bytes is not None:
                    st.download_button(
                        "Download current paper",
                        data=existing_bytes,
                        file_name=existing_file,
                        mime="application/octet-stream",
                        use_container_width=True
                    )

    with doc_tab2:
        st.markdown("### TRIPOD assessment")

        st.write(
            "You can either upload a completed TRIPOD assessment document or complete a quick structured checklist below."
        )

        tripod_file = st.file_uploader(
            "Upload completed TRIPOD assessment",
            type=["pdf", "docx", "xlsx", "xls", "csv", "txt"],
            key="doc_tripod_file"
        )

        if tripod_file is not None:
            calculator_info["tripod_assessment_filename"] = tripod_file.name
            calculator_info["tripod_assessment_bytes"] = tripod_file.read()
            st.success(f"TRIPOD assessment uploaded: {tripod_file.name}")

        existing_tripod = calculator_info.get("tripod_assessment_filename", "")
        if existing_tripod:
            st.write("Current TRIPOD file:", existing_tripod)
            tripod_bytes = calculator_info.get("tripod_assessment_bytes", None)
            if tripod_bytes is not None:
                st.download_button(
                    "Download current TRIPOD assessment",
                    data=tripod_bytes,
                    file_name=existing_tripod,
                    mime="application/octet-stream",
                    use_container_width=True
                )

        st.markdown("#### Quick TRIPOD checklist")

        tripod_items = {
            "Title identifies prediction model study": "tripod_title",
            "Abstract summarises objectives, source data, participants, predictors, outcome, and performance": "tripod_abstract",
            "Source of data described": "tripod_source_data",
            "Eligibility criteria described": "tripod_eligibility",
            "Outcome definition described": "tripod_outcome",
            "Candidate predictors clearly defined": "tripod_predictors",
            "Sample size explained": "tripod_sample_size",
            "Missing data handling described": "tripod_missing_data",
            "Model development method described": "tripod_model_method",
            "Model performance measures reported": "tripod_performance",
            "Validation approach described": "tripod_validation",
            "Model specification sufficient for use": "tripod_specification",
            "Limitations discussed": "tripod_limitations",
            "Clinical interpretation discussed": "tripod_interpretation",
        }

        checklist_results = calculator_info.get("tripod_checklist", {})

        for item, key in tripod_items.items():
            checklist_results[item] = st.selectbox(
                item,
                options=["Not assessed", "Yes", "Partial", "No"],
                index=["Not assessed", "Yes", "Partial", "No"].index(
                    checklist_results.get(item, "Not assessed")
                )
                if checklist_results.get(item, "Not assessed") in ["Not assessed", "Yes", "Partial", "No"]
                else 0,
                key=key
            )

        calculator_info["tripod_checklist"] = checklist_results

        scored_items = [v for v in checklist_results.values() if v != "Not assessed"]
        if len(scored_items) > 0:
            yes_count = sum(1 for v in scored_items if v == "Yes")
            partial_count = sum(1 for v in scored_items if v == "Partial")
            no_count = sum(1 for v in scored_items if v == "No")
            st.write(f"TRIPOD checklist summary: Yes={yes_count}, Partial={partial_count}, No={no_count}")

            checklist_df = pd.DataFrame([
                {"TRIPOD item": item, "Assessment": value}
                for item, value in checklist_results.items()
            ])

            st.download_button(
                "Download TRIPOD checklist as CSV",
                data=checklist_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{make_safe_column_name(selected_calculator)}_tripod_checklist.csv",
                mime="text/csv",
                use_container_width=True
            )

    st.session_state.established_calculators[selected_calculator] = calculator_info

    st.divider()

    col_back, col_library = st.columns(2)
    with col_back:
        if st.button("← Back to Established Model", use_container_width=True):
            go_to("established_model")
    with col_library:
        if st.button("Save documentation and return", type="primary", use_container_width=True):
            st.session_state.established_calculators[selected_calculator] = calculator_info
            st.success("Supporting documentation saved.")
            go_to("established_model")




# ============================================================
# FULL PAGE: CODE / STEPS VIEWER
# ============================================================

elif st.session_state.page == "page_code_viewer":
    target_page = st.session_state.get("code_viewer_target_page", "home")
    return_page = st.session_state.get("code_viewer_return_page", "home")

    st.header("💻 Page code / steps")
    st.caption(f"Showing workflow explanation and code for: `{target_page}`")

    col_back, col_refresh = st.columns([0.75, 0.25])

    with col_back:
        if st.button("← Back to previous page", use_container_width=True):
            go_to(return_page)

    with col_refresh:
        if st.button("Refresh code", use_container_width=True):
            st.rerun()

    st.divider()

    steps = PAGE_STEP_GUIDES.get(target_page, [
        "This page is part of the application workflow.",
        "Review the source code below to inspect the page logic."
    ])

    st.markdown("## What this page does")
    for i, step in enumerate(steps, start=1):
        st.markdown(f"**{i}.** {step}")

    st.divider()

    st.markdown("## Page code")
    st.caption(
        "This is the code block used to render the selected page. "
        "It is read directly from the current app.py file."
    )

    st.code(get_page_source_block(target_page), language="python")

    st.divider()

    if st.button("← Return", use_container_width=True):
        go_to(return_page)



# ============================================================
# PAGE 1: UPLOAD EXCEL
# ============================================================


# ============================================================
# CLINICAL MODULE - START / OPEN PROJECT
# ============================================================
elif st.session_state.page == "clinical_start_project":
    st.header("Clinical Module")
    st.subheader("Start / Open clinical project")

    st.write(
        "Create a local clinical-model project folder or reopen an existing one. "
        "This works like the voxel-based analysis project page: choose a save location on your PC, "
        "create/open the project, and all clinical outputs are saved inside workflow-matched folders."
    )

    def clinical_safe_folder_name(name):
        safe = str(name).strip()
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", safe)
        safe = safe.strip("._-")
        return safe or "Clinical_Model_Project"

    def browse_for_clinical_folder(dialog_title="Select folder", mustexist=True):
        """Open a Windows folder browser, with tkinter fallback."""
        selected_folder = ""

        try:
            import subprocess
            show_new_folder = "$true" if not mustexist else "$true"
            powershell_script = f"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = '{dialog_title}'
$dialog.ShowNewFolderButton = {show_new_folder}
$dialog.RootFolder = [System.Environment+SpecialFolder]::Desktop
$result = $dialog.ShowDialog()
if ($result -eq [System.Windows.Forms.DialogResult]::OK) {{
    Write-Output $dialog.SelectedPath
}}
"""
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-STA", "-Command", powershell_script],
                capture_output=True,
                text=True,
                timeout=120,
            )
            selected_folder = completed.stdout.strip().splitlines()[-1].strip() if completed.stdout.strip() else ""
        except Exception:
            selected_folder = ""

        if not selected_folder:
            try:
                import tkinter as tk
                from tkinter import filedialog

                root = tk.Tk()
                root.withdraw()
                root.update()
                try:
                    root.attributes("-topmost", True)
                except Exception:
                    pass
                selected_folder = filedialog.askdirectory(
                    parent=root,
                    title=dialog_title,
                    mustexist=mustexist,
                )
                root.destroy()
            except Exception as error:
                st.warning("Folder browser could not be opened from this Streamlit session. Paste the folder path manually instead.")
                st.caption(f"Folder browser error: {error}")
                selected_folder = ""

        return selected_folder

    mode = st.radio(
        "What do you want to do?",
        ["Create new project", "Open existing project"],
        horizontal=True,
        key="clinical_project_start_open_mode",
    )

    default_clinical_root = str(Path("data") / "clinical_projects")

    if mode == "Create new project":
        st.markdown("### Create new clinical project")

        if "clinical_project_root_folder_input" not in st.session_state:
            st.session_state.clinical_project_root_folder_input = st.session_state.get("clinical_project_root_folder", default_clinical_root)
        if "clinical_project_root_folder_selected" not in st.session_state:
            st.session_state.clinical_project_root_folder_selected = ""
        if "clinical_project_root_folder_selected_applied" not in st.session_state:
            st.session_state.clinical_project_root_folder_selected_applied = ""

        selected_root = st.session_state.get("clinical_project_root_folder_selected", "")
        applied_root = st.session_state.get("clinical_project_root_folder_selected_applied", "")
        if selected_root and selected_root != applied_root:
            st.session_state.clinical_project_root_folder_input = selected_root
            st.session_state.clinical_project_root_folder_selected_applied = selected_root

        st.markdown("### Project details")
        st.caption("Choose from common clinical-model project options, or select Other to type your own.")

        project_name_options = [
            "Proton photon neurocognitive clinical model",
            "Radiotherapy toxicity prediction model",
            "Survival prediction clinical model",
            "Treatment response prediction model",
            "Clinical risk stratification model",
            "Other / type your own",
        ]
        project_name_choice = st.selectbox(
            "Project name / template",
            options=project_name_options,
            key="clinical_project_name_choice",
        )
        if project_name_choice == "Other / type your own":
            project_name = st.text_input(
                "Type project name",
                value=st.session_state.get("clinical_project_name_input", ""),
                key="clinical_project_name_input",
                placeholder="Example: Proton photon neurocognitive clinical model",
            )
        else:
            project_name = project_name_choice
            st.caption(f"Selected project name: {project_name}")

        clinical_focus_options = [
            "Brain / CNS",
            "Head and neck",
            "Thorax / lung",
            "Breast",
            "Abdomen / GI",
            "Pelvis / prostate / gynaecology",
            "Paediatrics",
            "Multi-site",
            "Other / type your own",
        ]
        clinical_focus_choice = st.selectbox(
            "Clinical focus / treatment site",
            options=clinical_focus_options,
            key="clinical_project_focus_choice",
        )
        if clinical_focus_choice == "Other / type your own":
            clinical_focus = st.text_input(
                "Type clinical focus / treatment site",
                value=st.session_state.get("clinical_project_focus_input", ""),
                key="clinical_project_focus_input",
                placeholder="Example: Brain / CNS, Head and neck, Thorax",
            )
        else:
            clinical_focus = clinical_focus_choice

        outcome_options = [
            "Neurocognitive decline",
            "Acute toxicity",
            "Late toxicity",
            "Overall survival",
            "Progression-free survival",
            "Local control",
            "Treatment response",
            "Quality of life / PROMs",
            "Hospital admission / treatment interruption",
            "Other / type your own",
        ]
        outcome_choice = st.selectbox(
            "Outcome of interest",
            options=outcome_options,
            key="clinical_project_outcome_choice",
        )
        if outcome_choice == "Other / type your own":
            outcome_interest = st.text_input(
                "Type outcome of interest",
                value=st.session_state.get("clinical_project_outcome_input", ""),
                key="clinical_project_outcome_input",
                placeholder="Example: neurocognitive decline, toxicity, survival",
            )
        else:
            outcome_interest = outcome_choice

        description_templates = {
            "Build from selected fields": "",
            "Prediction model development": "This project develops and evaluates a clinical prediction model using patient-level clinical, treatment and outcome data.",
            "Radiotherapy toxicity model": "This project evaluates clinical and treatment-related predictors of radiotherapy toxicity and supports model development, validation and comparison.",
            "Neurocognitive outcome model": "This project evaluates predictors of neurocognitive outcome following radiotherapy, with emphasis on treatment-related and patient-level risk factors.",
            "External validation project": "This project validates an existing prediction model using local clinical data and compares performance with published or external validation results.",
            "Other / type your own": "",
        }
        description_choice = st.selectbox(
            "Short description template",
            options=list(description_templates.keys()),
            key="clinical_project_description_choice",
        )
        auto_description = description_templates.get(description_choice, "")
        if description_choice == "Build from selected fields":
            auto_description = (
                f"This project focuses on {clinical_focus.lower()} and evaluates {outcome_interest.lower()} using patient-level clinical data. "
                "The project supports clinical model development, comparison and validation."
            )
        short_description = st.text_area(
            "Short description",
            value=auto_description,
            height=110,
            key=f"clinical_project_description_input_{make_safe_column_name(description_choice)}",
        )

        st.markdown("### Save project location")
        col_path, col_browse = st.columns([0.78, 0.22])
        with col_path:
            parent_folder = st.text_input(
                "Save project inside folder",
                key="clinical_project_root_folder_input",
                help="Paste a Windows folder path, or click Browse to choose where the new clinical project folder will be created.",
            )
        with col_browse:
            st.write("")
            st.write("")
            if st.button("Browse", use_container_width=True, key="browse_new_clinical_project_parent"):
                selected = browse_for_clinical_folder("Select folder where the new clinical project will be saved", mustexist=True)
                if selected:
                    st.session_state.clinical_project_root_folder_selected = selected
                    st.rerun()

        preview_name = clinical_safe_folder_name(project_name)
        preview_folder = Path(parent_folder).expanduser() / preview_name if parent_folder else Path(preview_name)
        st.caption("New project will be created here:")
        st.code(str(preview_folder), language=None)

        if st.button("Create clinical project", type="primary", use_container_width=True, key="create_clinical_project_button"):
            try:
                if not str(parent_folder).strip():
                    raise ValueError("Select or paste the folder where the project should be saved.")
                safe_project_name = clinical_safe_folder_name(project_name)
                project_folder = Path(parent_folder).expanduser() / safe_project_name
                create_clinical_project_structure(project_folder)
                setup = {
                    "Project name": safe_project_name,
                    "Project folder": str(project_folder),
                    "Clinical focus": clinical_focus,
                    "Outcome of interest": outcome_interest,
                    "Short description": short_description,
                }
                st.session_state.clinical_project_folder = str(project_folder)
                st.session_state.clinical_project_root_folder = str(parent_folder)
                st.session_state.clinical_project_setup = setup
                note_path = write_clinical_project_setup_note(project_folder, setup)
                st.success(f"Clinical project created: {project_folder}")
                st.caption(f"Setup note saved: {note_path}")
            except Exception as error:
                st.error(f"Could not create project: {error}")

    else:
        st.markdown("### Open existing clinical project")

        if "clinical_existing_project_folder_input" not in st.session_state:
            st.session_state.clinical_existing_project_folder_input = st.session_state.get("clinical_project_folder", "")
        if "clinical_existing_project_folder_selected" not in st.session_state:
            st.session_state.clinical_existing_project_folder_selected = ""
        if "clinical_existing_project_folder_selected_applied" not in st.session_state:
            st.session_state.clinical_existing_project_folder_selected_applied = ""

        selected_existing = st.session_state.get("clinical_existing_project_folder_selected", "")
        applied_existing = st.session_state.get("clinical_existing_project_folder_selected_applied", "")
        if selected_existing and selected_existing != applied_existing:
            st.session_state.clinical_existing_project_folder_input = selected_existing
            st.session_state.clinical_existing_project_folder_selected_applied = selected_existing

        col_existing, col_existing_browse = st.columns([0.78, 0.22])
        with col_existing:
            existing_folder = st.text_input(
                "Existing clinical project folder",
                key="clinical_existing_project_folder_input",
                help="Paste the folder path of an existing BrainRT clinical project, or click Browse.",
            )
        with col_existing_browse:
            st.write("")
            st.write("")
            if st.button("Browse", use_container_width=True, key="browse_existing_clinical_project"):
                selected = browse_for_clinical_folder("Select existing BrainRT clinical project folder", mustexist=True)
                if selected:
                    st.session_state.clinical_existing_project_folder_selected = selected
                    st.rerun()

        if st.button("Open clinical project", type="primary", use_container_width=True, key="open_clinical_project_button"):
            try:
                project_folder = restore_clinical_project_from_folder(existing_folder)
                st.success(f"Clinical project opened: {project_folder}")
                if st.session_state.df is not None:
                    st.info("Saved clinical dataset was restored from the project folder.")
            except Exception as error:
                st.error(f"Could not open project: {error}")

    active_folder = st.session_state.get("clinical_project_folder", "")
    if active_folder:
        st.divider()
        st.markdown("### Active clinical project")
        st.code(active_folder, language=None)
        st.markdown("**Project folders**")
        folder_df = pd.DataFrame({"Workflow folder": clinical_project_subfolders()})
        st.dataframe(folder_df, use_container_width=True, hide_index=True)

        if st.button("Next: Upload Excel →", type="primary", use_container_width=True, key="clinical_project_next_upload"):
            go_to("clinical_upload")
    else:
        st.info("Create or open a clinical project to start saving outputs locally.")

    st.divider()
    if st.button("← Back to Model Development", use_container_width=True, key="clinical_project_back_model_dev"):
        go_to("model_development_home")

# ============================================================
# CLINICAL MODULE - STEP 1 UPLOAD EXCEL
# ============================================================
elif st.session_state.page == "clinical_upload":
    st.header("Clinical Module")
    st.subheader("Step 1: Upload clinical Excel file")

    active_clinical_project = st.session_state.get("clinical_project_folder", "")
    if active_clinical_project:
        st.caption(f"Active clinical project: {active_clinical_project}")
    else:
        st.warning("No active clinical project. Outputs will be shown in the app but not saved to a project folder until you create/open one.")

    st.write(
        "Upload the patient-level clinical dataset. After upload, the app will show a data preview "
        "and run a general QC check before you move to variable selection."
    )

    uploaded_file = st.file_uploader(
        "Upload Excel file",
        type=["xlsx", "xls"],
        key="clinical_excel_upload"
    )

    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            uploaded_bytes = uploaded_file.getvalue()
            st.session_state.clinical_uploaded_file_bytes = uploaded_bytes
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
            df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
            st.session_state.df = df
            st.session_state.clinical_uploaded_filename = uploaded_file.name
            reset_trained_model()

            st.success("Excel file loaded successfully. Review the preview/QC, then click Save this step to project at the end of the page.")

            st.subheader("Preview of uploaded data")
            st.caption("Showing the first 10 rows. Full variable selection is handled in Step 2.")
            st.dataframe(df.head(10), use_container_width=True)

            render_clinical_excel_qc(df)

        except Exception as error:
            st.error(f"Could not read the uploaded Excel file: {error}")
            st.session_state.df = None

    else:
        if st.session_state.df is not None:
            restored_path = st.session_state.get("clinical_restored_dataset_path", "")
            saved_folder = st.session_state.get("clinical_upload_saved_folder", "")
            uploaded_name = st.session_state.get("clinical_uploaded_filename", "Previously saved clinical dataset")

            st.success("A clinical dataset is already loaded from the active project.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", int(st.session_state.df.shape[0]))
            c2.metric("Columns", int(st.session_state.df.shape[1]))
            c3.metric("Source", uploaded_name)

            if restored_path:
                st.caption(f"Restored dataset: {restored_path}")
            elif saved_folder:
                st.caption(f"Saved Step 1 folder: {saved_folder}")

            with st.expander("Preview saved clinical dataset", expanded=True):
                st.dataframe(st.session_state.df.head(10), use_container_width=True)

            with st.expander("Re-run QC on saved clinical dataset", expanded=False):
                render_clinical_excel_qc(st.session_state.df)

            st.info("Upload a new Excel file only if you want to replace the current project dataset. Then click Save this step to clinical project to overwrite the saved Step 1 files.")
        else:
            st.info("Upload an Excel file to preview the dataset and run QC.")

    st.divider()

    st.markdown("### Save Step 1")
    save_step1_label = "💾 Save / overwrite Step 1 in clinical project" if st.session_state.df is not None else "💾 Save this step to clinical project"
    if st.button(save_step1_label, disabled=st.session_state.df is None, use_container_width=True, key="clinical_save_step1_upload"):
        saved_folder = save_clinical_upload_outputs(
            st.session_state.df,
            st.session_state.get("clinical_uploaded_filename", "clinical_data.xlsx"),
            st.session_state.get("clinical_uploaded_file_bytes", None),
        )
        if saved_folder is not None:
            st.success(f"Step 1 saved to: {saved_folder}")
        else:
            st.warning("Create or open a clinical project first, then save this step.")

    col_back, col_forward = st.columns(2)

    with col_back:
        if st.button("← Back to Start / Open project", use_container_width=True):
            go_to("clinical_start_project")

    with col_forward:
        if st.button(
            "Forward to variable selection →",
            disabled=st.session_state.df is None,
            use_container_width=True
        ):
            go_to("clinical_variables")


# ============================================================
# PAGE 2: OUTCOME MAPPING TABLE + DERIVE DECLINE
# ============================================================

# ============================================================
# CLINICAL MODULE - STEP 2 VARIABLES & OUTCOMES
# ============================================================
elif st.session_state.page == "clinical_variables":
    if st.session_state.df is None:
        st.warning("Please upload an Excel file first.")
        if st.button("Go to upload page"):
            go_to("clinical_upload")
        st.stop()

    df = st.session_state.df
    columns = get_columns()
    column_options = [""] + columns

    st.header("Clinical Model")
    st.subheader("Step 2: Define baseline and follow-up outcomes")

    left_col, right_col = st.columns([2.4, 1])

    with right_col:
        st.markdown("### Settings")

        input_variables = st.multiselect(
            "Select input predictors",
            options=columns,
            default=[
                col for col in st.session_state.input_variables
                if col in columns
            ],
            help="These are predictors such as age, sex, tumour features, dose variables, treatment variables, biomarkers, etc."
        )

        decline_threshold = st.number_input(
            "Decline threshold",
            min_value=0.1,
            max_value=30.0,
            value=float(st.session_state.decline_threshold),
            step=0.5,
            help="A patient is classified as declined if follow-up minus baseline is less than or equal to minus this threshold."
        )

    with left_col:
        st.markdown("### A. Outcome mapping table")
        st.write(
            "Map each baseline variable to one or more post-treatment time points. "
            "Each row can represent a different cognitive test/domain."
        )

        previous_mapping = st.session_state.get("outcome_mapping_rows", [])

        if previous_mapping:
            mapping_df = pd.DataFrame(previous_mapping)
        else:
            mapping_df = pd.DataFrame([
                {
                    "Baseline Variable": "",
                    "Follow-up 1": "",
                    "Follow-up 2": "",
                    "Follow-up 3": "",
                }
            ])

        for col in ["Baseline Variable", "Follow-up 1", "Follow-up 2", "Follow-up 3"]:
            if col not in mapping_df.columns:
                mapping_df[col] = ""

        mapping_df = mapping_df[["Baseline Variable", "Follow-up 1", "Follow-up 2", "Follow-up 3"]]

        edited_mapping_df = st.data_editor(
            mapping_df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=False,
            column_config={
                "Baseline Variable": st.column_config.SelectboxColumn(
                    "Baseline Variable",
                    options=column_options,
                    required=False,
                ),
                "Follow-up 1": st.column_config.SelectboxColumn(
                    "Follow-up 1",
                    options=column_options,
                    required=False,
                ),
                "Follow-up 2": st.column_config.SelectboxColumn(
                    "Follow-up 2",
                    options=column_options,
                    required=False,
                ),
                "Follow-up 3": st.column_config.SelectboxColumn(
                    "Follow-up 3",
                    options=column_options,
                    required=False,
                ),
            },
            key="outcome_mapping_editor"
        )

        clean_mapping_rows = []
        selected_baselines = []
        selected_followups = []

        for _, row in edited_mapping_df.fillna("").iterrows():
            row_dict = {
                "Baseline Variable": str(row.get("Baseline Variable", "")).strip(),
                "Follow-up 1": str(row.get("Follow-up 1", "")).strip(),
                "Follow-up 2": str(row.get("Follow-up 2", "")).strip(),
                "Follow-up 3": str(row.get("Follow-up 3", "")).strip(),
            }

            baseline = row_dict["Baseline Variable"]
            followups = [row_dict["Follow-up 1"], row_dict["Follow-up 2"], row_dict["Follow-up 3"]]
            followups = [f for f in followups if f != ""]

            if baseline != "" or len(followups) > 0:
                clean_mapping_rows.append(row_dict)

            if baseline != "":
                selected_baselines.append(baseline)

            selected_followups.extend(followups)

        selected_baselines = list(dict.fromkeys(selected_baselines))
        selected_followups = list(dict.fromkeys(selected_followups))

        mapping_errors = []

        for row_number, row in enumerate(clean_mapping_rows, start=1):
            baseline = row["Baseline Variable"]
            followups = [row["Follow-up 1"], row["Follow-up 2"], row["Follow-up 3"]]
            followups = [f for f in followups if f != ""]

            if baseline == "" and len(followups) > 0:
                mapping_errors.append(f"Row {row_number}: select a baseline variable.")

            if baseline != "" and len(followups) == 0:
                mapping_errors.append(f"Row {row_number}: select at least one follow-up variable.")

            if baseline in followups:
                mapping_errors.append(f"Row {row_number}: baseline and follow-up must be different.")

        repeated_predictors = [
            col for col in input_variables
            if col in selected_baselines or col in selected_followups
        ]

        if repeated_predictors:
            mapping_errors.append(
                "These outcome variables are also selected as predictors. Remove them from predictors: "
                + ", ".join(repeated_predictors)
            )

        selection_complete = (
            len(input_variables) > 0
            and len(clean_mapping_rows) > 0
            and len(selected_baselines) > 0
            and len(selected_followups) > 0
            and len(mapping_errors) == 0
        )

        if mapping_errors:
            for error in mapping_errors:
                st.warning(error)

        if selection_complete:
            derived_df, generated_changes, generated_outcomes, generated_summary = derive_declines_from_mapping(
                df,
                clean_mapping_rows,
                decline_threshold
            )

            if generated_summary.empty:
                st.warning("No change variables were generated. Check the mapping table.")
                selection_complete = False
            else:
                st.success("Change and decline outcomes created successfully.")

                st.markdown("### B. Generated change variables")
                st.dataframe(generated_summary, use_container_width=True)

                primary_options = generated_summary["Decline Variable"].tolist()
                previous_primary = st.session_state.get(
                    "primary_derived_outcome",
                    primary_options[0] if primary_options else ""
                )

                primary_index = (
                    primary_options.index(previous_primary)
                    if previous_primary in primary_options
                    else 0
                )

                primary_derived_outcome = st.selectbox(
                    "Primary modeling outcome",
                    options=primary_options,
                    index=primary_index,
                    help="This decline variable will become the default dependent variable for regression and machine learning."
                )

                st.markdown("### Primary derived outcome summary")
                st.write("0 = no decline")
                st.write("1 = neurocognitive decline")

                decline_counts = (
                    derived_df[primary_derived_outcome]
                    .value_counts(dropna=False)
                    .reset_index()
                )
                decline_counts.columns = ["Derived decline status", "Count"]
                st.dataframe(decline_counts)

                preview_columns = list(dict.fromkeys(
                    input_variables
                    + selected_baselines
                    + selected_followups
                    + generated_changes
                    + generated_outcomes
                    + ["Cognitive_Change", "Derived_Neurocognitive_Decline"]
                ))

                st.markdown("### Preview")
                safe_preview_columns = existing_columns(derived_df, preview_columns)

                if safe_preview_columns:
                    st.dataframe(derived_df[safe_preview_columns].head(), use_container_width=True)
                else:
                    st.info("No preview columns are available in the current uploaded dataset.")
        else:
            primary_derived_outcome = ""
            st.info(
                "Complete the mapping table: select at least one predictor, one baseline variable, "
                "and one or more follow-up variables."
            )

    st.divider()

    st.markdown("### Save Step 2")
    if st.button("💾 Save this step to clinical project", disabled=not selection_complete, use_container_width=True, key="clinical_save_step2_variables"):
        derived_df, generated_changes, generated_outcomes, generated_summary = derive_declines_from_mapping(
            df,
            clean_mapping_rows,
            decline_threshold
        )
        if primary_derived_outcome != "" and primary_derived_outcome in derived_df.columns:
            derived_df["Derived_Neurocognitive_Decline"] = derived_df[primary_derived_outcome]
        st.session_state.input_variables = input_variables
        st.session_state.outcome_mapping_rows = clean_mapping_rows
        st.session_state.baseline_variables = selected_baselines
        st.session_state.followup_variables = selected_followups
        st.session_state.decline_threshold = decline_threshold
        st.session_state.generated_change_columns = generated_changes
        st.session_state.generated_decline_columns = generated_outcomes
        st.session_state.generated_outcome_mapping_table = generated_summary
        st.session_state.primary_derived_outcome = primary_derived_outcome
        st.session_state.selected_outcome_variable = primary_derived_outcome
        st.session_state.df = derived_df
        save_folder = save_clinical_variables_outputs(
            derived_df, input_variables, clean_mapping_rows, selected_baselines, selected_followups,
            decline_threshold, generated_changes, generated_outcomes, generated_summary, primary_derived_outcome
        )
        if save_folder is not None:
            st.success(f"Step 2 saved to: {save_folder}")
        else:
            st.warning("Create or open a clinical project first, then save this step.")

    col_back, col_forward = st.columns(2)

    with col_back:
        if st.button("← Back to upload", use_container_width=True):
            go_to("clinical_upload")

    with col_forward:
        if st.button(
            "Forward to treatment selection →",
            disabled=not selection_complete,
            use_container_width=True
        ):
            derived_df, generated_changes, generated_outcomes, generated_summary = derive_declines_from_mapping(
                df,
                clean_mapping_rows,
                decline_threshold
            )

            # Use selected primary outcome as compatibility/default outcome.
            if primary_derived_outcome != "" and primary_derived_outcome in derived_df.columns:
                derived_df["Derived_Neurocognitive_Decline"] = derived_df[primary_derived_outcome]

                match_row = generated_summary.loc[
                    generated_summary["Decline Variable"] == primary_derived_outcome
                ]
                if not match_row.empty:
                    change_col = match_row.iloc[0]["Change Variable"]
                    derived_df["Cognitive_Change"] = derived_df[change_col]
                    st.session_state.baseline_variable = match_row.iloc[0]["Baseline"]
                    st.session_state.followup_variable = match_row.iloc[0]["Follow-up"]
                    st.session_state.primary_followup_variable = match_row.iloc[0]["Follow-up"]

            st.session_state.input_variables = input_variables
            st.session_state.outcome_mapping_rows = clean_mapping_rows
            st.session_state.baseline_variables = selected_baselines
            st.session_state.followup_variables = selected_followups
            st.session_state.decline_threshold = decline_threshold
            st.session_state.generated_change_columns = generated_changes
            st.session_state.generated_decline_columns = generated_outcomes
            st.session_state.generated_outcome_mapping_table = generated_summary
            st.session_state.primary_derived_outcome = primary_derived_outcome
            st.session_state.selected_outcome_variable = primary_derived_outcome
            st.session_state.df = derived_df
            save_folder = save_clinical_variables_outputs(
                derived_df,
                input_variables,
                clean_mapping_rows,
                selected_baselines,
                selected_followups,
                decline_threshold,
                generated_changes,
                generated_outcomes,
                generated_summary,
                primary_derived_outcome,
            )
            if save_folder is not None:
                st.success(f"Variables and outcomes saved to: {save_folder}")
            reset_trained_model()
            go_to("clinical_treatment")


# ============================================================
# PAGE 3: SELECT TREATMENT COLUMN
# ============================================================

# ============================================================
# CLINICAL MODULE - STEP 3 TREATMENT GROUPS
# ============================================================
elif st.session_state.page == "clinical_treatment":
    if st.session_state.df is None:
        st.warning("Please upload an Excel file first.")
        if st.button("Go to upload page"):
            go_to("clinical_upload")
        st.stop()

    df = st.session_state.df
    columns = get_columns()

    st.header("Clinical Model")
    st.subheader("Step 3: Select treatment column")

    treatment_options_for_dropdown = [""] + columns
    treatment_index = (
        treatment_options_for_dropdown.index(st.session_state.treatment_variable)
        if st.session_state.treatment_variable in treatment_options_for_dropdown
        else 0
    )

    treatment_variable = st.selectbox(
        "Select treatment column",
        options=treatment_options_for_dropdown,
        index=treatment_index
    )

    detected_treatment_values = get_treatment_options(df, treatment_variable)

    selected_treatment_values = []

    if treatment_variable != "":
        st.success("Treatment column selected.")

        st.subheader("Automatically detected treatment options")
        st.write(detected_treatment_values)

        if len(detected_treatment_values) < 2:
            st.warning("This column has fewer than two treatment groups.")
        else:
            st.write("Number of treatment groups detected:", len(detected_treatment_values))

            treatment_counts = (
                df[treatment_variable]
                .dropna()
                .astype(str)
                .str.strip()
                .value_counts()
                .reset_index()
            )

            treatment_counts.columns = ["Treatment option", "Count"]

            st.subheader("Treatment group counts")
            st.dataframe(treatment_counts)

            st.subheader("Include / exclude treatment groups")

            previous_selection = st.session_state.get("treatment_options", detected_treatment_values)
            default_selection = [
                group for group in previous_selection
                if group in detected_treatment_values
            ]

            if len(default_selection) == 0:
                default_selection = detected_treatment_values

            selected_treatment_values = st.multiselect(
                "Treatment groups to include in the analysis",
                options=detected_treatment_values,
                default=default_selection,
                help="All detected treatment groups are selected automatically. Untick any group you want to exclude."
            )

            excluded_groups = [
                group for group in detected_treatment_values
                if group not in selected_treatment_values
            ]

            if len(excluded_groups) > 0:
                st.warning(f"Excluded groups: {excluded_groups}")
            else:
                st.info("All detected treatment groups are included.")

    else:
        st.info("Select the column that contains the treatment groups.")

    treatment_complete = treatment_variable != "" and len(selected_treatment_values) >= 2

    st.divider()

    st.markdown("### Save Step 3")
    if st.button("💾 Save this step to clinical project", disabled=not treatment_complete, use_container_width=True, key="clinical_save_step3_treatment"):
        st.session_state.treatment_variable = treatment_variable
        st.session_state.treatment_options = selected_treatment_values
        save_folder = save_clinical_treatment_outputs(df, treatment_variable, selected_treatment_values)
        if save_folder is not None:
            st.success(f"Step 3 saved to: {save_folder}")
        else:
            st.warning("Create or open a clinical project first, then save this step.")

    col_back, col_forward = st.columns(2)

    with col_back:
        if st.button("← Back to decline definition", use_container_width=True):
            go_to("clinical_variables")

    with col_forward:
        if st.button(
            "Forward to statistics →",
            disabled=not treatment_complete,
            use_container_width=True
        ):
            st.session_state.treatment_variable = treatment_variable
            st.session_state.treatment_options = selected_treatment_values
            save_folder = save_clinical_treatment_outputs(df, treatment_variable, selected_treatment_values)
            if save_folder is not None:
                st.success(f"Treatment settings saved to: {save_folder}")
            reset_trained_model()
            go_to("clinical_analysis")


# ============================================================
# PAGE 4: STATISTICS
# ============================================================

# ============================================================
# CLINICAL MODULE - STEP 4 STATISTICS LANDING
# ============================================================
elif st.session_state.page == "clinical_analysis":
    if st.session_state.df is None:
        st.warning("Please upload an Excel file first.")
        if st.button("Go to upload page"):
            go_to("clinical_upload")
        st.stop()

    st.header("Clinical Model")
    st.subheader("Step 4: Statistics")

    st.write(
        "Choose a statistical analysis pathway. Machine learning is kept separate in Step 5."
    )

    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)

    with c1:
        st.markdown("### 4A. Descriptive statistics")
        st.write("Summarise clinical variables, treatment groups, outcomes, and distributions.")
        st.caption("Includes mean ± SD, median (IQR), % (n), normality testing, and graphs.")
        if st.button("Open descriptive statistics", use_container_width=True):
            go_to("clinical_descriptive")

    with c2:
        st.markdown("### 4B. Inferential statistics")
        st.write("Compare variables between treatment groups using appropriate statistical tests.")
        st.caption("Uses normality results to guide parametric or non-parametric testing.")
        if st.button("Open inferential statistics", use_container_width=True):
            go_to("clinical_inferential")

    with c3:
        st.markdown("### 4C. Timepoint analysis")
        st.write("Analyse baseline-to-follow-up changes across one or more post-treatment timepoints.")
        st.caption("For paired pre/post analysis, longitudinal change, and treatment differences over time.")
        if st.button("Open timepoint analysis", use_container_width=True):
            go_to("clinical_timepoint")

    with c4:
        st.markdown("### 4D. Domain-specific decline proportion")
        st.write("Compare the proportion of patients with decline in each domain across treatment groups.")
        st.caption("Useful for domain-level decline comparisons, for example memory, attention, language and orientation.")
        if st.button("Open domain-specific decline proportion", use_container_width=True):
            go_to("clinical_spider_chart")

    st.divider()

    col_back, col_forward = st.columns(2)

    with col_back:
        if st.button("← Back to treatment selection", use_container_width=True):
            go_to("clinical_treatment")

    with col_forward:
        if st.button("Forward to machine learning →", use_container_width=True):
            go_to("clinical_model_selection")



# ============================================================
# PAGE 4D: SPIDER CHART / EPROMs DOMAIN PROFILE
# ============================================================

elif st.session_state.page == "clinical_spider_chart":
    if st.session_state.df is None:
        st.warning("Please upload an Excel file first.")
        if st.button("Go to upload page"):
            go_to("clinical_upload")
        st.stop()

    df = st.session_state.df.copy()

    st.header("Clinical Model")
    st.subheader("Step 4D: Domain-specific decline proportion")

    st.write(
        "Compare the proportion of patients with decline in each selected domain across treatment groups."
    )

    st.info(
        "This is usually easier to interpret than a spider chart when comparing treatment groups."
    )

    numeric_columns = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]

    if len(numeric_columns) < 2:
        st.warning("At least two numeric columns are needed.")
    else:
        def clean_domain_label(label):
            text = str(label)
            for token in ["baseline", "followup", "follow_up", "follow-up", "3m", "6m", "12m", "score", "domain"]:
                text = text.replace(token, "").replace(token.upper(), "").replace(token.capitalize(), "")
            text = text.replace("_", " ").replace("-", " ")
            text = " ".join(text.split())
            return text if text else str(label)

        st.markdown("### 1. Select baseline domains")
        suggested_domains = [
            col for col in numeric_columns
            if any(
                keyword in str(col).lower()
                for keyword in [
                    "memory", "verbal", "language", "linguistic", "executive",
                    "attention", "processing", "speed", "physical",
                    "emotional", "social", "cognitive", "quality",
                    "qol", "global", "function", "domain", "moca", "naming",
                    "abstraction", "orientation", "recall", "visuospatial"
                ]
            )
        ]
        default_domains = suggested_domains[:8] if len(suggested_domains) >= 1 else numeric_columns[:6]

        baseline_domains = st.multiselect(
            "Select baseline domain variables",
            options=numeric_columns,
            default=default_domains,
            key="decline_baseline_domains"
        )

        st.markdown("### 2. Match follow-up domains")
        st.caption("For each selected baseline domain, choose the matching follow-up variable.")

        followup_mapping = {}
        for baseline_col in baseline_domains:
            cleaned_baseline = clean_domain_label(baseline_col).lower()
            likely_matches = [
                col for col in numeric_columns
                if col != baseline_col and (
                    cleaned_baseline in clean_domain_label(col).lower()
                    or clean_domain_label(col).lower() in cleaned_baseline
                )
            ]
            options = ["Do not include"] + numeric_columns
            default_index = options.index(likely_matches[0]) if likely_matches and likely_matches[0] in options else 0

            followup_mapping[baseline_col] = st.selectbox(
                f"Follow-up variable for: {baseline_col}",
                options=options,
                index=default_index,
                key=f"decline_followup_for_{baseline_col}"
            )

        valid_pairs = {
            baseline: followup
            for baseline, followup in followup_mapping.items()
            if followup != "Do not include"
        }

        st.markdown("### 3. Decline definition")
        decline_mode = st.radio(
            "How should decline be defined?",
            options=["Absolute score drop", "Percentage drop from baseline"],
            horizontal=True,
            key="decline_mode"
        )

        if decline_mode == "Absolute score drop":
            decline_threshold = st.number_input(
                "Patient is classified as declined if follow-up - baseline is less than or equal to:",
                value=-1.0,
                step=1.0,
                key="decline_threshold_absolute"
            )
        else:
            decline_threshold = st.number_input(
                "Patient is classified as declined if percentage change is less than or equal to:",
                value=-10.0,
                step=5.0,
                key="decline_threshold_percent"
            )

        st.markdown("### 4. Group comparison settings")
        candidate_group_columns = [
            col for col in df.columns
            if df[col].nunique(dropna=True) <= 12 and col not in list(valid_pairs.keys()) + list(valid_pairs.values())
        ]
        treatment_variable = st.session_state.get("treatment_variable", "")
        group_options = ["Use treatment selection"] + candidate_group_columns

        default_group_index = 0
        if treatment_variable and treatment_variable in group_options:
            default_group_index = group_options.index(treatment_variable)

        group_variable_choice = st.selectbox(
            "Grouping variable",
            options=group_options,
            index=default_group_index,
            key="decline_group_variable_choice"
        )

        if group_variable_choice == "Use treatment selection":
            group_variable = st.session_state.get("treatment_variable", "")
        else:
            group_variable = group_variable_choice

        if not group_variable or group_variable not in df.columns:
            st.warning("Please ensure a valid treatment/group variable is selected in Step 3.")
        elif len(valid_pairs) < 1:
            st.warning("Select at least one baseline/follow-up domain pair.")
        else:
            group_values = df[group_variable].dropna().astype(str)
            available_groups = sorted(group_values.unique().tolist())

            selected_groups = st.multiselect(
                "Groups to compare",
                options=available_groups,
                default=available_groups,
                key="decline_selected_groups"
            )

            if len(selected_groups) == 0:
                st.warning("Select at least one group.")
            else:
                summary_rows = []

                for baseline_col, followup_col in valid_pairs.items():
                    baseline_values = pd.to_numeric(df[baseline_col], errors="coerce")
                    followup_values = pd.to_numeric(df[followup_col], errors="coerce")

                    if decline_mode == "Absolute score drop":
                        change_values = followup_values - baseline_values
                        declined = change_values <= decline_threshold
                    else:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            percent_change = ((followup_values - baseline_values) / baseline_values.replace(0, np.nan)) * 100.0
                        declined = percent_change <= decline_threshold

                    for group in selected_groups:
                        group_mask = df[group_variable].astype(str) == str(group)
                        valid_mask = group_mask & baseline_values.notna() & followup_values.notna()

                        n_valid = int(valid_mask.sum())
                        if n_valid == 0:
                            decline_percent = np.nan
                            declined_n = 0
                        else:
                            declined_n = int(declined[valid_mask].fillna(False).sum())
                            decline_percent = 100.0 * declined_n / n_valid

                        summary_rows.append({
                            "Domain": clean_domain_label(baseline_col),
                            "Baseline variable": baseline_col,
                            "Follow-up variable": followup_col,
                            "Group": str(group),
                            "Patients with data": n_valid,
                            "Declined (n)": declined_n,
                            "Declined (%)": decline_percent,
                        })

                summary_df = pd.DataFrame(summary_rows)

                if not summary_df.empty:
                    # Make display domain labels unique if two selected variables clean to the same name.
                    # This avoids duplicate domain/group combinations in the plot.
                    domain_counts = {}
                    unique_domains = []

                    for _, row in summary_df.iterrows():
                        domain = str(row["Domain"])
                        baseline_var = str(row["Baseline variable"])

                        key = (domain, baseline_var)

                        if domain not in domain_counts:
                            domain_counts[domain] = set()

                        domain_counts[domain].add(baseline_var)

                    duplicated_display_domains = {
                        domain for domain, baseline_vars in domain_counts.items()
                        if len(baseline_vars) > 1
                    }

                    for _, row in summary_df.iterrows():
                        domain = str(row["Domain"])
                        baseline_var = str(row["Baseline variable"])

                        if domain in duplicated_display_domains:
                            unique_domains.append(f"{domain} ({baseline_var})")
                        else:
                            unique_domains.append(domain)

                    summary_df["Domain"] = unique_domains

                if summary_df.empty:
                    st.warning("No valid results could be calculated.")
                else:
                    st.markdown("### Summary table")
                    st.dataframe(summary_df, use_container_width=True)

                    # Use pivot_table rather than pivot so duplicate Domain + Group rows do not crash the app.
                    # Duplicates can happen when several variables clean to the same display name.
                    plot_df = summary_df.pivot_table(
                        index="Domain",
                        columns="Group",
                        values="Declined (%)",
                        aggfunc="mean"
                    ).fillna(0)

                    st.markdown("### Domain-specific decline proportion plot")
                    fig, ax = plt.subplots(figsize=(9, 5))

                    x = np.arange(len(plot_df.index))
                    n_groups = max(1, len(plot_df.columns))
                    total_width = 0.8
                    bar_width = total_width / n_groups

                    for i, group in enumerate(plot_df.columns):
                        positions = x - total_width / 2 + (i + 0.5) * bar_width
                        ax.bar(positions, plot_df[group].values, width=bar_width, label=str(group))

                    ax.set_xticks(x)
                    ax.set_xticklabels(plot_df.index.tolist(), rotation=35, ha="right")
                    ax.set_ylabel("Patients with decline (%)")
                    ax.set_ylim(0, 100)
                    ax.set_title("Domain-specific decline proportion by group")
                    ax.legend(title=group_variable)
                    st.pyplot(fig)
                    plt.close(fig)

                    st.caption(
                        "Decline is defined using the threshold selected above. "
                        "This plot compares the percentage of patients with decline in each domain across groups."
                    )

    st.divider()

    col_back, col_forward = st.columns(2)

    with col_back:
        if st.button("← Back to statistics", use_container_width=True):
            go_to("clinical_analysis")

    with col_forward:
        if st.button("Forward to machine learning →", use_container_width=True):
            go_to("clinical_model_selection")


# ============================================================
# PAGE 4A: DESCRIPTIVE STATISTICS
# ============================================================

# ============================================================
# CLINICAL MODULE - STEP 4A DESCRIPTIVE STATISTICS
# ============================================================
elif st.session_state.page == "clinical_descriptive":
    if st.session_state.df is None:
        st.warning("Please upload an Excel file first.")
        if st.button("Go to upload page"):
            go_to("clinical_upload")
        st.stop()

    df = st.session_state.df

    st.header("Clinical Model")
    st.subheader("Step 4A: Descriptive statistics")

    default_vars = list(dict.fromkeys(
        st.session_state.input_variables
        + [st.session_state.baseline_variable]
        + st.session_state.get("followup_variables", [])
        + st.session_state.get("generated_change_columns", [])
        + st.session_state.get("generated_decline_columns", [])
    ))

    descriptive_variables = st.multiselect(
        "Select variables to describe",
        options=list(df.columns),
        default=[col for col in default_vars if col in df.columns]
    )

    group_by_treatment = st.checkbox(
        "Show summaries by treatment group",
        value=st.session_state.treatment_variable != ""
    )

    if st.button("Run descriptive statistics", use_container_width=True):
        rows = []
        treatment_variable = st.session_state.treatment_variable
        treatment_groups = st.session_state.treatment_options

        if group_by_treatment and treatment_variable != "":
            for variable in descriptive_variables:
                is_numeric = is_numeric_column(df, variable)

                for group in treatment_groups:
                    group_data = df.loc[
                        df[treatment_variable].astype(str).str.strip() == group,
                        variable
                    ]

                    if is_numeric:
                        summary = summarize_numeric_extended(group_data)
                        row = {
                            "Variable": variable,
                            "Group": group,
                            "Type": "Numeric",
                            "N non-missing": summary["N non-missing"],
                            "Mean ± SD": summary["Mean ± SD"],
                            "Median (IQR)": summary["Median (IQR)"],
                            "Normality test": summary["Normality test"],
                            "Normality p-value": summary["Normality p-value"],
                            "Distribution": summary["Distribution"],
                            "% (n)": "",
                        }
                    else:
                        summary = summarize_categorical_percent_n(group_data)
                        row = {
                            "Variable": variable,
                            "Group": group,
                            "Type": "Categorical",
                            "N non-missing": summary["N non-missing"],
                            "Mean ± SD": "",
                            "Median (IQR)": "",
                            "Normality test": "",
                            "Normality p-value": "",
                            "Distribution": "",
                            "% (n)": summary["% (n)"],
                        }

                    rows.append(row)
        else:
            for variable in descriptive_variables:
                is_numeric = is_numeric_column(df, variable)

                if is_numeric:
                    summary = summarize_numeric_extended(df[variable])
                    row = {
                        "Variable": variable,
                        "Group": "Overall",
                        "Type": "Numeric",
                        "N non-missing": summary["N non-missing"],
                        "Mean ± SD": summary["Mean ± SD"],
                        "Median (IQR)": summary["Median (IQR)"],
                        "Normality test": summary["Normality test"],
                        "Normality p-value": summary["Normality p-value"],
                        "Distribution": summary["Distribution"],
                        "% (n)": "",
                    }
                else:
                    summary = summarize_categorical_percent_n(df[variable])
                    row = {
                        "Variable": variable,
                        "Group": "Overall",
                        "Type": "Categorical",
                        "N non-missing": summary["N non-missing"],
                        "Mean ± SD": "",
                        "Median (IQR)": "",
                        "Normality test": "",
                        "Normality p-value": "",
                        "Distribution": "",
                        "% (n)": summary["% (n)"],
                    }

                rows.append(row)

        results_df = pd.DataFrame(rows)

        desired_order = [
            "Variable",
            "Group",
            "N non-missing",
            "Mean ± SD",
            "Median (IQR)",
            "% (n)",
            "Type",
            "Normality test",
            "Normality p-value",
            "Distribution",
        ]
        results_df = results_df[[col for col in desired_order if col in results_df.columns]]

        st.subheader("Descriptive statistics table")
        st.caption(
            "Numeric variables show mean ± SD, median (IQR), and a normality test. "
            "Categorical variables show % (n)."
        )
        st.dataframe(results_df, use_container_width=True)
        st.session_state.descriptive_results_df = results_df.copy()
        saved_path = save_clinical_statistics_output(
            results_df,
            "descriptive_statistics.csv",
            settings={
                "descriptive_variables": list(descriptive_variables),
                "group_by_treatment": bool(group_by_treatment),
                "treatment_variable": st.session_state.get("treatment_variable", ""),
            },
        )
        if saved_path is not None:
            st.success(f"Descriptive statistics saved to: {saved_path}")

        normality_tests = sorted([
            test for test in results_df.get("Normality test", pd.Series(dtype=str)).dropna().unique().tolist()
            if str(test).strip() not in ["", "Not available"]
        ])

        if len(normality_tests) > 0:
            st.markdown("### Normality test explanations")
            selected_normality_test = st.selectbox(
                "Click/select a normality test to see what it means",
                options=normality_tests
            )
            st.info(explain_statistical_test(selected_normality_test))

        st.download_button(
            "Download descriptive statistics as CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="descriptive_statistics.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.divider()
    st.subheader("Step 4A graphs")

    if len(descriptive_variables) == 0:
        st.info("Select variables above before plotting.")
    else:
        graph_variables = st.multiselect(
            "Select variables to plot",
            options=descriptive_variables,
            default=descriptive_variables[:1],
            key="step4a_graph_variables"
        )

        graph_type = st.selectbox(
            "Select graph type",
            options=[
                "Histogram",
                "Boxplot by treatment/group",
                "Mean with 95% CI by treatment/group",
                "Bar chart",
                "Grouped bar chart by treatment/group",
                "Percentage bar chart by treatment/group",
            ],
            key="step4a_graph_type"
        )

        if st.button("Generate Step 4A graphs", use_container_width=True):
            for variable in graph_variables:
                st.markdown(f"#### {variable}")
                render_variable_plot(
                    df=df,
                    variable=variable,
                    graph_type=graph_type,
                    treatment_variable=st.session_state.treatment_variable
                )

    st.divider()
    st.markdown("### Save Step 4A")
    if st.button("💾 Save this step to clinical project", use_container_width=True, key="clinical_save_step4a_descriptive"):
        results_df = st.session_state.get("descriptive_results_df", pd.DataFrame())
        if results_df is None or results_df.empty:
            st.warning("Run descriptive statistics first, then save this step.")
        else:
            saved_path = save_clinical_statistics_output(results_df, "descriptive_statistics.csv")
            if saved_path is not None:
                st.success(f"Step 4A saved to: {saved_path}")
            else:
                st.warning("Create or open a clinical project first, then save this step.")

    col_back, col_forward = st.columns(2)
    with col_back:
        if st.button("← Back to statistics", use_container_width=True):
            go_to("clinical_analysis")
    with col_forward:
        if st.button("Forward to inferential statistics →", use_container_width=True):
            go_to("clinical_inferential")


# ============================================================
# PAGE 4B: INFERENTIAL STATISTICS
# ============================================================

# ============================================================
# CLINICAL MODULE - STEP 4B INFERENTIAL STATISTICS
# ============================================================
elif st.session_state.page == "clinical_inferential":
    if st.session_state.df is None:
        st.warning("Please upload an Excel file first.")
        if st.button("Go to upload page"):
            go_to("clinical_upload")
        st.stop()

    if st.session_state.treatment_variable == "":
        st.warning("Please select a treatment column first.")
        if st.button("Go to treatment selection"):
            go_to("clinical_treatment")
        st.stop()

    df = st.session_state.df
    treatment_variable = st.session_state.treatment_variable
    treatment_groups = st.session_state.treatment_options

    st.header("Clinical Model")
    st.subheader("Step 4B: Inferential statistics")

    st.write("Treatment column:", treatment_variable)
    st.write("Treatment groups included:", treatment_groups)

    if "step4a_normality_map" in st.session_state:
        st.success("Using Step 4A normality results to choose tests.")
    else:
        st.warning("Step 4A normality results not found. Run Step 4A first for the cleanest workflow.")


    default_analysis_variables = list(dict.fromkeys(
        st.session_state.input_variables
        + [st.session_state.baseline_variable]
        + st.session_state.get("followup_variables", [])
        + st.session_state.get("generated_change_columns", [])
        + st.session_state.get("generated_decline_columns", [])
        + ["Cognitive_Change", "Derived_Neurocognitive_Decline"]
    ))

    analysis_variables = st.multiselect(
        "Select variables to compare between treatment groups",
        options=list(df.columns),
        default=[col for col in default_analysis_variables if col in df.columns]
    )

    if len(analysis_variables) == 0:
        st.info("Select at least one variable to analyse.")
    else:
        if st.button("Run inferential statistics", use_container_width=True):
            results = []

            for variable in analysis_variables:
                is_numeric = is_numeric_column(df, variable)

                if is_numeric:
                    test_name, p_value, normality_basis, normality_p_value, test_rationale = choose_numeric_group_test_from_step4a(
                        df,
                        variable,
                        treatment_variable,
                        treatment_groups
                    )
                    summary_shown = (
                        "Mean ± SD"
                        if test_name in ["Welch t-test", "One-way ANOVA"]
                        else "Median (IQR)"
                    )
                else:
                    test_name, p_value = run_categorical_group_test(
                        df,
                        variable,
                        treatment_variable
                    )
                    test_rationale = explain_test_choice(test_name)
                    summary_shown = "% (n)"
                    normality_basis = "Not applicable"
                    normality_p_value = "Not applicable"

                formatted_p = format_p_value(p_value)

                row = {
                    "Variable": variable,
                    "Summary shown": summary_shown,
                    "Normality basis": normality_basis,
                    "Normality p-value": normality_p_value,
                    "Test": test_name,
                    "p-value": formatted_p,
                    "Significant": "Yes" if p_value_is_significant(formatted_p) else "No",
                }

                for group in treatment_groups:
                    group_data = df.loc[
                        df[treatment_variable].astype(str).str.strip() == str(group),
                        variable
                    ]

                    if is_numeric:
                        row[group] = inferential_numeric_summary_for_test(group_data, test_name)
                    else:
                        row[group] = inferential_categorical_summary(group_data)

                # Store rationale privately in session state for the selector.
                row["_Why this test"] = test_rationale

                results.append(row)

            results_df = pd.DataFrame(results)

            st.session_state.inferential_results_df = results_df.copy()
            st.session_state.inferential_test_explanations = {
                row["Test"]: row["_Why this test"]
                for _, row in results_df.iterrows()
                if row["Test"] not in ["", "Not available"]
            }
            saved_path = save_clinical_statistics_output(
                results_df.drop(columns=["_Why this test"], errors="ignore"),
                "inferential_statistics.csv",
                settings={
                    "analysis_variables": list(analysis_variables),
                    "treatment_variable": treatment_variable,
                    "treatment_groups": list(treatment_groups),
                },
            )
            if saved_path is not None:
                st.success(f"Inferential statistics saved to: {saved_path}")

        if "inferential_results_df" in st.session_state:
            results_df = st.session_state.inferential_results_df.copy()

            display_df = results_df.drop(columns=["_Why this test"], errors="ignore")

            st.subheader("Treatment group comparison table")
            st.caption(
                "Numeric summaries depend on the selected test: mean ± SD for parametric tests, "
                "median (IQR) for non-parametric tests. Categorical variables are shown as % (n). "
                "Rows with p < 0.05 are bold."
            )

            styled_results = display_df.style.apply(style_significant_rows, axis=1)
            st.dataframe(styled_results, use_container_width=True)

            unique_tests = sorted([
                test for test in display_df["Test"].dropna().unique().tolist()
                if test not in ["", "Not available"]
            ])

            if len(unique_tests) > 0:
                st.markdown("### Test explanations")
                selected_test = st.selectbox(
                    "Click/select a test to see what it means and why it was chosen",
                    options=unique_tests,
                    key="inferential_test_explanation_selector"
                )

                st.info(explain_statistical_test(selected_test))

                rationale = st.session_state.get("inferential_test_explanations", {}).get(
                    selected_test,
                    explain_test_choice(selected_test)
                )
                st.success(rationale)

            st.download_button(
                "Download inferential statistics as CSV",
                data=display_df.to_csv(index=False).encode("utf-8"),
                file_name="inferential_statistics.csv",
                mime="text/csv",
                use_container_width=True
            )

    st.divider()
    st.subheader("Step 4B graphs")

    if len(analysis_variables) == 0:
        st.info("Select variables above before plotting.")
    else:
        inferential_graph_variables = st.multiselect(
            "Select variables to plot",
            options=analysis_variables,
            default=analysis_variables[:1],
            key="step4b_graph_variables"
        )

        inferential_graph_type = st.selectbox(
            "Select graph type",
            options=[
                "Boxplot by treatment/group",
                "Mean with 95% CI by treatment/group",
                "Grouped bar chart by treatment/group",
                "Percentage bar chart by treatment/group",
                "Histogram",
                "Bar chart",
            ],
            key="step4b_graph_type"
        )

        if st.button("Generate Step 4B graphs", use_container_width=True):
            for variable in inferential_graph_variables:
                st.markdown(f"#### {variable}")
                render_variable_plot(
                    df=df,
                    variable=variable,
                    graph_type=inferential_graph_type,
                    treatment_variable=treatment_variable
                )

    st.divider()
    st.markdown("### Save Step 4B")
    if st.button("💾 Save this step to clinical project", use_container_width=True, key="clinical_save_step4b_inferential"):
        results_df = st.session_state.get("inferential_results_df", pd.DataFrame())
        if results_df is None or results_df.empty:
            st.warning("Run inferential statistics first, then save this step.")
        else:
            display_df = results_df.drop(columns=["_Why this test"], errors="ignore")
            saved_path = save_clinical_statistics_output(display_df, "inferential_statistics.csv")
            if saved_path is not None:
                st.success(f"Step 4B saved to: {saved_path}")
            else:
                st.warning("Create or open a clinical project first, then save this step.")

    col_back, col_forward = st.columns(2)
    with col_back:
        if st.button("← Back to statistics", use_container_width=True):
            go_to("clinical_analysis")
    with col_forward:
        if st.button("Forward to timepoint analysis →", use_container_width=True):
            go_to("clinical_timepoint")


# ============================================================

# ============================================================
# PAGE 4C: TIMEPOINT ANALYSIS
# ============================================================

# ============================================================
# CLINICAL MODULE - STEP 4C TIMEPOINT ANALYSIS
# ============================================================
elif st.session_state.page == "clinical_timepoint":
    if st.session_state.df is None:
        st.warning("Please upload an Excel file first.")
        if st.button("Go to upload page"):
            go_to("clinical_upload")
        st.stop()

    if st.session_state.treatment_variable == "":
        st.warning("Please select a treatment column first.")
        if st.button("Go to treatment selection"):
            go_to("clinical_treatment")
        st.stop()

    df = st.session_state.df
    treatment_variable = st.session_state.treatment_variable
    treatment_groups = st.session_state.treatment_options

    st.header("Clinical Model")
    st.subheader("Step 4C: Timepoint analysis")

    st.write(
        "This section analyses change over time and compares that change between treatment groups."
    )

    columns = list(df.columns)

    st.markdown("### 1. Select baseline and follow-up timepoints")

    baseline_col = st.selectbox(
        "Select baseline variable",
        options=columns,
        index=columns.index(st.session_state.baseline_variable)
        if st.session_state.get("baseline_variable", "") in columns
        else 0,
        key="timepoint_baseline_col"
    )

    suggested_followups = [
        col for col in st.session_state.get("followup_variables", [])
        if col in columns
    ]

    followup_cols = st.multiselect(
        "Select follow-up timepoint variables",
        options=[col for col in columns if col != baseline_col],
        default=suggested_followups,
        key="timepoint_followup_cols"
    )

    patient_id_options = ["None"] + columns
    patient_id_col = st.selectbox(
        "Patient ID column for spaghetti plot",
        options=patient_id_options,
        index=0,
        key="timepoint_patient_id_col"
    )

    st.markdown("### 2. Select analyses to run")

    run_between_treatment = st.checkbox(
        "Compare change scores between treatment groups",
        value=True,
        help="For each follow-up: change = follow-up - baseline, then compare change between treatment groups."
    )

    run_within_treatment = st.checkbox(
        "Test within-patient pre/post change inside each treatment group",
        value=True,
        help="For each treatment group: baseline vs follow-up paired test."
    )

    st.markdown("### 3. Select plots")

    plot_options = st.multiselect(
        "Plots to generate",
        options=[
            "Mean score over time by treatment",
            "Change-score boxplot by treatment",
            "Individual patient spaghetti plot",
        ],
        default=[
            "Mean score over time by treatment",
            "Change-score boxplot by treatment",
        ],
        key="timepoint_plot_options"
    )

    if len(followup_cols) == 0:
        st.info("Select at least one follow-up timepoint variable.")
    else:
        if st.button("Run timepoint analysis", use_container_width=True):
            between_rows = []
            within_rows = []
            generated_change_cols = []

            for followup_col in followup_cols:
                change_col = f"Timepoint_Change_{make_safe_column_name(baseline_col)}_to_{make_safe_column_name(followup_col)}"
                df[change_col] = pd.to_numeric(df[followup_col], errors="coerce") - pd.to_numeric(df[baseline_col], errors="coerce")
                generated_change_cols.append(change_col)

                if run_between_treatment:
                    test_name, p_value, normality_label, normality_p = between_treatment_change_test(
                        df,
                        change_col,
                        treatment_variable,
                        treatment_groups
                    )

                    group_summaries = summarize_change_by_group(
                        df,
                        change_col,
                        treatment_variable,
                        treatment_groups,
                        normality_label
                    )

                    row = {
                        "Baseline": baseline_col,
                        "Follow-up": followup_col,
                        "Change variable": change_col,
                        "Comparison": "Change between treatment groups",
                        "Normality of change": normality_label,
                        "Normality p-value": format_p_value(normality_p),
                        "Test": test_name,
                        "p-value": format_p_value(p_value),
                        "Significant": "Yes" if p_value_is_significant(format_p_value(p_value)) else "No",
                    }

                    for summary_row in group_summaries:
                        row[f"{summary_row['Treatment']} N"] = summary_row["N"]
                        row[f"{summary_row['Treatment']} change"] = summary_row["Change summary"]

                    between_rows.append(row)

                if run_within_treatment:
                    for group in treatment_groups:
                        group_df = df[df[treatment_variable].astype(str).str.strip() == str(group).strip()]

                        test_name, p_value, normality_label, normality_p = paired_test_for_change(
                            group_df[baseline_col],
                            group_df[followup_col]
                        )

                        baseline_summary = format_mean_sd(group_df[baseline_col]) if normality_label == "Normal" else format_median_iqr(group_df[baseline_col])
                        followup_summary = format_mean_sd(group_df[followup_col]) if normality_label == "Normal" else format_median_iqr(group_df[followup_col])
                        change_summary = format_mean_sd(group_df[change_col]) if normality_label == "Normal" else format_median_iqr(group_df[change_col])

                        within_rows.append({
                            "Treatment": group,
                            "Baseline": baseline_col,
                            "Follow-up": followup_col,
                            "N paired": int(pd.DataFrame({
                                "baseline": pd.to_numeric(group_df[baseline_col], errors="coerce"),
                                "followup": pd.to_numeric(group_df[followup_col], errors="coerce"),
                            }).dropna().shape[0]),
                            "Baseline summary": baseline_summary,
                            "Follow-up summary": followup_summary,
                            "Change summary": change_summary,
                            "Normality of change": normality_label,
                            "Normality p-value": format_p_value(normality_p),
                            "Test": test_name,
                            "p-value": format_p_value(p_value),
                            "Significant": "Yes" if p_value_is_significant(format_p_value(p_value)) else "No",
                        })

            st.session_state.timepoint_analysis_df = df.copy()
            st.session_state.timepoint_selected_baseline_col = baseline_col
            st.session_state.timepoint_selected_followup_cols = list(followup_cols)
            st.session_state.timepoint_change_cols = generated_change_cols
            st.session_state.timepoint_between_results = pd.DataFrame(between_rows)
            st.session_state.timepoint_within_results = pd.DataFrame(within_rows)
            save_folder = save_clinical_timepoint_outputs(
                df.copy(),
                st.session_state.timepoint_between_results,
                st.session_state.timepoint_within_results,
                settings={
                    "baseline_col": baseline_col,
                    "followup_cols": list(followup_cols),
                    "patient_id_col": patient_id_col,
                    "run_between_treatment": bool(run_between_treatment),
                    "run_within_treatment": bool(run_within_treatment),
                    "generated_change_cols": list(generated_change_cols),
                },
            )
            if save_folder is not None:
                st.success(f"Timepoint analysis saved to: {save_folder}")

        if "timepoint_between_results" in st.session_state and run_between_treatment:
            st.markdown("## A. Change-score comparison between treatment groups")
            between_results = st.session_state.timepoint_between_results
            if between_results.empty:
                st.info("No between-treatment results were generated.")
            else:
                st.dataframe(
                    between_results.style.apply(style_significant_rows, axis=1),
                    use_container_width=True
                )

                st.download_button(
                    "Download between-treatment change results as CSV",
                    data=between_results.to_csv(index=False).encode("utf-8"),
                    file_name="timepoint_between_treatment_change.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        if "timepoint_within_results" in st.session_state and run_within_treatment:
            st.markdown("## B. Within-treatment paired pre/post tests")
            within_results = st.session_state.timepoint_within_results
            if within_results.empty:
                st.info("No within-treatment paired results were generated.")
            else:
                st.dataframe(
                    within_results.style.apply(style_significant_rows, axis=1),
                    use_container_width=True
                )

                st.download_button(
                    "Download within-treatment paired results as CSV",
                    data=within_results.to_csv(index=False).encode("utf-8"),
                    file_name="timepoint_within_treatment_paired_tests.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        if "timepoint_analysis_df" in st.session_state and len(plot_options) > 0:
            st.markdown("## C. Timepoint graphs")

            # Use the saved analysis dataframe only if it still contains the current selections.
            # Otherwise fall back to the current uploaded Excel dataframe.
            saved_plot_df = st.session_state.get("timepoint_analysis_df", pd.DataFrame())
            required_plot_columns = [baseline_col] + list(followup_cols) + [treatment_variable]

            if (
                saved_plot_df is None
                or saved_plot_df.empty
                or any(col not in saved_plot_df.columns for col in required_plot_columns)
            ):
                plot_df = df.copy()
                st.info(
                    "The timepoint graph is using the current uploaded Excel data because the saved "
                    "timepoint analysis data did not contain the currently selected baseline/follow-up variables."
                )
            else:
                plot_df = saved_plot_df.copy()

            valid_followup_cols_for_plot = [
                col for col in followup_cols
                if col in plot_df.columns
            ]

            if baseline_col not in plot_df.columns:
                st.warning(
                    f"The selected baseline variable '{baseline_col}' is not available in the graph dataframe. "
                    "Please re-run the timepoint analysis after selecting baseline and follow-up variables."
                )
            elif len(valid_followup_cols_for_plot) == 0:
                st.warning("No selected follow-up variables are available in the graph dataframe.")
            else:
                # Ensure change columns exist for current follow-up choices.
                current_change_cols = []

                for followup_col in valid_followup_cols_for_plot:
                    change_col = f"Timepoint_Change_{make_safe_column_name(baseline_col)}_to_{make_safe_column_name(followup_col)}"

                    if change_col not in plot_df.columns:
                        plot_df[change_col] = (
                            pd.to_numeric(plot_df[followup_col], errors="coerce")
                            - pd.to_numeric(plot_df[baseline_col], errors="coerce")
                        )

                    current_change_cols.append(change_col)

                st.session_state.timepoint_change_cols = current_change_cols

            if "Mean score over time by treatment" in plot_options:
                st.markdown("### Mean score over time by treatment")
                plot_timepoint_mean_scores(
                    plot_df,
                    baseline_col,
                    valid_followup_cols_for_plot,
                    treatment_variable,
                    treatment_groups
                )

            if "Change-score boxplot by treatment" in plot_options:
                st.markdown("### Change-score boxplot by treatment")
                for followup_col, change_col in zip(
                    valid_followup_cols_for_plot,
                    st.session_state.get("timepoint_change_cols", [])
                ):
                    if change_col not in plot_df.columns:
                        continue

                    st.markdown(f"#### {baseline_col} to {followup_col}")
                    plot_timepoint_change_boxplot(
                        plot_df,
                        change_col,
                        treatment_variable,
                        treatment_groups
                    )

            if "Individual patient spaghetti plot" in plot_options:
                st.markdown("### Individual patient spaghetti plot")
                selected_spaghetti_group = st.selectbox(
                    "Select treatment group for spaghetti plot",
                    options=["All"] + treatment_groups,
                    key="timepoint_spaghetti_group"
                )

                plot_timepoint_spaghetti(
                    plot_df,
                    patient_id_col if patient_id_col != "None" else None,
                    baseline_col,
                    valid_followup_cols_for_plot,
                    treatment_variable,
                    selected_spaghetti_group
                )

    st.divider()
    col_back, col_forward = st.columns(2)
    with col_back:
        if st.button("← Back to statistics", use_container_width=True):
            go_to("clinical_analysis")
    with col_forward:
        if st.button("Forward to machine learning →", use_container_width=True):
            go_to("clinical_model_selection")


# ============================================================
# PAGE 5: MACHINE LEARNING MODEL SELECTION
# ============================================================

# ============================================================
# CLINICAL MODULE - STEP 5 MODEL SELECTION
# ============================================================
elif st.session_state.page == "clinical_model_selection":
    if st.session_state.df is None:
        st.warning("Please upload an Excel file first.")
        if st.button("Go to upload page"):
            go_to("clinical_upload")
        st.stop()

    st.header("Clinical Model")
    st.subheader("Step 5: Machine learning")

    st.write(
        "Select the model type you want to develop. All three models use the same workflow: "
        "outcomes created in Step 2, predictors selected in Step 2, and optional treatment from Step 3."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Regression")
        st.write("Logistic regression for a binary outcome.")
        st.caption("Coefficients, odds ratios, and validation metrics.")
        if st.button("Open Regression", use_container_width=True, type="primary"):
            st.session_state.analytics_method = "Regression"
            go_to("clinical_model_calculation")

    with col2:
        st.markdown("### Random Forest")
        st.write("Tree-based machine learning model.")
        st.caption("Feature importance and validation metrics.")
        if st.button("Open Random Forest", use_container_width=True):
            st.session_state.analytics_method = "Random Forest"
            go_to("clinical_model_calculation")

    with col3:
        st.markdown("### XGBoost")
        st.write("Gradient-boosted tree model for prediction.")
        if XGBOOST_AVAILABLE:
            st.caption("Feature importance and validation metrics.")
            if st.button("Open XGBoost", use_container_width=True):
                st.session_state.analytics_method = "XGBoost"
                go_to("clinical_model_calculation")
        else:
            st.warning("XGBoost is not installed on this computer.")
            st.code("py -3.13 -m pip install xgboost")
            st.button("Open XGBoost", disabled=True, use_container_width=True)

    st.divider()

    if st.button("← Back to statistics", use_container_width=True):
        go_to("clinical_analysis")


# ============================================================
# PAGE 6: MODEL DEVELOPMENT
# ============================================================

# ============================================================
# CLINICAL MODULE - STEP 5 MODEL CALCULATION
# ============================================================
elif st.session_state.page == "clinical_model_calculation":
    if st.session_state.df is None:
        st.warning("Please upload an Excel file first.")
        if st.button("Go to upload page"):
            go_to("clinical_upload")
        st.stop()

    df = st.session_state.df
    method = st.session_state.get("analytics_method", "Regression")

    st.header("Clinical Model")
    st.subheader(f"Step 5: {method} model calculation")

    st.info(
        f"This page calculates the selected {method} model using the chosen outcome, predictors and treatment variable."
    )

    columns = list(df.columns)

    st.markdown("### Variables used in this model")

    st.markdown("#### Outcome")

    step2_outcomes = []
    for col in st.session_state.get("generated_decline_columns", []):
        if col in columns:
            step2_outcomes.append(col)

    primary_outcome = st.session_state.get("primary_outcome_variable", "")
    selected_outcome = st.session_state.get("selected_outcome_variable", "")

    for col in [primary_outcome, selected_outcome, "Derived_Neurocognitive_Decline"]:
        if col in columns and col not in step2_outcomes:
            values = df[col].dropna()
            if len(values.unique()) == 2:
                step2_outcomes.insert(0, col)

    step2_outcomes = list(dict.fromkeys(step2_outcomes))

    if len(step2_outcomes) == 0:
        st.error(
            "No Step 2 decline outcomes were found. Please go back to Step 2 and create decline outcomes first."
        )
        st.stop()

    outcome_variable = st.selectbox(
        "Select outcome created in Step 2",
        options=step2_outcomes,
        index=0,
        key=f"{method}_outcome_variable"
    )

    outcome_counts = df[outcome_variable].value_counts(dropna=False).reset_index()
    outcome_counts.columns = ["Outcome value", "Count"]
    st.dataframe(outcome_counts, use_container_width=True)

    st.markdown("#### Predictors and treatment variable")

    step2_predictors = [
        col for col in st.session_state.get("input_variables", [])
        if col in columns and col != outcome_variable
    ]

    baseline_vars = []
    if st.session_state.get("baseline_variable", "") in columns:
        baseline_vars.append(st.session_state.get("baseline_variable", ""))

    for col in st.session_state.get("baseline_variables", []):
        if col in columns and col not in baseline_vars:
            baseline_vars.append(col)

    available_predictors = list(dict.fromkeys(step2_predictors + baseline_vars))
    available_predictors = [col for col in available_predictors if col != outcome_variable]

    if len(available_predictors) == 0:
        st.error(
            "No Step 2 predictors were found. Please go back to Step 2 and select input predictors first."
        )
        st.stop()

    predictors = st.multiselect(
        "Select predictors carried forward from Step 2",
        options=available_predictors,
        default=available_predictors,
        key=f"{method}_predictors"
    )

    treatment_variable_from_step3 = st.session_state.get("treatment_variable", "")
    treatment_variable = ""

    if treatment_variable_from_step3 != "" and treatment_variable_from_step3 in columns:
        include_treatment = st.checkbox(
            f"Include Step 3 treatment column: {treatment_variable_from_step3}",
            value=True,
            key=f"{method}_include_treatment"
        )
        treatment_variable = treatment_variable_from_step3 if include_treatment else ""

        if treatment_variable != "":
            st.write("Treatment included from Step 3:", treatment_variable)
            st.write("Treatment groups included:", st.session_state.get("treatment_options", []))
    else:
        st.info("No treatment column from Step 3 is available. Model will use Step 2 predictors only.")

    st.markdown("### Machine learning validation settings")

    validation_percent = st.slider(
        "Validation set percentage",
        min_value=10,
        max_value=50,
        value=20,
        step=5,
        key=f"{method}_validation_percent",
        help="Percentage of patients held out for validation. Suggested default: 20% validation / 80% training."
    )

    validation_size = validation_percent / 100

    train_percent = 100 - validation_percent

    split_col1, split_col2 = st.columns(2)
    split_col1.metric("Training set", f"{train_percent}%")
    split_col2.metric("Validation set", f"{validation_percent}%")

    st.caption(
        "Suggested default: 80% training and 20% validation. "
        "Use 70/30 for larger datasets or if you want a bigger validation set. "
        "Avoid very large validation splits for small datasets."
    )

    st.markdown("### Machine learning model parameters")

    model_settings = {}

    with st.expander("Parameter descriptions and suggested settings", expanded=True):
        st.dataframe(parameter_description_table(method), use_container_width=True)

    if method == "Regression":
        model_settings["class_weight_balanced"] = st.checkbox(
            "Use class balancing",
            value=True,
            key=f"{method}_class_weight_balanced",
            help="Useful when decline/no-decline groups are imbalanced."
        )
        st.caption("Suggested: On. Helps when one outcome class is much smaller than the other.")
        model_settings["max_iter"] = st.number_input(
            "Maximum iterations",
            min_value=100,
            max_value=10000,
            value=3000,
            step=100,
            key=f"{method}_max_iter"
        )
        st.caption("Suggested: 3000. Increase only if the regression model does not converge.")

    elif method == "Random Forest":
        st.caption("Random Forest settings control the number and complexity of decision trees.")

        rf_col1, rf_col2 = st.columns(2)

        with rf_col1:
            model_settings["n_estimators"] = st.slider(
                "Number of trees",
                min_value=50,
                max_value=2000,
                value=500,
                step=50,
                key="rf_n_estimators",
                help="More trees usually improve stability but take longer."
            )
            st.caption("Suggested: 500. Use more trees for stability; training will take longer.")

            use_automatic_depth = st.checkbox(
                "Automatic maximum depth",
                value=True,
                key="rf_auto_depth",
                help="If selected, trees can expand until other stopping rules are met."
            )
            st.caption("Suggested: On initially. Turn off and use depth 3-8 if the model overfits.")

            if use_automatic_depth:
                model_settings["max_depth"] = None
            else:
                model_settings["max_depth"] = st.slider(
                    "Maximum tree depth",
                    min_value=1,
                    max_value=30,
                    value=5,
                    step=1,
                    key="rf_max_depth",
                    help="Lower values reduce overfitting."
                )
                st.caption("Suggested: 3-8 for small/medium datasets.")

        with rf_col2:
            model_settings["min_samples_leaf"] = st.slider(
                "Minimum samples per leaf",
                min_value=1,
                max_value=20,
                value=2,
                step=1,
                key="rf_min_samples_leaf",
                help="Higher values make the model more conservative."
            )
            st.caption("Suggested: 2-5. Higher values reduce overfitting.")

            model_settings["min_samples_split"] = st.slider(
                "Minimum samples to split",
                min_value=2,
                max_value=50,
                value=2,
                step=1,
                key="rf_min_samples_split",
                help="Higher values reduce tree complexity."
            )
            st.caption("Suggested: 2-10. Higher values make trees less complex.")

            model_settings["class_weight_balanced"] = st.checkbox(
                "Use class balancing",
                value=True,
                key="rf_class_weight_balanced",
                help="Useful when the outcome classes are imbalanced."
            )
            st.caption("Suggested: On when decline/no-decline groups are imbalanced.")

        with st.expander("Advanced Random Forest settings"):
            model_settings["random_state"] = st.number_input(
                "Random seed",
                min_value=0,
                max_value=999999,
                value=42,
                step=1,
                key="rf_random_state",
                help="Keeps results reproducible."
            )

    elif method == "XGBoost":
        st.caption("XGBoost settings control boosted trees and regularisation.")

        xgb_col1, xgb_col2 = st.columns(2)

        with xgb_col1:
            model_settings["n_estimators"] = st.slider(
                "Number of boosting rounds / trees",
                min_value=50,
                max_value=2000,
                value=400,
                step=50,
                key="xgb_n_estimators",
                help="More trees can improve learning but may overfit."
            )
            st.caption("Suggested: 300-500. Use with a low learning rate.")

            model_settings["learning_rate"] = st.slider(
                "Learning rate",
                min_value=0.005,
                max_value=0.300,
                value=0.030,
                step=0.005,
                format="%.3f",
                key="xgb_learning_rate",
                help="Smaller values learn more slowly and usually need more trees."
            )
            st.caption("Suggested: 0.03-0.10. Lower values are safer but need more trees.")

            model_settings["max_depth"] = st.slider(
                "Maximum tree depth",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                key="xgb_max_depth",
                help="Controls model complexity."
            )
            st.caption("Suggested: 2-4 for clinical datasets.")

        with xgb_col2:
            model_settings["subsample"] = st.slider(
                "Subsample fraction",
                min_value=0.50,
                max_value=1.00,
                value=0.85,
                step=0.05,
                key="xgb_subsample",
                help="Uses a fraction of patients for each tree."
            )
            st.caption("Suggested: 0.80-0.90. Helps reduce overfitting.")

            model_settings["colsample_bytree"] = st.slider(
                "Column sample fraction",
                min_value=0.50,
                max_value=1.00,
                value=0.85,
                step=0.05,
                key="xgb_colsample_bytree",
                help="Uses a fraction of predictors for each tree."
            )
            st.caption("Suggested: 0.80-0.90. Helps reduce overfitting.")

            model_settings["random_state"] = st.number_input(
                "Random seed",
                min_value=0,
                max_value=999999,
                value=42,
                step=1,
                key="xgb_random_state",
                help="Keeps results reproducible."
            )

        with st.expander("Advanced XGBoost regularisation settings"):
            model_settings["min_child_weight"] = st.slider(
                "Minimum child weight",
                min_value=0.0,
                max_value=20.0,
                value=1.0,
                step=0.5,
                key="xgb_min_child_weight",
                help="Higher values make splits more conservative."
            )

            model_settings["gamma"] = st.slider(
                "Gamma",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.1,
                key="xgb_gamma",
                help="Minimum improvement required to make a split."
            )

            model_settings["reg_alpha"] = st.slider(
                "L1 regularisation alpha",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.1,
                key="xgb_reg_alpha",
                help="Can shrink some variables toward zero."
            )

            model_settings["reg_lambda"] = st.slider(
                "L2 regularisation lambda",
                min_value=0.0,
                max_value=20.0,
                value=1.0,
                step=0.5,
                key="xgb_reg_lambda",
                help="Reduces overfitting."
            )

    st.session_state[f"{method}_model_settings"] = model_settings

    st.markdown(f"### Calculate {method} model")

    if method == "Regression":
        st.caption("Regression outputs coefficients, odds ratios, and nomogram-style points.")
    else:
        st.caption(f"{method} outputs feature importance rather than odds ratios.")

    if len(predictors) == 0:
        st.warning("Select at least one predictor.")
    else:
        if st.button(f"Run {method} model", type="primary", use_container_width=True):
            try:
                X, y, final_predictors = prepare_model_data(
                    df,
                    predictors,
                    treatment_variable,
                    outcome_variable
                )
            except Exception as error:
                st.error(f"Could not prepare model data: {error}")
                st.stop()

            if X.empty:
                st.error("No complete rows are available after removing missing values.")
                st.stop()

            if y.nunique() < 2:
                st.error("The selected outcome has only one class after removing missing data.")
                st.stop()

            st.markdown(f"## A. Univariable {method} screening")
            univariable_results = run_univariable_screening(
                df,
                predictors,
                treatment_variable,
                outcome_variable,
                method
            )

            if univariable_results.empty:
                st.warning("No univariable models could be fitted.")
            else:
                st.dataframe(
                    univariable_results.style.format({
                        "Value": "{:.3f}",
                        "Odds Ratio": "{:.3f}",
                    }),
                    use_container_width=True
                )

            st.markdown(f"## B. Full {method} model")

            pipeline, model_results, X_model, y_model = run_multivariable_model(
                df,
                predictors,
                treatment_variable,
                outcome_variable,
                method,
                model_settings
            )

            if model_results is None:
                st.error("The selected model could not be fitted.")
                st.stop()

            if method == "Regression":
                st.dataframe(
                    model_results.style.format({
                        "Coefficient": "{:.3f}",
                        "Odds Ratio": "{:.3f}",
                        "Importance": "{:.3f}",
                    }),
                    use_container_width=True
                )
            else:
                st.dataframe(
                    model_results.style.format({
                        "Importance": "{:.4f}",
                    }),
                    use_container_width=True
                )

            st.markdown("## C. Training and validation")

            trained_pipeline, metrics, cm = evaluate_model(
                pipeline,
                X_model,
                y_model,
                validation_size,
                safe_int_setting(model_settings.get("random_state", 42), 42)
            )

            save_step5_calculated_model(
                method=method,
                pipeline=trained_pipeline,
                predictors=final_predictors,
                outcome_variable=outcome_variable,
                metrics=metrics,
                model_settings=model_settings,
                model_results=model_results,
            )
            model_save_folder = save_clinical_model_outputs(
                method=method,
                pipeline=trained_pipeline,
                predictors=final_predictors,
                outcome_variable=outcome_variable,
                metrics=metrics,
                model_settings=model_settings,
                model_results=model_results,
            )
            if model_save_folder is not None:
                st.success(f"Model artefacts saved to: {model_save_folder}")

            st.markdown("### Training and validation performance")

            performance_rows = metrics.get("Performance table", [])
            performance_df = pd.DataFrame(performance_rows)

            def _format_performance_value(value):
                if value is None:
                    return "Not available"
                try:
                    if pd.isna(value):
                        return "Not available"
                except Exception:
                    pass

                try:
                    numeric_value = float(value)
                    if numeric_value.is_integer() and numeric_value > 1:
                        return str(int(numeric_value))
                    return f"{numeric_value:.3f}"
                except Exception:
                    return str(value)

            if not performance_df.empty:
                performance_display = performance_df.copy()
                for col in ["Training", "Validation"]:
                    performance_display[col] = performance_display[col].apply(_format_performance_value)

                st.dataframe(performance_display, use_container_width=True, hide_index=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Validation AUC", "Not available" if pd.isna(metrics["AUC"]) else f"{metrics['AUC']:.3f}")
            m2.metric("Validation accuracy", f"{metrics['Accuracy']:.3f}")
            m3.metric("Validation sensitivity", f"{metrics['Sensitivity']:.3f}")
            m4.metric("Validation specificity", "Not available" if pd.isna(metrics["Specificity"]) else f"{metrics['Specificity']:.3f}")

            cm_tabs = st.tabs(["Validation confusion matrix", "Training confusion matrix"])

            with cm_tabs[0]:
                st.dataframe(
                    pd.DataFrame(
                        metrics.get("Validation confusion matrix", cm),
                        index=["Actual 0", "Actual 1"],
                        columns=["Predicted 0", "Predicted 1"]
                    ),
                    use_container_width=True
                )

            with cm_tabs[1]:
                st.dataframe(
                    pd.DataFrame(
                        metrics.get("Training confusion matrix", np.zeros((2, 2), dtype=int)),
                        index=["Actual 0", "Actual 1"],
                        columns=["Predicted 0", "Predicted 1"]
                    ),
                    use_container_width=True
                )

            st.markdown("## D. Model output table")
            output_table = create_nomogram_points(model_results)
            st.dataframe(
                output_table.style.format({
                    "Coefficient": "{:.3f}",
                    "Odds Ratio": "{:.3f}",
                    "Importance": "{:.4f}",
                    "Nomogram Points": "{:.1f}",
                }),
                use_container_width=True
            )

            st.download_button(
                "Download model table as CSV",
                data=output_table.to_csv(index=False).encode("utf-8"),
                file_name=f"{method.lower().replace(' ', '_')}_model_table.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.success(f"{method} model calculation complete.")

            if len(model_settings) > 0:
                st.markdown("## E. Model settings used")
                settings_df = pd.DataFrame([
                    {"Setting": key, "Value": value}
                    for key, value in model_settings.items()
                ])
                st.dataframe(settings_df, use_container_width=True)


    # Persistent post-calculation action.
    # This is outside the "Run model" button block so it remains clickable after Streamlit reruns.
    calculated_model = st.session_state.get("step5_calculated_model", None)

    if calculated_model is not None and calculated_model.get("method", "") == method:
        st.divider()
        st.success(f"{method} model is stored and ready for calculator generation.")

        if st.button("💾 Save this step to clinical project", use_container_width=True, key=f"clinical_save_step5_model_{method}"):
            model_save_folder = save_clinical_model_outputs(
                method=calculated_model.get("method", method),
                pipeline=calculated_model.get("pipeline", None),
                predictors=calculated_model.get("predictors", []),
                outcome_variable=calculated_model.get("outcome_variable", ""),
                metrics=calculated_model.get("metrics", {}),
                model_settings=calculated_model.get("model_settings", {}),
                model_results=calculated_model.get("model_results", pd.DataFrame()),
            )
            if model_save_folder is not None:
                st.success(f"Step 5 saved to: {model_save_folder}")
            else:
                st.warning("Create or open a clinical project first, then save this step.")

        col_risk, col_compare = st.columns(2)
        with col_risk:
            if st.button(
                "Generate risk calculator",
                type="primary",
                use_container_width=True,
                key=f"{method}_generate_risk_calculator_persistent"
            ):
                st.session_state.page = "generated_risk_calculator_page"
                st.rerun()
        with col_compare:
            if st.button(
                "Open Step 6: Model comparison",
                use_container_width=True,
                key=f"{method}_open_model_comparison"
            ):
                st.session_state.page = "clinical_model_comparison"
                st.rerun()



# ============================================================
# PAGE 6: MODEL COMPARISON
# ============================================================

# ============================================================
# CLINICAL MODULE - STEP 6 MODEL COMPARISON
# ============================================================
elif st.session_state.page == "clinical_model_comparison":
    st.header("Clinical Model")
    st.subheader("Step 6: Model comparison")

    st.write(
        "View all generated risk models side-by-side, compare validation results, export selected models "
        "to the Established Model library, and optionally compare predicted risk for one patient."
    )

    models = load_generated_clinical_models_for_comparison()

    if not models:
        st.warning("No generated clinical risk models were found yet. Run at least one Step 5 model first.")
        if st.button("← Back to Step 5: Machine learning", use_container_width=True, key="clinical_compare_back_no_models"):
            go_to("clinical_model_selection")
        st.stop()

    comparison_df = clinical_model_comparison_table(models)

    st.markdown("### 1. Generated model comparison")
    st.caption(
        "Models are loaded from the current session and from the active clinical project folder when available. "
        "Orange cells mark the best-performing model for each criterion. For Brier score, lower is better."
    )

    display_df = comparison_df.copy()
    numeric_cols = [
        "Training AUC", "Validation AUC", "Training accuracy", "Validation accuracy",
        "Validation sensitivity", "Validation specificity", "Validation F1", "Brier score"
    ]
    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce")

    matrix_df = clinical_model_comparison_matrix(display_df, include_export_row=True)
    render_clinical_model_comparison_interactive_matrix(matrix_df, models)

    if "Validation AUC" in display_df.columns and display_df["Validation AUC"].notna().any():
        best_auc_row = display_df.sort_values("Validation AUC", ascending=False).iloc[0]
        st.info(
            f"Highest validation AUC: {best_auc_row['Model']} "
            f"({best_auc_row['Validation AUC']:.3f}). Interpret cautiously if the validation set is small."
        )

    st.markdown("### 2. Inspect one model")
    selected_model_label = st.selectbox("Select model to inspect", options=list(models.keys()), key="clinical_compare_selected_model")
    selected_payload = models[selected_model_label]

    c1, c2, c3 = st.columns(3)
    c1.metric("Method", selected_payload.get("method", selected_model_label))
    c2.metric("Outcome", selected_payload.get("outcome_variable", ""))
    c3.metric("Predictors", len(selected_payload.get("predictors", [])))

    st.markdown("#### Predictors")
    st.dataframe(pd.DataFrame({"Predictor": selected_payload.get("predictors", [])}), use_container_width=True, hide_index=True)

    model_results = selected_payload.get("model_results", pd.DataFrame())
    if isinstance(model_results, pd.DataFrame) and not model_results.empty:
        st.markdown("#### Coefficients / feature importance")
        st.dataframe(model_results, use_container_width=True, hide_index=True)

    st.markdown("### 3. Compare predicted risk for one patient")
    st.caption("Enter one set of patient values. The app will calculate predicted risk for every model that has a saved fitted pipeline and all required predictors.")

    df = st.session_state.get("df", None)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.info("Load or reopen the clinical dataset to enable patient-level risk comparison.")
    else:
        all_predictors = []
        for payload in models.values():
            for pred in payload.get("predictors", []):
                if pred in df.columns and pred not in all_predictors:
                    all_predictors.append(pred)

        patient_values = {}
        if not all_predictors:
            st.info("No shared predictor inputs were available for risk comparison.")
        else:
            with st.expander("Enter patient values for risk comparison", expanded=True):
                for predictor in all_predictors:
                    if is_numeric_column(df, predictor):
                        values = pd.to_numeric(df[predictor], errors="coerce")
                        default_value = float(values.median()) if values.notna().any() else 0.0
                        patient_values[predictor] = st.number_input(
                            predictor,
                            value=default_value,
                            key=f"clinical_compare_patient_value_{predictor}",
                        )
                    else:
                        options = sorted([x for x in df[predictor].dropna().astype(str).str.strip().unique().tolist() if x != ""])
                        if not options:
                            options = [""]
                        patient_values[predictor] = st.selectbox(
                            predictor,
                            options=options,
                            key=f"clinical_compare_patient_value_{predictor}",
                        )

            if st.button("Calculate risk across generated models", type="primary", use_container_width=True, key="clinical_compare_calculate_risks"):
                risk_rows = []
                for label, payload in models.items():
                    pipeline = payload.get("pipeline", None)
                    predictors = payload.get("predictors", [])
                    if pipeline is None:
                        risk_rows.append({"Model": label, "Predicted risk": np.nan, "Status": "No saved fitted pipeline"})
                        continue
                    missing = [p for p in predictors if p not in patient_values]
                    if missing:
                        risk_rows.append({"Model": label, "Predicted risk": np.nan, "Status": "Missing inputs: " + "; ".join(missing)})
                        continue
                    try:
                        new_patient = pd.DataFrame([{p: patient_values[p] for p in predictors}])
                        risk = float(pipeline.predict_proba(new_patient)[0, 1])
                        risk_rows.append({"Model": label, "Predicted risk": risk, "Predicted risk (%)": risk * 100, "Status": "Calculated"})
                    except Exception as error:
                        risk_rows.append({"Model": label, "Predicted risk": np.nan, "Status": f"Failed: {error}"})
                risk_df = pd.DataFrame(risk_rows)
                st.session_state.clinical_model_risk_comparison_df = risk_df

            risk_df = st.session_state.get("clinical_model_risk_comparison_df", pd.DataFrame())
            if risk_df is not None and isinstance(risk_df, pd.DataFrame) and not risk_df.empty:
                st.dataframe(risk_df, use_container_width=True, hide_index=True)

    st.markdown("### 5. Save Step 6")
    if st.button("💾 Save this step to clinical project", use_container_width=True, key="clinical_save_step6_model_comparison"):
        risk_df = st.session_state.get("clinical_model_risk_comparison_df", pd.DataFrame())
        out_folder = save_clinical_model_comparison_outputs(comparison_df, risk_df)
        if out_folder is not None:
            matrix_to_save = clinical_model_comparison_matrix(display_df, include_export_row=True)
            if not matrix_to_save.empty:
                matrix_to_save.to_csv(Path(out_folder) / "clinical_model_comparison_matrix.csv", index=False)
            st.success(f"Step 6 saved to: {out_folder}")
        else:
            st.warning("Create or open a clinical project first, then save this step.")

    st.divider()
    back_col, established_col = st.columns(2)
    with back_col:
        if st.button("← Back to Step 5", use_container_width=True, key="clinical_compare_back_step5"):
            go_to("clinical_model_selection")
    with established_col:
        if st.button("Open Established Model library", use_container_width=True, key="clinical_compare_open_established"):
            go_to("established_model")


elif st.session_state.page == "generated_risk_calculator_page":
    render_step5_risk_calculator_only()


elif st.session_state.page == "model_export_page":
    render_model_export_page()


elif st.session_state.page == "risk_calculator_only":
    render_step5_risk_calculator_only()

elif st.session_state.page == "clinical_risk_calculator":
    render_step5_risk_calculator_only()

