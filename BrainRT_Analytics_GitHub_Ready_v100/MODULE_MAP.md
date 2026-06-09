# BrainRT Analytics module map

## Clinical module
Page ids:
- clinical_start_project
- clinical_upload
- clinical_variables
- clinical_treatment
- clinical_analysis
- clinical_descriptive
- clinical_inferential
- clinical_timepoint
- clinical_model_selection
- clinical_model_calculation
- clinical_model_comparison

## Voxel-based analysis module
Page ids:
- voxel_analysis_home
- voxel_start_project
- voxel_load_patient_data
- voxel_load_images
- voxel_registration_alignment
- voxel_reference_ccs
- voxel_batch_registration
- voxel_registration_qc
- voxel_warp_to_ccs
- voxel_dose_normalisation
- voxel_vba_ready_dataset
- voxel_statistical_analysis

## Established Model module
Page ids:
- established_model
- established_model_search
- established_model_compare
- established_model_risk_calculator
- established_model_validate
- established_model_external_validation
- established_model_documentation
- established_model_collaborate

## Knowledge Base module
Page ids:
- knowledge_base
- kb_outcome_assessment_tools
- kb_model_development_tools
- kb_model_evaluation_tools
- kb_other

## Navigation
The workflow navigation is controlled mainly by:
- `valid_pages`
- `get_current_workflow()`
- `get_navigation_steps()`
- `render_step_sidebar()`
