# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 21:58:03 2025
MLISS

@author: mdcob
"""
import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(layout="centered")

st.markdown("""
    <style>
    /* Widen the page a bit more */
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 4rem;
        padding-right: 4rem;
    }

    /* Make fonts more legible */
    html, body, [class*="css"]  {
        font-size: 1.0rem;
    }

    /* Optional: make input boxes bigger */
    input, select, textarea {
        font-size: 0.9rem !important;
    }

    /* Reduce vertical whitespace between form elements */
    label {
        margin-bottom: 0.2rem !important;
    }
    </style>
""", unsafe_allow_html=True)




# === Load Model and Scaler ===
with open('calibrated_model2.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler2.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title("RT-MLISS Score")


st.markdown(
    """
    <h4 style='margin-top: -8px; color: gray;'>
        A real-time mortality prediction tool for trauma patients
    </h4>
    <p style='font-size:18px; color: #555; font-style: bold;'>
        Developed by <b>MD Cobler-Lichter MD MSDS</b>, JM Delamater MD MPH, AM Reyes MD MPH, TR Arcieri MD,
        JP Meizoso MD MSPH, CI Schulman MD PhD MSPH, BM Parker DO, KG Proctor PhD, N Namias MD MBA
    </p>
    <p style='font-size:16x; color: #555; font-style: italic;'>
        Our models are calibrated such that the outputted scores reflects an accurate prediction
        of the true probability of in-hospital mortality based on our training data
    </p>
    """,
    unsafe_allow_html=True
)

    
# === Helper Functions ===
def int_input(label, key, min_val=None, max_val=None):
    """Integer input that returns np.nan if user leaves it blank."""
    # Use a small text_input "shim" to allow true blank, otherwise number_input forces a default.
    raw = st.text_input(label, value="", key=key, help="Enter a whole number or leave blank")
    if raw == "":
        return np.nan
    try:
        v = int(raw)
        if (min_val is not None and v < min_val) or (max_val is not None and v > max_val):
            return np.nan
        return v
    except ValueError:
        return np.nan


def yes_no_radio(label, key):
    return st.radio(label, ['No', 'Yes'], index=0, horizontal=True, key=key)




# === Main Clinical Input Fields ===
label_map = {
    'TRAUMATYPE': "Trauma Type",
    'AGEYEARS': "Age",
    'TOTALGCS': "Arrival GCS",
    'SBP': "Arrival SBP",
    'TEMPERATURE': "Arrival Temperature (°C)",
    'PULSERATE': "Arrival Heart Rate",
    'WEIGHT': "Weight (kg)"
}

user_inputs = {}
sbp_val = None
pulse_val = None


# === Frontend display names mapped to backend variable names ===
frontend_labels = {
    "Intracranial Vascular Injury": "IntracranialVascularInjury",
    "Brain Stem Injury": "BrainStemInjury",
    "Epidural Hematoma (EDH)": "EDH",
    "Subarachnoid Hemorrhage (SAH)": "SAH",
    "Subdural Hematoma (SDH)": "SDH",
    "Skull Fracture": "SkullFx",
    "Diffuse Axonal Injury (DAI)": "DAI",
    "Intraparenchymal Hemorrhage (IPH)": "IPH",
    "Neck Vascular Injury": "NeckVascularInjury",
    "Aerodigestive Injury": "AeroDigestiveInjury",
    "Spinal Cord Injury (SCI)": "SCI",
    "Spine Fracture": "SpineFx",
    "Thoracic Vascular Injury": "ThoracicVascularInjury",
    "Cardiac Injury": "CardiacInjury",
    "Lung Injury": "LungInjury",
    "Rib Fracture": "RibFx",
    "Abdominal Vascular Injury": "AbdominalVascular",
    "Kidney Injury": "KidneyInjury",
    "Stomach Injury": "StomachInjury",
    "Spleen Injury": "SpleenInjury",
    "Urogenital Injury": "UroGenInternalInjury",
    "Pelvic Fracture": "PelvicFx",
    "Pancreas Injury": "PancreasInjury",
    "Liver Injury": "LiverInjury",
    "Colorectal Injury": "ColorectalInjury",
    "Small Bowel Injury": "SmallBowelInjury",
    "Upper Extremity Amputation (other than finger)": "UEAmputation",
    "Upper Extremity Vascular Injury": "UEVascularInjury",
    "Upper Extremity Long Bone Fracture": "UELongBoneFx",
    "Lower Extremity Vascular Injury": "LEVascularInjury",
    "Lower Extremity Amputation (other than toe)": "LEAmputation",
    "Lower Extremity Long Bone Fracture": "LELongBoneFx"
}

# === Injury Regions and Subquestions with frontend names ===
injury_categories_display = {
    "Head injury?": [
        "Intracranial Vascular Injury", "Brain Stem Injury", "Epidural Hematoma (EDH)", 
        "Subarachnoid Hemorrhage (SAH)", "Subdural Hematoma (SDH)", "Skull Fracture", 
        "Diffuse Axonal Injury (DAI)", "Intraparenchymal Hemorrhage (IPH)"
    ],
    "Neck/Back injury?": [
        "Neck Vascular Injury", "Aerodigestive Injury", "Spinal Cord Injury (SCI)", 
        "Spine Fracture"
    ],
    "Thoracic injury?": [
        "Thoracic Vascular Injury", "Cardiac Injury", "Lung Injury", "Rib Fracture"
    ],
    "Abdominal/Pelvic injury?": [
        "Abdominal Vascular Injury", "Kidney Injury", "Stomach Injury", "Spleen Injury",
        "Urogenital Injury", "Pelvic Fracture", "Pancreas Injury", "Liver Injury",
        "Colorectal Injury", "Small Bowel Injury"
    ],
    "Extremity injury?": [
        "Upper Extremity Amputation (other than finger)", "Upper Extremity Vascular Injury", 
        "Upper Extremity Long Bone Fracture", "Lower Extremity Vascular Injury", 
        "Lower Extremity Amputation (other than toe)", "Lower Extremity Long Bone Fracture"
    ]
}

injury_inputs = {}


# === Two-Column Layout ===
col1, spacer, col2 = st.columns([2, 1, 2])

with col1:
    st.subheader("Patient Info & Vitals")
    for var, label in label_map.items():
        if var == 'TRAUMATYPE':
            trauma_type = st.radio(label, ['Blunt', 'Penetrating'], index=0, horizontal=True, key='TRAUMATYPE')
            user_inputs['Penetrating'] = 1 if trauma_type == 'Penetrating' else 0
        else:
            val = int_input(label, var)
            user_inputs[var] = val
            if var == 'SBP':
                sbp_val = val
            elif var == 'PULSERATE':
                pulse_val = val

    # === Derive ShockIndex ===
    if sbp_val == 0 or pulse_val == 0:
        user_inputs['ShockIndex'] = 2.0
    elif sbp_val is not None and pulse_val is not None:
        user_inputs['ShockIndex'] = pulse_val / sbp_val
    else:
        user_inputs['ShockIndex'] = np.nan

with col2:
    st.subheader("Injury Pattern")

    for region_label, subquestions in injury_categories_display.items():
        has_injury = yes_no_radio(region_label, key=region_label)
        if has_injury == 'Yes':
            with st.expander(f"Specify injuries for {region_label.replace(' injury?', '')}"):
                for display_name in subquestions:
                    backend_var = frontend_labels[display_name]
                    injury_inputs[backend_var] = 1 if yes_no_radio(display_name, key=backend_var) == 'Yes' else 0
        else:
            for display_name in subquestions:
                backend_var = frontend_labels[display_name]
                injury_inputs[backend_var] = 0


# === Calculate NumberOfInjuries ===
user_inputs['NumberOfInjuries'] = sum(injury_inputs.values())

# === Merge Inputs ===
user_inputs.update(injury_inputs)
input_df = pd.DataFrame([user_inputs])



# === Check for missing inputs ===
if any(pd.isna(val) for val in user_inputs.values()):
    st.markdown(
        "<span style='color:red; font-weight:bold;'>"
        "⚠️ One or more of the input variables are missing. "
        "A score will still be calculated but it may be inaccurate."
        "</span>",
        unsafe_allow_html=True
    )
    
# Always show the heading
st.markdown("### RT-MLISS Score (Predicted Mortality):")

# Placeholder for the dynamic result
mortality_output = st.empty()

# Prediction button logic
if st.button("Predict Mortality"):
    try:
        input_df = input_df[scaler.feature_names_in_]
        input_scaled = scaler.transform(input_df)
        prediction_proba = model.predict_proba(input_scaled)[:, 1][0]

        # Output with large font size
        mortality_output.markdown(
            f"<p style='font-size:36px; font-weight:bold; color:#d62728;'>{prediction_proba:.1%}</p>",
            unsafe_allow_html=True
        )
    except Exception as e:
        mortality_output.error(f"Error: {e}")

# === Reset Button ===
if st.button("Reset Form"):
    for key in st.session_state.keys():
        del st.session_state[key]

    st.rerun()




