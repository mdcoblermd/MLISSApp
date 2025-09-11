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
import re

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
@st.cache_resource
def load_artifacts():
    import pickle
    with open("calibrated_model2.pkl","rb") as f:
        model = pickle.load(f)
    with open("scaler2.pkl","rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

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
def int_input_live(label, key, min_val=None, max_val=None, placeholder=""):
    raw_key = f"{key}__raw"           # separate raw string storage

    # initialize once; DO NOT pass `value=` into st.text_input on every rerun
    if raw_key not in st.session_state:
        st.session_state[raw_key] = ""

    raw = st.text_input(label, key=raw_key, placeholder=placeholder).strip()

    if raw == "":
        val = np.nan
    elif re.fullmatch(r"\d+", raw):
        v = int(raw)
        val = np.nan if (min_val is not None and v < min_val) or (max_val is not None and v > max_val) else v
    else:
        val = np.nan

    st.session_state[key] = val       # parsed integer your app uses
    return val
    
def float_input_live(label, key, min_val=None, max_val=None, placeholder=""):
    """
    Text input that validates floats on every keystroke.
    Returns np.nan when blank/invalid; else a float within optional bounds.
    """
    raw_key = f"{key}__raw"
    if raw_key not in st.session_state:
        st.session_state[raw_key] = ""

    raw = st.text_input(label, key=raw_key, placeholder=placeholder).strip()

    if raw == "":
        val = np.nan
    elif re.fullmatch(r"\d+(\.\d+)?", raw):   # digits with optional .decimal
        v = float(raw)
        if (min_val is not None and v < min_val) or (max_val is not None and v > max_val):
            val = np.nan
        else:
            val = v
    else:
        val = np.nan

    st.session_state[key] = val
    return val

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
bounds = {
    'AGEYEARS': (0, 110),
    'TOTALGCS': (3, 15),
    'SBP': (40, 260),
    'TEMPERATURE': (30, 43),   # realistic °C range
    'PULSERATE': (20, 220),
    'WEIGHT': (2, 400),
}

with col1:
    st.subheader("Patient Info & Vitals")
    for var, label in label_map.items():
        if var == 'TRAUMATYPE':
            trauma_type = st.radio(label, ['Blunt', 'Penetrating'],
                                   index=0, horizontal=True, key='TRAUMATYPE')
            user_inputs['Penetrating'] = 1 if trauma_type == 'Penetrating' else 0
        else:
            lo, hi = bounds.get(var, (None, None))
            if var == 'TEMPERATURE':
                val = float_input_live(label, var, min_val=lo, max_val=hi)
            else:
                val = int_input_live(label, var, min_val=lo, max_val=hi)
            user_inputs[var] = val
            if var == 'SBP': sbp_val = val
            if var == 'PULSERATE': pulse_val = val

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



# # === Check for missing inputs ===
# if any(pd.isna(val) for val in user_inputs.values()):
#     st.markdown(
#         "<span style='color:red; font-weight:bold;'>"
#         "⚠️ One or more of the input variables are missing. "
#         "A score will still be calculated but it may be inaccurate."
#         "</span>",
#         unsafe_allow_html=True
#     )
    
# Always show the heading
st.markdown("### RT-MLISS Score (Predicted Mortality):")

# Placeholder for the dynamic result
mortality_output = st.empty()

# Prediction button logic
st.markdown("### RT-MLISS Score (Predicted Mortality):")
mortality_output = st.empty()

# Which fields must be present to enable Predict?
required = ['AGEYEARS','TOTALGCS','SBP','TEMPERATURE','PULSERATE','WEIGHT']

# Build the one-row frame in the model’s expected column order
try:
    X = pd.DataFrame([user_inputs], columns=scaler.feature_names_in_)
except Exception as e:
    mortality_output.error(f"Column alignment error: {e}")
    X = None

inputs_ready = X is not None and not X.isna().any(axis=1).item()

# Session storage for last prediction and last inputs signature
if 'last_pred' not in st.session_state:
    st.session_state['last_pred'] = None
if 'last_sig' not in st.session_state:
    st.session_state['last_sig'] = None

def make_signature(df_row: pd.DataFrame) -> tuple:
    # Deterministic, NaN-safe signature of current inputs (in model column order)
    s = df_row.iloc[0].astype(object)
    return tuple(None if pd.isna(v) else float(v) for v in s)

current_sig = make_signature(X) if X is not None else None
changed_since_last = (current_sig is not None and 
                      st.session_state['last_sig'] is not None and 
                      current_sig != st.session_state['last_sig'])

# Predict button
clicked = st.button("Predict Mortality", disabled=not inputs_ready)

if clicked and inputs_ready:
    try:
        X_scaled = scaler.transform(X)
        pred = float(model.predict_proba(X_scaled)[:, 1][0])
        st.session_state['last_pred'] = pred
        st.session_state['last_sig']  = current_sig
    except Exception as e:
        st.session_state['last_pred'] = None
        mortality_output.error(f"Error during prediction: {e}")

# Show result (persist across reruns)
if st.session_state['last_pred'] is not None:
    mortality_output.markdown(
        f"<p style='font-size:36px; font-weight:bold; color:#d62728;'>{st.session_state['last_pred']:.1%}</p>",
        unsafe_allow_html=True
    )
    if changed_since_last:
        st.caption("Inputs changed since last prediction — press **Predict Mortality** to refresh.")
else:
    if not inputs_ready:
        missing = [k for k in required if pd.isna(user_inputs.get(k))]
        if missing:
            st.info("Enter all required fields to enable **Predict Mortality**.")

# # === Reset Button ===
# if st.button("Reset Form"):
#     for key in st.session_state.keys():
#         del st.session_state[key]

#     st.rerun()
















