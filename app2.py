import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model, encoders, scaler
with open("best_model.pkl", "rb") as f:
    best_model = pickle.load(f)
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)
with open("scaler_selective.pkl", "rb") as f:
    scaler = pickle.load(f)

# Config
st.set_page_config(page_title="Autism Prediction", layout="wide")
st.title("🧠 Autism Prediction System (AQ-10 Based)")

# Column lists
training_columns = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
                    'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'age', 'gender',
                    'ethnicity', 'jundice', 'austim', 'contry_of_res', 'used_app_before',
                    'result', 'relation']
numerical_cols_to_standardize = ['age', 'result']
non_standardized_cols = [col for col in training_columns if col not in numerical_cols_to_standardize]

# Questions
aq_questions = [
    "Q1. I often notice small sounds when others do not",
    "Q2. I usually concentrate more on the whole picture, rather than the small details",
    "Q3. I find it easy to do more than one thing at once",
    "Q4. If there's an interruption, I can switch back to what I was doing very quickly",
    "Q5. I find it easy to ‘read between the lines’ when someone is talking to me",
    "Q6. I know how to tell if someone listening to me is getting bored",
    "Q7. When I’m reading a story, I find it difficult to work out the characters’ intentions",
    "Q8. I like to collect information about categories of things (e.g. types of car, bird, train, plant etc)",
    "Q9. I find it easy to work out what someone is thinking or feeling just by looking at their face",
    "Q10. I find it difficult to work out people’s intentions"
]

response_map = {
    "Definitely Agree": 1,
    "Slightly Agree": 2,
    "Slightly Disagree": 3,
    "Definitely Disagree": 4
}

ethnicity_choices = [
    "White-European", "Latino", "Others", "Black", "Asian",
    "Middle Eastern", "Pasifika", "South Asian", "Hispanic", "Turkish"
]

country_choices = [
    "United States", "Brazil", "Spain", "Egypt", "New Zealand", "Bahamas",
    "Burundi", "Austria", "Argentina", "Jordan", "Ireland", "United Arab Emirates",
    "Afghanistan", "Lebanon", "United Kingdom", "South Africa", "Italy",
    "Pakistan", "Bangladesh", "Chile", "France", "China", "Australia", "Canada",
    "Saudi Arabia", "Netherlands", "Romania", "Sweden", "Tonga", "Oman", "India",
    "Philippines", "Sri Lanka", "Sierra Leone", "Ethiopia", "Viet Nam", "Iran",
    "Costa Rica", "Germany", "Mexico", "Russia", "Armenia", "Iceland", "Nicaragua",
    "Hong Kong", "Japan", "Ukraine", "Kazakhstan", "AmericanSamoa", "Uruguay",
    "Serbia", "Portugal", "Malaysia", "Ecuador", "Niger", "Belgium", "Bolivia",
    "Aruba", "Finland", "Turkey", "Nepal", "Indonesia", "Angola", "Azerbaijan",
    "Iraq", "Czech Republic", "Cyprus"
]

# Scoring function (official AQ-10)
def aq_to_binary(index, value):
    if index in [0, 6, 7, 9]:
        return 1 if value in [1, 2] else 0
    elif index in [1, 2, 3, 4, 5, 8]:
        return 1 if value in [3, 4] else 0
    return 0

# Form
with st.form("aq_form"):
    st.subheader("📋 AQ-10 Questionnaire")
    aq_scores_raw = []
    missing_fields = []

    for i in range(0, 10, 2):
        col1, col2 = st.columns(2)
        for j, col in enumerate([col1, col2]):
            if i + j < len(aq_questions):
                q_text = aq_questions[i + j]
                options = ["Select an option"] + list(response_map.keys())
                ans = col.selectbox(q_text, options, key=f"q{i + j}")
                if ans == "Select an option":
                    aq_scores_raw.append(None)
                    missing_fields.append(f"{q_text}")
                else:
                    aq_scores_raw.append(response_map[ans])

    st.subheader("👤 Personal Information")
    col1, col2 = st.columns(2)
    age = col1.number_input("Q11. Age", min_value=1, max_value=120, value=25)
    gender = col2.selectbox("Q12. Gender", ["Select an option", "m", "f"])
    ethnicity = col1.selectbox("Q13. Ethnicity", ["Select an option"] + ethnicity_choices)
    jundice = col2.selectbox("Q14. Jaundice at birth?", ["Select an option", "yes", "no"])
    austim = col1.selectbox("Q15. Family member diagnosed with autism?", ["Select an option", "yes", "no"])
    contry_of_res = col2.selectbox("Q16. Country of residence", ["Select an option"] + country_choices)
    used_app_before = col1.selectbox("Q17. Used screening app before?", ["Select an option", "yes", "no"])
    relation = col2.selectbox("Q18. Relation with candidate", ["Select an option", "Self", "Parent", "Relative", "Health care professional", "Others"])

    # Buttons
    submit = st.form_submit_button("🔍 Predict")
    reset = st.form_submit_button("🔄 Refresh")

if reset:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

if submit:
    # Check demographic questions
    demo_fields = {
        "Q12. Gender": gender,
        "Q13. Ethnicity": ethnicity,
        "Q14. Jaundice at birth?": jundice,
        "Q15. Family member diagnosed with autism?": austim,
        "Q16. Country of residence": contry_of_res,
        "Q17. Used screening app before?": used_app_before,
        "Q18. Relation with candidate": relation
    }
    for label, val in demo_fields.items():
        if val == "Select an option":
            missing_fields.append(label)

    if None in aq_scores_raw or missing_fields:
        st.warning("⚠️ Please complete all questions before submitting.")
        for f in missing_fields:
            st.error(f"❌ {f} is unanswered.")
    else:
        try:
            # Score conversion
            binary_scores = [aq_to_binary(i, val) for i, val in enumerate(aq_scores_raw)]
            result_score = sum(binary_scores)

            # Input dictionary
            input_data = {
                'A1_Score': binary_scores[0], 'A2_Score': binary_scores[1], 'A3_Score': binary_scores[2],
                'A4_Score': binary_scores[3], 'A5_Score': binary_scores[4], 'A6_Score': binary_scores[5],
                'A7_Score': binary_scores[6], 'A8_Score': binary_scores[7], 'A9_Score': binary_scores[8],
                'A10_Score': binary_scores[9], 'age': age, 'gender': gender,
                'ethnicity': ethnicity, 'jundice': jundice, 'austim': austim,
                'contry_of_res': contry_of_res, 'used_app_before': used_app_before,
                'result': result_score, 'relation': relation
            }

            input_df = pd.DataFrame([input_data])
            input_df_raw = input_df.copy()  # Save for debug view

            # Clean & encode
            input_df["ethnicity"] = input_df["ethnicity"].replace({"?": "Others", "others": "Others"})
            input_df["relation"] = input_df["relation"].replace({
                "?": "Others", "Relative": "Others", "Parent": "Others", "Health care professional": "Others"
            })

            for column in input_df.select_dtypes(include="object").columns:
                if column in encoders:
                    try:
                        input_df[column] = encoders[column].transform(input_df[column])
                    except:
                        input_df[column] = -1

            # Scale
            input_scaled = scaler.transform(input_df[numerical_cols_to_standardize])
            input_scaled_df = pd.DataFrame(input_scaled, columns=numerical_cols_to_standardize, index=input_df.index)
            input_final = pd.concat([input_scaled_df, input_df[non_standardized_cols]], axis=1)

            # Debug
            #st.subheader("🧪 Debug: Input Data")
            #st.code(input_df_raw.to_string(index=False), language="text")
            #st.markdown(f"**AQ-10 Binary Scores:** {binary_scores}")
            #st.markdown(f"**Total AQ Score:** {result_score}/10")

            # Predict
            prediction = best_model.predict(input_final)[0]
            # Extract the single numerical value from the 'result' Series
            result_score = input_df["result"].iloc[0] # Or .item() if preferred


            if result_score <= 3:
                risk_level = "Low Risk"
            elif result_score <= 5:
                risk_level = "Moderate Risk"
            else:
                risk_level = "High Risk"

            prob = best_model.predict_proba(input_final)[0][1]

            # Output
            st.subheader("🎯 Prediction Result")
            if prediction == 1:
                st.error("🔴 The person is **Autistic**.")
            else:
                st.success(f"🟢 The person is **Not Autistic**.\n\nEstimated risk level based on AQ-10 score : *{risk_level}**.\n\nEstimated Risk: **{result_score*10:.2f}%**")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
