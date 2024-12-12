import streamlit as st
import joblib

st.set_page_config(
    page_icon='logo.jpeg',
    page_title='Infant Diagnosis'
)

# Load the model and vectorizer
model = joblib.load('infant_sickness_model.pkl')
vectorizer = joblib.load('infant_symptoms_vectorizer.pkl')

# Extract all symptoms and sicknesses
all_symptoms = vectorizer.get_feature_names_out()
all_sicknesses = model.classes_

# Mappings for prescriptions, causes, and symptoms
prescription_mapping = {
    "Teething Fever": "Cool compress, hydration, teething gel",
    "Common Cold": "Saline drops, humidifier, rest",
    "Stomach Flu": "Oral rehydration salts, hydration",
    "Roseola": "Acetaminophen, hydration, see doctor if needed",
    "Ear Infection": "Pain relief drops, antibiotics for infection",
    "Bronchiolitis": "Nebulizer, consult doctor",
    "Infant Colic": "Burping after meals, massage tummy",
    "Gastroesophageal Reflux": "Small frequent feeds, consult doctor",
    "Conjunctivitis": "Clean eyes with warm water, antibiotic drops",
    "Measles": "Acetaminophen, hydration, see doctor",
    "Tonsillitis": "Pain relievers, hydration, consult doctor",
    "Brain Tumor": "Surgery, radiation, chemotherapy",
    "Stroke": "Emergency care, physical therapy, blood thinners",
    "Multiple Sclerosis": "Disease-modifying drugs, physical therapy",
    "Meningitis": "Antibiotics, fluids, rest",
    "Kidney Infection": "Antibiotics, pain relief, hydration",
    "Food Poisoning": "Hydration, rest, anti-nausea medication",
    "Gastritis": "Antacids, small meals, avoid spicy food",
    "Hepatitis": "Rest, hydration, antiviral medication",
    "Lymphoma": "Chemotherapy, radiotherapy, consultation with oncologist",
    "Tuberculosis": "Antibiotics, isolation, follow-up with doctor",
    "Allergic Reaction": "Antihistamines, epinephrine if severe"
}

causes_mapping = {
    "Teething Fever": "Teething process causing gum inflammation",
    "Common Cold": "Viral infection, exposure to infected individuals",
    "Stomach Flu": "Viral gastroenteritis, contaminated food or water",
    "Roseola": "Human herpesvirus 6 or 7",
    "Ear Infection": "Bacterial or viral infection in the middle ear",
    "Bronchiolitis": "RSV infection, common in infants during winter",
    "Infant Colic": "Digestive issues, overfeeding, or air swallowed during feeding",
    "Gastroesophageal Reflux": "Immature digestive system causing acid reflux",
    "Conjunctivitis": "Bacterial or viral infection, allergens, or irritants",
    "Measles": "Measles virus, lack of vaccination",
    "Tonsillitis": "Bacterial or viral infection of the tonsils",
    "Brain Tumor": "Abnormal cell growth, genetic factors",
    "Stroke": "Blocked blood flow to the brain, burst blood vessels",
    "Multiple Sclerosis": "Immune system attacks the central nervous system, genetic factors",
    "Meningitis": "Bacterial or viral infection affecting brain membranes",
    "Kidney Infection": "Bacterial infection, untreated UTI",
    "Food Poisoning": "Contaminated food or water, bacterial toxins",
    "Gastritis": "Helicobacter pylori infection, overuse of NSAIDs, spicy foods",
    "Hepatitis": "Viral infections (A, B, C), alcohol abuse, toxins",
    "Lymphoma": "Abnormal lymphocyte growth, genetic factors",
    "Tuberculosis": "Mycobacterium tuberculosis infection, airborne transmission",
    "Allergic Reaction": "Exposure to allergens (food, pollen, medications)"
}

symptoms_mapping = {
    "Teething Fever": "Mild fever, drooling, irritability",
    "Common Cold": "Runny nose, congestion, sneezing, mild fever",
    "Stomach Flu": "Vomiting, diarrhea, abdominal cramps",
    "Roseola": "High fever followed by rash, irritability",
    "Ear Infection": "Ear pain, fever, difficulty sleeping",
    "Bronchiolitis": "Wheezing, rapid breathing, coughing",
    "Infant Colic": "Prolonged crying, clenching fists, gas",
    "Gastroesophageal Reflux": "Frequent spit-ups, irritability during feeds",
    "Conjunctivitis": "Red eyes, discharge, tearing",
    "Measles": "Fever, rash, cough, runny nose",
    "Tonsillitis": "Sore throat, swollen tonsils, fever",
    "Brain Tumor": "Headache, nausea, vision problems",
    "Stroke": "Weakness on one side, trouble speaking, dizziness",
    "Multiple Sclerosis": "Muscle weakness, numbness, coordination issues",
    "Meningitis": "Stiff neck, fever, headache, sensitivity to light",
    "Kidney Infection": "Fever, back pain, painful urination",
    "Food Poisoning": "Nausea, vomiting, diarrhea, stomach pain",
    "Gastritis": "Stomach pain, nausea, bloating",
    "Hepatitis": "Jaundice, fatigue, abdominal pain",
    "Lymphoma": "Swollen lymph nodes, fatigue, weight loss",
    "Tuberculosis": "Persistent cough, night sweats, weight loss",
    "Allergic Reaction": "Itching, swelling, difficulty breathing"
}

# Streamlit app
st.title("Infant Sickness Prediction App")
st.write("Enter the symptoms of the infant to get a diagnosis, prescription, possible causes, and related symptoms.")
st.image('bg-removebg-preview.png', width=300)

# Display all possible symptoms for user guidance
st.subheader("Possible Symptoms:")
st.write(", ".join(all_symptoms))

# User input
symptoms = st.text_area("Symptoms (e.g., fever, cough, fatigue):")

if st.button("Predict"):
    if symptoms.strip():
        # Vectorize input and predict
        symptoms_vec = vectorizer.transform([symptoms])
        sickness = model.predict(symptoms_vec)[0]
        prescription = prescription_mapping.get(sickness, "Consult a pediatrician")
        causes = causes_mapping.get(sickness, "Unknown causes. Consult a pediatrician.")
        related_symptoms = symptoms_mapping.get(sickness, "No specific symptoms listed.")
        
        # Display results
        st.subheader(f"Predicted Sickness: {sickness}")
        st.write(f"Prescription: {prescription}")
        st.write(f"Possible Causes: {causes}")
        st.write(f"Related Symptoms: {related_symptoms}")
    else:
        st.error("Please enter symptoms.")
