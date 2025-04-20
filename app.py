from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd  # <--- make sure this import is present

# Load model and preprocessor
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

app = Flask(__name__)

# Label mapping from model output
label_map = {
    0: "Very Dissatisfied",
    1: "Dissatisfied",
    2: "Neutral",
    3: "Satisfied",
    4: "Very Satisfied"
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Extract input from form
        input_data = [
            request.form.get('branch'),
            request.form.get('coding'),
            request.form.get('multidisciplinary'),
            request.form.get('hackathons'),
            request.form.get('field_interest'),
            request.form.get('entrepreneurship'),      
            request.form.get('govt_jobs'),
            request.form.get('sports'),
            request.form.get('social_work'),
            request.form.get('career_plan'),
            request.form.get('study_env'),
            request.form.get('hostel'),
            request.form.get('motivation'),
            request.form.get('native_state'),
            request.form.get('city_life'),
            request.form.get('infra'),
            request.form.get('research'),
        ]

        # Convert input to DataFrame to match training format
        input_df = pd.DataFrame([input_data], columns=preprocessor.feature_names_in_)

        # === Input Normalization ===
        import json

        # Mapping: form field name → preprocessor column name
        column_mapping = {
            'branch': 'What is your branch',
            'coding': 'Enjoyment of Coding & Logical Problem Solving',
            'multidisciplinary': 'Interest in Multidisciplinary Learning (Beyond Your Branch)  ',
            'hackathons': 'How Often Do You Participate in Hackathons/Technical Competitions?  ',
            'field_interest': '  Field of Interest ',
            'entrepreneurship': 'Interest in Entrepreneurship & Startups ',
            'govt_jobs': 'Interest in Government Jobs & Civil Services ',
            'sports': '  Sports & Fitness Activity Level  ',
            'social_work': '  Interest in Social Work & Volunteering  ',
            'career_plan': 'Career Preferences & Future Plans  ',
            'study_env': 'Preferred Study Environment',
            'hostel': '  Do You Enjoy Living in a Hostel?  ',
            'motivation': 'What’s Your Main Motivation for Attending College?',
            'native_state': 'Do you belong to the state where MNIT is located?',
            'city_life': 'Do you enjoy city life',
            'infra': 'Satisfied with college Infrastructure',
            'research': 'Interest in Research field'
        }

        # Load expected categories
        with open("preprocessor_categories.json") as f:
            expected_categories = json.load(f)

        # Fix any mismatched values
        for form_key, col_name in column_mapping.items():
        # Normalize quotes (straight to curly) for fair matching
            user_val = input_df.at[0, col_name]
            normalized_val = user_val.replace("'", "’").strip()

            valid_vals = expected_categories[col_name]

            # Try exact match, case-insensitive match, or substring match
            if normalized_val not in valid_vals:
                fixed = next((v for v in valid_vals if v.lower() == normalized_val.lower()), None)
                if not fixed:
                    fixed = next((v for v in valid_vals if normalized_val.lower() in v.lower()), None)
                if fixed:
                    input_df.at[0, col_name] = fixed
                else:
                    raise ValueError(f"Invalid value '{user_val}' for column '{col_name}'. Expected one of: {valid_vals}")
            else:
                input_df.at[0, col_name] = normalized_val  # 💥 Add this line to handle exact matches


        # Transform and predict
        input_encoded = preprocessor.transform(input_df)
        print("=== INPUT TO MODEL ===")
        print(input_df)
        print("=== ENCODED INPUT ===")
        print(input_encoded)


        # Show class probabilities for debugging
        proba = model.predict(input_encoded)
        print("Prediction probabilities:", proba)
        predicted_class = int(np.argmax(proba, axis=1)[0])

        print("Raw Prediction:", model.predict(input_encoded))


        output = model.predict(input_encoded)
        predicted_class = int(np.argmax(output, axis=1)[0])
        prediction = label_map[predicted_class]


    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
