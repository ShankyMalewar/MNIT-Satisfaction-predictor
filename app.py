from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import time
import json

app = Flask(__name__)

# Label mapping from model output
label_map = {
    0: "Very Dissatisfied",
    1: "Dissatisfied",
    2: "Neutral",
    3: "Satisfied",
    4: "Very Satisfied"
}

# Lazy load model and preprocessor
model = None
preprocessor = None

def get_model():
    global model
    if model is None:
        print("📦 Loading model...")
        with open('rf.pkl', 'rb') as f:
            model = pickle.load(f)
    return model

def get_preprocessor():
    global preprocessor
    if preprocessor is None:
        print("🔧 Loading preprocessor...")
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
    return preprocessor

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            start_time = time.time()
            print("🚀 Starting prediction pipeline...")

            model = get_model()
            preprocessor = get_preprocessor()

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

            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data], columns=preprocessor.feature_names_in_)

            # Normalize values
            for form_key, col_name in column_mapping.items():
                user_val = input_df.at[0, col_name]
                normalized_val = user_val.replace("'", "’").strip()
                valid_vals = expected_categories[col_name]

                if normalized_val not in valid_vals:
                    fixed = next((v for v in valid_vals if v.lower() == normalized_val.lower()), None)
                    if not fixed:
                        fixed = next((v for v in valid_vals if normalized_val.lower() in v.lower()), None)
                    if fixed:
                        input_df.at[0, col_name] = fixed
                    else:
                        print(f"❌ Invalid value: {user_val} for column: {col_name}")
                        return render_template('index.html', prediction=f"Invalid input: '{user_val}' for '{col_name}'")
                else:
                    input_df.at[0, col_name] = normalized_val

            print("🧹 Input cleaned:", input_df)

            input_encoded = preprocessor.transform(input_df)
            print("🔢 Encoded input:", input_encoded)

            output = model.predict(input_encoded)
            predicted_class = int(np.argmax(output, axis=1)[0])
            prediction = label_map[predicted_class]

            print("✅ Prediction complete in", time.time() - start_time, "seconds.")
            print("🎯 Predicted:", prediction)

        except Exception as e:
            print("🔥 Exception during prediction:", str(e))
            return render_template('index.html', prediction="Error during prediction: " + str(e))

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
