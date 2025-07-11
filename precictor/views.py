import os
import pickle
import pandas as pd
import difflib
from django.shortcuts import render
from django.conf import settings
from googletrans import Translator
from sklearn.preprocessing import LabelEncoder
import speech_recognition as sr

# Load model and dataset
data_path = os.path.join(settings.BASE_DIR, 'precictor', 'ml', 'crop_disease_data.csv')
model_path = os.path.join(settings.BASE_DIR, 'precictor', 'ml', 'model.pkl')

# Load the model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load and preprocess the dataset
df = pd.read_csv(data_path)
translator = Translator()

# Clean dataset
df['symptoms'] = df['symptoms'].str.lower().str.strip()
df['crop'] = df['crop'].str.lower().str.strip()

# Label encoding
m = LabelEncoder()
n = LabelEncoder()
m.fit(df['symptoms'])
n.fit(df['crop'])

# Helper: Fuzzy match to known values
def fuzzy_match(input_text, known_list):
    matches = difflib.get_close_matches(input_text, known_list, n=1, cutoff=0.6)
    return matches[0] if matches else None

# Voice Input View
def v_input(request):
    prediction = None
    spoken_text = None

    if request.method == 'POST' and request.FILES.get('voice_input'):
        audio_file = request.FILES['voice_input']
        recognizer = sr.Recognizer()

        temp_path = os.path.join(settings.BASE_DIR, 'media', 'temp_voice.wav')
        with open(temp_path, 'wb+') as f:
            for chunk in audio_file.chunks():
                f.write(chunk)

        with sr.AudioFile(temp_path) as source:
            audio = recognizer.record(source)

        try:
            spoken_text = recognizer.recognize_google(audio)
            translated_symptom = translator.translate(spoken_text, dest='en').text.strip().lower()
            matched_symptom = fuzzy_match(translated_symptom, list(m.classes_))

            default_crop = 'wheat'  # or handle crop input via UI later
            matched_crop = fuzzy_match(default_crop, list(n.classes_))

            if matched_symptom and matched_crop:
                symptom_encoded = m.transform([matched_symptom])[0]
                crop_encoded = n.transform([matched_crop])[0]
                disease = model.predict([[symptom_encoded, crop_encoded]])[0]
                remedy = df[df['disease'] == disease]['remedy'].values[0]
                prediction = disease
            else:
                spoken_text = "Could not match symptom/crop"

        except sr.UnknownValueError:
            spoken_text = "Sorry, could not understand."
        except sr.RequestError:
            spoken_text = "Speech recognition service failed."

    return render(request, 'index.html', {
        'spoken_text': spoken_text,
        'prediction': prediction
    })

# Text Input View
def predict_disease(request):
    context = {}

    if request.method == "POST":
        symptom = request.POST.get('symptom', '').strip().lower()
        crop = request.POST.get('crop', '').strip().lower()
        lang = request.POST.get('lang', 'en')

        translated_symptom = translator.translate(symptom, dest='en').text.strip().lower()
        matched_symptom = fuzzy_match(translated_symptom, list(m.classes_))
        matched_crop = fuzzy_match(crop, list(n.classes_))

        if not matched_symptom or not matched_crop:
            context['error'] = "Unknown symptom or crop. Please try again with valid inputs."
            return render(request, 'index.html', context)

        symptom_encoded = m.transform([matched_symptom])[0]
        crop_encoded = n.transform([matched_crop])[0]

        disease = model.predict([[symptom_encoded, crop_encoded]])[0]
        disease_translated = translator.translate(disease, dest=lang).text
        remedy = df[df['disease'] == disease]['remedy'].values[0]
        remedy_translated = translator.translate(remedy, dest=lang).text

        context = {
            'disease': disease_translated,
            'remedy': remedy_translated
        }

    return render(request, 'index.html', context)
