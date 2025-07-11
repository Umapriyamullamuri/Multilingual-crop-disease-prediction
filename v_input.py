import speech_recognition as sr
import pyaudio

def mic_to_text():
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            print("Speak now...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source)

        try:
            # Use Google Web Speech API to recognize audio
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text

        except sr.UnknownValueError:
            print("Could not understand your voice.")
            return None

        except sr.RequestError as e:
            print(f"Google service error: {e}")
            return None

    except OSError as mic_error:
        print(f"Microphone error: {mic_error}")
        print("Make sure PyAudio is installed and your microphone is working.")
        return None

if __name__ == "__main__":
    mic_to_text()
