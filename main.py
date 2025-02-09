import pyttsx3 as p
import speech_recognition as sr
from selenium_web import Info
from News import news
import randfacts
from jokes import joke
from weather import temp, des
import datetime
import sys

# Initialize Text-to-Speech
engine = p.init()
engine.setProperty('rate', 130)
engine.setProperty('voice', engine.getProperty('voices')[1].id)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def wishme():
    hour = int(datetime.datetime.now().hour)
    if 0 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 16:
        return "Afternoon"
    else:
        return "Evening"

# Greet User
today_date = datetime.datetime.now()
speak("Hello, Good " + wishme() + ". I'm your voice assistant.")
speak("Today is " + today_date.strftime("%d") + " of " + today_date.strftime("%B") + ", And it's currently " +
      today_date.strftime("%I") + ":" + today_date.strftime("%M") + " " + today_date.strftime("%p"))
speak("Temperature in Hyderabad is " + str(temp()) + " degree Celsius and with " + str(des()))
speak("What can I do for you?")

# Main Listening and Processing Loop
r = sr.Recognizer()

while True:
    with sr.Microphone() as mic_source:
        r.energy_threshold = 4000
        r.adjust_for_ambient_noise(mic_source)
        print("Listening for your command...")
        try:
            audio = r.listen(mic_source, timeout=5)
            command = r.recognize_google(audio).lower()
            print(f"You said: {command}")

            if "information" in command:
                speak("You need information on what topic?")
                with sr.Microphone() as topic_source:
                    r.energy_threshold = 4000
                    r.adjust_for_ambient_noise(topic_source, duration=1)
                    print("Listening for topic...")
                    try:
                        topic_audio = r.listen(topic_source)
                        infor = r.recognize_google(topic_audio)

                        speak(f"Searching {infor} in Wikipedia")
                        print(f"Searching {infor} in Wikipedia")

                        # Using selenium_web to search for the information
                        assist = Info()  # Creating an instance of Info
                        assist.get_info(infor)  # Search for the topic
                    except sr.UnknownValueError:
                        speak("Sorry, I couldn't understand the topic. Please try again.")
                    except sr.RequestError:
                        speak("There was an issue with the speech recognition service.")
            elif "news" in command:
                speak("Fetching the latest news.")
                arr = news()
                for item in arr:
                    print(item)
                    speak(item)
            elif "fact" in command:
                speak("Here's a fun fact.")
                fact = randfacts.get_fact()
                print(fact)
                speak(f"Did you know? {fact}")
            elif "joke" in command:
                speak("Let me tell you a joke.")
                jokes = joke()
                for j in jokes:
                    print(j)
                    speak(j)
            elif "exit" in command or "quit" in command:
                speak("Goodbye! Have a great day!")
                sys.exit()
            else:
                speak("I'm sorry, I didn't understand. Can you please repeat?")
        except sr.UnknownValueError:
            speak("Sorry, I couldn't understand. Could you repeat?")
        except sr.RequestError:
            speak("There was an issue with the speech recognition service.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            speak("An unexpected error occurred. Please try again.")