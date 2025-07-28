import speech_recognition as sr
import os


def recognize_speech(recognizer, microphone):
    with microphone as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        print("Recognizing...")
        command = recognizer.recognize_google(audio)
        print(f"You said: {command}")
        return command.lower()
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
        return None
    except sr.RequestError:
        print("Sorry, there was an error with the speech recognition service.")
        return None


def open_application(command):
    if "dashboard" in command:
        os.system("start http://103.20.212.103:8889/")
    elif "detail" in command:
        os.system("start http://103.20.212.103:8889/dashboard/DetailDashboard/")
    elif "panel" in command:
        os.system("start http://103.20.212.103:8889/AdminPanel/")
    elif "approvals" in command:
        os.system("start http://103.20.212.103:8889/Approvals/")
    elif "mapping" in command:
        os.system("start http://103.20.212.103:8889/Mapping/")
    elif "report" in command:
        os.system("start http://103.20.212.103:8889/Reports/")
    elif "holidays" in command:
        os.system("start http://103.20.212.103:8889/holidays/")
    elif "sfc" in command:
        os.system("start http://103.20.212.103:8889/sfc/")
    else:
        print("Command not recognized. Please try again.")


def main():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    print("Voice navigation script is running!")
    print("You can say commands like 'open blog', 'open contact', 'open home', 'open website'.")

    while True:
        command = recognize_speech(recognizer, microphone)
        if command:
            if "exit" in command:
                print("Exiting...")
                break
            open_application(command)
        else:
            print("No valid command recognized. Listening again...")


if __name__ == "__main__":
    main()
