import pyaudio
import wave
import threading

# Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
FILENAME = "output.wav"

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Function to record audio
def record_audio():
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    while not stop_recording:
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    
    with wave.open(FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

# Flag to stop recording
stop_recording = False

# Start recording in a separate thread
recording_thread = threading.Thread(target=record_audio)
recording_thread.start()

# Wait for user to press Enter
input("Press Enter to stop recording...\n")
stop_recording = True
recording_thread.join()

# Terminate PyAudio
audio.terminate()
