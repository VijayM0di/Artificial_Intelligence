import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pyaudio
import wave

def plot_waveform(signal, rate):
    if signal.ndim == 2:
        signal = signal.mean(axis=1)  # Convert to mono by averaging channels
    time = np.arange(0, len(signal)) / rate

    plt.figure(figsize=(10, 4))
    plt.plot(time, signal)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.show()

def plot_spectrum(signal, rate):
    if signal.ndim == 2:
        signal = signal.mean(axis=1)  # Convert to mono by averaging channels
    n = len(signal)
    k = np.arange(n)
    T = n/rate
    frq = k/T
    frq = frq[range(n//2)]

    Y = np.fft.fft(signal)/n
    Y = Y[range(n//2)]

    plt.figure(figsize=(10, 4))
    plt.plot(frq, abs(Y))
    plt.xlabel('Freq (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Frequency Spectrum')
    plt.show()

def record_audio(file_path, record_seconds=60, sample_rate=44100):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=1024)
    frames = []

    for i in range(0, int(sample_rate / 1024 * record_seconds)):
        data = stream.read(1024)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

def analyze_audio(file_path):
    rate, data = wavfile.read(file_path)
    plot_waveform(data, rate)
    plot_spectrum(data, rate)

# Example usage
file_path = r"C:\Users\dazau\Desktop\Taylor Swift - Blank Space.wav"
record_audio(file_path)
analyze_audio(file_path)
