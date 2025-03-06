import numpy as np
import librosa
import scipy.signal
import matplotlib.pyplot as plt
import io
import base64

def process_audio_signal(file_path):
    # Charger le signal audio
    y, sr = librosa.load(file_path)
    
    # Analyse spectrale
    stft = librosa.stft(y)
    spectrogram = librosa.amplitude_to_db(np.abs(stft))
    
    # Spectrogramme
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogramme')
    spectro_plot = get_image_base64()
    
    # Transformée de Fourier
    ft = np.fft.fft(y)
    frequencies = np.fft.fftfreq(len(y), 1/sr)
    
    # Plot de la Transformée de Fourier
    plt.figure(figsize=(12, 8))
    plt.plot(frequencies[:len(frequencies)//2], np.abs(ft)[:len(ft)//2])
    plt.title('Transformée de Fourier')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Amplitude')
    ft_plot = get_image_base64()
    
    # Extraction de caractéristiques
    features = {
        'duration': librosa.get_duration(y=y, sr=sr),
        'sample_rate': sr,
        'total_samples': len(y),
        'rms_energy': np.sqrt(np.mean(y**2)),
        'zero_crossing_rate': librosa.feature.zero_crossing_rate(y)[0][0],
        'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    }
    
    return {
        'features': features,
        'spectrogramme': spectro_plot,
        'fourier_transform': ft_plot
    }

def get_image_base64():
    """Convertir le plot matplotlib en base64"""
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return image_base64