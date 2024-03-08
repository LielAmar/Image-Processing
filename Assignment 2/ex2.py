import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt

from scipy.fft import fft, ifft

def plot_frequencies(audio_data, sample_rate):
  fourier_coeff = np.fft.fft(audio_data)
  frequencies = np.fft.fftfreq(len(fourier_coeff))

  plt.figure(figsize=(10, 4))
  plt.plot(frequencies, np.abs(fourier_coeff))
  plt.title('Fourier Transform - Magnitude Spectrum')
  plt.xlabel('Frequency (Hz)')
  plt.ylabel('Magnitude')
  plt.show()

def plot_spectogram(audio_data, sample_rate):
  frequencies, times, fourier_coeff = signal.stft(audio_data, fs=sample_rate)

  plt.pcolormesh(times, frequencies, np.log10(np.abs(fourier_coeff) + 0.01), 
                 shading='auto')
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')
  plt.title('STFT Spectrogram')
  plt.colorbar(label='Magnitude')
  plt.show()

def q1(audio_path) -> np.array:
    """
    :param audio_path: path to q1 audio file
    :return: return q1 denoised version
    """

    #  Read the audio file
    sample_rate, audio_data = wav.read(audio_path)

    # Compute one-dimensional FFT of the audio data
    # Returns a list of complex numbers
    fourier_coeff = fft(audio_data)

    # Find the indeces of the frequency with the maximum magnitude, and then
    # take the symmetric frequency as well
    max_freq_index = np.argmax(np.absolute(fourier_coeff))
    max_freq_indices = [max_freq_index, -max_freq_index]

    # Set the magnitude of the max frequencies to 0
    fourier_coeff[max_freq_indices] = 0

    # Compute the inverse FFT
    denoised_audio_data = ifft(fourier_coeff).real
    
    # Save the denoised audio to a file
    # wav.write('output/q1.wav', sample_rate, denoised_audio_data)

    return denoised_audio_data


def q2(audio_path) -> np.array:
    """
    :param audio_path: path to q2 audio file
    :return: return q2 denoised version
    """

    # Read the audio file
    sample_rate, audio_data = wav.read(audio_path)

    # Compute the STFT of the audio data, which returns 3 arrays,
    # frequencies (1D), times (1D) and Zxx (2D)
    frequencies, times, fourier_coeff = signal.stft(audio_data, fs=sample_rate, nperseg=250)

    # Loop over all times between (~0.5, 2) seconds, and set the magnitude of
    # all frequencies with mag < 0.02, which are between (650, 850) to 0.
    for i in range(int(len(times)/10)*3, len(times)):
      freq_indices = np.argwhere((frequencies < 650) 
                                     & (frequencies > 550))
      fourier_coeff[freq_indices, i] = 0

    # Compute the inverse STFT
    _, denoised_audio_data = signal.istft(fourier_coeff, fs=sample_rate)

    # Save the denoised audio to a file
    # wav.write('output/q2.wav', sample_rate, denoised_audio_data)

    return denoised_audio_data

# if __name__ == '__main__':
#   q1('input/q1.wav')
#   q2('input/q2.wav')