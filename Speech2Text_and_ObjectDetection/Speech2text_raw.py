import sounddevice as sd
import numpy as np
from pathlib import Path
from openvino.runtime import Core, Tensor
import librosa
import scipy
import os
import sys
import cv2

def audio_to_mel(audio, sampling_rate):
    assert sampling_rate == 16000, "Only 16 KHz audio supported"
    preemph = 0.97
    preemphased = np.concatenate([audio[:1], audio[1:] - preemph * audio[:-1].astype(np.float32)])

    # Calculate the window length.
    win_length = round(sampling_rate * 0.02)

    # Based on the previously calculated window length, run short-time Fourier transform.
    spec = np.abs(librosa.stft(preemphased, n_fft=512, hop_length=round(sampling_rate * 0.01),
                  win_length=win_length, center=True, window=scipy.signal.windows.hann(win_length), pad_mode='reflect'))

    # Create mel filter-bank, produce transformation matrix to project current values onto Mel-frequency bins.
    mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=512, n_mels=64, fmin=0.0, fmax=8000.0, htk=False)
    return mel_basis, spec

def mel_to_input(mel_basis, spec, padding=16):
    # Convert to a logarithmic scale.
    log_melspectrum = np.log(np.dot(mel_basis, np.power(spec, 2)) + 2 ** -24)

    # Normalize the output.
    normalized = (log_melspectrum - log_melspectrum.mean(1)[:, None]) / (log_melspectrum.std(1)[:, None] + 1e-5)

    # Calculate padding.
    remainder = normalized.shape[1] % padding
    if remainder != 0:
        return np.pad(normalized, ((0, 0), (0, padding - remainder)))[None]
    return normalized[None]

def ctc_greedy_decode(predictions):
    previous_letter_id = blank_id = len(alphabet) - 1
    transcription = list()
    for letter_index in predictions:
        if previous_letter_id != letter_index != blank_id:
            transcription.append(alphabet[letter_index])
        previous_letter_id = letter_index
    return ''.join(transcription)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("Camera")

# Set up OpenVINO model
model_folder = "model"
precision = "FP16"
model_name = "quartznet-15x5-en"
alphabet = " abcdefghijklmnopqrstuvwxyz'~"

ie = Core()
model = ie.read_model(
    model=f"{model_folder}/public/{model_name}/{precision}/{model_name}.xml"
)
model_input_layer = model.input(0)
shape = model_input_layer.partial_shape
shape[2] = -1
model.reshape({model_input_layer: shape})
compiled_model = ie.compile_model(model=model, device_name="CPU")
output_layer_ir = compiled_model.output(0)

# Setup stream for mic
fs = 16000  # Sample rate
seconds = 3  # Duration of recording

while (cap.isOpened()):
    key = cv2.waitKey(1) & 0xFF
    ret, frame = cap.read()
    if ret is False:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Display
    cv2.imshow("Camera",frame)

    # if the `s` key was pressed, start recording
    if key == ord("s"):
        print("Recording started.")
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    # print("Listening...")
    # myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    # if the `e` key was pressed, stop recording
    if key == ord("e"):
        print("Recording ended.")
        sd.wait()  # Wait until recording is finished
        # Process and print the recording.
        print("Processing...")
    
        # use the recording here as `audio`
        if max(np.abs(myrecording)) <= 1:
            myrecording = (myrecording * (2**15 - 1))
        myrecording = myrecording.astype(np.int16)

        mel_basis, spec = audio_to_mel(audio=myrecording.flatten(), sampling_rate=fs)
        audio = mel_to_input(mel_basis=mel_basis, spec=spec)
        character_probabilities = compiled_model([Tensor(audio)])[output_layer_ir]

        # Remove unnececery dimension
        character_probabilities = np.squeeze(character_probabilities)

        # Run argmax to pick most possible symbols
        character_probabilities = np.argmax(character_probabilities, axis=1)
        transcription = ctc_greedy_decode(character_probabilities)
        print(transcription)
    
    # sd.wait()  # Wait until recording is finished
    # print("Recording ended.")
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()