import sounddevice as sd
import numpy as np
from pathlib import Path
from openvino.runtime import Core, Tensor
import librosa
import scipy
import cv2
import collections
import time

def process_results(frame, results, thresh = 0.6):
	# The size of the original frame.
	h, w = frame.shape[:2]
	# The 'results' variable is a [1, 1, 100, 7] tensor.
	results = results.squeeze()
	boxes = []
	labels = []
	scores = []
	for _, label, score, xmin, ymin, xmax, ymax in results:
		# Create a box with pixels coordinates from the box with normalized coordinates [0,1].
		boxes.append(
			tuple(map(int, (xmin * w, ymin * h, (xmax - xmin) * w, (ymax - ymin) * h)))
		)
		labels.append(int(label))
		scores.append(float(score))

	# Apply non-maximum suppression to get rid of many overlapping entities.
	# See https://paperswithcode.com/method/non-maximum-suppression
	# This algorithm returns indices of objects to keep.
	indices = cv2.dnn.NMSBoxes(
		bboxes = boxes, scores = scores, score_threshold = thresh, nms_threshold = 0.6
	)

	# If there are no boxes.
	if len(indices) == 0:
		return []

	# Filter detected objects.
	return [(labels[idx], scores[idx], boxes[idx]) for idx in indices.flatten()]


def draw_boxes(frame, boxes, target_label, changed_label):
	for label, score, box in boxes:
		color = (255, 255, 255)
		if label >= 0 and label < len(classes):
			color = tuple(map(int, colors[label]))
		# Draw a box.
		x2 = box[0] + box[2]
		y2 = box[1] + box[3]
		cv2.rectangle(img = frame, pt1 = box[:2], pt2 = (x2, y2), color = color, thickness = 3)
		# Ensure the label is within the bounds of the classes list
		if label >= 0 and label < len(classes):
			# Choose color for the label.
			label_text = classes[label] if target_label is None or classes[label] != target_label else changed_label
		else:
			print(f"Label {label} is out of range")
		# Draw a label name inside the box.
		# if not target_classes:
		cv2.putText(
			img = frame,
			# text = f"{classes[label]} {score:.2f}",
			text = f"{label_text} {score:.2f}",
			org = (box[0] + 10, box[1] + 30),
			fontFace = cv2.FONT_HERSHEY_COMPLEX,
			fontScale = frame.shape[1] / 1000,
			color = color,
			thickness = 1,
			lineType = cv2.LINE_AA,
		)

	return frame

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

# Set up OpenVINO model
# Speech to text
model_folder = "model_speech2text"
precision = "FP16"
model_name = "quartznet-15x5-en"
alphabet = " abcdefghijklmnopqrstuvwxyz'~"

ie = Core()
model = ie.read_model(
	model=f"{model_folder}/public/{model_name}/{precision}/{model_name}.xml"
)
compiled_model = ie.compile_model(model=model, device_name="CPU")
model_input_layer = model.input(0)
shape = model_input_layer.partial_shape
shape[2] = -1
model.reshape({model_input_layer: shape})
output_layer_ir = compiled_model.output(0)

# Object detection
model_folder2 = "model_detection"
model_name2 = "ssdlite_mobilenet_v2_fp16"

ie = Core()
model2 = ie.read_model(
	model=f"{model_folder2}/{model_name2}.xml"
)
compiled_model2 = ie.compile_model(model=model2, device_name="CPU")
model_input_layer2 = model2.input(0)
output_layer_ir2 = compiled_model2.output(0)

height, width = list(model_input_layer2.shape)[1:3]

# 클래스 명칭 리스트 설정
classes = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 
		   'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
		   'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
		   'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 
		   'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 
		   'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
		   'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 
		   'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 
		   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 
		   'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
		   'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 
		   'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Colors for the classes above (Rainbow Color Map).
colors = cv2.applyColorMap(
	src = np.arange(0, 255, 255 / len(classes), dtype = np.float32).astype(np.uint8),
	colormap = cv2.COLORMAP_RAINBOW,
).squeeze()

# # Setup stream for mic
fs = 16000  # Sample rate
buffer = []

def record_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    buffer.append(indata.copy())

# For use with sd.InputStream() as callback
stream = sd.InputStream(callback=record_callback, channels=1, samplerate=fs)

command = None
target_label = None  # target_label is defined after recording starts
changed_label = None  # changed_label is defined after recording starts
processing_times = collections.deque()

while True:
	ret, frame = cap.read()
	if not ret:
		print("Can't receive frame (stream end?). Exiting ...")
		break

	# flip the frame if needed
	frame = cv2.flip(frame, 1)
	#height, width = frame.shape[:2]

	# resize the frame to a smaller size for speed purposes
	scale = 1280 / max(frame.shape)
	if scale < 1:
		frame = cv2.resize(
			src=frame, 
			dsize=None,
			fx=scale, 
			fy=scale, 
			interpolation=cv2.INTER_AREA
		)

	# Resize the image and change dims to fit neural network input.
	resized_img = cv2.resize(src=frame, dsize=(width, height), interpolation=cv2.INTER_AREA)
	resized_img = resized_img[np.newaxis, ...]  # Add batch dimension

	# Measure processing time.
	start_time = time.time()
	# Get the results.
	results = compiled_model2([resized_img])[output_layer_ir2]
	stop_time = time.time()

	# Get poses from network results.
	boxes = process_results(frame = frame, results = results)

	# Draw boxes on a frame.
	frame = draw_boxes(
		frame = frame,
		boxes = boxes, 
		target_label=target_label, 
		changed_label=changed_label
	)

	processing_times.append(stop_time - start_time)
	# Use processing times from last 200 frames.
	if len(processing_times) > 200:
		processing_times.popleft()

	_, f_width = frame.shape[:2]
	# Mean processing time [ms].
	processing_time = np.mean(processing_times) * 1000
	fps = 1000 / processing_time
	cv2.putText(
		img = frame,
		text = f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
		org = (20, 40),
		fontFace = cv2.FONT_HERSHEY_COMPLEX,
		fontScale = f_width / 1000,
		color = (0, 0, 255),
		thickness = 1,
		lineType = cv2.LINE_AA,
	)
		
	# show the frame
	cv2.imshow('Object Detection', frame)

	# get the key press
	key = cv2.waitKey(1)
	# Check if the 's' key was pressed to start recording
	if key == ord("s"):
		print("Recording started")
		# myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
		buffer.clear()  # Reset buffer
		stream.start()
		start_time2 = time.time()

	# Check if the 'e' key was pressed to stop recording
	if key == ord("e"):
		print("Recording ended")
		stream.stop()
		end_time2 = time.time()
		myrecording = np.concatenate(buffer, axis=0)
		print(f"Recorded for {end_time2 - start_time2} seconds")

		# Process and print the recording
		print("Processing...")
		if max(np.abs(myrecording)) <= 1:
			myrecording = myrecording * (2 ** 15 - 1)
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

		# Check if the command is "change"
		if transcription and "change" in transcription:
			words = transcription.split(" ")
			if len(words) > 1: #change person to man
				target_label = words[-3]
				changed_label = words[-1]
				print(f"Target class '{target_label}' changed to '{changed_label}'")
			# Check if target_label and changed_label are in classes
			if target_label not in classes:
				print(f"'{target_label}' is not a valid class.")
				target_label = None
				changed_label = None
			if changed_label in classes:
				print(f"'{changed_label}' already exists in classes. Choose a different label.")
				target_label = None
				changed_label = None
				continue

	# if the `q` key was pressed, break from the loop
	if key == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()