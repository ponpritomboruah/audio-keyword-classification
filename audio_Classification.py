import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 

DATASET_PATH = pathlib.Path("data/mini_speech_commands")
ALT_DATASET_PATH = pathlib.Path("data/mini_speech_commands_extracted/mini_speech_commands")
MINI_SPEECH_COMMANDS_SHA256 = "49650f2341b26d886b46b3f4fb8fed59e30300b17550f1ee4a768b3106cf93a0"

if not DATASET_PATH.exists() and not ALT_DATASET_PATH.exists():
    tf.keras.utils.get_file(
        "mini_speech_commands.zip",
        origin="https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        file_hash=MINI_SPEECH_COMMANDS_SHA256,
        hash_algorithm="sha256",
        extract=True,
        cache_dir=".",
        cache_subdir="data",
    )

if ALT_DATASET_PATH.exists():
    DATASET_PATH = ALT_DATASET_PATH

commands = sorted(item.name for item in DATASET_PATH.iterdir() if item.is_dir())
print("Commands:", commands)


#This module uses only the yes and no classes.

yes_files = sorted((DATASET_PATH / "yes").glob("*.wav"))
no_files = sorted((DATASET_PATH / "no").glob("*.wav"))

print(f"Yes examples: {len(yes_files)}")
print(f"No examples: {len(no_files)}")


yes_files = sorted((DATASET_PATH / "yes").glob("*.wav"))
no_files = sorted((DATASET_PATH / "no").glob("*.wav"))

print(f"Yes examples: {len(yes_files)}")
print(f"No examples: {len(no_files)}")

#Convert a WAV file to a tensor

def load_waveform(file_path):
    audio_binary = tf.io.read_file(str(file_path))
    waveform, sample_rate = tf.audio.decode_wav(
        audio_binary,
        desired_channels=1,
        desired_samples=16000,
    )
    waveform = tf.squeeze(waveform, axis=-1)
    return waveform, sample_rate


example_file = yes_files[0]
waveform, sample_rate = load_waveform(example_file)

print("Waveform shape:", waveform.shape)
print("Sample rate:", sample_rate.numpy())
print("Value range:", float(tf.reduce_min(waveform)), "to", float(tf.reduce_max(waveform)))



#waveform

timescale = np.arange(waveform.shape[0])

plt.figure(figsize=(12, 4))
plt.plot(timescale, waveform.numpy())
plt.title("Waveform for a 'yes' example")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.xlim([0, 16000])
plt.show()

#Visualizing and transforming data

import pathlib
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

SOURCE_DATASET_PATH = pathlib.Path("data/mini_speech_commands")
ALT_SOURCE_DATASET_PATH = pathlib.Path("data/mini_speech_commands_extracted/mini_speech_commands")
BINARY_DATASET_PATH = pathlib.Path("data/speech_commands_yes_no")
MINI_SPEECH_COMMANDS_SHA256 = "49650f2341b26d886b46b3f4fb8fed59e30300b17550f1ee4a768b3106cf93a0"

if not SOURCE_DATASET_PATH.exists() and not ALT_SOURCE_DATASET_PATH.exists():
    tf.keras.utils.get_file(
        "mini_speech_commands.zip",
        origin="https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        file_hash=MINI_SPEECH_COMMANDS_SHA256,
        hash_algorithm="sha256",
        extract=True,
        cache_dir=".",
        cache_subdir="data",
    )

if ALT_SOURCE_DATASET_PATH.exists():
    SOURCE_DATASET_PATH = ALT_SOURCE_DATASET_PATH

for label in ("no", "yes"):
    target_dir = BINARY_DATASET_PATH / label
    target_dir.mkdir(parents=True, exist_ok=True)

    for source_file in (SOURCE_DATASET_PATH / label).glob("*.wav"):
        target_file = target_dir / source_file.name
        if not target_file.exists():
            shutil.copy2(source_file, target_file)




#Load audio files as TensorFlow datasets

SEED = 42
BATCH_SIZE = 64

tf.random.set_seed(SEED)
np.random.seed(SEED)

train_ds, validation_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=BINARY_DATASET_PATH,
    label_mode="int",
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="both",
    seed=SEED,
    output_sequence_length=16000,
)

label_names = np.array(train_ds.class_names)
print("Label names:", label_names)

holdout_ds = validation_ds.unbatch().cache()
val_ds = holdout_ds.shard(num_shards=2, index=0).batch(BATCH_SIZE)
test_ds = holdout_ds.shard(num_shards=2, index=1).batch(BATCH_SIZE)


#The dataset returns audio tensors with a channel dimension. Because these files are mono, remove the extra channel dimension before creating spectrograms.



def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels


train_ds = train_ds.map(squeeze, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(squeeze, num_parallel_calls=tf.data.AUTOTUNE)

for example_audio, example_labels in train_ds.take(1):
    print("Audio batch shape:", example_audio.shape)
    print("Label batch shape:", example_labels.shape)




#Visualize a waveform

example_waveform = example_audio[0]
example_label = label_names[example_labels[0].numpy()]

plt.figure(figsize=(12, 4))
plt.plot(example_waveform.numpy())
plt.title(f"Waveform for '{example_label}'")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.xlim([0, 16000])
plt.show()

#Create spectrograms

def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(
        waveform,
        frame_length=255,
        frame_step=128,
    )
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


example_spectrogram = get_spectrogram(example_waveform)
print("Spectrogram shape:", example_spectrogram.shape)


#Visualize a spectrogram

def plot_spectrogram(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        spectrogram = np.squeeze(spectrogram, axis=-1)

    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    time_steps = np.arange(width)
    frequency_bins = np.arange(height)
    ax.pcolormesh(time_steps, frequency_bins, log_spec)
    ax.set_xlabel("Time frame")
    ax.set_ylabel("Frequency bin")


fig, axes = plt.subplots(2, figsize=(12, 8))

axes[0].plot(example_waveform.numpy())
axes[0].set_title("Waveform")
axes[0].set_xlim([0, 16000])

plot_spectrogram(example_spectrogram.numpy(), axes[1])
axes[1].set_title("Spectrogram")

plt.suptitle(example_label.title())
plt.show()

#Create spectrogram datasets

def make_spectrogram_dataset(dataset):
    return dataset.map(
        map_func=lambda audio, label: (get_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


train_spectrogram_ds = make_spectrogram_dataset(train_ds)
val_spectrogram_ds = make_spectrogram_dataset(val_ds)
test_spectrogram_ds = make_spectrogram_dataset(test_ds)

train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(1000, seed=SEED).prefetch(tf.data.AUTOTUNE)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

for spectrograms, labels in train_spectrogram_ds.take(1):
    print("Spectrogram batch shape:", spectrograms.shape)
    print("Label batch shape:", labels.shape)


#Inspect the model input


import pathlib

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

for example_spectrograms, example_labels in train_spectrogram_ds.take(1):
    input_shape = example_spectrograms.shape[1:]
    break

num_labels = len(label_names)

print("Input shape:", input_shape)
print("Number of labels:", num_labels)
print("Labels:", label_names)



#Create the model


normalization_layer = layers.Normalization()
normalization_layer.adapt(
    data=train_spectrogram_ds.map(lambda spectrogram, label: spectrogram)
)

model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Resizing(32, 32),
    normalization_layer,
    layers.Conv2D(32, 3, activation="relu"),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])

model.summary()

#Compile and train the model - Use the Adam optimizer and sparse categorical cross-entropy. Although the task has two classes, this model uses two output logits, one for no and one for yes, rather than one sigmoid output. That design matches the integer labels created with label_mode="int" in the previous unit. The model's final Dense layer outputs raw logits (no softmax activation), so set from_logits=True so the loss applies a numerically stable softmax internally.



model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history = model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=10,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )
    ],
)



#Plot training history

metrics = history.history

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.epoch, metrics["loss"], label="Training loss")
plt.plot(history.epoch, metrics["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.epoch, metrics["accuracy"], label="Training accuracy")
plt.plot(history.epoch, metrics["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()


#Evaluate on the test set

test_metrics = model.evaluate(test_spectrogram_ds, return_dict=True)
print(test_metrics)


#Confusion Matrix

predicted_batches = []
true_batches = []

for spectrograms, labels in test_spectrogram_ds:
    logits = model(spectrograms, training=False)
    predicted_batches.append(tf.argmax(logits, axis=1))
    true_batches.append(labels)

predicted_labels = tf.concat(predicted_batches, axis=0)
true_labels = tf.concat(true_batches, axis=0)

confusion_matrix = tf.math.confusion_matrix(
    true_labels,
    predicted_labels,
    num_classes=num_labels,
)
print(confusion_matrix.numpy())

#Run inference on one audio file

def load_waveform(file_path):
    audio_binary = tf.io.read_file(str(file_path))
    waveform, sample_rate = tf.audio.decode_wav(
        audio_binary,
        desired_channels=1,
        desired_samples=16000,
    )
    waveform = tf.squeeze(waveform, axis=-1)
    return waveform, sample_rate


sample_file = next((BINARY_DATASET_PATH / "no").glob("*.wav"))

sample_waveform, sample_rate = load_waveform(sample_file)
sample_spectrogram = get_spectrogram(sample_waveform)

logits = model(sample_spectrogram[tf.newaxis, ...], training=False)
predicted_index = tf.argmax(logits[0]).numpy()
predicted_label = label_names[predicted_index]

print("Sample file:", sample_file)
print("Predicted label:", predicted_label)




#Optional: test your own voice
#You can test the model with your own WAV files. Record short clips of yourself saying "yes" and "no". Keep each clip close to one second and minimize background noise. Export the files as 16 kHz mono 16-bit PCM WAV files so they match the training data and can be decoded by tf.audio.decode_wav. If your recording tool exports a different sample rate, resample the files to 16 kHz before using this code. The desired_samples=16000 argument pads or trims samples; it doesn't convert a 44.1 kHz or 48 kHz recording to 16 kHz audio. Update the paths in custom_files to match the files you create.

# def load_voice_sample(file_path):
#     audio_binary = tf.io.read_file(str(file_path))
#     waveform, sample_rate = tf.audio.decode_wav(
#         audio_binary,
#         desired_channels=1,
#         desired_samples=16000,
#     )

#     if int(sample_rate.numpy()) != 16000:
#         raise ValueError("Use a 16 kHz WAV file, or resample the audio to 16 kHz before inference.")

#     waveform = tf.squeeze(waveform, axis=-1)
#     return waveform


# custom_files = {
#     "no": pathlib.Path("data/myvoice/no.wav"),
#     "yes": pathlib.Path("data/myvoice/yes.wav"),
# }

# missing_files = [file_path for file_path in custom_files.values() if not file_path.exists()]

# if missing_files:
#     print("Create these WAV files before running the optional custom-voice example:")
#     for file_path in missing_files:
#         print(file_path)
# else:
#     for expected_label, file_path in custom_files.items():
#         waveform = load_voice_sample(file_path)
#         spectrogram = get_spectrogram(waveform)
#         logits = model(spectrogram[tf.newaxis, ...], training=False)
#         predicted_label = label_names[tf.argmax(logits[0]).numpy()]

#         print(f"Expected: {expected_label}; predicted: {predicted_label}")

