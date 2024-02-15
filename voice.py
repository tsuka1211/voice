from google.colab import drive
drive.mount('/content/drive')
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ファイル読み込み
filename = "/content/drive/MyDrive/konolab/test.wav"
y, sr = librosa.load(filename)
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
times = librosa.times_like(f0)

#@title
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
fig, ax = plt.subplots()
# データ表示
img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)

ax.set(title='pYIN fundamental frequency estimation')
fig.colorbar(img, ax=ax, format="%+2.f dB")

# 基本周波数表示
ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
ax.legend(loc='upper right')

#@title
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
fig, ax = plt.subplots()
# データ表示
img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)

ax.set(title='pYIN fundamental frequency estimation')
# fig.colorbar(img, ax=ax, format="%+2.f dB")

# 基本周波数表示
ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
ax.legend(loc='upper right')

# 拡大
plt.xlim(225,227)
plt.show()

df_f0 = pd.Series(f0)
# df_f0 = df_f0.dropna(how='all')
# print(df_f0.describe())
df_f0.tail()

times

left = 0
right = 0
count = 0
threshold = 5 * (times.size / librosa.get_duration(y=y, sr=sr))
highlight = list()
right_array = list()
empty_array = list()

for item in f0:
    if not np.isnan(item):
      right += item
    right_array.append(item)
    empty_array.append(np.nan)
    count += 1
    if count >= threshold:
        if right - left > 50:
          highlight = highlight + right_array
        else:
          highlight = highlight + empty_array
        left,right = right, 0
        right_array = list()
        empty_array = list()
        right = 0
        count = 0

highlight = highlight + empty_array
highlight = np.array(highlight)

highlight

D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
fig, ax = plt.subplots()
# データ表示
img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)

ax.set(title='pYIN fundamental frequency estimation')
fig.colorbar(img, ax=ax, format="%+2.f dB")

# 基本周波数表示
ax.plot(times, f0, label='f0', color='cyan', linewidth=2)
ax.plot(times, highlight, label='highlight', color='green', linewidth=2)
ax.legend(loc='upper right')

# 拡大
plt.xlim(0, 1)
plt.show()

voice_freq = list()

started = False
start_time = -1
for index,item in enumerate(highlight):
  if not np.isnan(item) and not started:
      time = index * (librosa.get_duration(y=y, sr=sr) / times.size)
      # print(f"started: {time}")
      start_time = time
      started = True
  elif np.isnan(item) and started:
      time = index * (librosa.get_duration(y=y, sr=sr) / times.size)
      # print(f"ended: {time}")
      started = False
      voice_freq.append([start_time,time])
      # print(f"{(start_time,time)}\n")

