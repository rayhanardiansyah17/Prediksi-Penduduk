import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Data jumlah penduduk
years = np.array([2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023], dtype=float)
population = np.array([2833, 2848, 2862, 2874, 2885, 2896, 2874, 2880, 2887, 2893], dtype=float)

# Normalisasi data
years_min, years_max = years.min(), years.max()
population_min, population_max = population.min(), population.max()

years_normalized = (years - years_min) / (years_max - years_min)
population_normalized = (population - population_min) / (population_max - population_min)

# Membuat model
model = Sequential()
model.add(Dense(128, input_dim=1, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))  # Menambahkan Dropout untuk regularisasi
model.add(Dense(128, activation='relu'))
model.add(Dense(1))


# Kompilasi model
model.compile(optimizer='adam', loss='mean_squared_error')

# Melatih model
model.fit(years_normalized, population_normalized, epochs=1000, verbose=0)

# Simpan model
model.save('population.h5')