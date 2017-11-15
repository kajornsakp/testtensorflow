import coremltools as coreml
from keras.models import load_model

model = coreml.converters.keras.convert("keras_model.h5")
model.save('test_keras_datagen.mlmodel')

