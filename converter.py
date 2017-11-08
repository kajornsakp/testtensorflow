import coremltools as coreml

model = coreml.converters.keras.convert('first_keras.h5')
model.save('keras_first.mlmodel')