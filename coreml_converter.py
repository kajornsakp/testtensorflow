import coremltools as coreml
from keras.engine import InputLayer
from keras.models import load_model

output_labels =['0','1','2','3','4','5','6','7',
                '8','9','10','11','12','13','14',
                '15','16','17','18','19','20',
                '21','22','23','24','25','26',
                '27','28','29','30','31']


coreml_model = coreml.converters.keras.convert('keras_model.h5',
                                               input_names="image",
                                               image_input_names='image',
                                               output_names='output',
                                               class_labels=output_labels,
                                               image_scale=1/255.)

coreml_model.save('keras_coreml.mlmodel')
