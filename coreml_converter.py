import coremltools as coreml
from keras.engine import InputLayer
from keras.models import load_model
from PIL import Image
output_labels =['0','1','2','3','4','5','6','7',
                '8','9','10','11','12','13','14',
                '15','16','17','18','19','20',
                '21','22','23','24','25','26',
                '27','28','29','30','31']



coreml_model = coreml.converters.keras.convert("2017-12-05 13:59:25.028514.h5",image_input_names="input1",class_labels=output_labels)
coreml_model.author = "Kajornsak Peerapathananont"
coreml_model.license = "MIT"
coreml_model.input_description['input1'] = "Image of leave"
print coreml_model

image = Image.open('leaves_data/full/10/1438.jpg')
# coreml_model.save('keras_third.mlmodel')
prediction = coreml_model.predict({"input1":image})
print prediction
# coreml_model = coreml.converters.keras.convert('2017-12-04 23:39:47.685856.h5',
#                                                input_names="image",
#                                                image_input_names='image',
#                                                output_names='output',
#                                                class_labels=output_labels,
#                                                image_scale=1/255.)

# coreml_model.save('keras_second.mlmodel')
