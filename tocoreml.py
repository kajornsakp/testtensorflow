from coremltools.models import MLModel
from coremltools.models.neural_network import NeuralNetworkBuilder
import coremltools.models.datatypes as datatypes

# ...

def make_mlmodel(variables):
    # Specify the inputs and outputs (there can be multiple).
    # Each name corresponds to the input_name/output_name of a layer in the network so
    # that Core ML knows where to insert and extract data.
    input_features = [('image', datatypes.Array(1, IMAGE_HEIGHT, IMAGE_WIDTH))]
    output_features = [('labelValues', datatypes.Array(NUM_LABEL_INDEXES))]
    builder = NeuralNetworkBuilder(input_features, output_features, mode=None)

    # The "name" parameter has no effect on the function of the network. As far as I know
    # it's only used when Xcode fails to load your mlmodel and gives you an error telling
    # you what the problem is.
    # The input_names and output_name are used to link layers to each other and to the
    # inputs and outputs of the model. When adding or removing layers, or renaming their
    # outputs, always make sure you correct the input and output names of the layers
    # before and after them.
    builder.add_elementwise(name='add_layer',
                            input_names=['image'], output_name='add_layer', mode='ADD',
                            alpha=-0.5)

    # Although Core ML internally uses weight matrices of shape
    # (outputChannels, inputChannels, height, width) (as can be found by looking at the
    # protobuf specification comments), add_convolution takes the shape
    # (height, width, inputChannels, outputChannels) (as can be found in the coremltools
    # documentation). The latter shape matches what TensorFlow uses so we don't need to
    # reorder the matrix axes ourselves.
    builder.add_convolution(name='conv2d_1', kernel_channels=1,
                            output_channels=32, height=3, width=3, stride_height=1,
                            stride_width=1, border_mode='same', groups=0,
                            W=variables['W_conv1'].eval(), b=variables['b_conv1'].eval(),
                            has_bias=True, is_deconv=False, output_shape=None,
                            input_name='add_layer', output_name='conv2d_1')

    builder.add_activation(name='relu_1', non_linearity='RELU', input_name='conv2d_1',
                           output_name='relu_1', params=None)

    builder.add_pooling(name='maxpool_1', height=2, width=2, stride_height=2,
                        stride_width=2, layer_type='MAX', padding_type='SAME',
                        input_name='relu_1', output_name='maxpool_1')

    # ...

    builder.add_flatten(name='maxpool_3_flat', mode=1, input_name='maxpool_3',
                        output_name='maxpool_3_flat')

    # We must swap the axes of the weight matrix because add_inner_product takes the shape
    # (outputChannels, inputChannels) whereas TensorFlow uses
    # (inputChannels, outputChannels). Unlike with add_convolution (see the comment
    # above), the shape add_inner_product expects matches what the protobuf specification
    # requires for inner products.
    builder.add_inner_product(name='fc1',
                              W=tf_fc_weights_order_to_mlmodel(variables['W_fc1'].eval())
                                .flatten(),
                              b=variables['b_fc1'].eval().flatten(),
                              input_channels=6*6*64, output_channels=1024, has_bias=True,
                              input_name='maxpool_3_flat', output_name='fc1')

    # ...

    builder.add_softmax(name='softmax', input_name='fc2', output_name='labelValues')

    model = MLModel(builder.spec)

    model.short_description = 'Model for recognizing a variety of images drawn on screen with one\'s finger'

    model.input_description['image'] = 'A gesture image to classify'
    model.output_description['labelValues'] = 'The "probability" of each label, in a dense array'

    return model