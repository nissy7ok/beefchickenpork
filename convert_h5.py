import coremltools
from keras.models import load_model

coreml_model = coremltools.converters.convert('animal_cnn_aug.h5', input_names='image', image_input_names='image', output_names='Prediction', class_labels=['cow', 'chicken', 'pig'])

coreml_model.save('./animal_cnn_aug.mlmodel')