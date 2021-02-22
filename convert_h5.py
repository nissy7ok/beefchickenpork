import coremltools as ct

class_labels = ['cow', 'chicken', 'pig']
classifier_config = ct.ClassifierConfig(class_labels)

coreml_model = ct.converters.convert('animal_cnn_aug.h5',
    inputs=[ct.ImageType()],
    classifier_config=classifier_config,
    source='TensorFlow',)

coreml_model.save('./animal_cnn_aug.mlmodel')