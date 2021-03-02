import coremltools as ct

class_labels = ['cow', 'chicken', 'pig']
classifier_config = ct.ClassifierConfig(class_labels)

coreml_model = ct.converters.convert('vgg16_transfer.h5',
    inputs=[ct.ImageType()],
    classifier_config=classifier_config,
    source='TensorFlow',)

coreml_model.save('./vgg16_transfer.mlmodel')