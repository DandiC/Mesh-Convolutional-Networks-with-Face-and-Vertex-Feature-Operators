def create_model(opt):
    from .mesh_classifier import ClassifierModel # todo - get rid of this ?
    from .mesh_generative import GenerativeModel
    from .mesh_autoencoder import AutoencoderModel
    if opt.dataset_mode == "generative":
        if opt.arch == 'meshunet':
            model = AutoencoderModel(opt)
        else:
            model = GenerativeModel(opt)
    else:
        model = ClassifierModel(opt)
    return model
