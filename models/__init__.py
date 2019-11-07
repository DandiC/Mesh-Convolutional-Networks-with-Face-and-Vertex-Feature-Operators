def create_model(opt):
    from .mesh_classifier import ClassifierModel # todo - get rid of this ?
    from .mesh_generative import GenerativeModel
    if opt.dataset_mode == "generative":
        model = GenerativeModel(opt)
    else:
        model = ClassifierModel(opt)
    return model
