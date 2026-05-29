from .behalfnet import behalfnet

def get_model(model_name, dataset_name, patch_size):
    if model_name == 'behalfnet':
        model = behalfnet(dataset_name)

    else:
        raise KeyError("{} model is not supported yet".format(model_name))

    return model

