from src.val.VGG16 import val_VGG16
from src.val.Bert import val_Bert
from src.val.ViT import val_ViT
from src.val.KWT import val_KWT

def get_val(model_name, data_name, state_dict_full, logger):
    if model_name == 'Bert':
        val_Bert(data_name, state_dict_full, logger)
        return True
    elif model_name == 'VGG16':
        val_VGG16(data_name, state_dict_full, logger)
        return True
    elif model_name == 'ViT':
        val_ViT(data_name, state_dict_full, logger)
        return True
    elif model_name == 'KWT':
        val_KWT(data_name, state_dict_full, logger)
        return True
    else:
        return False

