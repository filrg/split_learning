from src.val.VGG16 import val_VGG16
from src.val.BERT import val_BERT
from src.val.KWT import val_KWT

def get_val(model_name, data_name, state_dict_full, logger):
    if model_name == 'BERT':
        val_BERT(data_name, state_dict_full, logger)
        return True
    elif model_name == 'VGG16':
        val_VGG16(data_name, state_dict_full, logger)
        return True
    elif model_name == 'KWT':
        val_KWT(data_name, state_dict_full, logger)
        return True
    else:
        return False

