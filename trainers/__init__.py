import importlib

def create_trainer(config):
    lib = importlib.import_module('trainers.{}'.format(config.trainer))
    model = lib.Trainer(config)
    
    return model
