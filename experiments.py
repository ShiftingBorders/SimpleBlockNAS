from NAS.data.dataset_loaders import EuroSAT
from NAS.pipelines.pipeline import NAS as NAS_GA
from NAS.core.nas_blocks_v2 import Linear, Convolution2d, Flatten
from NAS.nets.net_base_class import CNNClassifier

NAS = NAS_GA('EuroSAT-SW', gen_error_suppress=True, in_jupyter=False)
NAS.set_io_size([3, 64, 64], [10])
if __name__ == '__main__':
    cnn_model = CNNClassifier
    NAS.set_learning_params(cnn_model, {'num_classes': 10})
    linear_relu = Linear([512,512],'ReLU',dropout=False)
    linear = Linear([10,10],'None')
    flatten = Flatten(dim=0)
    conv = Convolution2d([128,128],'ReLU',custom_connections=['Flatten'],batch_norm=False,param_randomise=True)

    import ssl # For EuroSAT
    ssl._create_default_https_context = ssl._create_unverified_context # For EuroSAT
    Loaders = EuroSAT()
    train_loader, val_loader = Loaders.get_dataloaders(32, num_workers = 4)
    NAS.run(conv, linear, [conv, flatten, linear_relu], train_loader, val_loader,7, 5)
