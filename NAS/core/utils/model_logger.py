import json
import os
from pathlib import Path
class ModelLogger:

    def __init__(self, name):
        """
        Initialize ModelLogger with a target directory for logs.
        Args:
            name (str): Name of the model/log directory.
        """
        target_path =  Path(f'./reports/models_logs/{name}').resolve()
        os.makedirs(target_path, exist_ok=True)
        self.path = target_path

    def make_epoch_dir(self, num_epoch):
        """
        Create a directory for the given epoch if it does not exist.
        Args:
            num_epoch (int): Epoch number.
        """
        path_to_dir = os.path.join(self.path, str(num_epoch))
        if not os.path.exists(path_to_dir):
            os.mkdir(path_to_dir)

    def dump_data(self, model_data, num_epoch):
        """
        Dump model data to a JSON file in the epoch directory.
        Args:
            model_data (dict): Model information to save.
            num_epoch (int): Epoch number.
        """
        path_to_dir = self.path / f'{num_epoch}'
        num_model = len(os.listdir(path_to_dir)) + 1
        path_to_model = path_to_dir / f'{num_model}.json'
        with open(path_to_model, 'w') as f:
            json.dump(model_data, f)

    def parse_block_params(self, block):
        """
        Parse parameters of a block and return as a dictionary.
        Args:
            block: Block object with parameters.
        Returns:
            dict: Block parameters and metadata.
        """
        block_params = {'layer_name': block.block_name}
        for param in block.block_parameters.keys():
            block_params[param] = block.block_parameters[param].get_value()
        block_params['allowed_connections'] = block.allowed_connections
        block_params['activation_function'] = block.activation_function_name
        block_params['has_dropout'] = block.has_dropout
        block_params['has_batchnorm'] = block.has_batchnorm
        return block_params

    def parse_model_architecture(self, model_architecture):
        """
        Parse the architecture of a model and return a list of block descriptions.
        Args:
            model_architecture (list): List of block objects.
        Returns:
            list: List of block parameter dictionaries.
        """
        model_description = []
        for block in model_architecture:
            block_description = self.parse_block_params(block)
            model_description.append(block_description)
        return model_description

    def save_model_info(self, unique_id, model_arch, loss_hist, acc_hist, epoch):
        """
        Save model information, architecture, and metrics for a given epoch.
        Args:
            unique_id: Unique identifier for the model.
            model_arch (list): Model architecture blocks.
            loss_hist (list): Loss history.
            acc_hist (list): Accuracy history.
            epoch (int): Epoch number.
        """
        loss_train, loss_val, loss_test = loss_hist
        acc_train, acc_val, acc_test = acc_hist
        max_acc_train, max_acc_val, max_acc_test = max(acc_train), max(acc_val), max(acc_test)
        model_blocks_description = self.parse_model_architecture(model_arch)
        model_metrics = {
            'loss_train': loss_train,
            'loss_val': loss_val,
            'loss_test': loss_test,
            'acc_train': acc_train,
            'acc_val': acc_val,
            'acc_test': acc_test,
            'max_acc_train': max_acc_train,
            'max_acc_val': max_acc_val,
            'max_acc_test': max_acc_test,
        }
        model_description = {
            'unique_id': unique_id,
            'model_description': model_blocks_description,
            'model_metrics': model_metrics,
        }
        self.make_epoch_dir(epoch)
        self.dump_data(model_description, epoch)

    def copy_previous(self, model_unique_id, current_epoch):
        """
        Copy previous epoch's model info to the current epoch if unique_id matches.
        Args:
            model_unique_id: Unique identifier for the model.
            current_epoch (int): Current epoch number.
        """
        self.make_epoch_dir(current_epoch)
        search_epoch = current_epoch - 1
        path_to_dir = self.path / f'{search_epoch}'
        path_to_dir_new = self.path / f'{current_epoch}'
        files = os.listdir(path_to_dir)
        for file in files:
            with open(os.path.join(path_to_dir, file), 'r') as f:
                model_json = json.load(f)
                if model_json['unique_id'] == model_unique_id:
                    num_files = len(os.listdir(path_to_dir_new))
                    new_name = str(num_files) + '.json'
                    with open(os.path.join(path_to_dir_new, new_name), 'w') as f2:
                        json.dump(model_json, f2)
                        break
