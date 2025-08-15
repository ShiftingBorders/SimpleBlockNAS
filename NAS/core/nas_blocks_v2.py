import torch
import torch.nn as nn
from NAS.core.utils.data_shape import ShapeCalculator
from NAS.core.network_parameter import NetParameter as parameter
import numpy as np

class BasicBlock:

    def __init__(self):
        """
        Initialize BasicBlock with default parameters and settings.
        """

        self.is_special = False
        self.block_id = -1
        self.block_name = ''
        self.allowed_connections = []
        self.block_parameters = {
            'input_size': parameter(256, 256, True),
            'output_size': parameter(256, 256, True),
            'dropout_probability': parameter(0.2, 0.9999999, True, change_min_value=0),
            'output_channels': parameter(64, 64, True),
            'kernel_size': parameter(3, 5, True),
            'padding': parameter(1, 1, False),
            'stride': parameter(2, 2, False)

        }
        self.datashape_calc = None
        self.block_stored_weights = None
        self.block_stored_bias = None
        self.has_dropout = False
        self.has_batchnorm = False
        self.selected_activation_function = None
        self.parameter_randomize = False
        self.allows_skip_connection = True
        self.weights_save_support = False
        self.activation_functions = {
            'None': nn.Identity(),
            'ReLU': nn.ReLU(),
            'Softmax': nn.Softmax()
        }

    #def __str__(self):
    #    print(f'Block: {self.block_name}')
    #    for param in self.block_parameters:
    #        param_value = self.block_parameters[param].get_value()
     #       print(f'\t{param}: {param_value}')

    def init_block(self, block_id, allowed_connections):
        """
        Initialize block ID and allowed connections.
        Args:
            block_id (int): Block identifier.
            allowed_connections (list): List of allowed connection names.
        """
        self.block_id = block_id
        self.allowed_connections = allowed_connections

    def randomize_parameters(self):
        """
        Randomize parameters if parameter_randomize is True.
        """
        if not self.parameter_randomize:
            return
        for i in self.block_parameters.keys():
            self.block_parameters[i].re_randomize()

    def calc_out_shape(self, in_data_shape):
        """
        Calculate output shape using datashape_calc and block parameters.
        Args:
            in_data_shape (MultivaluedParameter): Input data shape.
        Returns:
            MultivaluedParameter: Output shape.
        """
        new_shape = self.datashape_calc.calculate_new_shape(in_data_shape, self.block_parameters)
        return new_shape

    def get_block_id(self):
        """
        Returns the block ID.
        Returns:
            int: Block identifier.
        """
        return self.block_id

    def get_block_name(self):
        """
        Returns the block name.
        Returns:
            str: Block name.
        """
        return self.block_name

    def get_allowed_connections(self):
        """
        Returns the allowed connections for this block.
        Returns:
            list: Allowed connection names.
        """
        return self.allowed_connections

    def get_mutable_parameters(self):
        """
        Returns a list of mutable parameter names.
        Returns:
            list: Names of mutable parameters.
        """
        mutable_parameters = []
        for param_name in self.block_parameters.keys():
            param = self.block_parameters[param_name]
            if not param.check_lock() and param.check_ga_support():
                mutable_parameters.append(param_name)
        return mutable_parameters

    def get_parameter_value(self, param_name):
        """
        Returns the value of a parameter by name.
        Args:
            param_name (str): Name of the parameter.
        Returns:
            value: Parameter value.
        """
        return self.block_parameters[param_name].get_value()

    def change_parameter_value(self, param_name, value):
        """
        Change the value of a parameter and run self_check.
        Args:
            param_name (str): Name of the parameter.
            value: New value for the parameter.
        """
        self.block_parameters[param_name].change_value(value)
        self.block_parameters[param_name].self_check()

    def update_block_id(self, new_id):
        """
        Update the block ID.
        Args:
            new_id (int): New block identifier.
        """
        self.block_id = new_id

    def update_weights(self, new_weight, new_bias):
        """
        Update stored weights and bias for the block.
        Args:
            new_weight: New weights.
            new_bias: New bias.
        """
        self.block_stored_weights = new_weight
        self.block_stored_bias = new_bias

    def get_input_shape(self):
        """
        Returns the input shape value.
        Returns:
            value: Input shape.
        """
        return self.block_parameters['input_size'].get_value()

    def update_input_shape(self, new_input_shape, avoid_self_check = False):
        """
        Update the input shape value.
        Args:
            new_input_shape: New input shape value.
            avoid_self_check: If True, avoid value self-check.
        """
        self.block_parameters['input_size'].change_value(new_input_shape, avoid_self_check = avoid_self_check)

    def get_output_shape(self):
        """
        Returns the output shape value.
        Returns:
            value: Output shape.
        """
        return self.block_parameters['output_size'].get_value()

    def update_output_shape(self, new_output_shape, avoid_self_check = False):
        """
        Update the output shape value.
        Args:
            new_output_shape: New output shape value.
            avoid_self_check: If True, avoid value self-check.
        """
        self.block_parameters['output_size'].change_value(new_output_shape, avoid_self_check = avoid_self_check)

    def weights_size_mismatch(self, new_weight, new_bias=None):
        """
        Check if the new weights and bias sizes mismatch with stored ones.
        Args:
            new_weight: New weights.
            new_bias: New bias (optional).
        Returns:
            bool: True if mismatch, False otherwise.
        """
        weights_ok = False
        if self.block_stored_weights is None:
            return True
        old_weights_shape = self.block_stored_weights.shape
        new_weights_shape = new_weight.shape
        weights_check = np.equal(old_weights_shape, new_weights_shape).all()
        weights_ok = weights_check
        if weights_ok:
            return False
        else:
            return True

    def get_weights(self, new_weights, new_bias):
        """
        Get weights and bias as nn.Parameter, updating stored values if mismatch.
        Args:
            new_weights: New weights.
            new_bias: New bias.
        Returns:
            tuple: (weights, bias) as nn.Parameter.
        """
        mismatch = self.weights_size_mismatch(new_weights, new_bias)
        if mismatch:
            self.block_stored_weights = new_weights
            self.block_stored_bias = new_bias
        else:
            pass

        new_weights_p = nn.Parameter(new_weights)
        if new_bias is not None:
            new_bias_p = nn.Parameter(new_bias)
        else:
            new_bias_p = None
        return new_weights_p, new_bias_p

    def check_weight_save_support(self):
        """
        Returns whether weight save is supported.
        Returns:
            bool: True if supported, False otherwise.
        """
        return self.weights_save_support

class Linear(BasicBlock):

    def __init__(self, io_size, activation_function, block_name='Linear', custom_connections=[], dropout=False,
                 dropout_prob=0.2, param_randomise=False):
        """
        Initialize Linear block with parameters and settings.
        """
        super().__init__()
        allowed_connections = ['Linear'] + custom_connections
        self.activation_function_name = activation_function
        self.selected_activation_function = self.activation_functions[activation_function]
        self.init_block(0, allowed_connections)
        self.block_name = block_name
        self.block_parameters['dropout_probability'].change_value(dropout_prob)
        self.has_dropout = dropout
        self.parameter_randomize = param_randomise
        self.datashape_calc = ShapeCalculator('linear')
        input_size = io_size[0]
        output_size = io_size[1]
        self.block_parameters['input_size'].change_value(input_size)
        self.block_parameters['input_size'].change_value(input_size,new_upper=True)
        self.block_parameters['output_size'].change_value(output_size)
        self.block_parameters['output_size'].change_value(output_size,new_upper=True)
        self.weights_save_support = True

    def get_layer(self):
        """
        Returns the layer(s) for the Linear block as nn.ModuleList.
        Returns:
            nn.ModuleList: List of layers.
        """
        input_size = self.get_input_shape()
        output_size = self.get_output_shape()
        modules = None
        layer = nn.Linear(input_size, output_size)
        weights, bias = layer.weight, layer.bias
        new_weight, new_bias = self.get_weights(weights, bias)
        layer.weight = new_weight
        layer.bias = new_bias
        modules = nn.ModuleList([layer])

        if self.has_dropout:
            modules.append(self.selected_activation_function)
            p = self.block_parameters['dropout_probability'].get_value()
            modules.append(nn.Dropout(p=p))
            return modules
        else:
            modules.append(self.selected_activation_function)
            return modules

class Flatten(BasicBlock):

    def __init__(self, block_name='Flatten', dim=1, custom_connections=[]):
        """
        Initialize Flatten block with parameters and settings.
        """
        super().__init__()
        self.block_name = block_name
        allowed_connections = ['Linear'] + custom_connections
        self.dim = dim
        self.init_block(0, allowed_connections)
        self.datashape_calc = ShapeCalculator('flatten', flatten_dim=dim)
        self.activation_function_name = 'None'
    def get_layer(self):
        """
        Returns the flatten layer as nn.ModuleList.
        Returns:
            nn.ModuleList: List containing flatten layer.
        """
        flatten = nn.Flatten(start_dim=1)
        flatten.weight = torch.zeros(size=(1,))
        return nn.ModuleList([flatten])

class Convolution2d(BasicBlock):

    def __init__(self, io_size, activation_function, block_name='Conv2d', custom_connections=[], batch_norm=False, dropout = False,
                 param_randomise=False):
        """
        Initialize Convolution2d block with parameters and settings.
        """
        super().__init__()
        allowed_connections = ['Conv2d'] + custom_connections
        self.activation_function_name = activation_function
        self.selected_activation_function = self.activation_functions[activation_function]
        self.init_block(0, allowed_connections)
        self.block_name = block_name
        self.has_batchnorm = batch_norm
        self.has_dropout = dropout
        self.parameter_randomize = param_randomise
        self.datashape_calc = ShapeCalculator('conv')
        #self.block_parameters['padding'].force_value(0)
        input_size = io_size[0]
        output_size = io_size[1]
        self.block_parameters['input_size'].change_value(input_size)
        self.block_parameters['input_size'].change_value(input_size,new_upper=True)
        self.block_parameters['output_size'].change_value(output_size)
        self.block_parameters['output_size'].change_value(output_size,new_upper=True)
        self.weights_save_support = True

    def get_layer(self):
        """
        Returns the layer(s) for the Convolution2d block as nn.ModuleList.
        Returns:
            nn.ModuleList: List of layers.
        """
        input_size = self.get_input_shape()
        output_size = self.get_output_shape()
        modules = None
        kernel_size = self.block_parameters['kernel_size'].get_value()
        padding = self.block_parameters['padding'].get_value()
        stride = self.block_parameters['stride'].get_value()
        conv_2d = nn.Conv2d(input_size, output_size, kernel_size, padding=padding, stride=stride, bias=False)
        weights, bias = conv_2d.weight, conv_2d.bias
        new_weight, new_bias = self.get_weights(weights, bias)
        conv_2d.weight = new_weight
        conv_2d.bias = new_bias
        modules = nn.ModuleList([conv_2d])
        if self.has_batchnorm:
            modules.append(nn.BatchNorm2d(output_size))
        modules.append(self.selected_activation_function)
        if self.has_dropout:
            p = self.block_parameters['dropout_probability'].get_value()
            modules.append(nn.Dropout(p=p))
        return modules

class MaxPool2d(BasicBlock):

    def __init__(self, io_size, activation_function, block_name='MaxPool2d', custom_connections=[], batch_norm=False, dropout = False,
                 param_randomise=False):
        """
        Initialize MaxPool2d block with parameters and settings.
        """
        super().__init__()
        allowed_connections = ['Conv2d'] + custom_connections
        self.activation_function_name = activation_function
        self.selected_activation_function = self.activation_functions[activation_function]
        self.init_block(0, allowed_connections)
        self.block_name = block_name
        self.has_batchnorm = batch_norm
        self.has_dropout = dropout
        self.parameter_randomize = param_randomise
        self.datashape_calc = ShapeCalculator('pool')
        self.block_parameters['kernel_size'].force_value(io_size[0], lock=True)
        self.block_parameters['padding'].force_value(0, lock=True)
        self.block_parameters['stride'].force_value(io_size[0], lock=True)

    def get_layer(self):
        """
        Returns the layer(s) for the MaxPool2d block as nn.ModuleList.
        Returns:
            nn.ModuleList: List of layers.
        """
        input_size = self.get_input_shape()
        output_size = self.get_output_shape()
        modules = None
        kernel_size = self.block_parameters['kernel_size'].get_value()
        padding = self.block_parameters['padding'].get_value()
        stride = self.block_parameters['stride'].get_value()
        maxpool_2d = nn.MaxPool2d(kernel_size, padding=padding, stride=stride)

        modules = nn.ModuleList([maxpool_2d])
        #modules.append(self.selected_activation_function)
        if self.has_dropout:
            p = self.block_parameters['dropout_probability'].get_value()
            modules.append(nn.Dropout(p=p))
        return modules
