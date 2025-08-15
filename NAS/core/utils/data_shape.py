import numpy as np
from NAS.core.network_parameter import MultivaluedParameter

class ShapeCalculator:

    FLATTEN_DIMENSION_OFFSET = 3

    def __init__(self, mode, flatten_dim=0):
        """
        Initialize ShapeCalculator with mode and flatten dimension.
        Args:
            mode (str): Calculation mode ('linear', 'conv', 'pool', 'flatten').
            flatten_dim (int): Dimension to start flattening from.
        """
        self.mode = mode
        self.flatten_dim = flatten_dim

    def get_mode(self):
        """
        Returns the current calculation mode.
        Returns:
            str: The mode of the calculator.
        """
        return self.mode

    def data_size_to_param(self, data):
        """
        Converts data size to a MultivaluedParameter.
        Args:
            data: Data object with size() method.
        Returns:
            MultivaluedParameter: Parameter with data sizes.
        """
        sizes = data.size()
        sizes_list = []
        for size in sizes:
            sizes_list.append(size)
        return MultivaluedParameter(sizes_list)

    def print_param(self, param):
        """
        Prints all parameter values.
        Args:
            param (MultivaluedParameter): Parameter to print.
        """
        len_param = param.get_num_parameters()
        for i in range(len_param):
            print(f'Value {i}: {param.get_value(i)}')

    def calculate_linear(self, initial_shape, num_out_neurons):
        """
        Calculates new shape for linear layer.
        Args:
            initial_shape (MultivaluedParameter): Initial shape.
            num_out_neurons (int): Number of output neurons.
        Returns:
            MultivaluedParameter: New shape.
        """
        initial_shape.change_value(0, num_out_neurons)
        return initial_shape

    def calculate_convolution(self, initial_shape, output_channels, kernel_size, stride, padding):
        """
        Calculates new shape for convolutional layer.
        Args:
            initial_shape (MultivaluedParameter): Initial shape.
            output_channels (int): Number of output channels.
            kernel_size (int): Kernel size.
            stride (int): Stride value.
            padding (int): Padding value.
        Returns:
            MultivaluedParameter: New shape.
        """
        if stride < 1:
            stride = 1
        new_channels = output_channels
        new_height = int(np.floor((initial_shape.get_value(1) - kernel_size + 2 * padding) / stride + 1))
        new_width = int(np.floor((initial_shape.get_value(2) - kernel_size + 2 * padding) / stride + 1))
        return MultivaluedParameter([new_channels, new_height, new_width])

    def calculate_pooling(self, initial_shape, kernel_size, stride, padding):
        """
        Calculates new shape for pooling layer.
        Args:
            initial_shape (MultivaluedParameter): Initial shape.
            kernel_size (int): Kernel size.
            stride (int): Stride value.
            padding (int): Padding value.
        Returns:
            MultivaluedParameter: New shape.
        """
        if stride < 1:
            stride = 1
        new_channels = initial_shape.get_value(0)
        new_height = int(np.floor((initial_shape.get_value(1) - kernel_size + 2 * padding) / stride + 1))
        new_width = int(np.floor((initial_shape.get_value(2) - kernel_size + 2 * padding) / stride + 1))
        return MultivaluedParameter([new_channels, new_height, new_width])

    def calculate_flatten(self, initial_shape, sdim):
        """
        Calculates new shape for flatten operation.
        Args:
            initial_shape (MultivaluedParameter): Initial shape.
            sdim (int): Start dimension for flattening.
        Returns:
            MultivaluedParameter: New shape.
        """
        if sdim < 0:
            sdim = (sdim) + self.FLATTEN_DIMENSION_OFFSET
        val = 1
        for i in range(sdim, initial_shape.get_num_parameters(), 1):
            val *= initial_shape.get_value(i)
        new_vals = []
        for i in range(0, sdim, 1):
            new_vals.append(initial_shape.get_value(i))
        new_vals.append(val)
        return MultivaluedParameter(new_vals)

    def calculate_new_shape(self, old_shape, block_params):
        """
        Calculates new shape based on the current mode and block parameters.
        Args:
            old_shape (MultivaluedParameter): Initial shape.
            block_params (dict): Block parameters.
        Returns:
            MultivaluedParameter: New shape.
        """
        if self.mode == 'linear':
            output_neurons = block_params['output_size'].get_value()
            new_shape = self.calculate_linear(old_shape, output_neurons)
            return new_shape
        if self.mode == 'flatten':
            sdim = self.flatten_dim
            new_shape = self.calculate_flatten(old_shape, sdim)
            return new_shape
        if self.mode == 'conv':
            output_channels = block_params['output_size'].get_value()
            kernel_size = block_params['kernel_size'].get_value()
            padding = block_params['padding'].get_value()
            stride = block_params['stride'].get_value()
            new_shape = self.calculate_convolution(old_shape, output_channels, kernel_size, stride, padding)
            return new_shape
        if self.mode == 'pool':
            kernel_size = block_params['kernel_size'].get_value()
            stride = block_params['stride'].get_value()
            padding = block_params['padding'].get_value()
            new_shape = self.calculate_pooling(old_shape, kernel_size, stride, padding)
            return new_shape
