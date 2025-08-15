from NAS.core.network_parameter import MultivaluedParameter

class ParameterAligner:

    def __init__(self):
        """
        Initialize ParameterAligner instance.
        """
        pass

    def rule_IO_align_block(self, net, initial_shape):
        """
        Aligns input/output shapes for each block in the network sequentially.
        Args:
            net (list): List of network blocks.
            initial_shape (list): Initial input shape.
        Returns:
            list: Network with aligned shapes.
        """
        def align_fn(net, b_1_index, b_2_index, ini_shape):
            block_1 = net[b_1_index]
            block_2 = net[b_2_index]
            shape_new = block_1.calc_out_shape(ini_shape)
            if block_1.block_name == 'Flatten':
                dim = block_1.dim
                slice = shape_new.values
                r = 1
                for elem in slice:
                    r *= elem
                block_1_output_size = r
            else:
                block_1_output_size = shape_new.get_value(0)
            net[b_2_index].update_input_shape(block_1_output_size, avoid_self_check=True)
            return shape_new
        shape_previous = MultivaluedParameter(initial_shape)
        for i in range(0, len(net)-1):
            shape_previous = align_fn(net, i, i+1, shape_previous)
        shape_previous = align_fn(net, len(net)-2, len(net)-1, shape_previous)
        return net

    def rule_IO_align(self, net, IS, OS):
        """
        Aligns input and output shapes for the first and last blocks in the network.
        Args:
            net (list): List of network blocks.
            IS (list): Input shape.
            OS (list): Output shape.
        Returns:
            list: Network with aligned input/output shapes.
        """
        net[0].update_input_shape(IS[0])
        net[-1].update_output_shape(OS[0])
        return net

    def align(self, net, initial_shape, final_shape):
        """
        Aligns the network by sequentially aligning block shapes and then input/output shapes.
        Args:
            net (list): List of network blocks.
            initial_shape (list): Initial input shape.
            final_shape (list): Final output shape.
        Returns:
            list: Fully aligned network.
        """
        net = self.rule_IO_align_block(net, initial_shape)
        net = self.rule_IO_align(net, initial_shape, final_shape)
        return net
