def net_validator(net_blocks):
    """
    Validates the network by checking allowed connections between blocks.
    Args:
        net_blocks (list): List of network blocks.
    Returns:
        tuple: (True, 0) if valid, (False, index) if invalid at index.
    """
    for i in range(len(net_blocks) - 1):
        block_1 = net_blocks[i]
        block_2 = net_blocks[i + 1]
        block_1_allowed_connections = block_1.get_allowed_connections()
        block_2_name = block_2.get_block_name()
        if block_2_name not in block_1_allowed_connections:
            return False, i
    return True, 0


