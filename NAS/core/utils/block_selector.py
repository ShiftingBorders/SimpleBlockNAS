import numpy as np


def block_selector_available_connections(block_list, starting_block, ending_block=None):
    """
    Returns a list of block names that are available for connection from the starting block.
    If ending_block is provided, filters connections that do not allow ending_block.
    Args:
        block_list (list): List of block objects.
        starting_block: Block object to start from.
        ending_block (optional): Block object to end at.
    Returns:
        list: Names of selectable connections.
    """
    starting_block_connections = starting_block.get_allowed_connections()
    available_connections = []
    for i in block_list:
        name = i.get_block_name()
        available_connections.append(name)
    selectable_connections = []
    for i in starting_block_connections:
        if i in available_connections:
            selectable_connections.append(i)
    selectable_connections = list(set(selectable_connections))
    if ending_block is not None:
        ending_block_name = ending_block.get_block_name()
        for i in block_list:
            block_name = i.get_block_name()
            block_available_connections = i.get_allowed_connections()
            if block_name in selectable_connections and ending_block_name not in block_available_connections:
                selectable_connections.remove(block_name)
    return selectable_connections


def block_selector_random(block_list, starting_block, ending_block=None):
    """
    Selects a random block name from available connections.
    If ending_block is provided, selects from connections compatible with ending_block.
    Args:
        block_list (list): List of block objects.
        starting_block: Block object to start from.
        ending_block (optional): Block object to end at.
    Returns:
        str or list: Selected block name or list if no valid connection.
    """
    selectable_connections = block_selector_available_connections(block_list, starting_block)
    if ending_block is not None:
        selected_block = __selector_with_ending_block(ending_block, selectable_connections)
        return selected_block
    selected_block = np.random.choice(selectable_connections)
    return selected_block


def __selector_with_ending_block(ending_block, selectable_connections):
    """
    Helper function to select a block name compatible with ending_block.
    Args:
        ending_block: Block object to end at.
        selectable_connections (list): List of block names.
    Returns:
        str or list: Selected block name or empty list if none found.
    """
    confirmed_connections = []
    ending_block_connections = ending_block.get_allowed_connections()
    for i in selectable_connections:
        if i in ending_block_connections:
            confirmed_connections.append(i)
    if len(confirmed_connections) == 0:
        return confirmed_connections
    selected_block = np.random.choice(confirmed_connections, 1)
    return selected_block
