import copy

import NAS.core.utils.net_validator as validator
import numpy as np
from NAS.core.utils.block_selector import block_selector_available_connections
from NAS.core.Exceptions import NASException

class SNGException(NASException):
    pass


# Algorithm: Depth-First Search (DFS)

class InitPopGenerator():

    def __init__(self, suppress_errors=False):
        """
        Initialize InitPopGenerator.
        Args:
            suppress_errors (bool): If True, suppresses errors during generation.
        """
        self.suppress_errors = suppress_errors

    def visualize(self, architectures):
        """
        Print a visual representation of architectures.
        Args:
            architectures (list): List of architectures to visualize.
        """
        for i in range(len(architectures)):
            print(f'Набор {i}:')
            architecture = architectures[i]
            for g in architecture:
                print(f'{g.get_block_name()}')

    def dfs(self, starting_block, ending_block, blocks, net_length, num_architectures):
        """
        Depth-First Search to generate valid architectures.
        Args:
            starting_block: Initial block.
            ending_block: Final block.
            blocks (list): List of available blocks.
            net_length (int): Desired architecture length.
            num_architectures (int): Number of architectures to generate.
        Returns:
            list: List of valid architectures.
        Raises:
            SNGException: If no valid candidates are found and errors are not suppressed.
        """
        candidates = []
        possible_architectures = [[starting_block]]
        while len(possible_architectures[0]) < net_length - 1:
            new_combinations = []
            for p in range(len(possible_architectures)):
                combinations = self.generate_combinations(possible_architectures[p], blocks)
                new_combinations.append(combinations)
            possible_architectures = self.decompose(new_combinations)
        for i in range(len(possible_architectures)):
            candidate = copy.deepcopy(possible_architectures[i])
            candidate.append(ending_block)
            validation_result, _ = validator.net_validator(candidate)
            if validation_result is True:
                candidates.append(candidate)
        true_number_of_candidates = len(candidates)
        if true_number_of_candidates == 0 and not self.suppress_errors:
            raise SNGException('No candidates, check IO and architecture length')
        if num_architectures > true_number_of_candidates:
            if self.suppress_errors:
                d = num_architectures - true_number_of_candidates
                candidates = self.append_architectures(candidates, d)
            else:
                SNGException("Can't generate enough samples, check architecture length and blocks")
        indexes = np.linspace(0, len(candidates) - 1, len(candidates), dtype=np.int16)
        selected_indexes = np.random.choice(indexes, num_architectures, replace=False)
        selected_candidates = []
        for i in selected_indexes:
            selected_candidates.append(candidates[i])
        return self.fix_ids(selected_candidates)

    def append_architectures(self, archi_for_append, num_for_append):
        """
        Appends copies of architectures to reach the required number.
        Args:
            archi_for_append (list): List of architectures to append to.
            num_for_append (int): Number of architectures to append.
        Returns:
            list: Extended list of architectures.
        """
        indexes = np.linspace(0, len(archi_for_append) - 1, len(archi_for_append), dtype=np.int16)
        selected_indexes = np.random.choice(indexes, num_for_append, replace=True)
        for i in selected_indexes:
            ap_arch = copy.deepcopy(archi_for_append[i])
            archi_for_append.append(ap_arch)
        return archi_for_append

    def generate_combinations(self, block_sequence, available_blocks):
        """
        Generates all possible combinations by adding blocks to the sequence.
        Args:
            block_sequence (list): Current block sequence.
            available_blocks (list): List of available blocks.
        Returns:
            list: List of new block sequences.
        """
        current_last_block = block_sequence[-1]
        selectable_connections = block_selector_available_connections(available_blocks, current_last_block)
        new_combinations = []
        for i in range(len(selectable_connections)):
            new_combination = copy.deepcopy(block_sequence)
            name_new_block = selectable_connections[i]
            for g in range(len(available_blocks)):
                possible_block = available_blocks[g]
                possible_block_name = possible_block.get_block_name()
                if possible_block_name == name_new_block:
                    possible_block = copy.deepcopy(possible_block)
                    new_id = new_combination[-1].get_block_id() + 1
                    possible_block.update_block_id(new_id)
                    possible_block.randomize_parameters()
                    new_combination.append(possible_block)
                    new_combinations.append(new_combination)
                    break
        return new_combinations

    def decompose(self, sequence):
        """
        Flattens a nested list of architectures into a single list.
        Args:
            sequence (list): Nested list of architectures.
        Returns:
            list: Flattened list of architectures.
        """
        new_sequence = []
        for i in range(len(sequence)):
            sub_sequence = sequence[i]
            for g in range(len(sub_sequence)):
                new_sequence.append(sub_sequence[g])
        return new_sequence

    def fix_ids(self, architectures):
        """
        Updates block IDs in all architectures to ensure uniqueness.
        Args:
            architectures (list): List of architectures.
        Returns:
            list: Architectures with updated block IDs.
        """
        for i in range(len(architectures)):
            architecture = architectures[i]
            for g in range(len(architecture)):
                architecture[g].update_block_id(g)
        return architectures

    def generate_starting_population(self, input_block, output_block, blocks, net_length, n_architectures):
        """
        Generates the starting population of architectures.
        Args:
            input_block: Initial block.
            output_block: Final block.
            blocks (list): List of available blocks.
            net_length (int): Desired architecture length.
            n_architectures (int): Number of architectures to generate.
        Returns:
            list: Starting population of architectures.
        """
        return self.dfs(input_block, output_block, blocks, net_length, n_architectures)
