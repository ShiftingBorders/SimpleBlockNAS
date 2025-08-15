import copy
import numpy as np

from NAS.core.utils.block_selector import block_selector_available_connections
from NAS.core.ga.ga_utils import get_gray_code, convert_to_int
from NAS.core.ga.ga_exceptions import GAException
from NAS.core.ga.ga_exceptions import TripletLengthException
class DistributionParameterNumber(GAException):
    """
    Exception raised when random distribution parameters are incorrectly specified.
    
    This exception is thrown when the random_distribution_params list does not
    contain exactly 2 elements, which is required for real number mode mutation.
    """
    pass

class MutationOperator:
    """
    A genetic algorithm mutation operator for neural architecture search.
    
    This class implements various mutation operations for neural network architectures,
    including parameter mutation, block addition, and block replacement. It supports
    both Gray code and real number mutation modes.
    
    Attributes:
        real_numbers_mode (bool): If True, uses real number mutation instead of Gray code
        parameter_mutation_probability (float): Probability of mutating individual parameters
        add_new_block_probability (float): Probability of adding a new block
        change_block_probability (float): Probability of replacing an existing block
        dist_p1 (float): First parameter for real number distribution (when real_numbers_mode is True)
        dist_p2 (float): Second parameter for real number distribution (when real_numbers_mode is True)
    """

    def __init__(self, param_mut_prob: float, add_block_prob:float = 0, change_block_prob:float = 0,
                 real_numbers_mode:bool = False, random_distribution_params:bool = None) -> None:
        """
        Initialize the mutation operator.
        
        Args:
            param_mut_prob (float): Probability of mutating individual parameters
            add_block_prob (float): Probability of adding a new block (default: 0)
            change_block_prob (float): Probability of replacing an existing block (default: 0)
            real_numbers_mode (bool): If True, uses real number mutation instead of Gray code (default: False)
            random_distribution_params (list): Parameters for real number distribution [p1, p2] (default: [0, 1])
            
        Raises:
            DistributionParameterNumber: If random_distribution_params does not have exactly 2 elements
        """
        if random_distribution_params is None:
            random_distribution_params = [0, 1]
        if len(random_distribution_params) != 2:
            raise DistributionParameterNumber("Random distribution params must have 2 elements")
        self.real_numbers_mode = real_numbers_mode
        self.parameter_mutation_probability = param_mut_prob
        self.add_new_block_probability = add_block_prob
        self.change_block_probability = change_block_prob

        # Real number mode params
        if self.real_numbers_mode:
            self.dist_p1 = random_distribution_params[0]
            self.dist_p2 = random_distribution_params[1]

    def gray_code_swap_bits(self, val:str)->str:
        """
        Mutate a Gray code string by randomly flipping bits.
        
        This method iterates through each bit in the Gray code string and
        randomly flips it based on the parameter mutation probability.
        
        Args:
            val (str): Gray code string to mutate
            
        Returns:
            str: Mutated Gray code string
        """
        val = list(val)
        for i in range(len(val)):
            val_bit = val[i]
            check_change = np.random.rand() < self.parameter_mutation_probability
            if check_change is True:
                if val_bit == '0':
                    val[i] = '1'
                else:
                    val[i] = '0'
        new_val = ''.join(str(bit) for bit in val)
        return new_val

    def mutate_params(self, block):
        """
        Mutate the parameters of a neural network block.
        
        This method identifies unlocked parameters that support genetic algorithm
        operations and mutates them according to the specified mutation mode
        (Gray code or real numbers).
        
        Args:
            block: Neural network block whose parameters should be mutated
            
        Returns:
            block: Copy of the block with mutated parameters
        """
        params = block.block_parameters
        unlocked_params = []

        for i in params.keys():
            if params[i].check_lock() is False and params[i].check_ga_support() is True:
                unlocked_params.append(i)

        for i in unlocked_params:
            param_val = params[i].get_value()
            param_val_upper_limit = params[i].get_upper_value()
            if self.real_numbers_mode:
                change_check = np.random.rand() < self.parameter_mutation_probability
                if change_check is False:
                    continue
                if isinstance(param_val, int):
                    new_val = np.random.randint(self.dist_p1,self.dist_p2) + param_val
                else:
                    new_val = np.random.uniform(self.dist_p1,self.dist_p2) + param_val
                params[i].change_value(new_val)
                params[i].self_check()
            else:
                param_val_gray_code, was_float = get_gray_code(param_val)
                param_val_upper_limit_gray_code, _ = get_gray_code(param_val_upper_limit)

                d = len(param_val_upper_limit_gray_code) - len(param_val_gray_code)

                if d > 0:
                    param_val_gray_code = '0' * d + param_val_gray_code  # Выравнивание по длине для доступа ко всем значениям

                new_val = self.gray_code_swap_bits(param_val_gray_code)
                decoded_val = convert_to_int(new_val, was_float)
                params[i].change_value(decoded_val)
                params[i].self_check()

        block.block_parameters = params
        return copy.deepcopy(block)

    def add_block(self, left_block, right_block, blocks_list):
        """
        Add a new block between two existing blocks.
        
        This method randomly selects a block from the available blocks list that
        can connect to both the left and right blocks, and inserts it between them.
        
        Args:
            left_block: Block to the left of the insertion point
            right_block: Block to the right of the insertion point
            blocks_list (list): List of available blocks to choose from
            
        Returns:
            block or None: Selected block to insert, or None if no suitable block found
        """
        check_change = np.random.rand() < self.add_new_block_probability
        if check_change is False:
            return None

        available_connections = left_block.get_allowed_connections()
        candidates = []
        for i in range(len(blocks_list)):
            block_name = blocks_list[i].get_block_name()
            if block_name in available_connections:
                candidates.append(copy.deepcopy(blocks_list[i]))

        selected_candidates = []
        right_block_name = right_block.get_block_name()
        for i in range(len(candidates)):
            candidate_allowed_connections = candidates[i].get_allowed_connections()
            if right_block_name in candidate_allowed_connections:
                selected_candidates.append(candidates[i])

        if len(selected_candidates) == 0:
            return None

        final_candidate = copy.deepcopy(np.random.choice(selected_candidates))
        return final_candidate

    def change_block(self, target_block, left_block, right_block, blocks_list):
        """
        Replace an existing block with a different compatible block.
        
        This method finds blocks that can connect to both the left and right blocks
        and randomly selects one to replace the target block.
        
        Args:
            target_block (list): List containing the block to be replaced
            left_block: Block to the left of the target block
            right_block: Block to the right of the target block
            blocks_list (list): List of available blocks to choose from
            
        Returns:
            block or None: Selected replacement block, or None if no suitable block found
        """
        check_change = np.random.rand() < self.change_block_probability
        if check_change is False:
            return None
        candidates = []
        available_blocks = block_selector_available_connections(blocks_list, left_block, ending_block=right_block)
        for i in range(len(blocks_list)):
            block_name = blocks_list[i].get_block_name()
            if block_name in available_blocks and block_name != target_block[0].get_block_name():
                candidates.append(blocks_list[i])
        if len(candidates) == 0:
            return None

        final_candidate = copy.deepcopy(np.random.choice(candidates))
        return final_candidate

    def apply_mutation(self, triplet, blocks):
        """
        Apply mutation operations to a triplet of blocks.
        
        This is the main mutation method that combines block replacement, block addition,
        and parameter mutation. It first attempts to replace the central block, then
        tries to add a new block, and finally mutates the parameters of all blocks.
        
        Args:
            triplet (list): Triplet of blocks [left_block, central_block, right_block]
            blocks (list): List of available blocks for replacement/addition
            
        Returns:
            list: New triplet after mutation operations
        """
        if len(triplet) != 3:
            raise TripletLengthException('Triplet have size smaller than three.'
                                         'Check initial size of architectures and generated architectures. ')
        main_block = [triplet[1]]
        main_mutation_complete = False

        if main_mutation_complete is False:
            new_block = self.change_block(main_block, triplet[0], triplet[2], blocks)
            if new_block is not None:
                main_block[0] = new_block
                main_mutation_complete = True

        if main_mutation_complete is False:
            insert_before_main = np.random.randint(0, 2)
            if insert_before_main == 0:
                left_block = triplet[0]
                right_block = triplet[1]
            else:
                left_block = triplet[1]
                right_block = triplet[2]
            new_block = self.add_block(left_block, right_block, blocks)
            if new_block is not None:
                if insert_before_main == 0:
                    main_block.insert(0, new_block)
                else:
                    main_block.append(new_block)
                main_mutation_complete = True

        for i in range(len(main_block)):
            main_block[i] = self.mutate_params(main_block[i])

        new_triplet = []
        new_triplet.append(copy.deepcopy(triplet[0]))
        for i in range(len(main_block)):
            new_triplet.append(copy.deepcopy(main_block[i]))
        new_triplet.append(copy.deepcopy(triplet[2]))

        return new_triplet
