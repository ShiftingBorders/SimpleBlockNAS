import copy
import numpy as np

from NAS.core.ga.ga_utils import get_gray_code, convert_to_int
from NAS.core.network_parameter import NetParameter
from NAS.core.ga.ga_exceptions import TripletLengthException
from NAS.core.Exceptions import NASException


class ValuesTypeMismatch(NASException):
    """
    Exception raised when parameter values have incompatible types during crossover.
    
    This exception is thrown when attempting to perform BLX-α crossover between
    parameters of different types (e.g., one integer and one float).
    """
    pass


class CrossoverOperator:
    """
    A genetic algorithm crossover operator for neural architecture search.
    
    This class implements crossover operations for neural network architectures
    represented as triplets of blocks. It supports both parameter crossover
    and block swapping operations. The crossover can work in two modes:
    - Gray code mode: Uses binary representation for parameter crossover
    - Real numbers mode: Uses BLX-α crossover for continuous parameters
    
    Attributes:
        prob (float): Probability of performing block swap crossover (0.0 to 1.0)
        real_numbers_mode (bool): If True, uses BLX-α crossover instead of Gray code
        blx_alpha (float): Alpha parameter for BLX-α crossover (controls offspring range)
    """

    def __init__(self, crossover_prob:float = 0.5, real_numbers_mode:bool = False, blx_alpha:float = 0.1) -> None:
        """
        Initialize the crossover operator.
        
        Args:
            crossover_prob (float): Probability of performing block swap crossover (0.0 to 1.0, default: 0.5)
            real_numbers_mode (bool): If True, uses BLX-α crossover instead of Gray code (default: False)
            blx_alpha (float): Alpha parameter for BLX-α crossover, controls offspring range (default: 0.1)
        """
        self.prob = crossover_prob
        self.real_numbers_mode = real_numbers_mode
        self.blx_alpha = blx_alpha

    def check_block_swap(self, seq1:list, seq2:list) -> bool:
        """
        Check if two triplets can safely swap their central blocks.
        
        This method verifies that swapping the central blocks between two triplets
        would result in valid connections between all blocks in both sequences.
        It checks compatibility in both directions: left->center and center->right.
        
        Args:
            seq1 (list): First triplet [left_block, central_block, right_block]
            seq2 (list): Second triplet [left_block, central_block, right_block]
            
        Returns:
            bool: True if blocks can be safely swapped, False otherwise
            
        Note:
            This method assumes that the input triplets have exactly 3 elements.
            The validation should be done by the calling method.
        """
        central_block_1 = seq1[1]
        central_block_2 = seq2[1]

        seq1_available_central_blocks = seq1[0].get_allowed_connections()
        seq2_available_central_blocks = seq2[0].get_allowed_connections()

        if central_block_1.get_block_name() not in seq2_available_central_blocks:
            return False

        if central_block_2.get_block_name() not in seq1_available_central_blocks:
            return False

        new_central_seq1_block_connections = central_block_2.get_allowed_connections()
        new_central_seq2_block_connections = central_block_1.get_allowed_connections()

        if seq1[2].get_block_name() not in new_central_seq1_block_connections:
            return False

        if seq2[2].get_block_name() not in new_central_seq2_block_connections:
            return False

        return True

    def blx_alpha_crossover(self, param1: NetParameter, param2:NetParameter) -> tuple[NetParameter,NetParameter]:
        """
        Perform BLX-α crossover between two network parameters.
        
        BLX-α (Blend Crossover with Alpha) is a real-valued crossover operator
        that creates offspring within an extended range around the parent values.
        The range is extended by α times the distance between parents.
        
        Args:
            param1 (NetParameter): First parameter to crossover
            param2 (NetParameter): Second parameter to crossover
            
        Returns:
            tuple[NetParameter, NetParameter]: Two new parameters after BLX-α crossover
            
        Raises:
            ValuesTypeMismatch: If the two parameters have different types (int vs float)
        """
        value1, value2 = param1.get_value(), param2.get_value()
        min_value = min(value1,value2)
        max_value = max(value1,value2)
        delta = max_value - min_value
        lower_bound = min_value - delta*self.blx_alpha
        upper_bound = max_value + delta*self.blx_alpha
        if isinstance(value1,int) and isinstance(value2,int):
            upper_bound = int(upper_bound)
            lower_bound = int(lower_bound)
            if lower_bound >= upper_bound:
                upper_bound = lower_bound + 1
            new_values_ = np.random.randint(high = upper_bound, low = lower_bound, size = (2,))
            new_values = [int(x) for x in new_values_]
        elif isinstance(value1,float) and isinstance(value2,float):
            new_values_ = np.random.uniform(high = upper_bound, low = lower_bound, size = (2,))
            new_values = [float(x) for x in new_values_]
        else:
            raise ValuesTypeMismatch('Values types in crossover operator mismatch')

        new_param1, new_param2 = copy.deepcopy(param1), copy.deepcopy(param2)
        new_param1.change_value(new_values[0])
        new_param2.change_value(new_values[1])
        return new_param1, new_param2


    def uniform_crossover_bits(self, p1:NetParameter, p2: NetParameter) -> tuple[NetParameter,NetParameter]:
        """
        Perform crossover between two network parameters using Gray code representation.
        
        This method converts parameters to Gray code, performs uniform bit-wise crossover,
        and converts back to the original parameter format. It handles length
        alignment and ensures the resulting values are valid. Each bit position
        has a 50% chance of being inherited from either parent.
        
        Args:
            p1 (NetParameter): First parameter to crossover
            p2 (NetParameter): Second parameter to crossover
            
        Returns:
            tuple[NetParameter, NetParameter]: Two new parameters after Gray code crossover
            
        Note:
            This method performs deep copies of the input parameters to avoid
            modifying the original objects.
        """
        p1 = copy.deepcopy(p1)
        p2 = copy.deepcopy(p2)
        val1 = p1.get_value()
        val2 = p2.get_value()
        val1_max_val = p1.get_upper_value()
        val2_max_val = p2.get_upper_value()

        val1_gray, was_float_val1 = get_gray_code(val1)
        val2_gray, was_float_val2 = get_gray_code(val2)

        val1_max_gray, _ = get_gray_code(val1_max_val)
        val2_max_gray, _ = get_gray_code(val2_max_val)

        # Length alignment
        d1 = len(val1_max_gray) - len(val1_gray)
        if d1 > 0:
            val1_gray = '0' * d1 + val1_gray

        d2 = len(val2_max_gray) - len(val2_gray)
        if d2 > 0:
            val2_gray = '0' * d2 + val2_gray

        dd = len(val1_gray) - len(val2_gray)
        if dd > 0:
            val2_gray = '0' * abs(dd) + val2_gray
        if dd < 0:
            val1_gray = '0' * abs(dd) + val1_gray

        new_val1_gray = ''
        new_val2_gray = ''

        for i in range(len(val1_gray)):
            coin_flip = np.random.randint(0, 2)
            if coin_flip == 0:
                new_val1_gray += val1_gray[i]
                new_val2_gray += val2_gray[i]
            else:
                new_val1_gray += val2_gray[i]
                new_val2_gray += val1_gray[i]

        new_val1 = convert_to_int(new_val1_gray, was_float_val1)
        new_val2 = convert_to_int(new_val2_gray, was_float_val2)

        new_p1, new_p2 = copy.deepcopy(p1), copy.deepcopy(p2)

        new_p1.change_value(new_val1)
        new_p2.change_value(new_val2)

        return new_p1, new_p2

    def parameter_crossover(self, seq1:list, seq2:list) -> tuple[list, list]:
        """
        Perform parameter crossover between two triplets.
        
        This method identifies common free parameters between the central blocks
        of two triplets and performs crossover on those parameters. It handles
        both real number mode (BLX-α) and Gray code mode. Only parameters that
        are unlocked and support genetic algorithm operations are considered.
        
        Args:
            seq1 (list): First triplet [left_block, central_block, right_block]
            seq2 (list): Second triplet [left_block, central_block, right_block]
            
        Returns:
            tuple[list, list]: Two new triplets with crossed-over parameters
            
        Note:
            This method performs deep copies of the input triplets to avoid
            modifying the original objects. Only the central blocks are modified.
        """
        block_1 = seq1[1]
        block_2 = seq2[1]
        new_seq1, new_seq2 = copy.deepcopy(seq1), copy.deepcopy(seq2)
        new_block_1 = copy.deepcopy(block_1)
        new_block_2 = copy.deepcopy(block_2)

        block_1_parameters = copy.deepcopy(block_1.block_parameters)
        block_2_parameters = copy.deepcopy(block_2.block_parameters)

        free_parameters_b1 = []

        for i in block_1_parameters.keys():
            if block_1_parameters[i].check_lock() is False and block_1_parameters[i].check_ga_support() is True:
                free_parameters_b1.append(i)

        free_parameters_b2 = []

        for i in block_2_parameters.keys():
            if block_2_parameters[i].check_lock() is False and block_2_parameters[i].check_ga_support() is True:
                free_parameters_b2.append(i)

        common_free_parameters = [element for element in free_parameters_b1 if element in free_parameters_b2]

        for i in common_free_parameters:
            p1 = block_1_parameters[i]
            p2 = block_2_parameters[i]
            if self.real_numbers_mode:
                new_p1, new_p2 = self.blx_alpha_crossover(p1, p2)
            else:
                new_p1, new_p2 = self.uniform_crossover_bits(p1, p2)
            block_1_parameters[i] = new_p1
            block_2_parameters[i] = new_p2

        new_block_1.block_parameters = block_1_parameters
        new_block_2.block_parameters = block_2_parameters

        new_seq1[1] = new_block_1
        new_seq2[1] = new_block_2

        return new_seq1, new_seq2

    def apply_crossover(self, triplet_1:list , triplet_2: list) -> tuple[list, list]:
        """
        Apply crossover operation between two triplets.
        
        This is the main crossover method that combines block swapping and
        parameter crossover. It first validates the triplet lengths, then
        checks if the triplets can swap their central blocks based on the
        crossover probability. Finally, it performs parameter crossover on
        the resulting triplets.
        
        The crossover process:
        1. Validate that both triplets have exactly 3 elements
        2. Check if block swapping is possible and desired (based on probability)
        3. If swapping is performed, create new triplets with swapped central blocks
        4. Apply parameter crossover to the central blocks of both triplets
        
        Args:
            triplet_1 (list): First triplet [left_block, central_block, right_block]
            triplet_2 (list): Second triplet [left_block, central_block, right_block]
            
        Returns:
            tuple[list, list]: Two new triplets after crossover operation
            
        Raises:
            TripletLengthException: If either triplet has fewer than 3 elements
        """
        if len(triplet_1) != 3 or len(triplet_2) != 3:
            raise TripletLengthException('One of the triplets have size smaller than three.'
                                         'Check initial size of architectures and generated architectures. ')
        check_swap = self.check_block_swap(triplet_1, triplet_2)
        swap_chance = self.prob > np.random.rand()
        if (check_swap and swap_chance) is True:
            new_triplet_1 = copy.deepcopy([triplet_1[0], triplet_2[1], triplet_1[2]])
            new_triplet_2 = copy.deepcopy([triplet_2[0], triplet_1[1], triplet_2[2]])
            new_triplet_1, new_triplet_2 = self.parameter_crossover(new_triplet_1, new_triplet_2)
            return new_triplet_1, new_triplet_2
        else:
            new_triplet_1, new_triplet_2 = self.parameter_crossover(triplet_1, triplet_2)
            return new_triplet_1, new_triplet_2
