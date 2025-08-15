import copy
import torch

class PopulationUnit:
    """
    A unit representing a single individual in a genetic algorithm population.
    
    This class encapsulates a neural network architecture along with its
    fitness score (adaptation) and unique identifier. It provides methods
    for managing the architecture and its associated weights.
    
    Attributes:
        stored_architecture (list): List of neural network blocks representing the architecture
        adaptation (float): Fitness score of the architecture
        unique_id (int): Unique identifier for this population unit
    """

    def __init__(self, unique_id:int) -> None:
        """
        Initialize a new population unit.
        
        Args:
            unique_id (int): Unique identifier for this population unit
        """
        self.stored_architecture = []
        self.adaptation = 0
        self.unique_id = unique_id

    def __len__(self) -> int:
        """
        Get the number of blocks in the stored architecture.
        
        Returns:
            int: Number of blocks in the architecture
        """
        return len(self.stored_architecture)

    def set_architecture(self, new_arch:list) -> None:
        """
        Set the neural network architecture.
        
        Args:
            new_arch (list): List of neural network blocks representing the new architecture
        """
        self.stored_architecture = copy.deepcopy(new_arch)

    def get_architecture(self) -> list:
        """
        Get the current neural network architecture.
        
        Returns:
            list: List of neural network blocks representing the architecture
        """
        return self.stored_architecture

    def set_adaptation(self, new_adaptation:float) -> None:
        """
        Set the fitness score (adaptation) of this architecture.
        
        Args:
            new_adaptation (float): New fitness score
        """
        self.adaptation = new_adaptation

    def get_adaptation(self) -> float:
        """
        Get the current fitness score (adaptation) of this architecture.
        
        Returns:
            float: Current fitness score
        """
        return self.adaptation

    def get_id(self) -> int:
        """
        Get the unique identifier of this population unit.
        
        Returns:
            int: Unique identifier
        """
        return self.unique_id

    def block_update_weights(self, block_num:int, weights: torch.tensor,biases: torch.tensor) -> None:
        """
        Update the weights and biases of a specific block in the architecture.
        
        Args:
            block_num (int): Index of the block to update
            weights (torch.tensor): New weights for the block
            biases (torch.tensor): New biases for the block
        """
        self.stored_architecture[block_num].update_weights(weights,biases)

    def block_weights_support(self, block_num: int) -> bool:
        """
        Check if a specific block supports weight updates.

        Args:
            block_num (int): Index of the block to check
            
        Returns:
            bool: True if the block supports weight updates, False otherwise
        """
        return self.stored_architecture[block_num].check_weight_save_support()

