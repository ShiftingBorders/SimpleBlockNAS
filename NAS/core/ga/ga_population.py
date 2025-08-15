import pandas as pd
from NAS.core.ga.ga_population_unit import PopulationUnit
import os
import numpy as np
from NAS.config import Config
from NAS.core.Exceptions import NASException
from pathlib import Path
import errno

class MissingArchitectureException(NASException):
    """
    Exception raised when an architecture is not found in the population.
    
    This exception is thrown when attempting to access or modify an architecture
    that does not exist in the current population.
    """
    pass

class Population:
    """
    A genetic algorithm population for neural architecture search.
    
    This class manages a collection of neural network architectures (PopulationUnit objects)
    and provides methods for population management, statistics tracking, and reporting.
    Each architecture in the population has a unique identifier and can be tracked
    for fitness scores and other metrics.
    
    Attributes:
        population (dict): Dictionary mapping unique IDs to PopulationUnit objects
        population_name (str): Name identifier for this population
        solutions_ids (list): List of solution IDs (currently unused)
        unique_id_length (int): Length of unique identifiers
        save_weights (bool): Whether to save weights for architectures
        path (Path): Path for saving population reports
    """

    def __init__(self, name:str) -> None:
        """
        Initialize a new population.
        
        Args:
            name (str): Name identifier for this population
        """
        self.population = {}
        self.population_name = name
        self.solutions_ids = []
        self.unique_id_length = Config.unique_id_length
        self.save_weights = Config.save_weights
        target_path = Path('.').resolve() / 'reports' / 'populations_reports'
        os.makedirs(target_path, exist_ok=True)
        self.path = target_path

    def generate_unique_id(self) -> int:
        """
        Generate a unique identifier for a new population unit.
        
        This method creates a random numeric ID of the specified length and
        ensures it is unique within the current population.
        
        Returns:
            int: Unique identifier that is not already in use
        """
        id_generated_successfully = False
        while not id_generated_successfully:
            numbers = np.random.randint(0,10, size=(self.unique_id_length, )).astype(str)
            number = int(''.join(numbers))
            if number not in list(self.population.keys()):
                id_generated_successfully = True
                return number

    def get_population_size(self) -> int:
        """
        Get the current size of the population.
        
        Returns:
            int: Number of individuals in the population
        """
        return len(self.population)

    def get_all_ids(self) -> list:
        """
        Get all unique identifiers in the population.
        
        Returns:
            list: List of all unique IDs currently in the population
        """
        return list(self.population.keys())

    def add_new_element(self, arch) -> int:
        """
        Add a new architecture to the population.
        
        Args:
            arch: Neural network architecture to add
            
        Returns:
            int: Unique identifier assigned to the new architecture
        """
        arch_unique_id = self.generate_unique_id()
        new_elem = PopulationUnit(arch_unique_id)
        new_elem.set_architecture(arch)
        self.population[arch_unique_id] = new_elem
        return arch_unique_id

    def delete_by_id(self, id_) -> None:
        """
        Remove an architecture from the population by its unique identifier.
        
        Args:
            id_ (int): Unique identifier of the architecture to remove
        """
        del self.population[id_]

    def change_adapt_val(self, id_:int, adapt_val:float) -> None:
        """
        Update the fitness score (adaptation value) of an architecture.
        
        Args:
            id_ (int): Unique identifier of the architecture
            adapt_val (float): New fitness score
        """
        self.population[id_].set_adaptation(adapt_val)

    def get_model_by_id(self, id_:int) -> PopulationUnit:
        return self.population[id_]

    def get_data_by_id(self, id_) ->tuple[list,float,int]:
        """
        Get the complete data for an architecture by its unique identifier.
        
        Args:
            id_ (int): Unique identifier of the architecture
            
        Returns:
            tuple[list, float, int]: Tuple containing (architecture, fitness_score, unique_id)
        """
        arch = self.population[id_].get_architecture()
        adapt = self.population[id_].get_adaptation()
        unique_id = self.population[id_].get_id()
        return arch, adapt, unique_id

    def update_weights(self,model_id:int,new_weights:list,new_biases:list) -> None:
        """
        Update the weights and biases of an architecture.
        
        This method updates the weights and biases of all blocks in an architecture
        that support weight updates. The operation is skipped if weight saving is
        disabled or if the model ID is not found in the population.
        
        Args:
            model_id (int): Unique identifier of the architecture to update
            new_weights (list): List of new weight tensors for each block
            new_biases (list): List of new bias tensors for each block
        """
        if not self.save_weights or model_id not in self.population.keys():
            return
        current_weights_num = 0
        for i in range(len(self.population[model_id])):
            if self.population[model_id].block_weights_support(i):
                self.population[model_id].block_update_weights(i,new_weights[current_weights_num],new_biases[current_weights_num])
                current_weights_num += 1

    def get_population(self) -> dict:
        """
        Get the complete population dictionary.
        
        Returns:
            dict: Dictionary mapping unique IDs to PopulationUnit objects
        """
        return self.population

    def population_stats(self) -> tuple[float,float,float,int,int,int]:
        """
        Calculate statistical information about the population.
        
        This method computes minimum, average, and maximum values for both
        fitness scores and architecture sizes across the entire population.
        
        Returns:
            tuple[float, float, float, int, int, int]: Tuple containing
                (min_adaptation, average_adaptation, max_adaptation, 
                 min_size, average_size, max_size)
        """
        min_adaptation = 1e100
        average_adaptation = 0
        max_adaptation = 0
        min_size = 1e100
        average_size = 0
        max_size = 0

        pop_size = self.get_population_size()
        if pop_size == 0:
            pop_size = 1

        for id_ in self.population:
            arch, adapt, _ = self.get_data_by_id(id_)
            arch_len = len(arch)
            average_adaptation += adapt
            average_size += arch_len

            if adapt > max_adaptation:
                max_adaptation = adapt
            if adapt < min_adaptation:
                min_adaptation = adapt
            if arch_len > max_size:
                max_size = arch_len
            if arch_len < min_size:
                min_size = arch_len

        average_adaptation /= pop_size
        average_size /= pop_size
        return min_adaptation, average_adaptation, max_adaptation, min_size, average_size, max_size

    def create_snapshot(self, verbose:bool=False) -> None:
        """
        Create and save a snapshot of population statistics.
        
        This method calculates population statistics and saves them to a CSV file.
        The snapshot includes information about fitness scores and architecture sizes.
        
        Args:
            verbose (bool): If True, print the statistics to console (default: False)
        """
        min_adaptation, average_adaptation, max_adaptation, min_size, average_size, max_size = self.population_stats()
        df_new = pd.DataFrame(
            {'min_adapt': min_adaptation, 'avg_adapt': average_adaptation, 'max_adapt': max_adaptation,
             'min_size': min_size, 'avg_size': int(average_size), 'max_size': max_size}, index=[0])
        new_row = {'min_adapt': min_adaptation, 'avg_adapt': average_adaptation, 'max_adapt': max_adaptation,
                   'min_size': min_size, 'avg_size': int(average_size), 'max_size': max_size}
        population_path = self.path / f'{self.population_name}.csv'
        try:
            df = pd.read_csv(population_path, index_col=False)
            df.loc[len(df)] = new_row
            df.to_csv(population_path, index=False)
        except OSError as e:
            if e.errno == errno.ENOENT:
                df_new.to_csv(f'{self.path}/{self.population_name}.csv', index=False)
            else:
                print(f'Encountered error during population report saving: \n {e}')


        if verbose is True:
            print(f'''
                  Minimal adaptation: {min_adaptation}
                  Average adaptation: {average_adaptation}
                  Maximum adaptation: {max_adaptation}
                  Minimal size: {min_size}
                  Average size: {average_size}
                  Maximum size: {max_size}
                  ''')

