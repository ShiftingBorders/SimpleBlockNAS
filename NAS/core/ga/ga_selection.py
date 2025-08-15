import copy
import numpy as np

class SelectorException(Exception):
    def __init__(self, message):
        """
        Initialize SelectorException with a custom error message.
        Args:
            message (str): Error message to display.
        """
        super().__init__(message)

class TournamentSelection:

    def __init__(self, bracket_size):
        """
        Initialize TournamentSelection with a bracket size.
        Args:
            bracket_size (int): Number of candidates in each tournament bracket.
        """
        self.bracket_size = bracket_size

    def bracket(self, selected_elements):
        """
        Select the best candidate from a list using adaptation score.
        Args:
            selected_elements (list): List of candidate elements.
        Returns:
            The candidate with the highest adaptation score.
        """
        best_score = -1
        selected_winner = None
        for i in range(self.bracket_size):
            candidate = selected_elements[i]
            score = candidate.get_adaptation()
            if score > best_score:
                best_score = score
                selected_winner = candidate
        return selected_winner

    def select(self, pool, req_num_samples):
        """
        Select a number of samples from the pool using tournament selection.
        Args:
            pool (list or Population): Pool of candidates to select from.
            req_num_samples (int): Number of samples to select.
        Returns:
            list: Selected samples.
        Raises:
            SelectorException: If not enough samples are available for selection.
        """
        selected_samples = []
        selected_ids = []
        bracket = []
        bracket_ids = []
        if type(pool) is not list:
            old_pool = copy.deepcopy(pool)
            pool_size = old_pool.get_population_size()
            pool = []
            for i in old_pool.get_all_ids():
                elem = old_pool.get_model_by_id(i)
                pool.append(elem)
        if (len(pool) - (req_num_samples - 1)) < self.bracket_size:
            raise SelectorException('Too few samples, change bracket size or increase number of samples')
        while len(selected_samples) < req_num_samples:
            sample = np.random.choice(pool)
            sample_id = sample.get_id()
            if sample_id not in selected_ids and sample_id not in bracket_ids:
                bracket.append(sample)
                bracket_ids.append(sample_id)
            if len(bracket) == self.bracket_size:
                winner = self.bracket(bracket)
                winner_id = winner.get_id()
                selected_samples.append(winner)
                selected_ids.append(winner_id)
                bracket.clear()
                bracket_ids.clear()
        return selected_samples
