import copy
import numpy as np
from NAS.core.utils.block_selector import block_selector_available_connections
from NAS.core.utils.net_validator import net_validator
from NAS.core.ga.ga_crossover import CrossoverOperator
from NAS.core.ga.ga_mutation import MutationOperator
from NAS.core.ga.ga_selection import TournamentSelection

class GAPipeline:

    def __init__(self, blocks, param_mut_prob, add_block_prob=0, change_block_prob=0,
                 cross_prob=0.5, cross_real_values = False, mutation_real_values=False, mutation_real_values_dist = None):
        self.blocks = blocks
        self.mo = MutationOperator(param_mut_prob, add_block_prob, change_block_prob,
                                   real_numbers_mode= mutation_real_values, random_distribution_params= mutation_real_values_dist)
        self.co = CrossoverOperator(crossover_prob=cross_prob, real_numbers_mode=cross_real_values)
        self.selector = TournamentSelection(2)

    def mutate(self, seq):
        new_seq = []
        blocks = copy.deepcopy(self.blocks)
        new_seq.append(copy.deepcopy(seq[0]))
        for i in range(1, len(seq) - 1):
            tr = [new_seq[-1]] + seq[i:i + 2]
            new_tr = self.mo.apply_mutation(tr, blocks)
            if len(new_tr) == 4:
                new_seq.append(new_tr[1])
                new_seq.append(new_tr[2])
            if len(new_tr) == 3:
                new_seq.append(new_tr[1])

        new_seq.append(copy.deepcopy(seq[-1]))
        return new_seq

    def __crossover_add_missing_part(self, seq, part):

        simple_combination = seq + part
        validation1, _ = net_validator(simple_combination)
        if validation1 is True:
            return simple_combination

        selectable_gluing_blocks = block_selector_available_connections(self.blocks, seq[-2], ending_block=part[0])

        if len(selectable_gluing_blocks) != 0:
            new_gluing_block_name = np.random.choice(selectable_gluing_blocks)
            for i in self.blocks:
                if i.get_block_name() == new_gluing_block_name:
                    new_gluing_block = copy.deepcopy(i)
            seq[-1] = new_gluing_block
            return seq + part

        selectable_gluing_blocks = block_selector_available_connections(self.blocks, seq[-1], ending_block=part[1])

        if len(selectable_gluing_blocks) != 0:
            new_gluing_block_name = np.random.choice(selectable_gluing_blocks)
            for i in self.blocks:
                if i.get_block_name() == new_gluing_block_name:
                    new_gluing_block = copy.deepcopy(i)
            part[0] = new_gluing_block
            return seq + part

        return []

    def crossover(self, seq1, seq2):
        new_seq1 = []
        new_seq2 = []
        seq1_l = len(seq1)
        seq2_l = len(seq2)
        d = seq1_l - seq2_l
        if d > 0:
            selected_length = seq2_l
        if d < 0 or d == 0:
            selected_length = seq1_l
        new_seq1.append(copy.deepcopy(seq1[0]))
        new_seq2.append(copy.deepcopy(seq2[0]))
        for i in range(1, selected_length - 1):
            tr1 = [new_seq1[-1]] + seq1[i:i + 2]
            tr2 = [new_seq2[-1]] + seq2[i:i + 2]
            new_tr1, new_tr2 = self.co.apply_crossover(tr1, tr2)
            new_seq1.append(new_tr1[1])
            new_seq2.append(new_tr2[1])

        new_seq1.append(copy.deepcopy(seq1[-1]))
        new_seq2.append(copy.deepcopy(seq2[-1]))

        if d < 0:
            new_seq2 = self.__crossover_add_missing_part(new_seq2, seq2[selected_length:])
            if len(new_seq2) == 0:
                new_seq2 = copy.deepcopy(seq2)

        if d > 0:
            new_seq1 = self.__crossover_add_missing_part(new_seq1, seq1[selected_length:])
            if len(new_seq1) == 0:
                new_seq1 = copy.deepcopy(seq1)

        n_seq1_val = net_validator(new_seq1)
        if not n_seq1_val:
            new_seq1 = copy.deepcopy(seq1)

        n_seq2_val = net_validator(new_seq2)
        if not n_seq2_val:
            new_seq2 = copy.deepcopy(seq2)

        return new_seq1, new_seq2

    def pipeline_expand_population(self, population, num_selected_parents, num_siblings):
        selected_parents = self.selector.select(population, num_selected_parents)
        siblings = []
        while len(siblings) < num_siblings:
            pars = np.random.choice(selected_parents, size=(num_selected_parents,))
            arch1, arch2 = pars[0].get_architecture(), pars[1].get_architecture()
            sib1, sib2 = self.crossover(arch1, arch2)
            sib1, sib2 = self.mutate(sib1), self.mutate(sib2)
            siblings.append(sib1)
            siblings.append(sib2)

        for i in range(len(siblings)):
            population.add_new_element(siblings[i])

        return population

    def pipeline_cut_population(self, population, target_size):

        while population.get_population_size() > target_size:
            worst_adapt = 1e100
            worst_adapt_id = -1
            for i in population.get_all_ids():
                _, adapt, _ = population.get_data_by_id(i)
                if adapt < worst_adapt:
                    worst_adapt_id = i
                    worst_adapt = adapt
            population.delete_by_id(worst_adapt_id)

        return population
