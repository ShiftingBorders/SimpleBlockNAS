class Config:
    data_path = '/tmp/data'
    save_weights = False
    unique_id_length = 8
    population_size = 100
    num_generations = 100
    num_parents_mating = 20
    keep_parents = 10

class ConfigGeneticAlgorithm:
    evolution_epochs = 10
    num_parents = 6
    num_siblings = 10
    population_target_size = 10
    crossover_probability = 0.5
    crossover_real_numbers_mode = False
    mutation_parameter_probability = 0.5
    mutation_add_block_probability = 0.2
    mutation_change_block_probability = 0.2
    mutation_remove_block_probability = 0.0
    mutation_real_numbers_mode = True
    mutation_distribution_params = [-1,1]

class ConfigLearning:
    learning_epochs = 10
    learning_rate = 1e-3
    accelerator = 'gpu'
    precision = 'bf16-true'
    val_every_n_epochs = 2





