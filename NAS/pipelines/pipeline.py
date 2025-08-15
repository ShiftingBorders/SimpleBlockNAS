import copy
from NAS.core.utils.net_starting_generator import InitPopGenerator
from NAS.core.utils.net_parameter_aligner import ParameterAligner
from NAS.core.ga.ga_population import Population
from NAS.core.ga.ga_pipeline import GAPipeline
from NAS.core.utils.model_logger import ModelLogger
from NAS.nets.net_base_class import ModelTrainer
from tqdm import tqdm, tqdm_notebook
from NAS.config import ConfigGeneticAlgorithm, ConfigLearning

class NAS:

    def __init__(self, population_name, gen_error_suppress=True, in_jupyter=False):
        """
        Initialize NAS pipeline with population name and configuration.
        Args:
            population_name (str): Name of the population.
            gen_error_suppress (bool): Suppress errors during generation.
            in_jupyter (bool): If running in Jupyter environment.
        """
        self.population_generator = InitPopGenerator(suppress_errors=gen_error_suppress)
        self.aligner = ParameterAligner()
        self.population = Population(population_name)
        self.models_logger = ModelLogger(population_name)

        self.in_jupyter = in_jupyter

        self.io_size_set = False
        self.input_shape = None
        self.output_shape = None

        self.learning_rate = ConfigLearning.learning_rate
        self.num_epochs = ConfigLearning.learning_epochs
        self.base_model = None
        self.accelerator = ConfigLearning.accelerator
        self.precision = ConfigLearning.precision
        self.model_args_payload = None


    def set_io_size(self, input_size, output_size):
        """
        Set input and output sizes for the pipeline.
        Args:
            input_size: Input shape.
            output_size: Output shape.
        """
        self.io_size_set = True
        self.input_shape = input_size
        self.output_shape = output_size

    def set_learning_params(self, base_model, model_args_payload):
        """
        Set learning parameters for the pipeline.
        Args:
            base_model: Base model class.
            model_args_payload (dict): Arguments for the model.
        """
        self.base_model = base_model
        self.model_args_payload = model_args_payload

    def __init_trainer(self):
        """
        Initialize the model trainer for the pipeline.
        """
        self.ModelTrainer = ModelTrainer(val_every_n_epoch=ConfigLearning.val_every_n_epochs)

    def eval_population(self, population_, current_epoch, train_loader, val_loader):
        """
        Evaluate the population for the current epoch.
        Args:
            population_ (Population): Population to evaluate.
            current_epoch (int): Current epoch number.
            train_loader: Training data loader.
            val_loader: Validation data loader.
        Returns:
            Population: Evaluated population.
        """
        population_size = population_.get_population_size()
        if self.in_jupyter:
            population_eval_progress = tqdm_notebook(total = population_size, desc = 'Population evaluation progress', leave = False)
        else:
            population_eval_progress = tqdm(total=population_size, desc='Population evaluation progress', leave=False)
        for architecture_number in self.population.get_all_ids():
            architecture, adaptation, unique_id = population_.get_data_by_id(architecture_number)
            aligned_architecture = self.aligner.align(architecture, self.input_shape, self.output_shape)
            if adaptation > 0:
                self.models_logger.copy_previous(unique_id, current_epoch)
                population_eval_progress.update(1)
                continue

            model = copy.deepcopy(self.base_model)
            local_payload = copy.deepcopy(self.model_args_payload)
            local_payload['blocks'] = aligned_architecture
            local_payload['lr'] = ConfigLearning.learning_rate
            loaded_model = model(local_payload)
            try:
                train_results = self.ModelTrainer.train_net(loaded_model,
                                                            ConfigLearning.accelerator,
                                                            ConfigLearning.precision,
                                                            train_loader, val_loader,
                                                            ConfigLearning.learning_epochs,
                                                            )
            except Exception as e:
                print(f'Failed to train model, reason: {e}')
                population_.change_adapt_val(architecture_number, 0)
                population_eval_progress.update(1)
                continue
            best_adaptation = train_results['highest_acc']
            population_.change_adapt_val(architecture_number, best_adaptation)
            population_.update_weights(architecture_number, train_results['weights'], train_results['biases'])
            acc_data = [[0], train_results['accuracy'], [0]]
            loss_data = [[0], train_results['loss'], [0]]
            self.models_logger.save_model_info(unique_id, architecture, loss_data, acc_data, current_epoch)
            population_eval_progress.update(1)
        return population_

    def run(self, start_block, end_block, blocks, train_loader, val_loader, gen_arch, arch_init_len):
        """
        Run the NAS pipeline for architecture search and evolution.
        Args:
            start_block: Starting block.
            end_block: Ending block.
            blocks (list): List of available blocks.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            gen_arch (int): Number of architectures to generate.
            arch_init_len (int): Initial architecture length.
        """
        self.__init_trainer()
        initial_candidates = self.population_generator.generate_starting_population(start_block, end_block, blocks, arch_init_len, gen_arch)
        for i in initial_candidates:
            self.population.add_new_element(i)

        genetic_algorithm = GAPipeline(
                blocks = blocks,
                param_mut_prob = ConfigGeneticAlgorithm.mutation_parameter_probability,
                add_block_prob = ConfigGeneticAlgorithm.mutation_add_block_probability,
                change_block_prob = ConfigGeneticAlgorithm.mutation_change_block_probability,
                cross_prob = ConfigGeneticAlgorithm.crossover_probability,
                cross_real_values = ConfigGeneticAlgorithm.crossover_real_numbers_mode,
                mutation_real_values = ConfigGeneticAlgorithm.mutation_real_numbers_mode,
                mutation_real_values_dist = ConfigGeneticAlgorithm.mutation_distribution_params,
            )
        if self.in_jupyter:
            progressbar_population = tqdm_notebook(range(ConfigGeneticAlgorithm.evolution_epochs), desc = 'GA progress')
        else:
            progressbar_population = tqdm(range(ConfigGeneticAlgorithm.evolution_epochs), desc = 'GA progress')
        self.population = self.eval_population(self.population, 0, train_loader, val_loader)
        for i in range(ConfigGeneticAlgorithm.evolution_epochs):
            self.population = genetic_algorithm.pipeline_expand_population(
                self.population,
                ConfigGeneticAlgorithm.num_parents,
                ConfigGeneticAlgorithm.num_siblings)
            self.population = self.eval_population(self.population, i+1, train_loader, val_loader)
            self.population = genetic_algorithm.pipeline_cut_population(self.population,ConfigGeneticAlgorithm.population_target_size)
            self.population.create_snapshot()
            progressbar_population.update(1)
            _, avg_adapt, _, _, _, _ = self.population.population_stats()
            progressbar_population.set_postfix_str(avg_adapt)






