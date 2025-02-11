#Project 1
#David Pham
import numpy as np

class GeneticAlgorithm:
    def __init__(self, eval_function, population_size=100, vector_length=5, mutation_probability=.05, crossover_probability=.05, population_lower_bound = 0, 
                 population_upper_bound = 10, mutation_increment = 10, iteration_number = 100):
        self.population_size = population_size
        self.vector_length = vector_length
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.eval_function = eval_function
        self.population_lower_bound = population_lower_bound
        self.population_upper_bound = population_upper_bound
        self.population = self.initialize_population()
        self.mutation_increment = mutation_increment
        self.iteration_number = iteration_number

    def initialize_population(self):
        population = np.random.uniform(self.population_lower_bound, self.population_upper_bound, (self.population_size, self.vector_length))
        return population

    def selection(self):
        selection_vector = np.zeros(self.population_size)

        #Calculate fitness
        for x in np.arange(0, self.population_size):
            selection_vector[x] = self.evaluate(self.population[x])

        #Calculate fitness probabilities
        total_fitness = np.sum(selection_vector)
        for vector in selection_vector:
            vector = vector/total_fitness
        next_population = np.zeros((self.population_size, self.vector_length))

        #Calculate cumulative probabilities
        temp_sum = 0
        for fitness_value in selection_vector:
            fitness_value = fitness_value + temp_sum
            temp_sum += fitness_value

        for idx in np.shape(next_population):
            random_float = np.random.random()
            for fitness in selection_vector:
                if random_float < fitness:
                    next_population[idx] = self.population[idx]
                    break

        self.population = next_population

    def evaluate(self, vector):
        #I messed up, re-evaulate what you understand about the objective function
        result = 0
        variable_index = np.arange(len(self.eval_function)-1) #leaves out last element which is constant
        for term in variable_index:
            result += (vector[term]**(self.eval_function[term])[1]) * (self.eval_function[term])[0] #Adds polynomial terms
        result += self.eval_function[-1] #Adds constant

        return result

    def crossover_selection(self):
        def crossover(idx, paired_index):
            vector1 = self.population[idx]
            vector2 = self.population[paired_index]
            crossover_point = np.random.randint(0, np.shape(vector1)[0])
            parent1 = np.concatenate((vector1[0:crossover_point], vector2[crossover_point:]))
            parent2 = np.concatenate((vector2[0:crossover_point], vector1[crossover_point:]))

            self.population[idx] = parent1
            self.population[paired_index] = parent2

        crossover_counter = 0
        paired_index = 0
        for idx in np.arange(0, np.shape(self.population)[0]):
            crossover_chance = np.random.random()
            if(crossover_chance < self.crossover_probability):
                crossover_counter += 1
                if(crossover_counter % 2 == 0):
                    crossover(idx, paired_index)
                else:
                    paired_index = idx

    
    def mutation(self):
        def mutate(population_index):
            random_gene = np.random.randint(0, np.shape(self.population[population_index])[0])
            self.population[population_index][random_gene] = self.population[population_index][random_gene] + (np.random.random()*self.mutation_increment)

        for idx in np.arange(0, np.shape(self.population)[0]):
            mutation_chance = np.random.random()
            if(mutation_chance < self.mutation_probability):
                mutate(idx)

    def predict(self):
        iterations = 0
        while iterations < self.iteration_number:
            self.selection
            self.crossover_selection
            self.mutation
            iterations += 1


first_term = [1, 2] #Encodes to 1x^2
second_term = [1, 2]
third_term = [1, 2]
third_term = [1, 2]
fourth_term = [1, 2]
fifth_term = [1, 2]
constant = 0
test_vector = [1, 2, 3, 4, 5]

eval_function = [first_term, second_term, third_term, fourth_term, fifth_term, constant]
prediction = GeneticAlgorithm(eval_function)
print(prediction.evaluate(test_vector))





