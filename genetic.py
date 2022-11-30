import random
import numpy as np


# В этом проекте используется десятичное представление хромосомы
# def calculate_population_fitness(equation_inputs, pop):
#     # Вычисление значения пригодности каждого решения в текущей популяции.
#     # Фитнес-функция вычисляет сумму произведений между каждым входом и его соответствующим весом.
#     fitness = np.sum(pop * equation_inputs, axis=1)
#     return fitness


# def fitness_assessment(population, coef_num, y):
#     p = []
#     for i in range(len(population)):
#         s = 0
#         for j in range(coef_num):
#             s += population[i][j] * diofantov_expr[j]
#         r = np.abs(y - s) + 1
#         p.append(1 / r)
#     return p


def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    # Точка, в которой происходит кроссовер между двумя родителями. Обычно она находится в центре
    crossover_point = np.uint8(offspring_size[1] / 2)
    # print(crossover_point)

    for k in range(offspring_size[0]):
        # Индекс первого сопряженного родителя.
        parent1_idx = k % len(parents)
        # Индекс второго родителя для спаривания.
        parent2_idx = (k + 1) % len(parents)

        for i in range(crossover_point):
            offspring[k][i] = parents[parent1_idx][i]
            # print(i)

        for i in range(crossover_point, offspring_size[1]):
            offspring[k][i] = parents[parent2_idx][i]
            # print(i)
        # У нового потомства первая половина генов будет взята от первого родителя.
        # offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # # У нового потомства вторая половина генов будет взята от второго родителя
        # offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


# def mutation(offspring_crossover, num_mutations):
#     mutations_counter = np.uint8(offspring_crossover.shape[1] / num_mutations)
#     # print(mutations_counter)
#     # print(offspring_crossover)
#
#     # Мутация изменяет количество генов, как определено аргументом num_mutations. Изменения случайны.
#     for idx in range(offspring_crossover.shape[0]):
#         gene_idx = mutations_counter - 1
#         # print(gene_idx)
#         for mutation_num in range(num_mutations):
#             # Случайное значение, добавляемое к гену.
#             random_value = np.random.uniform(-1.0, 1.0, 1)
#             offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
#             gene_idx = gene_idx + mutations_counter
#             # print(gene_idx)
#
#     return offspring_crossover


def fitness_assessment(population, y):
    p = []
    for i in range(len(population)):
        s = 0
        for j in range(len(diofantov_expr)):
            s += population[i][j] * diofantov_expr[j]
        r = np.abs(y - s) + 1
        p.append(1 / r)
    return p


# def select_mating_pool(pop, fitness, num_parents):
#     # Выбор лучших особей текущего поколения в качестве родителей для производства потомства следующего поколения.
#     parents = np.empty((num_parents, pop.shape[1]))
#
#     for parent_num in range(num_parents):
#         max_fitness_idx = np.where(fitness == np.max(fitness))
#         max_fitness_idx = max_fitness_idx[0][0]
#         parents[parent_num, :] = pop[max_fitness_idx, :]
#         fitness[max_fitness_idx] = -99999999999
#
#     return parents

def mutation_test(pop_after_cross, mutation_rate):
    population_nextgen = []
    for i in range(0, len(pop_after_cross)):
        chromosome = pop_after_cross[i]
        for j in range(len(chromosome)):
            if random.random() < mutation_rate:
                random_value = np.random.uniform(-1.0, 1.0, 1)
                # chromosome[j] = not chromosome[j]
                chromosome[j] = chromosome[j] + random_value
        population_nextgen.append(chromosome)
    return population_nextgen


def parents_selection(population, parents_num, p):
    choose_parents = []
    for i in range(parents_num):
        max_ind = [j for j in range(len(p)) if p[j] == max(p)][0]
        choose_parents.append(population[max_ind])
        p.remove(max(p))
    return choose_parents


if __name__ == '__main__':
    diofantov_expr = [4, -2, 3.5, 5, -11, -4.7]
    y = 30

    diofantov_weights = len(diofantov_expr)

    # count_solution = 8

    sol_per_pop = 12
    # num_parents_mating = 8

    # Определение численности населения.
    population_size = (sol_per_pop, diofantov_weights)
    # У популяции будет хромосома sol_per_pop, где каждая хромосома имеет num_weights генов.

    # Создание начальной популяции.
    new_population = np.random.uniform(low=-len(diofantov_expr), high=len(diofantov_expr), size=population_size)
    # print(new_population)

    count_iteration = 1000
    for iteration in range(count_iteration):
        # print(new_population)
        fitness = fitness_assessment(new_population, y)

        # print("Текущая эффективность выборки : ", fitness)
        new_parents = parents_selection(new_population, 6, fitness)
        # print(new_parents)
        # print(len(new_parents))
        new_offspring_cross = crossover(parents=new_parents,
                                        offspring_size=(population_size[0] - len(new_parents), diofantov_weights))

        # print(new_offspring_cross)

        new_offspring_mut = mutation_test(new_offspring_cross, 0.1)
            # mutation(offspring_crossover=new_offspring_cross, num_mutations=2)

        # print(new_offspring_mut)


        for i in range(len(new_parents)):
            new_population[i] = new_parents[i]

        current_count = 0

        for i in range(len(new_parents), len(new_parents) + len(new_offspring_mut)):
            new_population[i] = new_offspring_mut[current_count]
            current_count += 1

    fitness = fitness_assessment(new_population, y)

    max_fitness = max(fitness)
    need_index = fitness.index(max_fitness)

    print("Лучшая хромосома(подборка весов): ", new_population[need_index])
    print("Лучшая пригодность: ", max_fitness)
    # best_match_idx = numpy.where(fitness == numpy.max(fitness))
    # print(fitness)
    # print(best_match_idx)
    #
    # print(new_population)
    # print(new_population[best_match_idx, :])