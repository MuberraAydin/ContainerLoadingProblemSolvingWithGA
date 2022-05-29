import json
import visualize_plotly as vis
import visualize as vis2
import matplotlib.pyplot as plt
from tabulate import tabulate
from copy import deepcopy
import matplotlib.tri as mtri
import random
from operator import itemgetter

#tekrar etme
NUM_OF_ITERATIONS = 1
#birey sayısı
NUM_OF_INDIVIDUALS = 36
#nesil sayısı
NUM_OF_GENERATIONS = 400
PC = int(0.8 * NUM_OF_INDIVIDUALS)
PM1 = 0.2
PM2 = 0.02
K = 2
ROTATIONS = 6  # 1 or 2 or 6

#dolu alanları hesaplar
def calc_crowding_dis(solutions):
    obj = len(solutions[0]) - 1
    ranks = {val[3] for val in solutions.values()}
    dict1 = {}
    for rank in ranks:
        group = {k: v for k, v in solutions.items() if v[3] == rank}
        for value in group.values():
            value.append(0)
        for i in range(obj):
            sorted_group = dict(sorted(group.items(), key=lambda x: x[1][i]))
            list1 = list(sorted_group.values())
            list1[0][4] = 5000
            list1[-1][4] = 5000
            for j in range(1, len(list1) - 1):
                list1[j][4] += (list1[j + 1][i] - list1[j - 1][i]) / 100
        dict1.update(group)
    return dict1

#baskın çözüm
def get_dominant_solution(p, q):
    obj = len(p)
    dominance = []
    for i in range(obj):
        if p[i] >= q[i]:
            dominance.append(True)
        else:
            dominance.append(False)

    if True in dominance and False not in dominance:
        return p
    elif True not in dominance and False in dominance:
        return q
    else:
        return None

#sıralama
def rank(population, individuals):
    ranked_solutions = {}
    frontal = set()
    for key, current_solution in individuals.items():
        ranked_solutions[key] = {'Sp': set(), 'Np': 0}

        for index, solution in individuals.items():
            if current_solution[0:3] != solution[0:3]:
                dominant = get_dominant_solution(current_solution[0:3], solution[0:3])
                if dominant is None:
                    continue
                if dominant == current_solution[0:3]:
                    ranked_solutions[key]['Sp'].add(index)
                elif dominant == solution[0:3]:
                    ranked_solutions[key]['Np'] += 1

        if ranked_solutions[key]['Np'] == 0:
            ranked_solutions[key]['Rank'] = 1
            individuals[key].append(1)
            frontal.add(key)

    i = 1
    while len(frontal) != 0:
        sub = set()
        for sol in frontal:
            for dominated_solution in ranked_solutions[sol]['Sp']:
                ranked_solutions[dominated_solution]['Np'] -= 1
                if ranked_solutions[dominated_solution]['Np'] == 0:
                    ranked_solutions[dominated_solution]['Rank'] = i + 1
                    individuals[dominated_solution].append(i + 1)
                    sub.add(dominated_solution)
        i += 1
        frontal = sub

    result = calc_crowding_dis(individuals)

    for key, value in result.items():
        population[key]['Rank'] = value[3]
        population[key]['CD'] = value[4]

    return population


#survivor selection
def select(population, offsprings, truck, boxes, total_value, count):
    survivors = {}
    offspring, fitness = evaluate(offsprings, truck, boxes, total_value)
    offspring = rank(offspring, fitness)
    pool = list(population.values()) + list(offspring.values())
    i = 1
    while len(survivors) < count:
        group = [values for values in pool if values['Rank'] == i]

        # If length of the group is lesser append the whole group
        if len(group) <= count - len(survivors):
            j = 0
            for index in range(len(survivors), len(survivors)+len(group)):
                survivors[index] = group[j]
                j += 1

        # If length of the group is bigger than needed, sort according to CD
        else:
            group = sorted(group, key=lambda x: x['CD'], reverse=True)
            j = 0
            for index in range(len(survivors), count):
                survivors[index] = group[j]
                j += 1
        i += 1

    return survivors

#rekombinasyon
def recombine(parents):
    offsprings = {}
    keys = list(parents.keys())
    random.shuffle(keys)
    for x in range(0, len(parents), 2):
        k1 = random.choice(keys)
        o1 = deepcopy(parents[k1]['order'])
        r1 = deepcopy(parents[k1]['rotate'])
        keys.remove(k1)
        k2 = random.choice(keys)
        o2 = deepcopy(parents[k2]['order'])
        r2 = deepcopy(parents[k2]['rotate'])
        keys.remove(k2)

        i = random.randint(1, int(len(o1) / 2) + 1)
        j = random.randint(i + 1, int(len(o1) - 1))
        # print("Values of i is {} and j is {}".format(i, j))
        co1 = [-1] * len(o1)
        co2 = [-1] * len(o2)
        cr1 = [-1] * len(r1)
        cr2 = [-1] * len(r2)

        co1[i:j + 1] = o1[i:j + 1]
        co2[i:j + 1] = o2[i:j + 1]
        cr1[i:j + 1] = r1[i:j + 1]
        cr2[i:j + 1] = r2[i:j + 1]
        pos = (j + 1) % len(o2)
        for k in range(len(o2)):
            if o2[k] not in co1 and co1[pos] == -1:
                co1[pos] = o2[k]
                pos = (pos + 1) % len(o2)
        pos = (j + 1) % len(o2)
        for k in range(len(o1)):
            if o1[k] not in co2 and co2[pos] == -1:
                co2[pos] = o1[k]
                pos = (pos + 1) % len(o1)
        pos = (j + 1) % len(o2)
        for k in range(len(r2)):
            if cr1[pos] == -1:
                cr1[pos] = r2[k]
                pos = (pos + 1) % len(r2)
        pos = (j + 1) % len(o2)
        for k in range(len(r1)):
            if cr2[pos] == -1:
                cr2[pos] = r1[k]
                pos = (pos + 1) % len(r1)
        offsprings[x] = {'order': deepcopy(co1), 'rotate': deepcopy(cr1)}
        offsprings[x + 1] = {'order': deepcopy(co2), 'rotate': deepcopy(cr2)}
    return offsprings


#parent seçimi
def select_parents(individuals, num, k):
    parents = {}
    for each in range(num):
        pool = random.sample(individuals, k)
        if pool[0]['Rank'] > pool[1]['Rank']:
            parents[each] = pool[0]
            individuals.remove(pool[0])
        elif pool[0]['Rank'] < pool[1]['Rank']:
            parents[each] = pool[1]
            individuals.remove(pool[1])
        elif pool[0]['CD'] > pool[1]['CD']:
            parents[each] = pool[0]
            individuals.remove(pool[0])
        else:
            parents[each] = pool[1]
            individuals.remove(pool[1])
    return parents


#Çaprazlama
def crossover(population, pc, k=3):
    p = select_parents(list(population.values()), pc, k)
    child = recombine(p)
    return child


#mutasyon
def mutate(offsprings, pm1, pm2,rotation=6):
    """
    :param offsprings: dictionary of the offsprings
    :param pm1: mutation probability constant 1
    :param pm2: mutation probability constant 1
    :param rotation: degrees of allowed rotations
    :return: dictionary of mutated offsprings
    """
    for child in offsprings.values():
        order = child['order']
        rotate = child['rotate']
        if random.uniform(0, 1) <= pm1:
            i = random.randint(1, int(len(order) / 2) + 1)
            j = random.randint(i + 1, int(len(order) - 1))
            order[i:j + 1] = order[j:i - 1:-1]
            rotate[i:j + 1] = rotate[j:i - 1:-1]

        # Second level of mutation
        for i in range(len(rotate)):
            if random.uniform(0, 1) <= pm2:
                rotate[i] = random.randint(0, rotation)

    return offsprings


#uygunluk hesabı
def evaluate(population, truck_dimension, boxes, total_value):

    container_vol = truck_dimension[0] * truck_dimension[1] * truck_dimension[2]
    ft = {}
    for key, individual in population.items():
        dblf = [[0, 0, 0] + truck_dimension]
        occupied_vol = 0
        number_boxes = 0
        value = 0
        result = []
        for box_number, r in zip(individual['order'], individual['rotate']):
            dblf = sorted(dblf, key=itemgetter(3))
            dblf = sorted(dblf, key=itemgetter(5))
            dblf = sorted(dblf, key=itemgetter(4))
            for pos in dblf:
                current = deepcopy(pos)
                space_vol = pos[3] * pos[4] * pos[5]
                box_vol = boxes[box_number][3]
                box_value = boxes[box_number][4]
                if r == 0:
                    l, w, h = boxes[box_number][0:3]
                elif r == 1:
                    w, l, h = boxes[box_number][0:3]
                elif r == 2:
                    l, h, w = boxes[box_number][0:3]
                elif r == 3:
                    h, l, w = boxes[box_number][0:3]
                elif r == 4:
                    h, w, l = boxes[box_number][0:3]
                else:
                    w, h, l = boxes[box_number][0:3]
                if space_vol >= box_vol and pos[3] >= l and pos[4] >= w and pos[5] >= h:
                    result.append(pos[0:3] + [l, w, h])
                    occupied_vol += box_vol
                    number_boxes += 1
                    value += box_value
                    top_space = [pos[0], pos[1], pos[2] + h, l, w, pos[5] - h]
                    beside_space = [pos[0], pos[1] + w, pos[2], l, pos[4] - w, pos[5]]
                    front_space = [pos[0] + l, pos[1], pos[2], pos[3] - l, pos[4], pos[5]]
                    dblf.remove(current)
                    dblf.append(top_space)
                    dblf.append(beside_space)
                    dblf.append(front_space)
                    break
        fitness = [round((occupied_vol / container_vol * 100), 2), round((number_boxes / len(list(boxes.keys())) * 100), 2),
                   round((value / total_value * 100), 2)]
        ft[key] = fitness
        population[key]['fitness'] = deepcopy(fitness)
        population[key]['result'] = result
    return population, ft


def plot_stats(average_fitness, title=""):
    x1 = range(len(average_fitness))
    avg_freespace = []
    avg_number = []
    avg_value = []

    for item in average_fitness:
        avg_freespace.append(item[0])
        avg_number.append(item[1])
        avg_value.append(item[2])

    plt.plot(x1, avg_freespace, label='Average Occupied Volume')
    plt.plot(x1, avg_number, label='Average Number of Boxes')
    plt.plot(x1, avg_value, label='Average Value of Boxes')
    plt.xlabel('Number of Generations')
    plt.ylabel('Fitness Values')
    plt.xticks(ticks=[t for t in x1 if t % 20 == 0])
    plt.title(title)
    plt.legend()
    plt.show()


#ortalama uygunluk
def calc_average_fitness(individuals):
    fitness_sum = [0.0, 0.0, 0.0]
    count = 0
    for key, value in individuals.items():
        if value['Rank'] == 1:
            count += 1
            fitness_sum[0] += value['fitness'][0]
            fitness_sum[1] += value['fitness'][1]
            fitness_sum[2] += value['fitness'][2]
    average = [number / count for number in fitness_sum]
    return average


def draw_pareto(population):
    fitness = []
    number = []
    weight = []
    fitness2 = []
    number2 = []
    weight2 = []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = []

    for key, value in population.items():
        if value['Rank'] == 1:
            fitness.append(value['fitness'][0])
            number.append(value['fitness'][1])
            weight.append(value['fitness'][2])
            colors.append('red')
        else:
            colors.append('blue')
            fitness2.append(value['fitness'][0])
            number2.append(value['fitness'][1])
            weight2.append(value['fitness'][2])

    if len(fitness) > 2:
        try:
            ax.scatter(fitness2, number2, weight2, c='b', marker='o')
            ax.scatter(fitness, number, weight, c='r', marker='o')
            triang = mtri.Triangulation(fitness, number)
            ax.plot_trisurf(triang, weight, color='red')
            ax.set_xlabel('occupied space')
            ax.set_ylabel('no of boxes')
            ax.set_zlabel('value')
            plt.show()
        except:
            print(
                "ERROR : Please try increasing the number of individuals as the unique Rank 1 solutions is less than 3")




#kutuların hacim, uzunluk, genişlik, yükseklik ve değer değerleri 
#azalan düzende sıralanarak beş farklı diziye dayalı beş farklı birey oluşturulur. 
#Bireylerin geri kalanı rastgele bir sırayla oluşturulur.
def generate_pop(box_params, count, rotation=5):
 
    population = {}
    if count > 5:
        x = 5
    else:
        x = count
    for i in range(0, x):
        sorted_box = dict(sorted(box_params.items(), key=lambda x: x[1][i]))
        population[i] = {"order": list(sorted_box.keys()),
                         "rotate": [random.randint(0, rotation - 1) for r in range(len(box_params))]}

    keys = list(box_params.keys())
    for i in range(5, count):
        random.shuffle(keys)
        population[i] = {"order": deepcopy(keys),
                         "rotate": [random.randint(0, rotation - 1) for r in range(len(box_params))]}
    return population





if __name__ == "__main__":
    with open('input.json', 'r') as outfile:
        data = json.load(outfile)
    problem_indices = list(data.keys())

    for p_ind in problem_indices:

        print("Running Problem Set {}".format(p_ind))
        print(tabulate([['Generations', NUM_OF_GENERATIONS], ['Individuals', NUM_OF_INDIVIDUALS],
                        ['Rotations', ROTATIONS], ['Crossover Prob.', PC], ['Mutation Prob1', PM1],
                        ['Mutation Prob2', PM2], ['Tournament Size', K]], headers=['Parameter', 'Value'],
                       tablefmt="github"))
        print()
        
        # Extracting inputs from the json file
        truck_dimension = data[p_ind]['truck dimension']
        packages = data[p_ind]['solution']
        boxes = data[p_ind]['boxes']
        total_value = data[p_ind]['total value']
        box_count = data[p_ind]['number']
        box_params = {}
        for index in range(len(boxes)):
            box_params[index] = boxes[index]

        # Storing the average values over every single iteration
        average_vol = []
        average_num = []
        average_value = []

        for i in range(NUM_OF_ITERATIONS):
            # Generate the initial population 
            population = generate_pop(box_params, NUM_OF_INDIVIDUALS, ROTATIONS)
            gen = 0
            average_fitness = []
            while gen < NUM_OF_GENERATIONS:
                population, fitness = evaluate(population, truck_dimension, box_params, total_value)
                population = rank(population, fitness)
                offsprings = crossover(deepcopy(population), PC, k=K)
                offsprings = mutate(offsprings, PM1, PM2, ROTATIONS)
                population = select(population, offsprings, truck_dimension, box_params, total_value,
                                       NUM_OF_INDIVIDUALS)
                average_fitness.append(calc_average_fitness(population))
                gen += 1
            results = []

            # Storing the final Rank 1 solutions
            for key, value in population.items():
                if value['Rank'] == 1:
                    results.append(value['result'])

            # Plot using plotly
            color_index = vis.draw_solution(pieces=packages)
            vis.draw(results, color_index)

            # Plot using matplotlib
            color_index = vis2.draw(pieces=packages, title="True Solution Packing")
            for each in results:
                vis2.draw(each, color_index, title="Rank 1 Solution Packing For Iteration {}".format(i))
            draw_pareto(population)
            average_vol.append(average_fitness[-1][0])
            average_num.append(average_fitness[-1][1])
            average_value.append(average_fitness[-1][2])
            plot_stats(average_fitness,
                       title="Average Fitness Values for Run {} over {} generations".format(i + 1,
                                                                                            NUM_OF_GENERATIONS))

        print(tabulate(
            [['Problem Set', p_ind], ['Runs', NUM_OF_ITERATIONS], ['Avg. Volume%', sum(average_vol) / len(average_vol)],
             ['Avg. Number%', sum(average_num) / len(average_num)],
             ['Avg. Value%', sum(average_value) / len(average_value)]],
            headers=['Parameter', 'Value'], tablefmt="github"))