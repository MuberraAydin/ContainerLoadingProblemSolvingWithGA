import json
import random

MIN_BOXES = 10
MAX_BOXES = 36
MIN_VALUE = 50
MAX_VALUE = 500
MAX_TRUCK_LEN = 600
MIN_TRUCK_LEN = 50
MAX_TRUCK_WID = 600
MIN_TRUCK_WID = 50
MAX_TRUCK_HT = 600
MIN_TRUCK_HT = 50


def generateboxes(container, num):
    retry = 500 * num
    while num > 1:
        cuboid = random.choice(container)
        while cuboid[3] <= 11 or cuboid[4] <= 11 or cuboid[5] <= 11:
            retry -= 1
            if retry == 0:
                print("Cannot partition into packages. Please try again")
                return
            cuboid = random.choice(container)
        container.remove(cuboid)
        prob = random.uniform(0, 1)
        x1 = cuboid[0]
        y1 = cuboid[1]
        z1 = cuboid[2]
        x2 = cuboid[3]
        y2 = cuboid[4]
        z2 = cuboid[5]
        if prob < 0.35:
            # Split in length
            t = random.randint(5, int(x2 / 2))
            package1 = [x1 + t, y1, z1, x2 - t, y2, z2]
            package2 = [x1, y1, z1, t, y2, z2]
        elif prob < 0.65:
            # Split in width
            t = random.randint(5, int(y2 / 2))
            package1 = [x1, y1 + t, z1, x2, y2 - t, z2]
            package2 = [x1, y1, z1, x2, t, z2]

        else:
            # Split in height
            t = random.randint(5, int(z2 / 2))
            package1 = [x1, y1, z1 + t, x2, y2, z2 - t]
            package2 = [x1, y1, z1, x2, y2, t]

        container.append(package1)
        container.append(package2)
        num -= 1

    return container




truck_dim = [[random.randint(MIN_TRUCK_LEN, MAX_TRUCK_LEN), random.randint(MIN_TRUCK_WID, MAX_TRUCK_WID),
              random.randint(MIN_TRUCK_HT, MAX_TRUCK_HT)] for _ in range(5)]
NUM_BOXES = [
    [random.randint(MIN_BOXES, MAX_BOXES), random.randint(MIN_BOXES, MAX_BOXES), random.randint(MIN_BOXES, MAX_BOXES),
     random.randint(MIN_BOXES, MAX_BOXES), random.randint(MIN_BOXES, MAX_BOXES)] for _ in range(5)]
dataset = {}
i = 0
for cont, counts in zip(truck_dim, NUM_BOXES):
    for number in counts:
        packages = generateboxes([[0, 0, 0] + cont], number)
        boxes = []
        total_value = 0
        for each in packages:
            l, w, h = each[3:]
            vol = l * w * h
            value = random.randint(MIN_VALUE, MAX_VALUE)
            total_value += value
            boxes.append([l, w, h, vol, value])
        dataset[i] = {'truck dimension': cont, 'number': number, 'boxes': boxes, 'solution': packages,
                      'total value': total_value}
        i += 1

with open('input.json', 'w') as outfile:
    json.dump(dataset, outfile)
