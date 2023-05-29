import numpy


def generate_triplet_pairs(values, ids, dup: int = 1):
    """
    generate list of triplet from array values,
    ids list indicates which value are from the same class
    :param values: list contains specific sample
    :param ids: id of sample (e.g: specify its cluster)
    :param dup: number of picking negative sample for same (anchor, positive) pair
    """
    if not values:
        return []
    value2id = {}
    id2value = {}
    for idx, value in enumerate(values):
        vid = ids[idx]
        value2id[value] = vid

        if vid not in id2value:
            id2value[vid] = []
        id2value[vid].append(value)

    triplets = []

    def random_except(current, array, size=1):
        """
        get a list of `size` random elements in list except `current` element
        """
        if current in array and len(array) == 1:
            raise Exception('no more option!')
        x = numpy.random.choice(array, size + 1, replace=False).tolist()
        if current in x:
            x.remove(current)
        return x[:size]

    for id1, values in id2value.items():
        for i in range(len(values)):
            for j in range(len(values)):
                if i == j: continue
                n_pids = random_except(id1, list(id2value.keys()), dup)
                anchor, positive = values[i], values[j]
                for pid2 in n_pids:
                    negative = numpy.random.choice(id2value[pid2], 1)[0]
                    triplets.append((anchor, positive, negative))

    return triplets
