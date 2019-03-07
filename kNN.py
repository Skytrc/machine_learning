import numpy as np
import operator


#%% create dataset
def create_dataset():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    label = ['A', 'A', 'B', 'B']
    return group, label


#%% create kNN algorithm
def classify0(inx, dataset, labels, k):
    # distance calculation
    # return the numbers of the rows of dataset
    dataset_size = dataset.shape[0]
    # in order to make test set equals dataset,then calculate different
    diff_mat = np.tile(inx, (dataset_size, 1)) - dataset
    sq_diffmat = diff_mat ** 2
    sq_distances = sq_diffmat.sum(axis=1)
    distances = sq_distances ** 0.5
    sorted_dist_indicies = distances.argsort()
    class_count={}
    for i in range(k):
        # choose k point with the smallest distance
        vote_ilabel = labels[sorted_dist_indicies[i]]
        class_count[vote_ilabel] = class_count.get(vote_ilabel, 0) + 1
    # sort
    sorted_class_count = sorted(class_count.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


#%% test
group, labels = create_dataset()
test = classify0([0, 0], group, labels, 3)
print(test)


#%% put file records to matrix
def file2maxtix(filename):
    fr = open(filename)
    array_olines = fr.readline()
    # get the number of flies lines
    number_of_lines = len(array_olines)
    # create and return numpy matrix
    return_mat = np.zeros((number_of_lines, 3))
    class_label_vector = {}
    index = 0
    for line in array_olines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index:] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector
