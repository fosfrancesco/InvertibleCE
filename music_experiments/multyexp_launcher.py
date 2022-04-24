from experiments_script import start_experiment

reducers = ["NMF", "NTD"]
nmf_ranks = ["3", "6", "10"]
ntd3_ranks = ["[3,20,100]", "[6,20,100]", "[10,20,100]"]
ntd4_ranks = ["[3, 2, 3, 20]", "[6, 2, 3, 20]", "[10, 2, 3, 20]"]
max_iter = 200
dimensions = [3, 4]
targets = "[5,6]"

# NMF experiment

start_experiment(
    reducer, max_iter, gpu_number, targets, dimension, rank, layer, batch_size
)

