from experiments_script import start_experiment_noclick
import numpy as np
from pathlib import Path
import pandas as pd
from ICE.utils import TARGET_COMPOSERS

gpu_number = 1
layer = "layer4"
batch_size = 10
max_iter = 500
composer_names = [["Chopin","Bach"],["Chopin","Beethoven"],["Beethoven","Rachmaninoff"],["Scriabin","Schubert"]]


asap_basepath = Path("/share/cp/projects/concept_composers/asap-dataset")
#get a list of performances such as there are not 2 performances of the same piece
df = pd.read_csv(Path(asap_basepath,"metadata.csv"))
df = df.drop_duplicates(subset=["title","composer"])


reducers = ["NMF", "NTD"]
nmf_ranks = ["1", "2", "3", "4", "5", "6", "10", "8", "12"]
# nmf_ranks = []
ntd3_ranks = ["1", "2", "3", "4", "5", "6", "10", "8", "12"]
# ntd3_ranks = [
#     "[3,20,100]",
#     "[6,20,100]",
#     "[10,20,100]",
#     "[3,20,25]",
#     "[6,20,25]",
#     "[10,20,25]",
#     "[3,30,100]",
#     "[6,30,100]",
#     "[10,30,100]",
#     "[6,39,80]",
#     "[10,39,80]",
#     "[6,39,200]",
#     "[10,39,200]",
#     "[8,39,100]",
#     "[10,39,100]",
#     "[10,39,375]",
#     "[8,39,375]",
#     "[6,39,375]",
#     "[3,39,375]",
#     "[2,39,375]",
#     "[1,39,375]",
#     "[4,20,25]",
#     "[5,20,25]",
#     "[4,30,100]",
#     "[5,30,100]",
#     "[4,39,375]",
#     "[5,39,375]",
# ]
ntd4_ranks = ["1", "2", "3", "4", "5", "6", "10", "8", "12"]
# ntd4_ranks = [
#     "[3, 3, 2, 25]",
#     "[1, 3, 2, 25]",
#     "[2, 3, 2, 25]",
#     "[3, 5, 3, 25]",
#     "[3, 13, 3, 25]",
#     "[6, 13, 3, 25]",
#     "[3, 3, 3, 25]" "[3, 2, 3, 20]",
#     "[6, 2, 3, 20]",
#     "[10, 2, 3, 20]",
#     "[3, 2, 3, 25]",
#     "[6, 2, 3, 25]",
#     "[10, 2, 3, 25]",
#     "[10, 13, 3, 375]",
#     "[10, 3, 3, 375]",
#     "[8, 13, 3, 375]",
#     "[6, 13, 3, 375]",
#     "[3, 13, 3, 375]",
#     "[1, 13, 3, 375]",
#     "[2, 13, 3, 375]",
#     "[4, 13, 3, 375]",
#     "[5, 13, 3, 375]",
#     "[4, 2, 3, 20]",
#     "[5, 2, 3, 20]",
#     "[4, 2, 3, 25]",
#     "[5, 2, 3, 25]",
# ]
dimensions = [3, 4]


# targets = "[5,6]"
# targets_list = ["[6,9]", "[0,4]", "[5,6]", "[9,12]", "[9,11]"]

for comp_targets in composer_names:
    targets = str([TARGET_COMPOSERS.index(comp) for comp in comp_targets])
    datasets_size = df[df['composer'].isin(comp_targets)].shape[0] * 2
    # NMF experiment
    for r in nmf_ranks:
        try:
            start_experiment_noclick(
                reducers[0],
                max_iter,
                gpu_number,
                targets,
                dimensions[0],
                r,
                layer,
                batch_size,
            )
        except Exception as e:
            print("!!!!!!!!!!!")
            print(e)

    # NTD3 experiment
    for r in ntd3_ranks:
        try:
            start_experiment_noclick(
                reducers[1],
                max_iter,
                gpu_number,
                targets,
                dimensions[0],
                f"[{r},39,{datasets_size}]",
                layer,
                batch_size,
            )
        except Exception as e:
            print("!!!!!!!!!!!")
            print(e)

    # NTD4 experiment
    for r in ntd4_ranks:
        try:
            start_experiment_noclick(
                reducers[1],
                max_iter,
                gpu_number,
                targets,
                dimensions[1],
                f"[{r},13,3,{datasets_size}]",
                layer,
                batch_size,
            )
        except Exception as e:
            print("!!!!!!!!!!!")
            print(e)
