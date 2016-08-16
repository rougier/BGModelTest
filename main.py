import numpy as np
from tqdm import tqdm
from time import sleep
from model import Model


class Task(object):

    def __init__(self):

        self.results_mot = []
        self.results_cog = []

    def process(self, trial, cog, mot, RT, model):

        # print("cog", cog, "mot", mot, "RT", RT)
        self.results_cog.append(cog)
        self.results_mot.append(mot)


def main():

    task = Task()

    trial = {"mot": np.ones(4), "ass": np.ones((4, 4)), "cog": np.ones(4)}
    trial["ass"][:] = [[1, 0, 1, 0],
                       [1, 0, 0, 1],
                       [0, 1, 1, 0],
                       [0, 1, 0, 1]]
    trial["mot"][:] = [1, 1, 0, 0]
    model = Model(filename="model-topalidou-original.json")

    t_max = 100
    for i in tqdm(range(t_max)):
        model.process(task=task, trial=trial)

    sleep(0.1)
    print()

    print("Cog results: ",
          "0:", task.results_cog.count(0)/len(task.results_cog),
          "; 1:", task.results_cog.count(1) / len(task.results_cog),
          "; 2:", task.results_cog.count(2) / len(task.results_cog),
          "; 3:", task.results_cog.count(3) / len(task.results_cog),
          "; -1:", task.results_cog.count(-1) / len(task.results_cog))

    print("Mot results: ",
          "0:", task.results_mot.count(0)/len(task.results_mot),
          "; 1:", task.results_mot.count(1) / len(task.results_mot),
          "; 2:", task.results_mot.count(2) / len(task.results_mot),
          "; 3:", task.results_mot.count(3) / len(task.results_mot),
          "; -1:", task.results_mot.count(-1) / len(task.results_mot))

if __name__ == "__main__":

    main()

