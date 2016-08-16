import numpy as np
from time import sleep
from tqdm import tqdm
from pylab import plt
from model__adapted__ import Model
from os import path, mkdir


class TheMostSimpleTask(object):

    @classmethod
    def run(cls):

        # Decision maker faces a single decision node (that's why possible moves are always the same).
        # For being rewarded, the decision maker needs to choose move '1'.

        model = Model(filename="economics-model-parameters.json")

        results_mot = []
        results_cog = []

        weight_list = []

        t_max = 500

        for i in tqdm(range(t_max)):

            choice = model.choose(
                possible_moves=np.array([1, 1, 0, 0], dtype=float),
                possible_strategies=np.array([1, 1, 1, 1], dtype=float)
            )

            results_cog.append(choice["cog"])
            results_mot.append(choice["mot"])

            if choice["mot"] == 1:
                model.learn(reward=1)
            else:
                model.learn(reward=0)

            weight_list.append(model["CTX:cog -> STR:cog"].weights.copy())

        sleep(0.1)
        print()

        print("Cog results: ",
              "0:", results_cog.count(0)/len(results_cog),
              "; 1:", results_cog.count(1) / len(results_cog),
              "; 2:", results_cog.count(2) / len(results_cog),
              "; 3:", results_cog.count(3) / len(results_cog),
              "; -1:", results_cog.count(-1) / len(results_cog))

        print("Mot results: ",
              "0:", results_mot.count(0)/len(results_mot),
              "; 1:", results_mot.count(1) / len(results_mot),
              "; 2:", results_mot.count(2) / len(results_mot),
              "; 3:", results_mot.count(3) / len(results_mot),
              "; -1:", results_mot.count(-1) / len(results_mot))

        cls.plot(results_mot)
        cls.plot_weights(weight_list)

    @classmethod
    def plot(cls, mot_results):

        if not path.exists("../figures-test"):
            mkdir("../figures-test")

        bool_mot_results = np.asarray(mot_results) == 1

        t_max = len(bool_mot_results)

        average_t = np.zeros(t_max)

        time_window = 10

        for t in range(t_max):

            if t < time_window:
                average_t[t] = np.mean(bool_mot_results[:t + 1])
            else:
                average_t[t] = np.mean(bool_mot_results[t - 10:t + 1])

        plt.plot(np.arange(t_max), average_t, linewidth=2)
        plt.ylim([-0.01, 1.01])
        plt.savefig("../figures-test/fig-behavior.pdf")
        plt.close()

    @classmethod
    def plot_weights(cls, weight_list):

        if not path.exists("../figures-test"):
            mkdir("../figures-test")

        weight_list = np.asarray(weight_list)

        t_max = len(weight_list)

        for i in range(4):

            plt.plot(np.arange(t_max), weight_list[:, i], linewidth=2)
            plt.ylim([-0.01, 1.01])
            plt.savefig("../figures-test/fig{}.pdf".format(i))
            plt.close()


class RewardInSecondStep(object):

    @classmethod
    def run(cls):

        # Multiple stage game

        # Best strategy is to choose '1' at the first stage, then '2' at the second.

        model = Model(filename="economics-model-parameters.json", hebbian=False)

        results_mot = []
        results_cog = []
        weight_list = []

        t_max = 500

        decision_node = 0

        reward = 2

        for i in tqdm(range(t_max)):

            if decision_node == 0:
                mot = -1
                while mot == -1:

                    choice = model.choose(
                        possible_moves=np.array([1, 1, 0, 0], dtype=float),
                        possible_strategies=np.array([1, 1, 1, 1], dtype=float),
                    )
                    mot = choice["mot"]

                results_cog.append(choice["cog"])
                results_mot.append(choice["mot"])

                if choice["mot"] == 1:
                    decision_node = 1

                model.learn(reward=0)
            else:
                mot = -1
                while mot == -1:
                    choice = model.choose(
                        possible_moves=np.array([0, 0, 1, 1], dtype=float),
                        possible_strategies=np.array([1, 1, 1, 1], dtype=float),
                    )
                    mot = choice["mot"]

                results_cog.append(choice["cog"])
                results_mot.append(choice["mot"])

                if choice["mot"] == 2:

                    model.learn(reward=reward)
                else:
                    model.learn(reward=0)

                decision_node = 0

            weight_list.append(model["CTX:cog -> STR:cog"].weights.copy())

        sleep(0.1)
        print()

        print("Cog results: ",
              "0:", results_cog.count(0) / len(results_cog),
              "; 1:", results_cog.count(1) / len(results_cog),
              "; 2:", results_cog.count(2) / len(results_cog),
              "; 3:", results_cog.count(3) / len(results_cog),
              "; -1:", results_cog.count(-1) / len(results_cog))

        print("Mot results: ",
              "0:", results_mot.count(0) / len(results_mot),
              "; 1:", results_mot.count(1) / len(results_mot),
              "; 2:", results_mot.count(2) / len(results_mot),
              "; 3:", results_mot.count(3) / len(results_mot),
              "; -1:", results_mot.count(-1) / len(results_mot))

        cls.plot(mot_results=results_mot)
        cls.plot_weights(weight_list)

    @classmethod
    def plot(cls, mot_results):

        if not path.exists("../figures-test"):
            mkdir("../figures-test")

        bool_mot_results_cond0 = np.asarray(mot_results) == 1
        bool_mot_results_cond1 = np.asarray(mot_results) == 2

        bool_mot_results = bool_mot_results_cond0 + bool_mot_results_cond1

        t_max = len(bool_mot_results)

        average_t = np.zeros(t_max)

        time_window = 10

        for t in range(t_max):

            if t < time_window:
                average_t[t] = np.mean(bool_mot_results[:t + 1])
            else:
                average_t[t] = np.mean(bool_mot_results[t - 10:t + 1])

        plt.plot(np.arange(t_max), average_t, linewidth=2)
        plt.ylim([-0.01, 1.01])
        plt.savefig("../figures-test/fig-behavior.pdf")
        plt.close()

    @classmethod
    def plot_weights(cls, weight_list):

        if not path.exists("../figures-test"):
            mkdir("../figures-test")

        weight_list = np.asarray(weight_list)

        t_max = len(weight_list)

        for i in range(4):

            plt.plot(np.arange(t_max), weight_list[:, i], linewidth=2)
            plt.ylim([-0.01, 1.01])
            plt.savefig("../figures-test/fig{}.pdf".format(i))
            plt.close()


def main():

    # # Uncomment for the most simple task
    # TheMostSimpleTask.run()

    RewardInSecondStep.run()

if __name__ == "__main__":

    main()

