import datetime
from microgrid.environments.data_center.data_center_env import DataCenterEnv
import pulp
import numpy as np

class DataCenterAgent:
    def __init__(self, env: DataCenterEnv):
        self.env = env

    def take_decision(self,
                      state,
                      previous_state=None,
                      previous_action=None,
                      previous_reward=None):

        l_IT = state["consumption_prevision"]
        lambdas = state["manager_signal"]
        p_HW = state["hotwater_price_prevision"]
        """Resolution par optimisation lineaire
        print(l_IT)
        print(lambdas)
        print(p_HW)

        #test pas de temps
        #print(self.env.delta_t/datetime.timedelta(hours=1))

        problem1 = pulp.LpProblem("data_center", pulp.LpMinimize)
        alphas = [0 for i in range(0,self.env.nb_pdt)]

        for i in range(self.env.nb_pdt):
            var_name = "alpha_" + str(i)
            alphas[i] = pulp.LpVariable(var_name, 0.0, 1.0)

        l_NF = [0 for i in range(self.env.nb_pdt)]
        h_r = [0 for i in range(self.env.nb_pdt)]
        l_HP = [0 for i in range(self.env.nb_pdt)]
        h_DC = [0 for i in range(self.env.nb_pdt)]

        for t in range(self.env.nb_pdt):
            l_NF[t] = (1 + 1 / (self.env.EER * (self.env.delta_t / datetime.timedelta(hours=1)))) * l_IT[t]
            h_r[t] = l_IT[t] * self.env.COP_CS / self.env.EER
            l_HP[t] = alphas[t] * h_r[t] / ((self.env.COP_HP - 1) * (self.env.delta_t / datetime.timedelta(hours=1)))
            h_DC[t] = self.env.COP_HP * (self.env.delta_t / datetime.timedelta(hours=1)) * l_HP[t]

            cons_name = "production limite en " + str(t)
            problem1 += h_DC[t] <= self.env.max_transfert, cons_name

        problem1 += sum([lambdas[i] * (l_NF[i] + l_HP[i]) - p_HW[i] * h_DC[i] for i in range(self.env.nb_pdt)]), "objectif"

        problem1.solve()
        alphas2 = [0 for i in range(self.env.nb_pdt)]
        for i in range(self.env.nb_pdt):
            alphas2[i] = alphas[i].varValue
        print(alphas2)
        print("Prix total : ", pulp.value(problem1.objective))

        return alphas2

        """

        #Resolution heuristique
        cop_hp = self.env.COP_HP
        cop_cs = self.env.COP_CS
        dt = self.env.delta_t / datetime.timedelta(hours=1)
        eer = self.env.EER

        alphas = [0 for t in range(self.env.nb_pdt)]
        for t in range(self.env.nb_pdt):
            if (lambdas[t] < p_HW[t]*cop_hp*dt) :
                alphas[t] = self.env.max_transfert * (cop_hp - 1) * eer / (cop_hp * cop_cs * l_IT[t])
            else:
                alphas[t] = 0

        return alphas


if __name__ == "__main__":
    delta_t = datetime.timedelta(minutes=30)
    time_horizon = datetime.timedelta(days=1)
    N = time_horizon // delta_t
    data_center_config = {
        'scenario': 10,
    }
    env = DataCenterEnv(data_center_config, nb_pdt=N)
    agent = DataCenterAgent(env)
    cumulative_reward = 0
    now = datetime.datetime.now()
    state = env.reset(now, delta_t)
    for i in range(N*2):
        action = agent.take_decision(state)
        state, reward, done, info = env.step(action)
        cumulative_reward += reward
        if done:
            break
        print(f"action: {action}, reward: {reward}, cumulative reward: {cumulative_reward}")
        print("State: {}".format(state))