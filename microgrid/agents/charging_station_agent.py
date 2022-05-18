import datetime
from microgrid.environments.charging_station.charging_station_env import ChargingStationEnv


class ChargingStationAgent:
    def __init__(self, env: ChargingStationEnv):
        self.env = env

    def take_decision(self,
                      state,
                      previous_state=None,
                      previous_action=None,
                      previous_reward=None):
        nbvoitures = self.env.nb_evs
        T = self.env.nb_pdt
        dt = self.env.delta_t / datetime.timedelta(hours=1)
        penalité = 5.
        t_dep_arr = []  # de départ de d'arrivée
        t_ini = []  # temps de branchement initial
        lambdaa = state["manager_signal"]
        a_initial = state["soc"]
        l = {}
        a = {}
        penalite = {}

        for j in range(1, nbvoitures):
            depart = []
            t = 0

            while t <= T - 2:
                k = 0
                while t < T - k - 1 and state["is_plugged_prevision"][j - 1][t + k] == \
                        state["is_plugged_prevision"][j - 1][t]:
                    k += 1
                if state["is_plugged_prevision"][j - 1][t] == 0 and t != 0:
                    depart.append([t, t + k])
                t += k + 1

            if depart != [] and depart[-1][-1] == T - 1:
                depart[-1][-1] = None

            t_dep_arr.append(depart)
            t_initial = 1
            while t_initial <= T - 1 and state["is_plugged_prevision"][j - 1][t_initial - 1] == 0:
                t_initial += 1
            t_ini.append(t_initial)

        lp = pulp.LpProblem("charging_station" + ".lp", pulp.LpMinimize)
        lp.setSolver()

        # Création des variables
        for j in range(1, nbvoitures):
            l[j] = {}
            a[j] = {}
            penalite[j] = {}
            for (i, p) in enumerate(t_dep_arr[j - 1]):
                var_name = "fined_" + str(j) + '_' + str(i)
                penalite[j][i] = pulp.LpVariable(var_name, cat="Binary")  # booléen pour savoir si pénalité il y a
            for t in range(1, T):
                var_name = "l_" + str(j) + "_" + str(t)
                l[j][t] = pulp.LpVariable(var_name, self.env.evs[j - 1].battery.pmin,
                                          self.env.evs[j - 1].battery.pmax)  # puissance pour j à t
                var_name = "a_" + str(j) + "_" + str(t)
                a[j][t] = pulp.LpVariable(var_name, 0.,
                                          self.env.evs[j - 1].battery.capacity)  # état batterie j au temps t

        # CONTRAINTES
        for t in range(1, T):
            # capacité stations
            const_name = "borne supérieur" + str(t)
            lp += pulp.lpSum([l[j][t] for j in range(1, nbvoitures)]) <= self.env.pmax_site, const_name
            const_name = "borne inférieure" + str(t)
            lp += pulp.lpSum([l[j][t] for j in range(1, nbvoitures)]) >= - self.env.pmax_site, const_name

        for j in range(1, nbvoitures):
            if t_dep_arr[j - 1] != []:  # S'il y a un départ pour la voiture j
                for (i, p) in enumerate(t_dep_arr[j - 1]):

                    const_name = "recharge minimum" + str(j) + "_" + str(i)
                    lp += (a[j][p[0]] >= self.env.evs[j - 1].battery.capacity * 0.25 * (1 - penalite[j][i])), const_name

                    const_name = "énergie minimum" + str(j) + '_' + str(i)  # as-t-on assez d'énergie ?
                    lp += a[j][p[0]] >= 4, const_name

                    if p[-1] != None:
                        const_name = "consommation" + str(j) + "_" + str(p[1]) + "_" + str(i)
                        lp += (a[j][p[1]] == a[j][p[0]] - 4), const_name

                        # relation entre les états de la batterie
                        const_name = "evolution" + str(j) + '_' + str(p[1])
                        lp += (a[j][p[1] + 1] == a[j][p[1]] + self.env.evs[j - 1].battery.efficiency * l[j][
                            p[1]] * dt), const_name

            # état de charge initial
            const_name = "initial_state_ev_" + str(j)
            lp += (a[j][t_ini[j - 1]] == a_initial[j - 1]), const_name

            for t in range(1, T):
                if not (state["is_plugged_prevision"][j - 1][t - 1]):  # pas en charge

                    const_name = "away_from_charging_station_" + str(j) + "_" + str(t)
                    lp += l[j][t] == 0., const_name
                else:
                    if t < T - 1:
                        const_name = "evolution_state_" + str(j) + '_' + str(t)
                        lp += (a[j][t + 1] == a[j][t] + self.env.evs[j - 1].battery.efficiency * l[j][
                            t] * 0.5), const_name

        lp.setObjective(pulp.lpSum(
            pulp.lpSum(l[j][t] * lambdaa[t - 1] for j in range(1, nbvoitures)) for t in range(1, T)) + pulp.lpSum(
            pulp.lpSum(penalite[j][i] for i in range(len(t_dep_arr[j - 1]))) for j in range(1, nbvoitures)) * penalité)

        lp.solve()
        sol = self.env.action_space.sample()

        for j in range(0, nbvoitures - 1):
            for t in range(0, T - 1):
                sol[j][t] = l[j + 1][t + 1].varValue

        return (sol)


if __name__ == "__main__":
    delta_t = datetime.timedelta(minutes=30)
    time_horizon = datetime.timedelta(days=1)
    N = time_horizon // delta_t
    evs_config = [
        {
            'capacity': 40,
            'pmax': 22,
        },
        {
            'capacity': 40,
            'pmax': 22,
        },
        {
            'capacity': 40,
            'pmax': 3,
        },
        {
            'capacity': 40,
            'pmax': 3,
        },
    ]
    station_config = {
        'pmax': 40,
        'evs': evs_config
    }
    env = ChargingStationEnv(station_config=station_config, nb_pdt=N)
    agent = ChargingStationAgent(env)
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
        print("Info: {}".format(action.sum(axis=0)))