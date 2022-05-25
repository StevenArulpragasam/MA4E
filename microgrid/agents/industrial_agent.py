import pulp
import datetime
from microgrid.environments.industrial.industrial_env import IndustrialEnv



class IndustrialAgent:
    def __init__(self, env: IndustrialEnv):
        self.env = env

    def take_decision(self,
                      state,
                      previous_state=None,
                      previous_action=None,
                      previous_reward=None):
        
        # définition des constantes du problème #
        
        p_max = self.env.battery.pmax
        pd = self.env.battery.efficiency
        pc = self.env.battery.efficiency
        C = self.env.battery.capacity
        T = self.env.nb_pdt
        dt = 24/T
        
        # récupération des données #
        
        l_dem = state[´conso_prevision’]
        prix = manager_signal = state[‘manager_signal’]
        
        # problème linéaire et résolution
        
        lp = pulp.LpProblem("Gestion optimale de la batterie",pulp.LpMinimize)
        lp.setSolver()
        lbat_pos = []
        lbat_neg = []
        est_pos = []
        for t in range(T):
            lbat_pos.append(pulp.LpVariable("charge batterie plus " + str(t)))
            lbat_neg.append(pulp.LpVariable("charge batterie moins " + str(t)))
            est_pos.append(pulp.LpVariable("est positive " + str(t),cat = 'Binary'))
            lp += lbat_pos[t] <= p_max*est_pos[t]
            lp += lbat_neg[t] <= p_max*(1-est_pos[t])
            lp += lbat_pos[t] >= 0
            lp += lbat_neg[t] >= 0
            lp += 0 <= dt*pulp.lpSum([pc*lbat_pos[i]-1/pd*lbat_neg[i] for i in range(t)])
            lp += dt*pulp.lpSum([pc*lbat_pos[i]-1/pd*lbat_neg[i] for i in range(t)]) <= C
        lp.setObjective(pulp.lpSum([prix[t]*(ldem[t]+lbat_pos[t]-lbat_neg[t]) for t in range(T)]))
        lp.solve()
        
        # traitement des résultats, qui sont par défaut dans le désordre #
        
        for v in lp.variables()[:T]:
            ordre.append(int(v.name[int(T/2):]))

        for v in lp.variables()[:2*T]:
            charge.append(v.varValue)
            
        charge_moins = charge[:T]
        charge_plus = charge[T:]
        
        charge_réelle = []

        for i in range(T):
            charge_réelle.append(charge_plus[i]-charge_moins[i])
            
        return charge_réelle


if __name__ == "__main__":
    delta_t = datetime.timedelta(minutes=30)
    time_horizon = datetime.timedelta(days=1)
    N = time_horizon // delta_t
    industrial_config = {
        'battery': {
            'capacity': 100,
            'efficiency': 0.95,
            'pmax': 25,
        },
        'building': {
            'site': 1,
        }
    }
    env = IndustrialEnv(industrial_config=industrial_config, nb_pdt=N)
    agent = IndustrialAgent(env)
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
