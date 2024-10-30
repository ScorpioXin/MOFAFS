from Env_Workshop import JobShop
from Draw import combine_chart

import statistics
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import matplotlib.pyplot as plt
import torch
import time
torch.manual_seed(1)


def timer(stime, cur_epi, tot_epi):
    running_time = time.time() - stime
    hour = int(running_time / 3600)
    minute = int((running_time % 3600) / 60)
    second = round(running_time % 3600 % 60, 2)
    print(f'\repisode:{cur_epi}     training progress:{cur_epi / tot_epi * 100:.1f}%  --->  running time:{hour}h{minute}m{second}s', end="")
    time.sleep(0.01)


class PolicyNet(nn.Module):     # actor
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actor_probs = F.softmax(x, dim=-1)
        return actor_probs


class ValueNet(nn.Module):      # critic
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        state_value = self.fc2(x)
        return state_value


class Proximal_Policy_Optimization:
    def __init__(self, env, state_dim, action_dim, hidden_dim=256, memory_episode_size=5, batch_size=32, clip_size=0.2,
                 alpha=0.6, update_size=10, gamma=0.98, actor_lr=1e-5, critic_lr=1e-5, replay_factor=0.25):
        self.env = env
        self.memory_episode_size = memory_episode_size
        self.batch_size = batch_size
        self.clip_size = clip_size
        self.alpha = alpha
        self.update_size = update_size
        self.gamma = gamma
        self.replay_factor = replay_factor
        self.actor_network = PolicyNet(state_dim, action_dim, hidden_dim)
        self.critic_network = ValueNet(state_dim, hidden_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor_network.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_network.parameters(), lr=critic_lr)
        self.prior_probs = None
        self.last_decision_time = None
        self.current_done = 0
        self.seed = 0

    def choose_action(self, state):
        # state = torch.tensor(state, dtype=torch.float)
        probs = self.actor_network(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()               # maybe it can be replaced by greedy-policy
        return action.item(), probs[action.item()].item()

    def get_state_value(self, state):
        # state = torch.tensor(state, dtype=torch.float)
        state_value = self.critic_network(state)
        return state_value

    def param_learn(self, states, actions, returns, old_probs):
        # returns = returns.view(-1, 1)
        state_value = self.get_state_value(states)
        delta = returns - state_value
        advantage = delta.detach()

        new_probs = self.actor_network(states).gather(1, actions)
        ratio = new_probs / old_probs
        surrogate_loss = ratio * advantage
        clip_loss = torch.clamp(ratio, 1-self.clip_size, 1+self.clip_size) * advantage
        action_loss = -torch.min(surrogate_loss, clip_loss).mean()
        critic_loss = F.mse_loss(state_value, returns)

        self.actor_optimizer.zero_grad()
        action_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for _ in range(len(advantage)):
            if advantage[_] < 0:
                advantage[_] = 1e-5
        prior_probs = advantage ** self.alpha
        return prior_probs.flatten()

    def param_update(self, states_buffer, actions_buffer, returns_buffer, probs_buffer, current_episode, total_episode):
        states_buffer = torch.tensor(states_buffer, dtype=torch.float)
        actions_buffer = torch.tensor(actions_buffer, dtype=torch.long).view(-1, 1)         # (batch_size, 1)
        returns_buffer = torch.tensor(returns_buffer, dtype=torch.float).view(-1, 1)
        probs_buffer = torch.tensor(probs_buffer, dtype=torch.float).view(-1, 1)

        self.prior_probs = torch.zeros(len(actions_buffer), dtype=torch.float)
        for _ in range(self.update_size):
            for idxs in BatchSampler(SubsetRandomSampler(range(len(actions_buffer))), self.batch_size, False):
                self.prior_probs[idxs] = self.param_learn(states_buffer[idxs], actions_buffer[idxs], returns_buffer[idxs], probs_buffer[idxs])
        prior_replay_probs = self.prior_probs / torch.sum(self.prior_probs)
        prior_replay_size = 1 + int(current_episode/total_episode * len(actions_buffer) * self.replay_factor)
        indices = torch.multinomial(prior_replay_probs, prior_replay_size, replacement=False)
        self.param_learn(states_buffer[indices], actions_buffer[indices], returns_buffer[indices], probs_buffer[indices])

    def step(self, available_machine):
        # if self.env.completed_decision_num == 0:
        #     self.env.machine_operation_time += (self.env.current_time - self.last_decision_time) *\
        #                                torch.clamp(self.env.machine_operation_end_time - self.env.current_time, 0, 1)
        s = self.env.state()
        a, p = self.choose_action(s)
        machine, category, individual, process_time = self.env.execute_action(a, available_machine)
        available_machine[machine] = 0
        before_makespan = max(self.env.machine_idle_time)
        self.env.env_update(machine, category, individual, process_time)
        after_makespan = max(self.env.machine_idle_time)
        r1, r2 = self.env.reward(before_makespan, after_makespan, machine, process_time)
        return s.tolist(), a, r1, r2, p

    def identify_available_machine(self, current_time):
        avail_mach = torch.zeros_like(self.env.machine_idle_time)
        self.env.machine_idle_yn = (self.env.machine_idle_time <= current_time) * 1
        for idx, yn in enumerate(self.env.machine_idle_yn):
            machine = idx + 1
            if yn == 1:  # machine state is idle
                for category, remain_order in enumerate(self.env.product_remain_order):
                    for order in remain_order:
                        if (order != 0) and (machine in self.env.available_machine[category][-order]):
                            avail_mach[idx] = 1  # symbolize machine is available
                            break
                    if avail_mach[idx] == 1:
                        break
        return avail_mach

    def save(self, episode):
        if episode % 500 == 0:
            model_name = 'Model/PPOModel'+str(episode)+'.pth'
            torch.save(self, model_name)

    def train(self, total_episode):
        stime = time.time()
        makespan_episode, utilization_episode = [], []
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))

        for current_episode in range(1, total_episode+1):
            timer(stime, current_episode, total_episode)
            states_buffer, actions_buffer, returns_buffer, probs_buffer = [], [], [], []
            memory_rb1, memory_rb2, utilization = [], [], []        # define to plot message
            for memory_episode in range(self.memory_episode_size):
                sb, ab, rb1, rb2, pb = [], [], [], [], []
                self.env.reset(self.seed)
                self.seed += 1

                self.last_decision_time = torch.tensor(0)
                while True:
                    machine_idle_time = self.env.machine_idle_time.clone()
                    for idle_time in sorted(machine_idle_time):
                        if idle_time > self.env.current_time or sum(machine_idle_time) == 0:
                            current_time = idle_time        # time moving
                            self.env.machine_operation_time += (current_time - self.last_decision_time) * \
                                                               torch.clamp(self.env.machine_operation_end_time - self.last_decision_time, 0, 1)
                            avail_mach = self.identify_available_machine(current_time)
                            self.last_decision_time = current_time
                            self.env.completed_decision_num = torch.tensor(0.)
                            # print(self.env.machine_idle_time, self.env.current_time, current_time, avail_mach)

                            if sum(avail_mach) == 0:
                                continue
                            while avail_mach.any():
                                self.env.current_time = current_time
                                s, a, r1, r2, p = self.step(avail_mach)
                                sb.append(s)
                                ab.append(a)
                                rb1.append(r1)
                                rb2.append(r2)
                                pb.append(p)
                                self.env.completed_decision_num += 1
                                avail_mach = self.identify_available_machine(current_time)
                        else:
                            continue
                        # self.env.current_time = current_time
                        break
                    if sum(self.env.product_remain_num) == 0:
                        break

                r = (torch.tensor(rb1) + torch.tensor(r2) * (sum(rb1)/sum(rb2))).tolist()
                Vs = 0
                rb = []
                for reward in r[::-1]:
                    Vs = reward + self.gamma * Vs
                    rb.insert(0, Vs)

                memory_rb1.append(sum(rb1))
                # memory_rb2.append(sum(rb2))
                # print(self.env.machine_operation_time <= self.env.machine_idle_time)
                utilization.append(torch.mean(self.env.machine_operation_time/self.env.machine_idle_time).item())

                if current_episode >= total_episode - 5:
                    # print(set(ab))
                    print(ab)

                states_buffer.extend(sb)
                actions_buffer.extend(ab)
                returns_buffer.extend(rb)
                probs_buffer.extend(pb)
            self.param_update(states_buffer, actions_buffer, returns_buffer, probs_buffer, current_episode, total_episode)
            self.save(current_episode)

            makespan_episode.append(statistics.mean(memory_rb1))
            utilization_episode.append(statistics.mean(utilization))
            # combine_chart(makespan_episode, utilization_episode, axs, window_size=10)
            # plt.pause(0.5)
        combine_chart(makespan_episode, utilization_episode, axs, window_size=10)
        plt.show()
        return makespan_episode, utilization_episode



if __name__ == "__main__":
    file_path = 'Beachmark\Mk01.fjs'
    workshop = JobShop(file_path)
    ppo = Proximal_Policy_Optimization(workshop, state_dim=50, action_dim=9)
    makespan_episode_return, consumption_episode_return = ppo.train(1000)
