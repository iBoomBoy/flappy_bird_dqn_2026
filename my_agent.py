import numpy as np
import pygame
from pytorch_mlp import MLPRegression
import argparse
from console import FlappyBirdEnv


# ── constants ──────────────────────────────────────────────────────────────────
INPUT_DIM  = 8          # features fed to the network (see BUILD_STATE)
OUTPUT_DIM = 2          # Q-values for [jump, do_nothing]
LR         = 1e-3       # Adam learning rate
HIDDEN     = (256, 512, 256)  # bigger net than default for lv5 complexity

SCREEN_W   = 400        # from config.yml — used for normalisation
SCREEN_H   = 600        # from config.yml — used for normalisation
MAX_VEL    = 15.0       # rough max |velocity| for normalisation


class MyAgent:
    def __init__(self, show_screen=False, load_model_path=None, mode=None):
        # ── do not modify these ───────────────────────────────────────────────
        self.show_screen = show_screen
        if mode is None:
            self.mode = 'train'
        else:
            self.mode = mode

        # ── storage D: list of transitions ───────────────────────────────────
        # Each entry is a dict:
        #   phi_t   : np.ndarray  — state features at time t
        #   action  : int         — action taken at time t
        #   reward  : float|None  — filled in by receive_after_action_observation
        #   q_next  : float|None  — filled in by receive_after_action_observation
        self.storage = []

        # ── Q network (online, the one we train) ─────────────────────────────
        self.network = MLPRegression(
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            hidden_dim=list(HIDDEN),
            learning_rate=LR
        )
        # ── Q_f network (fixed target, for stable TD targets) ─────────────────
        self.network2 = MLPRegression(
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            hidden_dim=list(HIDDEN),
            learning_rate=LR
        )
        # initialise Q_f's weights to match Q
        MyAgent.update_network_model(net_to_update=self.network2,
                                     net_as_source=self.network)

        # ── hyper-parameters ─────────────────────────────────────────────────
        self.epsilon         = 0.08   # ε — keep low; flappy bird is fragile
        self.epsilon_min     = 0.01   # floor for epsilon decay
        self.epsilon_decay   = 0.995  # multiply ε by this each episode
        self.n               = 64     # minibatch size
        self.discount_factor = 0.99   # γ

        # ── do not modify this ────────────────────────────────────────────────
        if load_model_path:
            self.load_model(load_model_path)

    # ══════════════════════════════════════════════════════════════════════════
    # BUILD_STATE  — the most critical design decision
    # ══════════════════════════════════════════════════════════════════════════
    def build_state(self, state: dict) -> np.ndarray:
        """
        Converts the raw game state dict into a normalised feature vector φ.

        Features (all normalised to roughly [-1, 1] or [0, 1]):
          0  bird_y          / SCREEN_H
          1  bird_velocity   / MAX_VEL
          2  dx to next pipe / SCREEN_W          (how far away is the danger)
          3  next pipe top   / SCREEN_H          (where is the gap top)
          4  next pipe bottom/ SCREEN_H          (where is the gap bottom)
          5  dx to 2nd pipe  / SCREEN_W
          6  2nd pipe top    / SCREEN_H
          7  2nd pipe bottom / SCREEN_H

        Using TWO pipes ahead gives the agent enough lookahead for lv5 where
        pipes come fast and the bird needs to plan its trajectory.
        """
        bird_y   = state['bird_y']   / SCREEN_H
        bird_vel = state['bird_velocity'] / MAX_VEL

        # ── find the pipes that are still ahead of the bird ──────────────────
        bird_right = state['bird_x'] + state['bird_width']
        # a pipe is "ahead" when its right edge hasn't fully passed the bird
        ahead = [p for p in state['pipes']
                 if p['x'] + p['width'] > state['bird_x']]
        # sort by x so the closest pipe is first
        ahead.sort(key=lambda p: p['x'])

        def pipe_features(pipe):
            dx  = (pipe['x'] - bird_right) / SCREEN_W
            top = pipe['top']    / SCREEN_H
            bot = pipe['bottom'] / SCREEN_H
            return dx, top, bot

        if len(ahead) >= 2:
            dx1, top1, bot1 = pipe_features(ahead[0])
            dx2, top2, bot2 = pipe_features(ahead[1])
        elif len(ahead) == 1:
            dx1, top1, bot1 = pipe_features(ahead[0])
            dx2, top2, bot2 = 1.0, 0.5, 0.75   # dummy — no second pipe yet
        else:
            # no pipes on screen at all (happens briefly at episode start)
            dx1, top1, bot1 = 1.0, 0.5, 0.75
            dx2, top2, bot2 = 1.0, 0.5, 0.75

        phi = np.array([bird_y, bird_vel,
                        dx1, top1, bot1,
                        dx2, top2, bot2], dtype=np.float32)
        return phi

    # ══════════════════════════════════════════════════════════════════════════
    # REWARD  — shaped to encourage survival and penalise bad deaths differently
    # ══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def reward(state_after: dict) -> float:
        """
        r_t based on s_{t+1}:
          • Alive each step          → +1      (encourages survival)
          • hit_pipe                 → -100    (worst: bird had no height awareness)
          • off_screen (top/bottom)  → -50     (bad but shows some awareness)
          • well_done                → +200    (completed the episode perfectly)
        """
        done_type = state_after['done_type']
        if done_type == 'hit_pipe':
            return -100.0
        elif done_type == 'offscreen':
            return -50.0
        elif done_type == 'well_done':
            return 200.0
        else:
            return 1.0   # 'not_done' — survived this step

    # ══════════════════════════════════════════════════════════════════════════
    # ONEHOT  — mask which output node to train towards
    # ══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def onehot(action: int, num_actions: int = OUTPUT_DIM) -> np.ndarray:
        """
        Returns a binary weight vector w_j of shape (num_actions,).
        Only the index corresponding to action a_j is 1; the rest are 0.
        This tells fit_step to only update the Q-value for the action taken,
        leaving the other output node's gradient as zero.
        """
        w = np.zeros(num_actions, dtype=np.float32)
        w[action] = 1.0
        return w

    # ══════════════════════════════════════════════════════════════════════════
    # CHOOSE_ACTION  — ε-greedy during train, greedy during eval
    # ══════════════════════════════════════════════════════════════════════════
    def choose_action(self, state: dict, action_table: dict) -> int:
        """
        Algorithm 2 — CHOOSE_ACTION pseudocode:
          phi_t = BUILD_STATE(s_t)
          if train:
              with prob ε  → random action
              else         → argmax_a Q(phi_t, a)
              store partial transition (phi_t, a_t, None, None) in D
          elif eval:
              a_t = argmax_a Q(phi_t, a)
          return a_t
        """
        phi_t = self.build_state(state)

        if self.mode == 'train':
            if np.random.rand() < self.epsilon:
                # explore: random between jump(0) and do_nothing(1) only
                a_t = np.random.choice([action_table['jump'],
                                        action_table['do_nothing']])
            else:
                # exploit: pick action with highest Q-value
                q_vals = self.network.predict(phi_t.reshape(1, -1))[0]
                a_t = int(np.argmax(q_vals))

            # store partial transition — reward and q_next filled in later
            self.storage.append({
                'phi_t':   phi_t,
                'action':  a_t,
                'reward':  None,
                'q_next':  None,
            })

        else:  # eval mode — pure greedy, no exploration, no storage
            q_vals = self.network.predict(phi_t.reshape(1, -1))[0]
            a_t = int(np.argmax(q_vals))

        return a_t

    # ══════════════════════════════════════════════════════════════════════════
    # RECEIVE_AFTER_ACTION_OBSERVATION  — core DQN update step
    # ══════════════════════════════════════════════════════════════════════════
    def receive_after_action_observation(self, state: dict,
                                         action_table: dict) -> None:
        """
        Algorithm 2 — RECEIVE_AFTER_ACTION_OBSERVATION pseudocode:
          phi_{t+1} = BUILD_STATE(s_{t+1})
          r_t       = REWARD(s_{t+1})
          if terminal: q_{t+1} = 0
          else:         q_{t+1} = max_a Q_f(phi_{t+1}, a)
          update last transition in D with (r_t, q_{t+1})

          sample minibatch from D
          for each (phi_j, a_j, r_j, q_{j+1}):
              w_j = ONEHOT(a_j)
              y_j = r_j + γ * q_{j+1}
              add to X, Y, W
          fit Q on (X, Y, W)
          [optionally decay ε]
        """
        if self.mode != 'train':
            return

        # ── nothing stored yet (edge case) ───────────────────────────────────
        if len(self.storage) == 0:
            return

        phi_next = self.build_state(state)

        # ── compute r_t ──────────────────────────────────────────────────────
        r_t = self.reward(state)

        # ── compute q_{t+1} using fixed target network Q_f ───────────────────
        is_terminal = state['done']
        if is_terminal:
            q_next = 0.0
        else:
            q_vals_next = self.network2.predict(phi_next.reshape(1, -1))[0]
            q_next = float(np.max(q_vals_next))

        # ── update the last (partial) transition in D ─────────────────────────
        self.storage[-1]['reward'] = r_t
        self.storage[-1]['q_next'] = q_next

        # ── only train once we have enough complete transitions ───────────────
        complete = [t for t in self.storage
                    if t['reward'] is not None and t['q_next'] is not None]
        if len(complete) < self.n:
            return

        # ── sample a random minibatch ─────────────────────────────────────────
        indices   = np.random.choice(len(complete), size=self.n, replace=False)
        minibatch = [complete[i] for i in indices]

        X = np.zeros((self.n, INPUT_DIM),  dtype=np.float32)
        Y = np.zeros((self.n, OUTPUT_DIM), dtype=np.float32)
        W = np.zeros((self.n, OUTPUT_DIM), dtype=np.float32)

        for j, transition in enumerate(minibatch):
            phi_j   = transition['phi_t']
            a_j     = transition['action']
            r_j     = transition['reward']
            q_jp1   = transition['q_next']

            w_j = self.onehot(a_j)
            y_j = r_j + self.discount_factor * q_jp1

            # current Q predictions — we only update the taken action's output
            q_current = self.network.predict(phi_j.reshape(1, -1))[0]

            X[j] = phi_j
            Y[j] = q_current          # start from current predictions …
            Y[j, a_j] = y_j           # … then overwrite with the TD target
            W[j] = w_j                # mask: only train the taken action

        # ── one gradient step on Q ────────────────────────────────────────────
        self.network.fit_step(X, Y, W)

    # ══════════════════════════════════════════════════════════════════════════
    # DECAY EPSILON  — call once per episode from the training loop
    # ══════════════════════════════════════════════════════════════════════════
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL SAVE / LOAD / SYNC  (do not modify save/load)
    # ══════════════════════════════════════════════════════════════════════════
    def save_model(self, path: str = 'my_model.ckpt'):
        self.network.save_model(path=path)

    def load_model(self, path: str = 'my_model.ckpt'):
        self.network.load_model(path=path)

    @staticmethod
    def update_network_model(net_to_update: MLPRegression,
                             net_as_source: MLPRegression):
        net_to_update.load_state_dict(net_as_source.state_dict())


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=5)
    args = parser.parse_args()

    # ── training environment ─────────────────────────────────────────────────
    env   = FlappyBirdEnv(config_file_path='config.yml',
                          show_screen=False,       # False = faster training
                          level=args.level,
                          game_length=10)
    agent = MyAgent(show_screen=False)

    EPISODES           = 10000
    CLEAR_MEMORY_EVERY = 5     # clear D every N episodes (keeps Q targets fresh)
    UPDATE_QF_EVERY    = 5     # sync Q_f ← Q every N episodes

    best_avg_score  = -1.0
    recent_scores   = []

    for episode in range(1, EPISODES + 1):
        env.play(player=agent)

        score   = env.score
        mileage = env.mileage
        recent_scores.append(score)
        if len(recent_scores) > 20:
            recent_scores.pop(0)

        avg_score = np.mean(recent_scores)

        # ── save best model ──────────────────────────────────────────────────
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            agent.save_model(path='my_model.ckpt')

        # ── periodic logging ─────────────────────────────────────────────────
        if episode % 100 == 0:
            print(f"Ep {episode:>5} | score={score} | "
                  f"avg20={avg_score:.2f} | best_avg={best_avg_score:.2f} | "
                  f"ε={agent.epsilon:.4f} | D={len(agent.storage)}")

        # ── clear replay memory ──────────────────────────────────────────────
        if episode % CLEAR_MEMORY_EVERY == 0:
            agent.storage = []

        # ── sync Q_f ← Q ─────────────────────────────────────────────────────
        if episode % UPDATE_QF_EVERY == 0:
            MyAgent.update_network_model(net_to_update=agent.network2,
                                         net_as_source=agent.network)

        # ── decay epsilon ─────────────────────────────────────────────────────
        agent.decay_epsilon()

    # ══════════════════════════════════════════════════════════════════════════
    # EVALUATION  (mirrors Gradescope evaluation)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n--- Evaluation ---")
    env2   = FlappyBirdEnv(config_file_path='config.yml',
                           show_screen=False, level=args.level)
    agent2 = MyAgent(show_screen=False,
                     load_model_path='my_model.ckpt',
                     mode='eval')

    scores = []
    for episode in range(10):
        env2.play(player=agent2)
        scores.append(env2.score)

    print(f"Max score:  {np.max(scores)}")
    print(f"Mean score: {np.mean(scores):.2f}")
