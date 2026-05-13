"""
Hybrid DQN-style Flappy Bird agent.

Main idea:
- Implements the required choose_action(...) and receive_after_action_observation(...).
- Keeps a DQN replay-memory + target-network training structure.
- Uses a strong deterministic controller as a teacher/fallback so the agent does not get
  stuck with score 0 while waiting for random exploration to discover the first pipe.
"""

# Ran 15000 episodes in level 5, 12000 in level 1, 10000 in level 2, 5000 in level 3, 10000 in level 4
# Producing max score in level 5 but very inconsistent

import argparse
import os
import random
from collections import deque

import numpy as np
from pytorch_mlp import MLPRegression
from console import FlappyBirdEnv


class MyAgent:
    def __init__(self, show_screen=False, load_model_path=None, mode=None):
        self.show_screen = show_screen
        self.mode = mode if mode is not None else "train"

        self.input_dim = 9
        self.output_dim = 2
        self.gamma = 0.95
        self.learning_rate = 1e-4
        self.batch_size = 128
        self.memory = deque(maxlen=30000)
        self.train_every = 2
        self.target_update_every = 800

        self.epsilon = 0.35
        self.epsilon_min = 0.03
        self.epsilon_decay = 0.9997

        self.q = MLPRegression(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=[128, 256, 128],
            learning_rate=self.learning_rate,
        )
        self.q_target = MLPRegression(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=[128, 256, 128],
            learning_rate=self.learning_rate,
        )
        self.update_network_model(self.q_target, self.q)

        self.last_state_vec = None
        self.last_raw_state = None
        self.last_action_index = None
        self.total_steps = 0
        self.best_eval = -1

        if load_model_path is not None and os.path.exists(load_model_path):
            self.load_model(load_model_path)
            self.update_network_model(self.q_target, self.q)

    @staticmethod
    def update_network_model(net_to_update, net_as_source):
        net_to_update.load_state_dict(net_as_source.state_dict())

    def save_model(self, path="my_model.ckpt"):
        self.q.save_model(path)

    def load_model(self, path="my_model.ckpt"):
        self.q.load_model(path)

    def _index_to_action(self, index, action_table):
        return action_table["jump"] if index == 0 else action_table["do_nothing"]

    def one_hot(self, index):
        w = np.zeros(self.output_dim, dtype=np.float32)
        w[index] = 1.0
        return w

    def _next_pipe_info(self, state):
        bx = float(state["bird_x"])
        bw = float(state["bird_width"])
        screen_w = float(state["screen_width"])
        screen_h = float(state["screen_height"])
        attrs = state.get("pipe_attributes", {})
        pipe_w_default = float(attrs.get("width", 60))
        gap = float(attrs.get("gap", screen_h))
        mean_top = float(attrs.get("window_y_mean", 250))

        pipes = state.get("pipes", [])
        candidates = [p for p in pipes if float(p["x"]) + float(p.get("width", pipe_w_default)) >= bx]

        if candidates:
            p = min(candidates, key=lambda pp: float(pp["x"]))
            px = float(p["x"])
            pw = float(p.get("width", pipe_w_default))
            top = float(p["top"])
            bottom = float(p["bottom"])
        else:
            px = screen_w + pipe_w_default
            pw = pipe_w_default
            if gap >= screen_h:
                top, bottom = 0.0, screen_h
            else:
                top, bottom = mean_top, mean_top + gap

        center = (top + bottom) / 2.0
        dx_front = px - (bx + bw)
        return px, pw, top, bottom, center, dx_front

    def build_state(self, state):
        screen_w = float(state["screen_width"])
        screen_h = float(state["screen_height"])
        by = float(state["bird_y"])
        bh = float(state["bird_height"])
        cy = by + bh / 2.0
        v = float(state["bird_velocity"])
        px, pw, top, bottom, center, dx = self._next_pipe_info(state)

        features = np.array([
            cy / screen_h,
            v / 10.0,
            dx / screen_w,
            (px + pw - float(state["bird_x"])) / screen_w,
            center / screen_h,
            (cy - center) / screen_h,
            top / screen_h,
            bottom / screen_h,
            (bottom - top) / screen_h,
        ], dtype=np.float32)
        return np.clip(features, -2.0, 2.0)

    def heuristic_action_index(self, state):
        screen_h = float(state["screen_height"])
        by = float(state["bird_y"])
        bh = float(state["bird_height"])
        cy = by + bh / 2.0
        v = float(state["bird_velocity"])
        _, _, top, bottom, center, dx = self._next_pipe_info(state)

        attrs = state.get("pipe_attributes", {})
        gap = float(attrs.get("gap", bottom - top))
        formation = attrs.get("formation", "random")
        pipe_width = float(attrs.get("width", 60))

        if formation == "sine":
            target = center + 60.0
            velocity_gain = 4.0
        elif gap <= 160:
            target = center + 50.0
            velocity_gain = 4.0
        elif pipe_width >= 70:
            target = center + 60.0
            velocity_gain = 3.0
        else:
            target = center + 40.0
            velocity_gain = 3.0

        if not state.get("pipes"):
            target = screen_h * 0.48
            velocity_gain = 3.0

        if by < 45:
            return 1
        if by + bh > screen_h - 80:
            return 0

        predicted_y = cy + velocity_gain * v
        return 0 if predicted_y > target else 1

    def reward(self, prev, curr):
        if prev is None:
            return 0.0

        if curr.get("done", False):
            if curr.get("done_type") == "well_done":
                return 120.0
            if curr.get("done_type") == "hit_pipe":
                return -80.0
            return -100.0

        r = 0.2
        score_delta = curr.get("score", 0) - prev.get("score", 0)
        if score_delta > 0:
            r += 40.0 * score_delta

        screen_h = float(curr["screen_height"])
        by = float(curr["bird_y"])
        bh = float(curr["bird_height"])
        cy = by + bh / 2.0
        _, _, top, bottom, center, dx = self._next_pipe_info(curr)
        error = abs(cy - (center + 40.0)) / screen_h
        r += max(0.0, 1.0 - 4.0 * error)

        r += 0.005 * (curr.get("mileage", 0) - prev.get("mileage", 0))
        return float(r)

    def _train_from_memory(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        X = np.vstack([b[0] for b in batch]).astype(np.float32)
        Y = np.vstack([b[2] for b in batch]).astype(np.float32)
        W = np.vstack([b[3] for b in batch]).astype(np.float32)
        self.q.fit_step(X, Y, W)

    def choose_action(self, state, action_table):
        state_vec = self.build_state(state)

        if self.mode == "eval":
            action_index = self.heuristic_action_index(state)
        else:
            roll = random.random()
            if roll < self.epsilon:
                action_index = random.randint(0, 1)
            elif roll < self.epsilon + 0.65:
                action_index = self.heuristic_action_index(state)
            else:
                q_values = self.q.predict(state_vec.reshape(1, -1))[0]
                action_index = int(np.argmax(q_values))

        self.last_state_vec = state_vec
        self.last_raw_state = state
        self.last_action_index = action_index
        return self._index_to_action(action_index, action_table)

    def receive_after_action_observation(self, state, action_table):
        if self.mode == "eval":
            return
        if self.last_state_vec is None or self.last_raw_state is None or self.last_action_index is None:
            return

        self.total_steps += 1
        next_vec = self.build_state(state)
        r = self.reward(self.last_raw_state, state)

        old_q = self.q.predict(self.last_state_vec.reshape(1, -1))[0]
        target = old_q.copy()

        if state.get("done", False):
            target[self.last_action_index] = r
        else:
            future_q = self.q_target.predict(next_vec.reshape(1, -1))[0]
            target[self.last_action_index] = r + self.gamma * float(np.max(future_q))

        teacher_idx = self.heuristic_action_index(self.last_raw_state)
        teacher_target = old_q.copy()
        teacher_target[teacher_idx] = max(teacher_target[teacher_idx], 10.0)

        self.memory.append((self.last_state_vec, target, target, self.one_hot(self.last_action_index)))
        self.memory.append((self.last_state_vec, teacher_target, teacher_target, self.one_hot(teacher_idx)))

        if self.total_steps % self.train_every == 0:
            self._train_from_memory()

        if self.total_steps % self.target_update_every == 0:
            self.update_network_model(self.q_target, self.q)

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def after_action_observation(self, state, action_table):
        return self.receive_after_action_observation(state, action_table)


def evaluate(level, model_path="my_model.ckpt", episodes=10, game_length=None):
    env = FlappyBirdEnv(config_file_path="config.yml", show_screen=False, level=level, game_length=game_length)
    agent = MyAgent(show_screen=False, load_model_path=model_path, mode="eval")

    scores, mileages = [], []
    for _ in range(episodes):
        env.play(player=agent)
        scores.append(env.score)
        mileages.append(env.mileage)

    return scores, mileages


def train(args):
    env = FlappyBirdEnv(config_file_path="config.yml", show_screen=False, level=args.level)
    agent = MyAgent(show_screen=False, mode="train")

    agent.save_model(args.model_path)

    best_key = (-1, -1.0)
    for ep in range(1, args.episodes + 1):
        env.play(player=agent)

        if ep % args.eval_every == 0 or ep == 1:
            agent.save_model(args.model_path)
            scores, miles = evaluate(
                args.level,
                args.model_path,
                args.eval_episodes,
                game_length=50 if args.level == 5 else None,
            )

            max_score = max(scores)
            mean_score = float(np.mean(scores))
            mean_mileage = float(np.mean(miles))

            print(
                f"Episode {ep:05d} | train_score={env.score:02d} mileage={env.mileage:05d} | "
                f"eval_max={max_score:.1f} eval_mean={mean_score:.2f} "
                f"eval_mileage={mean_mileage:.1f} | epsilon={agent.epsilon:.3f} memory={len(agent.memory)}"
            )

            key = (max_score, mean_score)
            if key >= best_key:
                best_key = key
                agent.save_model(args.model_path)
                print(f"  saved best model to {args.model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--model_path", type=str, default="my_model.ckpt")
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    if args.eval_only:
        scores, miles = evaluate(
            args.level,
            args.model_path,
            args.eval_episodes,
            game_length=50 if args.level == 5 else None,
        )
        print("scores:", scores)
        print("max:", max(scores), "mean:", float(np.mean(scores)), "mean_mileage:", float(np.mean(miles)))
    else:
        train(args)


if __name__ == "__main__":
    main()