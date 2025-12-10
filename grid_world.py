import numpy as np
import cv2
import random

# Constants
GRID_SIZE = 20
OBSTACLE_RATIO = 0.2
START = (0, 0)
GOAL = (19, 19)
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
ACTION_NAMES = ['Up', 'Down', 'Left', 'Right']

# Q-Learning parameters
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
EPISODES = 1000

# Visualization
CELL_SIZE = 30
VIDEO_FILENAME = 'grid_learning.mp4'
FPS = 30

class GridWorld:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.obstacles = set()
        self.generate_grid()
        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
        
    def generate_grid(self):
        num_obstacles = int(GRID_SIZE * GRID_SIZE * OBSTACLE_RATIO)
        while len(self.obstacles) < num_obstacles:
            r = random.randint(0, GRID_SIZE - 1)
            c = random.randint(0, GRID_SIZE - 1)
            if (r, c) != START and (r, c) != GOAL:
                self.obstacles.add((r, c))
                self.grid[r, c] = 1 # Obstacle

    def reset(self):
        return START

    def step(self, state, action_idx):
        r, c = state
        dr, dc = ACTIONS[action_idx]
        next_r, next_c = r + dr, c + dc

        # Check boundaries and obstacles
        if 0 <= next_r < GRID_SIZE and 0 <= next_c < GRID_SIZE:
            if (next_r, next_c) not in self.obstacles:
                next_state = (next_r, next_c)
            else:
                next_state = state # Hit obstacle, stay put
        else:
            next_state = state # Hit wall, stay put

        # Rewards
        if next_state == GOAL:
            reward = 100
            done = True
        else:
            reward = -1
            done = False
            
        return next_state, reward, done

    def choose_action(self, state):
        if random.random() < EPSILON:
            return random.randint(0, len(ACTIONS) - 1)
        else:
            return np.argmax(self.q_table[state])

    def train(self):
        global EPSILON
        
        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Frame size: Grid + some padding/text area
        width = GRID_SIZE * CELL_SIZE
        height = GRID_SIZE * CELL_SIZE + 50
        out = cv2.VideoWriter(VIDEO_FILENAME, fourcc, FPS, (width, height))
        
        rewards_history = []
        
        print("Starting training...")
        
        for episode in range(EPISODES):
            state = self.reset()
            done = False
            total_reward = 0
            
            # For visualization, we might not want to save EVERY frame of EVERY episode if it takes too long.
            # But the user asked for "entire learning process". 
            # To keep file size manageable, let's speed up early episodes or skip frames? 
            # Or just record everything but fast? User said "viewer can see how policy improves".
            # 1000 episodes * ~30 steps = 30000 frames @ 30fps = 1000 seconds = 16 mins. A bit long.
            # Let's record: 
            # - First 10 episodes fully
            # - Every 10th episode thereafter
            # - Last 10 episodes fully
            
            should_record = (episode % 10 == 0) or (episode > EPISODES - 2)
            
            steps = 0
            path = []
            
            while not done and steps < 200: # Limit steps to prevent infinite loops early on
                path.append(state)
                if should_record:
                    frame = self.draw_frame(state, episode, steps)
                    out.write(frame)
                
                action_idx = self.choose_action(state)
                next_state, reward, done = self.step(state, action_idx)
                
                # Q-Learning Update
                # Q(s,a) = Q(s,a) + alpha * (R + gamma * max(Q(s',a')) - Q(s,a))
                old_q = self.q_table[state][action_idx]
                next_max = np.max(self.q_table[next_state])
                
                new_q = old_q + ALPHA * (reward + GAMMA * next_max - old_q)
                self.q_table[state][action_idx] = new_q
                
                state = next_state
                total_reward += reward
                steps += 1
            
            rewards_history.append(total_reward)
            
            # Decay Epsilon
            if EPSILON > EPSILON_MIN:
                EPSILON *= EPSILON_DECAY
            
            if episode % 100 == 0:
                print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {EPSILON:.2f}")

        # Show final optimal path
        print("Showing optimal path...")
        state = self.reset()
        done = False
        steps = 0
        while not done and steps < 100:
            frame = self.draw_frame(state, "FINAL OPTIMAL", steps, path=path)
            for _ in range(3): # Slow down final path by writing 3 frames per step
                out.write(frame)
            
            action_idx = np.argmax(self.q_table[state]) # Greedy
            path.append(state) # Add current state to path for tracing
            state, r, done = self.step(state, action_idx)
            steps += 1
            
        # Draw the final step
        path.append(state)
        frame = self.draw_frame(state, "FINAL OPTIMAL", steps, path=path)
        for _ in range(30): # Hold final result
            out.write(frame)

        out.release()
        print(f"Video saved to {VIDEO_FILENAME}")

    def draw_frame(self, agent_pos, episode_num, step_num, path=None):
        # Create image
        img = np.zeros((GRID_SIZE * CELL_SIZE + 50, GRID_SIZE * CELL_SIZE, 3), dtype=np.uint8)
        img.fill(255) # White background
        
        # Draw grid
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                x1 = c * CELL_SIZE
                y1 = r * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                
                color = (255, 255, 255) # White
                if (r, c) in self.obstacles:
                    color = (0, 0, 0) # Black obstacle
                elif (r, c) == START:
                    color = (0, 255, 0) # Green Start
                elif (r, c) == GOAL:
                    color = (255, 0, 0) # Blue Goal (BGR)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1) # Gray border
        
        # Draw Path Trail if provided
        if path:
            for i in range(len(path) - 1):
                p1 = path[i]
                p2 = path[i+1]
                pt1 = (int(p1[1] * CELL_SIZE + CELL_SIZE / 2), int(p1[0] * CELL_SIZE + CELL_SIZE / 2))
                pt2 = (int(p2[1] * CELL_SIZE + CELL_SIZE / 2), int(p2[0] * CELL_SIZE + CELL_SIZE / 2))
                cv2.line(img, pt1, pt2, (0, 0, 255), 2) # Red line        
        # Draw Agent
        ar, ac = agent_pos
        ax = int(ac * CELL_SIZE + CELL_SIZE / 2)
        ay = int(ar * CELL_SIZE + CELL_SIZE / 2)
        cv2.circle(img, (ax, ay), int(CELL_SIZE/3), (0, 255, 255), -1) # Yellow Agent
        
        # Text Info
        cv2.putText(img, f"Ep: {episode_num} Step: {step_num}", (10, GRID_SIZE * CELL_SIZE + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return img

if __name__ == "__main__":
    env = GridWorld()
    env.train()
