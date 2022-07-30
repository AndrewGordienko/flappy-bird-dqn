import pygame
import sys
import numpy as np
import matplotlib.pyplot as plt
from gui import Bird, Pipe
from brain import Agent

WIDTH = 600
HEIGHT = 400
SIZE = WIDTH, HEIGHT
BLACK = 0, 0, 0
WHITE = 255, 255, 255
TIME_DELAY = 100
EPISODES = 250

screen = pygame.display.set_mode(SIZE)

agent = Agent()
score = 0
best_score = 0
scores = []
episode_numbers = []

for i in range(EPISODES):
    bird = Bird()
    pipe = Pipe()
    print(f"episode {i} last score {score} top score {best_score} epsilon {agent.exploration_rate}")
    score = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    bird.vert_v = bird.hop

        screen.fill(BLACK)
        bird.draw()
        pipe.draw(bird)
        bird.distances(pipe)
        pygame.display.flip()
        #pygame.time.delay(TIME_DELAY) # uncomment to slow down screen

        state = np.array([bird.d1, bird.d2, bird.d3, bird.d4, bird.bird_y])
        action = agent.choose_action(state)
        old_score = bird.passed

        if action == 1:
            bird.vert_v = bird.hop
        
        screen.fill(BLACK)
        bird.draw()
        pipe.draw(bird)
        bird.distances(pipe)
        pygame.display.flip()
        #pygame.time.delay(TIME_DELAY) # uncomment to slow down screen
        
        state_ = np.array([bird.d1, bird.d2, bird.d3, bird.d4, bird.bird_y])
        
        reward = abs(bird.passed - old_score) # simpler reward policy
        #reward = min(bird.d1, bird.d2)/max(bird.d1, bird.d2) + abs(bird.passed - old_score)
        done = False

        if bird.collision(pipe) == True or bird.bird_y < 0 or bird.bird_y > HEIGHT:
            done = True
            reward = -5
        
        agent.memory.add(state, action, reward, state_, done)
        agent.learn()

        if done == True:
            score = bird.passed
            scores.append(score)
            episode_numbers.append(i)
            if score > best_score:
                best_score = score
            break
    
        
plt.plot(episode_numbers, scores)
plt.show()

        
