import pygame
import sys
import random
import math

pygame.init()

WIDTH = 600
HEIGHT = 400
SIZE = WIDTH, HEIGHT
BLACK = 0, 0, 0
WHITE = 255, 255, 255

screen = pygame.display.set_mode(SIZE)

class Bird:
    def __init__(self):
        self.bird_x = 300
        self.bird_y = HEIGHT/2
        self.dimension = 20
        self.gravity = 10
        self.hop = -40
        self.vert_v = 0
        self.passed = 0
        self.d1, self.d2, self.d3, self.d4 = None, None, None, None
        self.collide = False

    def draw(self):
        self.bird_y += self.gravity
        self.bird_y += self.vert_v

        self.bird_y += self.vert_v
    
        pygame.draw.rect(screen, WHITE, pygame.Rect(self.bird_x, self.bird_y, self.dimension, self.dimension))

        if self.vert_v == self.hop:
            self.vert_v = 0
    
    def collision(self):
        self.collide = False
        bird_object = pygame.Rect(self.bird_x, self.bird_y, self.dimension, self.dimension)

        pipe1, pipe2 = pipe.pipes[self.passed][0], pipe.pipes[self.passed][1]
        bottom_pipe = pygame.Rect(pipe.pipe_xs[self.passed], 0, 50, pipe1)
        top_pipe = pygame.Rect(pipe.pipe_xs[self.passed], HEIGHT - pipe2, 50, pipe2)

        if self.collide == False:
            if bottom_pipe.contains(bird_object) == True or top_pipe.contains(bird_object) == True:
                self.collide = True
    
    def distances(self):
        bird_mid_x = self.bird_x + self.dimension/2
        bird_mid_y = self.bird_y + self.dimension/2
        bird_midpoint = (self.bird_x + self.dimension/2, self.bird_y + self.dimension/2)

        ending_position = (pipe.pipe_xs[self.passed], pipe.pipes[self.passed][0])
        self.d1 = math.sqrt((abs(bird_mid_x - pipe.pipe_xs[self.passed]))**2 + (abs(bird_mid_y - pipe.pipes[self.passed][0]))**2)
        pygame.draw.line(screen, (255, 0, 0), bird_midpoint, ending_position)

        ending_position = (pipe.pipe_xs[self.passed], pipe.pipes[self.passed][0] + pipe.space_vert)
        self.d2 = math.sqrt((abs(bird_mid_x - pipe.pipe_xs[self.passed]))**2 + (abs(bird_mid_y - pipe.pipes[self.passed][0] + pipe.space_vert))**2)
        pygame.draw.line(screen, (255, 0, 0), bird_midpoint, ending_position)

        ending_position = (pipe.pipe_xs[self.passed-1] + pipe.pipe_width, pipe.pipes[self.passed-1][0])
        self.d3 = math.sqrt((abs(bird_mid_x - pipe.pipe_xs[self.passed-1] + pipe.pipe_width))**2 + (abs(bird_mid_y - pipe.pipes[self.passed-1][0]))**2)
        pygame.draw.line(screen, (255, 0, 0), bird_midpoint, ending_position)

        ending_position = (pipe.pipe_xs[self.passed-1] + pipe.pipe_width, pipe.pipes[self.passed-1][0] + pipe.space_vert)
        self.d4 = math.sqrt((abs(bird_mid_x - pipe.pipe_xs[self.passed-1] + pipe.pipe_width))**2 + (abs(bird_mid_y - pipe.pipes[self.passed-1][0] + pipe.space_vert))**2)
        pygame.draw.line(screen, (255, 0, 0), bird_midpoint, ending_position)

        self.d1, self.d2, self.d3, self.d4 = round(self.d1, 2), round(self.d2, 2), round(self.d3, 2), round(self.d4, 2)
        
        print(self.d1, self.d2, self.d3, self.d4)

class Pipe:
    def __init__(self):
        self.pipes = []
        self.pipe_xs = []
        self.scroll = 10
        self.space_horz = 200
        self.space_vert = 150
        self.pipe_amount = 3
        self.pipe_width = 50

        for i in range(self.pipe_amount):
            self.pipes.append(self.two_pipes())
            self.pipe_xs.append(self.space_horz + self.space_horz * i)
    
    def two_pipes(self):
        max_pipe_height = HEIGHT - self.space_vert
        standard = max_pipe_height/2

        pipe1, pipe2 = standard, standard
        
        increment = random.randint(0, 10)
        growing = random.randint(0, 1)

        if growing == 0:
            pipe1 += increment * 10
            pipe2 -= increment * 10
        else:
            pipe2 += increment * 10
            pipe1 -= increment * 10
        
        return [pipe1, pipe2]
    
    def draw(self):
        passed_pipe = False

        for i in range(len(self.pipes)):
            pipe1, pipe2 = self.pipes[i][0], self.pipes[i][1]
        
            pygame.draw.rect(screen, WHITE, pygame.Rect(self.pipe_xs[i], 0, self.pipe_width, pipe1))
            pygame.draw.rect(screen, WHITE, pygame.Rect(self.pipe_xs[i], HEIGHT - pipe2, self.pipe_width, pipe2))

            self.pipe_xs[i] -= self.scroll

            if self.pipe_xs[i] + 25 <= bird.bird_x:
                self.pipes.append(self.two_pipes())
                self.pipe_xs.append(self.space_horz + self.space_horz * (len(self.pipes)-1))
            
            else:
                if passed_pipe == False:
                    passed_pipe = True
                    bird.passed = i

bird = Bird()
pipe = Pipe()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: 
            sys.exit()
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                bird.vert_v = bird.hop
    
    screen.fill(BLACK)
    bird.draw()
    pipe.draw()
    bird.collision()
    bird.distances()
    pygame.display.flip()
    pygame.time.delay(100)
