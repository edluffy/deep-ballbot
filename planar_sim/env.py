import sys, math

import pygame

import pymunk
import pymunk.pygame_util

class BallBotEnv():
    def __init__(self):
        # Parameters
        mB = 1000
        mW = 250
        mR = 10
        mC = 500

        rB = 60
        rW = 35
        l = 300 + rW

        # Starting oordinates
        xB, yB = (400, 635)
        xW, yW = (xB, yB-rB-rW)
        xC, yC = (xB, yB-l/2)

        pygame.init()
        self.screen = pygame.display.set_mode((800, 800))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("None", 32)
        self.space = pymunk.Space()
        self.space.gravity = (0, 900)
        self.options = pymunk.pygame_util.DrawOptions(self.screen)

        self.create_floor()
        self.create_ball(mass=mB, radius=rB)
        self.create_wheel(mass=mW, radius=rW)
        self.create_rod(mass=mR, length=l)
        self.create_com(mass=mC)

        self.ball.position = (xB, yB)
        self.wheel.position = (xW, yW)
        self.rod.position = (xC, yC)
        self.com.position = (xC, yC)

        self.pivot_joint(self.rod, self.ball, (0, l/2), (0, 0), False)
        self.pivot_joint(self.rod, self.wheel, (0, l/2 - rB-rW), (0, 0), False)
        self.pivot_joint(self.rod, self.com, (0, 0), (0, 0), False)

        self.motor = pymunk.SimpleMotor(self.ball, self.wheel, 0)
        self.space.add(self.motor)
        self.motor.rate = 1

        #self.states = 
        #self.actions = 
        #self.rewards =

    def render(self, stats=True):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
        self.screen.fill(pygame.Color('white'))
        self.space.debug_draw(self.options)

        if stats:
            self.draw_coords([self.ball, self.wheel, self.com])
            self.draw_phi()
            self.draw_theta()

        self.space.step(1/120)
        pygame.display.update()
        self.clock.tick(120)

    def create_ball(self, mass, radius):
        moment = pymunk.moment_for_circle(mass, 0, radius)
        self.ball = pymunk.Body(mass, moment, body_type=pymunk.Body.DYNAMIC)

        shape = pymunk.Circle(self.ball, radius)
        shape.friction = 1
        shape.color = pygame.Color('grey')
        self.space.add(self.ball, shape)

    def create_wheel(self, mass, radius):
        moment = pymunk.moment_for_circle(mass, 0, radius)
        self.wheel = pymunk.Body(mass, moment, body_type=pymunk.Body.DYNAMIC)

        shape = pymunk.Circle(self.wheel, radius)
        shape.friction = 1
        self.space.add(self.wheel, shape)

    def create_rod(self, mass, length):
        size = (10, length)
        moment = pymunk.moment_for_box(mass, size)
        self.rod = pymunk.Body(mass, moment, body_type=pymunk.Body.DYNAMIC)

        shape = pymunk.Poly.create_box(self.rod, size)
        shape.color = pygame.Color('black')
        self.space.add(self.rod, shape)

    def create_com(self, mass):
        radius = 15
        moment = pymunk.moment_for_circle(mass, 0, radius)
        self.com = pymunk.Body(mass, moment, body_type=pymunk.Body.DYNAMIC)

        shape = pymunk.Circle(self.com, radius)
        shape.color = pygame.Color('tan')
        self.space.add(self.com, shape)

    def create_floor(self):
        self.floor = pymunk.Segment(self.space.static_body, (-100, 700), (1000, 700), 5)
        self.floor.friction = 1.0
        self.space.add(self.floor)

    def pivot_joint(self, body1, body2, anchor1=(0, 0), anchor2=(0, 0), collide=True):
        joint = pymunk.PivotJoint(body1, body2, anchor1, anchor2)
        joint.collide_bodies = collide
        self.space.add(joint)

    def draw_coords(self, bodies):
        for body in bodies:
            x, y = body.position
            self.screen.blit(
                self.font.render("({}, {})".format(round(x), round(y)), True, pygame.Color("black")),
                (x-150, y),
            )

    def draw_phi(self):
        xB, yB = self.ball.position
        angles = [0, -self.ball.angle]
        pygame.draw.arc(self.screen, pygame.Color("black"), (xB-35, yB-35, 70, 70), min(angles), max(angles), 1)
        #pygame.draw.rect(screen, pygame.Color("black"), (x-35, y-35, 70, 70), 2)
        pygame.draw.line(self.screen, pygame.Color("black"), (xB, yB), (xB+70, yB))

        self.screen.blit(
            self.font.render("φ: {}°".format(round(math.degrees(self.ball.angle))), True, pygame.Color("black")),
            (xB+80, yB-20),
        )

    def draw_theta(self):
        xC, yC = self.rod.position
        xB, yB = self.ball.position
        w = abs(xC-xB)/2

        angles = [math.pi/2, math.pi/2-self.rod.angle] # start and stop angles
        pygame.draw.arc(self.screen, pygame.Color("black"), (min(xB-w, xC+w), yB-w, w*2, w*2), min(angles), max(angles), 1)
        #pygame.draw.rect(screen, pygame.Color("black"), (min(xB-w, xC+w), yB-w, w*2, w*2), 2)
        pygame.draw.line(self.screen, pygame.Color("black"), (xB, yB), (xB, yB-300))

        self.screen.blit(
            self.font.render("θ: {}°".format(round(math.degrees(self.rod.angle))), True, pygame.Color("black")),
            (xB+20, yB-w-20),
        )

env = BallBotEnv()

while True:
    env.render()
