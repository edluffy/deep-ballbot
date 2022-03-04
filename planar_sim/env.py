import sys, math

import numpy as np
import pygame

import pymunk
import pymunk.pygame_util

class BallBotEnv():
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 800))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("None", 32)
        self.options = pymunk.pygame_util.DrawOptions(self.screen)
        self.episode_count = 0

    def reset(self):
        self.episode_count += 1
        self.space = pymunk.Space()
        self.space.gravity = (0, 900)

        # Parameters
        mB = 1000
        mW = 250
        mR = 10
        mC = 500

        rB = 60
        rW = 35
        l = 300 + rW

        # Starting coordinates
        xB, yB = (400, 635)
        xW, yW = (xB, yB-rB-rW)
        xC, yC = (xB, yB-l/2)

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
        self.gear_joint(self.ball, self.wheel, 0, -rW/rB)

        self.state = [self.ball.angle, self.ball.angular_velocity,
                self.rod.angle, self.rod.angular_velocity]
        self.action = 0 # angular velocity of the motor
        self.reward = 0 # +1 reward per time step
        self.done = False

        #self.motor.max_force = 10e10
        self.wheel.angular_velocity = -1
        self.space.step(1/120)

        return self.state

    def render(self, stats=False):
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
            self.draw_mdp()

        pygame.display.update()

    def step(self, action):
        self.action = action

        #self.wheel.apply_impulse_at_local_point((action*1000, 0), (0, 35))
        #self.wheel.apply_impulse_at_local_point((-action*1000, 0), (0, 0))

        print(action)
        #action = -action*5*10e6#*-4.9*10e7
        action = -action*5.15*10e6
        self.wheel.torque = np.clip(action, -10e10, 10e10)

        self.space.step(1/120)
        self.clock.tick(120)

        self.state = [self.ball.angle, self.rod.angle,
                self.ball.angular_velocity, self.rod.angular_velocity]
        self.done = self.done or abs(self.rod.angle) >= math.pi / 6

        self.reward += math.cos(self.rod.angle)
        return self.state, math.cos(self.rod.angle), self.done

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
        self.floor.friction = 1
        self.space.add(self.floor)

    def pivot_joint(self, body1, body2, anchor1=(0, 0), anchor2=(0, 0), collide=True):
        joint = pymunk.PivotJoint(body1, body2, anchor1, anchor2)
        joint.collide_bodies = collide
        self.space.add(joint)

    def gear_joint(self, body1, body2, phase, ratio, collide=True):
        joint = pymunk.GearJoint(body1, body2, phase, ratio)
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

    def draw_mdp(self):
        x, y = (0, 0)
        self.screen.blit(self.font.render("state: [", True, pygame.Color("black")), (0, y))

        for s in self.state:
            x += 75
            self.screen.blit(self.font.render("{:<5.3f}".format(s), True, pygame.Color("black")), (x, y))

        x += 75
        self.screen.blit(self.font.render("]", True, pygame.Color("black")), (x, y))

        y += 25
        self.screen.blit(self.font.render("action: {:.3f}".format(self.action), True, pygame.Color("black")), (0, y))

        y += 25
        self.screen.blit(self.font.render("reward: {}".format(self.reward), True, pygame.Color("black")), (0, y))

        y += 25
        self.screen.blit(self.font.render("episode: {}".format(self.episode_count), True, pygame.Color("black")), (0, y))

