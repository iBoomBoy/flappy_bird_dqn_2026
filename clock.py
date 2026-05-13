"""
Author: Dr Zhibin Liao
Organisation: School of Computer Science and Information Technology, Adelaide University
Date: 12-Mar-2026
Description: This Python script is a wrapper of the game clock. This makes the game runnable without showing a screen.

The script is a part of Assignment 2 made for the course COMP 3027 Artificial Intelligence for the year
of 2026. Public distribution of this source code is strictly forbidden.
"""
import pygame


class ClockWrapper:
    def __init__(self, show_screen=False, frame_rate=30):
        self.show_screen = show_screen
        self.frame_rate = frame_rate

        if self.show_screen:
            self.clock = pygame.time.Clock()
        else:
            self.clock_counter = 0

    def current_time(self):
        if self.show_screen:
            return pygame.time.get_ticks()
        else:
            return self.clock_counter

    def tick(self):
        if self.show_screen:
            self.clock.tick(self.frame_rate)  # frame rate
        else:
            self.clock_counter += 1000 / self.frame_rate