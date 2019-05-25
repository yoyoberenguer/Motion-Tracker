
# !/usr/bin/env python


import os.path
import pygame
from pygame import sndarray, mixer
import numpy
from numpy import zeros
import matplotlib.pyplot as plt

import timeit

pygame.mixer.pre_init(44100, -16, 2, 1024)
pygame.mixer.init()


def make_echo(array_, mixer_samples_rate_):

    echo = 4
    samples = array_.shape[0]
    # echo start e.g (31712 / 8) * 4 = 15856 start at 1/2 of the first sampling
    # freq = 44100
    # period = 1 / freq = 22.6 u seconds
    # 15856 * 22.6 u seconds = 0.359 seconds (start of the first echo)
    start = int((samples / 8) * 8)
    print(start)
    total_time = echo * array_.shape[0] + start
    print(total_time)
    # size = (array_.shape[0] + int(echo_length * array_.shape[0]), array_.shape[1])
    size = (total_time, array_.shape[1])
    echoed_array = zeros(size, numpy.int16)
    echoed_array[0:samples] = array_

    for i in range(echo):
        echoed_array[start * i: samples + start * i] += array_ >> i + 1

    return sndarray.make_sound(echoed_array.astype(numpy.int16))


if __name__ == '__main__':

    main_dir = os.path.split(os.path.abspath(__file__))[0]

    pygame.init()
    mix = pygame.mixer.get_init()
    sound = mixer.Sound(os.path.join(main_dir, '', 'pulse.ogg'))

    while mixer.get_busy():
        pygame.time.wait(200)

    sound.play()

    while mixer.get_busy():
        pygame.time.wait(200)

    array = sndarray.array(sound).astype(dtype=numpy.int16)
    sound2 = make_echo(array_=array, mixer_samples_rate_=mix[0])

    sound2.play()

    while mixer.get_busy():
        pygame.time.wait(200)

    # print(timeit.timeit('make_echo(array, mix[0])', 'from __main__ import make_echo, array, mix', number=100) / 100)
    plt.plot(array)
    plt.show()
    plt.plot(pygame.sndarray.samples(sound2))
    plt.show()



