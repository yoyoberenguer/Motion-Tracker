# encoding: utf-8
"""

                   GNU GENERAL PUBLIC LICENSE

                       Version 3, 29 June 2007


 Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>

 Everyone is permitted to copy and distribute verbatim copies

 of this license document, but changing it is not allowed.
 """

__author__ = "Yoann Berenguer"
__copyright__ = "Copyright 2007, Cobra Project"
__credits__ = ["Yoann Berenguer"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Yoann Berenguer"
__email__ = "yoyoberenguer@hotmail.com"
__status__ = "Demo"


import numpy

import pygame
from pygame import *
from math import cos, sin, sqrt, atan2
import os
from pygame import freetype
import SoundServer


class Halo(pygame.sprite.Sprite):
    """
    Create a Halo sprite
    """

    images = []
    containers = None

    def __init__(self,
                 rect_,
                 timing_,
                 layer_: int = 0
                 ):
        """
        :param rect_: pygame.Rect representing the coordinates and sprite center
        :param timing_: integer representing the sprite refreshing time in ms
        :param layer_: Layer to use, default 0
        """
        pygame.sprite.Sprite.__init__(self, self.containers)

        if isinstance(GL.All, pygame.sprite.LayeredUpdates):
            GL.All.change_layer(self, layer_)

        self.images_copy = self.images.copy()
        self.image = self.images_copy[0]
        self.center = rect_.center
        self.rect = self.image.get_rect(center=self.center)
        self._blend = False     # blend mode
        self.dt = 0             # time constant
        self.index = 0          # list index
        self.timing = timing_

    def update(self):

        if self.dt > self.timing:

            self.image = self.images_copy[self.index]
            self.rect = self.image.get_rect(center=self.center)
            if self.index < len(self.images_copy) - 1:
                self.index += 1
            else:
                self.kill()

            self.dt = 0

        self.dt += GL.TIME_PASSED_SECONDS


def make_echo(array_):

    echo = 1
    samples = array_.shape[0]
    # echo start e.g (31712 / 8) * 4 = 15856 start at 1/2 of the first sampling
    # freq = 44100
    # period = 1 / freq = 22.6 u seconds
    # 15856 * 22.6 u seconds = 0.359 seconds (start of the first echo)
    start = int((samples / 8) * 8)
    total_time = echo * array_.shape[0] + start
    # size = (array_.shape[0] + int(echo_length * array_.shape[0]), array_.shape[1])
    size = (total_time, array_.shape[1])
    echoed_array = numpy.zeros(size, numpy.int16)
    echoed_array[0:samples] = array_

    for i in range(echo):
        echoed_array[start * i: samples + start * i] += array_ >> i + 1

    return sndarray.make_sound(echoed_array.astype(numpy.int16))


def reshape(sprite_, factor_):
    """
    Reshape a surface or every surfaces from a given array
    :param sprite_: Array full of pygame surface
    :param factor_:  can be a float (width * factor_, height * factor_) or a tuple representing the surface size
    :return: return a python array (array of surface). Surfaces conserve their transparency
    mode and bit depth
    """
    assert isinstance(sprite_, (list, pygame.Surface)), 'Argument sprite_ should be a list or a pygame.Surface.'
    assert isinstance(factor_, (tuple, float, int)), \
        'Argument factor_ should be a float, int or tuple got %s ' % type(factor_)

    if isinstance(factor_, (float, int)):
        if float(factor_) == 1.0:
            return sprite_
    else:
        assert factor_ > (0, 0)

    sprite_copy = sprite_.copy()
    if isinstance(sprite_copy, list):
        i = 0
        for surface in sprite_copy:
            w, h = sprite_[i].get_size()
            sprite_copy[i] = \
                pygame.transform.smoothscale(surface,
                                             (int(w * factor_), int(h * factor_)) if isinstance(factor_, (
                                                 float, int)) else factor_)
            i += 1
    else:
        w, h = sprite_.get_size()
        sprite_copy = \
            pygame.transform.smoothscale(sprite_copy,
                                         (int(w * factor_), int(h * factor_)) if isinstance(factor_, (
                                             float, int)) else factor_)
    return sprite_copy


def spread_sheet_fs8(file: str,
                     chunk: int,
                     rows_: int,
                     columns_: int,
                     tweak_: bool = False,
                     *args) -> list:
    """ surface fs8 without per pixel alpha channel """
    assert isinstance(file, str), 'Expecting string for argument file got %s: ' % type(file)
    assert isinstance(chunk, int), 'Expecting int for argument number got %s: ' % type(chunk)
    assert isinstance(rows_, int) and isinstance(columns_, int), 'Expecting int for argument rows_ and columns_ ' \
                                                                 'got %s, %s ' % (type(rows_), type(columns_))
    image_ = pygame.image.load(file)
    array = pygame.surfarray.array3d(image_)

    animation = []
    for rows in range(rows_):
        for columns in range(columns_):
            if tweak_:
                chunkx = args[0]
                chunky = args[1]
                array1 = array[columns * chunkx:(columns + 1) * chunkx, rows * chunky:(rows + 1) * chunky, :]
            else:
                array1 = array[columns * chunk:(columns + 1) * chunk, rows * chunk:(rows + 1) * chunk, :]
            surface_ = pygame.surfarray.make_surface(array1)
            animation.append(surface_.convert())
    return animation


def spread_sheet_per_pixel(file_: str,
                           chunk: int,
                           rows_: int,
                           columns_: int) -> list:
    surface = pygame.image.load(file_)
    buffer_ = surface.get_view('2')

    w, h = surface.get_size()
    source_array = numpy.frombuffer(buffer_, dtype=numpy.uint8).reshape((h, w, 4))
    animation = []

    for rows in range(rows_):
        for columns in range(columns_):
            array1 = source_array[rows * chunk:(rows + 1) * chunk, columns * chunk:(columns + 1) * chunk, :]

            surface_ = pygame.image.frombuffer(array1.copy(order='C'),
                                               (tuple(array1.shape[:2])), 'RGBA').convert_alpha()
            animation.append(surface_.convert(32, pygame.SWSURFACE | pygame.RLEACCEL | pygame.SRCALPHA))

    return animation


def load_per_pixel(file: str) -> pygame.Surface:
    """ Not compatible with 8 bit depth color surface"""

    assert isinstance(file, str), 'Expecting path for argument <file> got %s: ' % type(file)
    try:
        surface = pygame.image.load(file)
        buffer_ = surface.get_view('2')
        w, h = surface.get_size()
        source_array = numpy.frombuffer(buffer_, dtype=numpy.uint8).reshape((w, h, 4))

        surface_ = pygame.image.frombuffer(source_array.copy(order='C'),
                                   (tuple(source_array.shape[:2])), 'RGBA').convert_alpha()
        return surface_
    except pygame.error:
        raise SystemExit('\n[-] Error : Could not load image %s %s ' % (file, pygame.get_error()))


def make_array(rgb_array_: numpy.ndarray, alpha_: numpy.ndarray) -> numpy.ndarray:
    """
    This function is used for 24-32 bit pygame surface with pixel alphas transparency layer

    make_array(RGB array, alpha array) -> RGBA array

    Return a 3D numpy (numpy.uint8) array representing (R, G, B, A)
    values of all pixels in a pygame surface.

    :param rgb_array_: 3D array that directly references the pixel values in a Surface.
                       Only work on Surfaces that have 24-bit or 32-bit formats.
    :param alpha_:     2D array that directly references the alpha values (degree of transparency) in a Surface.
                       alpha_ is created from a 32-bit Surfaces with a per-pixel alpha value.
    :return:           Return a numpy 3D array (numpy.uint8) storing a transparency value for every pixel
                       This allow the most precise transparency effects, but it is also the slowest.
                       Per pixel alphas cannot be mixed with pygame method set_colorkey (this will have
                       no effect).
    """
    return numpy.dstack((rgb_array_, alpha_))


def make_surface(rgba_array: numpy.ndarray) -> pygame.Surface:
    """
    This function is used for 24-32 bit pygame surface with pixel alphas transparency layer

    make_surface(RGBA array) -> Surface

    Argument rgba_array is a 3d numpy array like (width, height, RGBA)
    This method create a 32 bit pygame surface that combines RGB values and alpha layer.

    :param rgba_array: 3D numpy array created with the method surface.make_array.
                       Combine RGB values and alpha values.
    :return:           Return a pixels alpha surface.This surface contains a transparency value
                       for each pixels.
    """
    return pygame.image.frombuffer((rgba_array.transpose(1, 0, 2)).copy(order='C').astype(numpy.uint8),
                                   (rgba_array.shape[:2][0], rgba_array.shape[:2][1]), 'RGBA').convert_alpha()


def blend_texture(surface_, interval_, color_) -> pygame.Surface:
    """
    Compatible with 24-32 bit pixel alphas texture.
    Blend two colors together to produce a third color.
    Alpha channel of the source image will be transfer to the destination surface (no alteration
    of the alpha channel)

    :param surface_: pygame surface
    :param interval_: number of steps or intervals, int value
    :param color_: Destination color. Can be a pygame.Color or a tuple
    :return: return a pygame.surface supporting per-pixels transparency only if the surface passed
                    as an argument has been created with convert_alpha() method.
                    Pixel transparency of the source array will be unchanged.
    """

    source_array = pygame.surfarray.pixels3d(surface_)
    alpha_channel = pygame.surfarray.pixels_alpha(surface_)
    diff = (numpy.full_like(source_array.shape, color_[:3]) - source_array) * interval_
    rgba_array = numpy.dstack((numpy.add(source_array, diff), alpha_channel)).astype(dtype=numpy.uint8)
    return pygame.image.frombuffer((rgba_array.transpose(1, 0, 2)).copy(order='C').astype(numpy.uint8),
                                   (rgba_array.shape[:2][0], rgba_array.shape[:2][1]), 'RGBA').convert_alpha()


# Add transparency value to all pixels including black pixels
def add_transparency_all(rgb_array: numpy.ndarray, alpha_: numpy.ndarray, value: int) -> pygame.Surface:
    """
    Increase transparency of a surface
    This method is equivalent to pygame.Surface.set_alpha() but conserve the per-pixel properties of a texture
    All pixels will be update with a new transparency value.
    If you need to increase transparency on visible pixels only, prefer the method add_transparency instead.
    :param rgb_array:
    :param alpha_:
    :param value:
    :return:
    """
    # if not 0 <= value <= 255:
    #     raise ERROR('\n[-] invalid value for argument value, should be 0 < value <=255 got %s '
    #                 % value)
    # method 1
    """
    mask = (alpha_ >= value)
    mask_zero = (alpha_ < value)
    alpha_[:][mask_zero] = 0
    alpha_[:][mask] -= value
    return make_surface(make_array(rgb_array, alpha_.astype(numpy.uint8)))
    """
    # method 2
    alpha_ = alpha_.astype(numpy.int16)
    alpha_ -= value
    numpy.putmask(alpha_, alpha_ < 0, 0)

    return make_surface(make_array(rgb_array, alpha_.astype(numpy.uint8))).convert_alpha()


class ERROR(BaseException):
    pass


def fish_eye(surface_, point_: pygame.math.Vector2):
    """ Fish eye algorithm
        surface pygame.Surface display, width and height dimension will be used below
        point_ is a vertex position (tuple) on the screen (or pixel position)
        return the new vertex position of in the fisheye texture
    """

    w, h = surface_.w, surface_.h
    w2 = w / 2
    h2 = h / 2
    y = point_.y
    # Normalize every pixels along y axis
    ny = ((2 * y) / h) - 1
    # ny * ny pre calculated
    ny2 = ny ** 2
    x = point_.x
    # Normalize every pixels along x axis
    nx = ((2 * x) / w) - 1
    # pre calculated nx * nx
    nx2 = nx ** 2
    # calculate distance from center (0, 0)
    r = sqrt(nx2 + ny2)
    # discard pixel if r below 0.0 or above 1.0
    x2, y2 = 0, 0
    if 0.0 <= r <= 1.0:
        nr = (r + 1 - sqrt(1 - r ** 2)) / 2
        if nr <= 1.0:
            theta = atan2(ny, nx)
            nxn = nr * cos(theta)
            nyn = nr * sin(theta)
            x2 = int(nxn * w2 + w2)
            y2 = int(nyn * h2 + h2)

        return (x2, y2), r
    else:
        return (None, None), None


class Radar(pygame.sprite.Sprite):
    images = None       # Radar sprite / bitmap
    containers = None   # sprite container
    active = False      # Radar active flag
    category = None     # dict
    hostiles = []       # list of hostile aircraft coordinates (position vector)
    inventory = []

    def __init__(self,
                 location_,             # tuple representing the radar centre position
                 gl_,                   # global variable class
                 player_,               # player instance (Player or Player2 using the radar?)
                 speed_: int = 1,       # refreshing rate in ms, 16ms correspond to 60fps
                 layer_: int = -1):     # layer to use

        pygame.sprite.Sprite.__init__(self, self.containers)

        if isinstance(gl_.All, pygame.sprite.LayeredUpdates):
            gl_.All.change_layer(self, layer_)

        self.image = Radar.images[0]                        # Radar bitmap to use
        self.images_copy = Radar.images.copy()              # Radar bitmap copy (org reference)
        self.rect = self.image.get_rect(center=location_)   # extract pygame.rectangle
        self.dt = 0                                         # timing constant
        self.location = location_                           # coordinates (radar center)
        self.speed = speed_
        self.player = player_
        self.gl = gl_
        Radar.active = True

        # Radar bitmap radius and sizes
        self.width, self.height = self.image.get_size()
        self.w2, self.h2 = self.width / 2, self.height / 2
        self.radius = sqrt(self.w2 ** 2 + self.h2 ** 2)

        # SCREENRECT radius
        self.Radius = sqrt(SCREENRECT.centerx ** 2 + SCREENRECT.centery ** 2)
        # ratio
        self.ratio = self.Radius / self.radius

        self.center = SCREENRECT.center

        self.index = 0
        # list of hostile aircraft coordinates (position vector)
        self.hostiles = []
        self.rad = 100

        self.val = 0
        self.screen = (pygame.display.get_surface()).get_rect()
        self.pulsating = False              # Default radar is not pulsating
        self.pulse_time = 70                # time between pulse, 70 frames
        self.last_pulse = 0                 # time of the last pulse (frame number)
        Radar.inventory.append(self)

    @staticmethod
    def destroy():
        # Reset values and kill all the instances
        if isinstance(Radar.inventory, list) and len(Radar.inventory) > 0:
            for instance in Radar.inventory:
                if hasattr(instance, 'active'):
                    instance.active = False
                if hasattr(instance, 'hostiles'):
                    instance.hostiles = []
                if hasattr(instance, 'kill'):
                    instance.kill()
                Radar.inventory.remove(instance)
        Radar.inventory = []

    def get_hostiles(self):
        global MOUSE_POS
        # emulation
        self.hostiles = []
        self.hostiles.append([MOUSE_POS, 'aircraft', 1])

    def quit(self):
        Radar.active = False
        self.kill()

    def is_still_loading_up(self):
        """ Pulsating at regular interval """
        global FRAME
        return FRAME - self.last_pulse > self.pulse_time

    def update(self):

        if self.dt > self.speed:

            self.image = self.images[int(self.index) % 31].copy()
            self.rect = self.image.get_rect(center=self.location)

            self.get_hostiles()

            for hostile in self.hostiles:

                vector, category, distance = list(hostile)

                if category is None or \
                        category not in list(Radar.category.keys()):
                    continue

                dot_shape = Radar.category[category]
                if isinstance(dot_shape, list):
                    dot_w, dot_h = dot_shape[0].get_size()
                    hrect = dot_shape[0].get_rect()
                else:
                    dot_w, dot_h = dot_shape.get_get_size()
                    hrect = dot_shape.get_rect()

                res, distance_ = fish_eye(self.screen, vector)

                if None not in res:
                    # if not SND.get_identical_id(id(pulse)):
                    if self.is_still_loading_up():
                        Halo(self.rect, 15, 0)
                        self.pulsating = True
                        self.last_pulse = FRAME
                        global MOUSE_POS
                        pulse_array = make_echo(pygame.sndarray.samples(pulse))
                        SND.play(sound_=pulse_array, loop_=False,
                                 priority_=0, volume_=min(1 - distance_, 1), fade_out_ms=0, panning_=True,
                                 name_='PULSE', x_=MOUSE_POS[0], object_id_=id(pulse))
                    else:
                        self.pulsating = False

                    hrect.center = pygame.math.Vector2(res) / self.ratio
                    hrect.center -= pygame.math.Vector2(dot_w, dot_h) / 2
                    self.image.blit(dot_shape[self.val % len(dot_shape) - 1] if isinstance(dot_shape, list) else
                                    dot_shape, hrect.center, special_flags=pygame.BLEND_RGB_ADD)

            self.index += 1.2

            self.dt = 0
            self.val += 1

        self.dt += self.gl.TIME_PASSED_SECONDS


class LayeredUpdatesModified(pygame.sprite.LayeredUpdates):

    def __init__(self):
        pygame.sprite.LayeredUpdates.__init__(self)

    def draw(self, surface_):
        """draw all sprites in the right order onto the passed surface

        LayeredUpdates.draw(surface): return Rect_list

        """
        spritedict = self.spritedict
        surface_blit = surface_.blit
        dirty = self.lostsprites
        self.lostsprites = []
        dirty_append = dirty.append
        init_rect = self._init_rect
        for spr in self.sprites():
            rec = spritedict[spr]

            if hasattr(spr, '_blend') and spr._blend is not None:
                newrect = surface_blit(spr.image, spr.rect, special_flags=spr._blend)
            else:
                newrect = surface_blit(spr.image, spr.rect)

            if rec is init_rect:
                dirty_append(newrect)
            else:
                if newrect.colliderect(rec):
                    dirty_append(newrect.union(rec))
                else:
                    dirty_append(newrect)
                    dirty_append(rec)
            spritedict[spr] = newrect
        return dirty


if __name__ == '__main__':

    class GL:
        All = None
        TIME_PASSED_SECONDS = None
        GROUP_UNION = None


    GAMEPATH = ''

    pygame.init()
    pygame.mixer.init()
    freetype.init(cache_size=64, resolution=72)
    SCREENRECT = pygame.Rect(0, 0, 600, 600)
    screen = pygame.display.set_mode(SCREENRECT.size, pygame.HWSURFACE, 32)
    BACKGROUND = []
    BACKGROUND1 = pygame.image.load(GAMEPATH +
                                    'Assets\\Alien_isolation1.jpg').convert()
    BACKGROUND.append(pygame.transform.smoothscale(BACKGROUND1, SCREENRECT.size))
    BACKGROUND2 = pygame.image.load(GAMEPATH +
                                    'Assets\\Alien_isolation2.jpg').convert()
    BACKGROUND.append(pygame.transform.smoothscale(BACKGROUND2, SCREENRECT.size))
    BACKGROUND3 = pygame.image.load(GAMEPATH +
                                    'Assets\\Alien_isolation3.jpg').convert()
    BACKGROUND.append(pygame.transform.smoothscale(BACKGROUND3, SCREENRECT.size))

    FONT = freetype.Font(os.path.join(GAMEPATH + 'Assets\\', 'ARCADE_R.ttf'), size=9)
    FONT.antialiased = True
    clock = pygame.time.Clock()
    screen.blit(BACKGROUND1, (0, 0))
    sprite_group = pygame.sprite.Group()
    GL.All = LayeredUpdatesModified()
    GL.GROUP_UNION = pygame.sprite.Group()


    class Player(pygame.sprite.Sprite):
        images = None
        mouse_pos = None

        def __init__(self):
            pygame.sprite.Sprite.__init__(self, GL.All)
            self.image = Player.images
            self.image_copy = self.image.copy()
            self.rect = self.images.get_rect(center=self.mouse_pos)

        def update(self):
            global screen
            self.image = self.image_copy.copy()
            self.rect = self.image.get_rect(center=self.mouse_pos)
            pygame.draw.line(screen, (255, 255, 255, 255), self.rect.midleft, self.rect.midright, 1)
            pygame.draw.line(screen, (255, 255, 255, 255), self.rect.midtop, self.rect.midbottom, 1)
            pygame.draw.rect(screen, (255, 255, 255, 255), self.rect, 1)


    MOUSE_POS = pygame.math.Vector2()
    Player.images = pygame.transform.rotozoom(pygame.image.load('Assets\\alien.png'), 0, 0.5)
    Player.containers = GL.All
    Player.mouse_pos = MOUSE_POS
    player = Player()
    GL.TIME_PASSED_SECONDS = 0

    Radar_interface = spread_sheet_fs8(
        GAMEPATH + 'Assets\\radar_animated1_.png', 180, 6, 6)
    Radar_interface = reshape(Radar_interface, (300, 300))

    Radar.containers = GL.All
    Radar.images = Radar_interface
    size = 18
    ground = spread_sheet_fs8(GAMEPATH +
                              'Assets\\RedSquare32x32_4x8_.png', 32, 4, 8)
    ground = reshape(ground, (size - 4, size - 4))
    boss = spread_sheet_fs8(GAMEPATH +
                            'Assets\\BrightDot_16x16_4x8_.png', 16, 2, 8)
    boss = reshape(boss, (size, size))

    aircraft = spread_sheet_fs8(GAMEPATH +
                                'Assets\\BrightDot_16x16_4x8_red_.png', 16, 2, 8)
    aircraft = reshape(aircraft, (size + 20, size + 20))

    missile = spread_sheet_fs8(GAMEPATH + 'Assets\\BrightDot_16x16_4x8_purple_.png', 16, 2, 8)
    missile = reshape(missile, (size, size))

    friend = spread_sheet_fs8(GAMEPATH +
                              'Assets\\BrightDot_16x16_4x8_green_.png', 16, 2, 8)
    friend = reshape(friend, (size, size))

    Radar.category = {'aircraft': aircraft, 'ground': ground, 'boss': boss, 'missile': missile, 'friend': friend}

    # WHITE HALO
    HALO_SPRITE = []
    HALO_SPRITE_ = load_per_pixel('Assets\\WhiteHalo.png')
    # HALO_SPRITE_ = pygame.image.load('Assets\\WhiteHalo_.png').convert()
    steps = numpy.array([0., 0.03333333, 0.06666667, 0.1, 0.13333333,
                         0.16666667, 0.2, 0.23333333, 0.26666667, 0.3,
                         0.33333333, 0.36666667, 0.4, 0.43333333, 0.46666667,
                         0.5, 0.53333333, 0.56666667, 0.6, 0.63333333,
                         0.66666667, 0.7, 0.73333333, 0.76666667, 0.8,
                         0.83333333, 0.86666667, 0.9, 0.93333333, 0.96666667])

    for number in range(30):
        # Blend red
        surface = blend_texture(HALO_SPRITE_, steps[number], pygame.Color(128, 255, 255, 255))
        rgb = pygame.surfarray.pixels3d(surface)
        alpha = pygame.surfarray.array_alpha(surface)
        surface = add_transparency_all(rgb, alpha, int(255 * steps[number] / 8))
        size = pygame.math.Vector2(surface.get_size())
        size *= (number / 10)
        surface1 = pygame.transform.smoothscale(surface, (int(size.x), int(size.y)))
        HALO_SPRITE.append(surface1)

    Halo.containers = GL.All
    Halo.images = HALO_SPRITE

    pulse = pygame.mixer.Sound('Assets\\pulse.ogg')
    SoundServer.SoundControl.SCREENRECT = SCREENRECT
    SND = SoundServer.SoundControl(10)

    ss = pygame.sprite.Sprite()
    ss.image = ground[0]
    ss.rect = ss.image.get_rect(center=(500, 100))
    GL.GROUP_UNION.add(ss)

    rad = Radar(location_=((SCREENRECT.w - Radar_interface[0].get_width() // 2),
                           (SCREENRECT.h - Radar_interface[0].get_height() // 2)),
                gl_=GL, player_=player, speed_=16, layer_=0)
    rad._blend = pygame.BLEND_RGB_ADD

    STOP_GAME = False

    FRAME = 0
    index = 0
    while not STOP_GAME:

        for event in pygame.event.get():
            keys = pygame.key.get_pressed()

            if event.type == QUIT:
                print('Quitting')
                STOP_GAME = True

            if keys[K_SPACE]:
                pass

            if keys[K_ESCAPE]:
                STOP_GAME = True

            if keys[K_F8]:
                pygame.image.save(screen, 'screenshot' + str(FRAME) + '.png')

            if event.type == pygame.MOUSEMOTION:
                MOUSE_POS = pygame.math.Vector2(event.pos)
                Player.mouse_pos = MOUSE_POS

        # screen.fill((0, 0, 0))
        screen.blit(BACKGROUND[index], (0, 0))
        if FRAME % 800 == 0:
            if index < len(BACKGROUND) - 1:
                index += 1
            else:
                index = 0

        pygame.draw.aaline(screen, (255, 255, 255, 255), SCREENRECT.midtop, SCREENRECT.midbottom, 1)
        pygame.draw.aaline(screen, (255, 255, 255, 255), SCREENRECT.midleft, SCREENRECT.midright, 1)
        pygame.draw.circle(screen, (255, 255, 255, 0), SCREENRECT.center, 300, 1)

        GL.All.update()
        GL.All.draw(screen)


        GL.TIME_PASSED_SECONDS = clock.tick(120)

        pygame.display.flip()
        FRAME += 1
        SND.update()

    pygame.quit()
