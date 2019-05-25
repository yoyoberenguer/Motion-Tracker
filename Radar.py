import numpy

import pygame
from pygame import *
from math import cos, sin, sqrt, atan2
import os
from pygame import freetype
import SoundServer


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
                    if FRAME % 35 == 0:
                        global MOUSE_POS
                        pulse_array = make_echo(pygame.sndarray.samples(pulse))
                        SND.play(sound_=pulse_array, loop_=False,
                                 priority_=0, volume_=min(1 - distance_, 1), fade_out_ms=0, panning_=True,
                                 name_='PULSE', x_=MOUSE_POS[0], object_id_=id(pulse))
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
    BACKGROUND = pygame.image.load(GAMEPATH +
                                   'Assets\\background1.jpg').convert()
    BACKGROUND = pygame.transform.smoothscale(BACKGROUND, SCREENRECT.size)
    FONT = freetype.Font(os.path.join(GAMEPATH + 'Assets\\', 'ARCADE_R.ttf'), size=9)
    FONT.antialiased = True
    clock = pygame.time.Clock()
    screen.blit(BACKGROUND, (0, 0))
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
    Player.images = pygame.transform.rotozoom(pygame.image.load('Assets\\Condor.png'), 180, 0.07)
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
    aircraft = reshape(aircraft, (size, size))

    missile = spread_sheet_fs8(GAMEPATH + 'Assets\\BrightDot_16x16_4x8_purple_.png', 16, 2, 8)
    missile = reshape(missile, (size, size))

    friend = spread_sheet_fs8(GAMEPATH +
                              'Assets\\BrightDot_16x16_4x8_green_.png', 16, 2, 8)
    friend = reshape(friend, (size, size))

    Radar.category = {'aircraft': aircraft, 'ground': ground, 'boss': boss, 'missile': missile, 'friend': friend}

    pulse = pygame.mixer.Sound('Assets\\pulse.ogg')
    SoundServer.SoundControl.SCREENRECT = SCREENRECT
    SND = SoundServer.SoundControl(10)

    ss = pygame.sprite.Sprite()
    ss.image = ground[0]
    ss.rect = ss.image.get_rect(center=(500, 100))
    GL.GROUP_UNION.add(ss)

    rad = Radar(location_=((SCREENRECT.w - Radar_interface[0].get_width() // 2),
                           (SCREENRECT.h - Radar_interface[0].get_height() // 2)),
                gl_=GL, player_=player, speed_=25, layer_=0)
    rad._blend = pygame.BLEND_RGB_ADD

    STOP_GAME = False

    FRAME = 0

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

            if event.type == pygame.MOUSEMOTION:
                MOUSE_POS = pygame.math.Vector2(event.pos)
                Player.mouse_pos = MOUSE_POS

        # screen.fill((0, 0, 0))
        screen.blit(BACKGROUND, (0, 0))

        pygame.draw.aaline(screen, (255, 255, 255, 255), SCREENRECT.midtop, SCREENRECT.midbottom, 1)
        pygame.draw.aaline(screen, (255, 255, 255, 255), SCREENRECT.midleft, SCREENRECT.midright, 1)
        pygame.draw.circle(screen, (255, 0, 0, 0), SCREENRECT.center, 300, 1)

        GL.All.update()
        GL.All.draw(screen)
        pygame.draw.circle(screen, (255, 255, 255, 255), SCREENRECT.center, 2)

        GL.TIME_PASSED_SECONDS = clock.tick(120)

        pygame.display.flip()
        FRAME += 1
        SND.update()

    pygame.quit()
