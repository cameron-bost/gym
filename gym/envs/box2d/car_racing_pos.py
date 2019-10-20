import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.envs.box2d.car_dynamics import Car
from gym.utils import colorize, seeding, EzPickle

import pyglet
from pyglet import gl

import random

# Easiest continuous control task to learn from pixels, a top-down racing environment.
# Discrete control is reasonable in this environment as well, on/off discretization is
# fine.
#
# State consists of STATE_W x STATE_H pixels.
#
# Reward is -0.1 every frame and +1000/N for every track tile visited, where N is
# the total number of tiles visited in the track. For example, if you have finished in 732 frames,
# your reward is 1000 - 0.1*732 = 926.8 points.
#
# Game is solved when agent consistently gets 900+ points. Track generated is random every episode.
#
# Episode finishes when all tiles are visited. Car also can go outside of PLAYFIELD, that
# is far off the track, then it will get -100 and die.
#
# Some indicators shown at the bottom of the window and the state RGB buffer. From
# left to right: true speed, four ABS sensors, steering wheel position and gyroscope.
#
# To play yourself (it's rather fast for humans), type:
#
# python gym/envs/box2d/car_racing.py
#
# Remember it's powerful rear-wheel drive car, don't press accelerator and turn at the
# same time.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

RAY_CAST_DISTANCE = 20
NUM_SENSORS = 5
RAY_CAST_INTERVALS = 5

DISTANCE_INTERVALS = 6
SPEED_INTERVALS = 10  # Number of intervals to discretize speed state into
MAX_SPEED = 100.
MIN_SPEED = 0.

STEER_INTERVALS = 3
STEER_MAX = 0.4
STEER_MIN = -0.4

STEER_ACTION = {0: 0.0, 1: -1.0, 2: 1.0}
GAS_ACTION = {0: 0.0, 1: 1.0}
BRAKE_ACTION = {0: 0.0, 1: 0.8}  # set 1.0 for wheels to block to zero rotation

STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 2.7  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)

TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.4, 0.4, 0.4]


class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        tile.color[0] = ROAD_COLOR[0]
        tile.color[1] = ROAD_COLOR[1]
        tile.color[2] = ROAD_COLOR[2]
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            # print tile.road_friction, "ADD", len(obj.tiles)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 1
                self.env.tile_visited_count += 1
        else:
            obj.tiles.remove(tile)
            # print tile.road_friction, "DEL", len(obj.tiles) -- should delete to zero when on grass (this works)


class CarRacingPoS(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second': FPS
    }

    def __init__(self, verbose=0):
        EzPickle.__init__(self)
        self.seed()
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=
                               [(0, 0), (1, 0), (1, -1), (0, -1)]))
        self.slowness = 0

        """
        Action Space:
        1) Steer: Discrete 3  - NOOP[0], Left[1], Right[2] - params: min: 0, max: 2
        2) Gas: Discrete 2 - NOOP[0], Go[1] - params: min: 0, max: 1
        3) Brake: Discrete 2  - NOOP[0], Brake[1] - params: min: 0, max: 1

        Observation Space:
        1) Speed: SPEED_INTERVALS + 1 discrete speeds
        2) Sensors: RAY_CAST_INTERVALS * NUM_SENSORS
        3) Wheel off or not ( for each wheel): 2
        4) Steering: STEER_INTERVALS

        """

        self.action_space = spaces.MultiDiscrete([3, 2, 2])
        self.observation_space = spaces.MultiDiscrete(
            # [DISTANCE_INTERVALS + 1,
            #  DISTANCE_INTERVALS + 1,
            [SPEED_INTERVALS + 1]
            + [RAY_CAST_INTERVALS] * NUM_SENSORS
            # + [2, 2]
            + [STEER_INTERVALS])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        self.car.destroy()

    def _create_track(self):
        CHECKPOINTS = 12

        # direction = -1 for right turns, 1 for left turns
        direction = self.track_direction

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            alpha = 2 * math.pi * c / CHECKPOINTS + self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)
            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD
            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))

        # print("\n".join(str(h) for h in checkpoints))
        # self.road_poly = [ (    # uncomment this to see checkpoints
        #    [ (tx,ty) for a,tx,ty in checkpoints ],
        #    (0.7,0.7,0.9) ) ]
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi
            while True:  # Find destination from checkpoints
                failed = True
                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break
                if not failed:
                    break
                alpha -= 2 * math.pi
                continue
            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            proj = r1x * dest_dx + r1y * dest_dy  # destination vector projected on rad
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, direction * x, direction * y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break
        # print "\n".join([str(t) for t in enumerate(track)])

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose == 1:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1:i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2])) +
            np.square(first_perp_y * (track[0][3] - track[-1][3])))
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (x1 - TRACK_WIDTH * math.cos(beta1), y1 - TRACK_WIDTH * math.sin(beta1))
            road1_r = (x1 + TRACK_WIDTH * math.cos(beta1), y1 + TRACK_WIDTH * math.sin(beta1))
            road2_l = (x2 - TRACK_WIDTH * math.cos(beta2), y2 - TRACK_WIDTH * math.sin(beta2))
            road2_r = (x2 + TRACK_WIDTH * math.cos(beta2), y2 + TRACK_WIDTH * math.sin(beta2))
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3)
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.road_visited = False
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (x1 + side * TRACK_WIDTH * math.cos(beta1), y1 + side * TRACK_WIDTH * math.sin(beta1))
                b1_r = (x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                        y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1))
                b2_l = (x2 + side * TRACK_WIDTH * math.cos(beta2), y2 + side * TRACK_WIDTH * math.sin(beta2))
                b2_r = (x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                        y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2))
                self.road_poly.append(([b1_l, b1_r, b2_r, b2_l], (1, 1, 1) if i % 2 == 0 else (1, 0, 0)))
        self.track = track
        return True

    def reset(self):
        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []
        self.track_direction = random.choice([-1, 1])

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose == 1:
                print("retry to generate track (normal if there are not many of this messages)")
        self.car = Car(self.world, *self.track[0][1:4])

        return self.step(None)[0]

    def step(self, action):
        if action is not None:
            self.car.steer(-STEER_ACTION[action[0]])
            self.car.gas(GAS_ACTION[action[1]])
            self.car.brake(BRAKE_ACTION[action[2]])

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        # Get distance to track tiles
        min_left_distance, min_right_distance, close_tiles = self.get_min_distances()
        min_distances = (min_left_distance, min_right_distance)
        wheel_distance_states = (min(DISTANCE_INTERVALS, int(min_left_distance)),
                                 min(DISTANCE_INTERVALS, int(min_right_distance)))

        speed = min(self.car.hull.linearVelocity.length, MAX_SPEED)
        # ceil division, +1 to keep between [0,SPEED_INTERVALS]
        speed_state = int(0 if speed <= 2 else math.ceil(speed / (MAX_SPEED / (SPEED_INTERVALS))))

        if action is not None:
            # Get raycast distances
            raycast_dist = self.get_raycast_points(close_tiles)
            raycast_dist_state = [int(dist // (RAY_CAST_DISTANCE / RAY_CAST_INTERVALS)) for dist in raycast_dist]
        else:
            raycast_dist_state = [RAY_CAST_INTERVALS - 1 for i in range(RAY_CAST_INTERVALS)]
        # print(raycast_dist)

        left_wheel_off = 1 if min_left_distance > DISTANCE_INTERVALS else 0
        right_wheel_off = 1 if min_right_distance > DISTANCE_INTERVALS else 0
        wheel_on_off_states = (left_wheel_off, right_wheel_off)

        # Steer interval states
        # Get wheel joint
        joint = self.car.wheels[0].joint
        joint_angle = np.clip(joint.angle, joint.lowerLimit, joint.upperLimit)
        joint_range = joint.upperLimit - joint.lowerLimit
        steer_state = min(int((joint_angle + joint_range / 2) // (joint_range / (STEER_INTERVALS))),
                          STEER_INTERVALS - 1)

        self.state = (speed_state,) + tuple(raycast_dist_state) + (steer_state,)

        step_reward = 0
        done = False
        reward_list = []

        if action is not None:  # First step without action, called from reset()

            ##REWARDS##

            # Negative reward based on time
            time_reward = -0.1
            reward_list.append(("time", time_reward))

            # positive speed reward
            speed_reward = (speed ** 0.5) / 50
            reward_list.append(("speed", speed_reward))

            # Increasingly bad reward for not moving
            if speed < 2:
                self.slowness += 1
            else:
                self.slowness = 0
            slow_reward = -0.01 * self.slowness
            if self.slowness > 10:
                # Slow for too long? shut 'er down
                self.slowness = 0
                slow_reward -= 10
                done = True
            reward_list.append(("slowness", slow_reward))

            # negative reward for steering away from away raypoint
            # Get max angle (or average of them if multiple)
            max_angles = []
            max_dist = max(raycast_dist)
            for index, ray in enumerate(raycast_dist):
                if ray == max_dist:
                    max_angles.append(self.raycast_angles[index])
            avg_angle = sum(max_angles) / len(max_angles)
            # Get steering angle
            abs_steer = joint_angle + self.car.hull.angle + math.pi / 2
            # Get difference between them, and score based on the difference
            angle_diff = abs(avg_angle - abs_steer)
            # Score proportional to sum of distances (more close sensors = bad)
            gas = self.car.wheels[2].gas + 1
            raycast_dist_normalized = 3 - (3 - 1) / (75 - 10) * (sum(raycast_dist) - 10) + 1
            angle_diff_normalized = 1 - 1 / (1.2) * (angle_diff)
            steer_reward = 0.3 * (speed_reward / 2 + .1) * angle_diff_normalized * raycast_dist_normalized
            reward_list.append(("steer", steer_reward))

            # Stop the run if both wheels outside bounds of track
            cnt = 0
            for dist in min_distances:
                if dist > DISTANCE_INTERVALS:
                    cnt += 1
            if cnt == len(min_distances):
                done = True
                step_reward -= 100

            # Positive reward for going toward the next tile.
            def minimum_distance(v, w, p):
                # Return minimum distance between line segment vw and point p
                l2 = np.sum((w - v) ** 2)
                if l2 == 0.0:
                    return np.linalg.norm(p - v)  # v == w
                t = max(0, min(1, np.dot(p - v, w - v) / l2))
                projection = v + t * (w - v)
                return np.linalg.norm(p - projection)

            try:
                next_tile = next(tile for tile in close_tiles if tile.road_visited is False)
                # Get closest vertex of next tile
                x, y = self.car.hull.position
                car_pos = np.array([x, y])
                verts = [np.array([vertex[0], vertex[1]])
                         for vertex in next_tile.fixtures[0].shape.vertices]
                i, min_vertex = min([(index, vertex) for index, vertex in enumerate(
                    verts)], key=lambda vert: np.linalg.norm(car_pos - vert[1]))
                opp_vertex = None
                if np.linalg.norm(verts[i] - verts[(i + 1) % 4]) > np.linalg.norm(verts[i] - verts[(i - 1) % 4]):
                    opp_vertex = verts[(i + 1) % 4]
                else:
                    opp_vertex = verts[(i - 1) % 4]

                cur_tile_dist = minimum_distance(min_vertex, opp_vertex, car_pos)
                if hasattr(self, "tile_dist"):
                    dtd = cur_tile_dist - self.tile_dist
                    if dtd <= 1:  # Filter out "jumps" by only updating when decreasing
                        self.dtile_dist = dtd
                    tile_dist_score = 0.1 * (abs(self.dtile_dist) ** 0.5)
                    reward_list.append(("tile_dist_score", tile_dist_score))
                self.tile_dist = cur_tile_dist

            except StopIteration:
                pass

            # Small negative reward based on raycast distance to wall
            raycast_reward = -.5 + 0.5 / (75 - 10) * (sum(raycast_dist) - 10) / (speed_state / 10 + 1)
            reward_list.append(("raycast_reward", raycast_reward))

            # Reduce score if wheel off track.
            wheel_reward = -1 * (0.2 if 1 in wheel_on_off_states else 0)
            reward_list.append(("wheel", wheel_reward))

            step_reward += sum([r for n, r in reward_list])
            # print(*[f"{n}:{r:.2f}" for n, r in reward_list])

            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            self.reward += step_reward
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track):
                done = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                done = True
                step_reward = -100

        return self.state, step_reward, done, self.tile_visited_count

    def render(self, mode='human'):
        assert mode in ['human', 'state_pixels', 'rgb_array']
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label('0000', font_size=36,
                                                 x=20, y=WINDOW_H * 2.5 / 40.00, anchor_x='left', anchor_y='center',
                                                 color=(255, 255, 255, 255))
            self.transform = rendering.Transform()

        if "t" not in self.__dict__: return  # reset() not called yet

        zoom = 0.3 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)  # Animate zoom first second
        zoom_state = ZOOM * SCALE * STATE_W / WINDOW_W
        zoom_video = ZOOM * SCALE * VIDEO_W / WINDOW_W
        scroll_x = self.car.hull.position[0]
        scroll_y = self.car.hull.position[1]
        angle = -self.car.hull.angle
        vel = self.car.hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(
            WINDOW_W / 2 - (scroll_x * zoom * math.cos(angle) - scroll_y * zoom * math.sin(angle)),
            WINDOW_H / 2 - (scroll_x * zoom * math.sin(angle) + scroll_y * zoom * math.cos(angle)))
        self.transform.set_rotation(angle)
        # self.transform.set_scale(2, 2)
        # self.transform.set_translation(WINDOW_W/2, WINDOW_H/2)
        self.car.draw(self.viewer, mode != "state_pixels")

        arr = None
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        t = self.transform
        if mode == 'rgb_array':
            VP_W = VIDEO_W
            VP_H = VIDEO_H
        elif mode == 'state_pixels':
            VP_W = STATE_W
            VP_H = STATE_H
        else:
            pixel_scale = 1
            if hasattr(win.context, '_nscontext'):
                pixel_scale = win.context._nscontext.view().backingScaleFactor()  # pylint: disable=protected-access
            VP_W = int(pixel_scale * WINDOW_W)
            VP_H = int(pixel_scale * WINDOW_H)

        gl.glViewport(0, 0, VP_W, VP_H)
        t.enable()
        self.render_road()
        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.onetime_geoms = []
        t.disable()
        self.render_indicators(WINDOW_W, WINDOW_H)
        self.render_raycasts()
        self.render_wall_segments()
        self.render_intersections()

        if mode == 'human':
            win.flip()
            return self.viewer.isopen

        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
        arr = arr.reshape(VP_H, VP_W, 4)
        arr = arr[::-1, :, 0:3]

        return arr

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def render_road(self):
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(0.4, 0.8, 0.4, 1.0)
        gl.glVertex3f(-PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, -PLAYFIELD, 0)
        gl.glVertex3f(-PLAYFIELD, -PLAYFIELD, 0)
        gl.glColor4f(0.4, 0.9, 0.4, 1.0)
        k = PLAYFIELD / 20.0
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                gl.glVertex3f(k * x + k, k * y + 0, 0)
                gl.glVertex3f(k * x + 0, k * y + 0, 0)
                gl.glVertex3f(k * x + 0, k * y + k, 0)
                gl.glVertex3f(k * x + k, k * y + k, 0)
        for poly, color in self.road_poly:
            gl.glColor4f(color[0], color[1], color[2], 1)
            for p in poly:
                gl.glVertex3f(p[0], p[1], 0)
        gl.glEnd()

    def render_indicators(self, W, H):
        gl.glBegin(gl.GL_QUADS)
        s = W / 40.0
        h = H / 40.0
        gl.glColor4f(0, 0, 0, 1)
        gl.glVertex3f(W, 0, 0)
        gl.glVertex3f(W, 5 * h, 0)
        gl.glVertex3f(0, 5 * h, 0)
        gl.glVertex3f(0, 0, 0)

        def vertical_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place + 0) * s, h + h * val, 0)
            gl.glVertex3f((place + 1) * s, h + h * val, 0)
            gl.glVertex3f((place + 1) * s, h, 0)
            gl.glVertex3f((place + 0) * s, h, 0)

        def horiz_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place + 0) * s, 4 * h, 0)
            gl.glVertex3f((place + val) * s, 4 * h, 0)
            gl.glVertex3f((place + val) * s, 2 * h, 0)
            gl.glVertex3f((place + 0) * s, 2 * h, 0)

        true_speed = np.sqrt(np.square(self.car.hull.linearVelocity[0]) + np.square(self.car.hull.linearVelocity[1]))
        vertical_ind(5, 0.02 * true_speed, (1, 1, 1))
        vertical_ind(7, 0.01 * self.car.wheels[0].omega, (0.0, 0, 1))  # ABS sensors
        vertical_ind(8, 0.01 * self.car.wheels[1].omega, (0.0, 0, 1))
        vertical_ind(9, 0.01 * self.car.wheels[2].omega, (0.2, 0, 1))
        vertical_ind(10, 0.01 * self.car.wheels[3].omega, (0.2, 0, 1))
        horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle, (0, 1, 0))
        horiz_ind(30, -0.8 * self.car.hull.angularVelocity, (1, 0, 0))
        gl.glEnd()
        self.score_label.text = "%04i" % self.reward
        self.score_label.draw()

    def render_raycasts(self):
        if hasattr(self, "raycasts"):
            for raycast in self.raycasts:
                path = [(raycast[0][0], raycast[0][1]), (raycast[1][0], raycast[1][1])]
                self.viewer.draw_line(start=path[0], end=path[1], color=(1, 0.0, 0.0), linewidth=3)

    def render_wall_segments(self):
        if hasattr(self, "wall_segments"):
            for path in self.wall_segments:
                self.viewer.draw_line(start=path[0], end=path[1], color=(0.0, 0.0, 1), linewidth=3)

    def render_intersections(self):
        if hasattr(self, "intersections"):
            for point in self.intersections:
                self.viewer.draw_circle(point, color=(0.0, 1, 0.0), radius=1)

    def get_min_distances(self):
        # Retrieves the distance to the nearest track tile centroid. Returns distance from left and right wheels, and close tiles
        wheels = self.car.wheels
        (front_left_wheel, front_right_wheel) = (wheels[0].position, wheels[1].position)
        min_left_distance = 9999
        min_right_distance = 9999
        close_tiles = []
        for road_tile in self.road:
            road_tile_position = road_tile.fixtures[0].shape.centroid
            lt_distance = math.sqrt(abs(road_tile_position.x - front_left_wheel.x) ** 2 + abs(
                road_tile_position.y - front_left_wheel.y) ** 2)
            if lt_distance < min_left_distance:
                min_left_distance = lt_distance
            rt_distance = math.sqrt(abs(road_tile_position.x - front_right_wheel.x) ** 2 + abs(
                road_tile_position.y - front_right_wheel.y) ** 2)
            if rt_distance < min_right_distance:
                min_right_distance = rt_distance

            if lt_distance < RAY_CAST_DISTANCE or rt_distance < RAY_CAST_DISTANCE:
                close_tiles.append((road_tile, lt_distance))
        close_tiles.sort(key=lambda x: x[1])
        close_tiles = [x[0] for x in close_tiles]

        return min_left_distance, min_right_distance, close_tiles

    def get_raycast_points(self, tiles):
        # Loop through my raycast sensors and find intersection distances for each sensor with the given tiles.
        # Angles are arc from -90deg to +90deg
        start_angle = -math.pi / 4
        end_angle = math.pi / 4
        interval = abs(start_angle - end_angle)
        rotation = math.pi / 2  # Correction factor
        angles = np.arange(start_angle, end_angle + interval / (NUM_SENSORS - 1), interval / (NUM_SENSORS - 1))
        # Add current orientation of car
        angles = [i + self.car.hull.angle + rotation for i in angles]
        # Get relative endpoints of raycast
        rel_endpts = [(math.cos(a) * RAY_CAST_DISTANCE, math.sin(a) * RAY_CAST_DISTANCE) for a in angles]
        # Get global enpoints of raycast
        endpts = [(x + self.car.hull.position.x, y + self.car.hull.position.y) for x, y in rel_endpts]
        # Get line segments from car to end of raycast
        raycasts = [((self.car.hull.position.x, self.car.hull.position.y), endpoint) for endpoint in endpts]

        self.raycasts = raycasts
        self.raycast_angles = angles

        # Get wall segments
        wall_segments = []
        for tile in tiles:
            verts = tile.fixtures[0].shape.vertices
            if len(verts) < 4:
                continue
            dist1 = math.sqrt((verts[0][0] - verts[3][0]) ** 2 +
                              (verts[0][1] - verts[3][1]) ** 2)
            dist2 = math.sqrt((verts[0][0] - verts[1][0]) ** 2 +
                              (verts[2][1] - verts[3][1]) ** 2)
            if dist1 < dist2:
                wall_segments.append((verts[0], verts[3]))
                wall_segments.append((verts[1], verts[2]))
            else:
                wall_segments.append((verts[0], verts[1]))
                wall_segments.append((verts[2], verts[3]))

        # Add them to  be drawn later
        self.wall_segments = []
        for wall_segment in wall_segments:
            path = [(wall_segment[0][0], wall_segment[0][1]), (wall_segment[1][0], wall_segment[1][1])]
            self.wall_segments.append(path)

        def intersection(seg1, seg2):
            # Based on this formula http://www-cs.ccny.cuny.edu/~wolberg/capstone/intersection/Intersection%20point%20of%20two%20lines.html
            x1, y1 = seg1[0]
            x2, y2 = seg1[1]
            x3, y3 = seg2[0]
            x4, y4 = seg2[1]

            denom = (x2 - x1) * (y4 - y3) - (x4 - x3) * (y2 - y1)
            if math.isclose(denom, 0):
                # Denominator close to 0 means lines parallel
                return None

            t_num = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            t = t_num / denom
            u_num = (x2 - x1) * (y1 - y3) - (x1 - x3) * (y2 - y1)
            u = u_num / denom

            if t >= 0.0 and t <= 1.0 and u >= 0.0 and u <= 1.0:
                return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

        # Loop through points, get the intersection of the closest wall
        int_dist = []
        self.intersections = []
        for raycast in raycasts:
            ray_int_points = []
            for wall in wall_segments:
                int_point = intersection(raycast, wall)
                if int_point is not None:
                    dist = math.sqrt(
                        (self.car.hull.position.x - int_point[0]) ** 2 + (self.car.hull.position.y - int_point[1]) ** 2)
                    ray_int_points.append((dist, [int_point[0], int_point[1]]))
            if ray_int_points:
                ray_int_points.sort()
                dist = ray_int_points[0][0]
                point = ray_int_points[0][1]
                self.intersections.append(point)
                int_dist.append(dist)
            else:
                int_dist.append(RAY_CAST_DISTANCE - 1)  # Max range

        return int_dist


class CarRacingPoSContinuousState(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second': FPS
    }

    def __init__(self, verbose=0):
        EzPickle.__init__(self)
        self.seed()
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=
                               [(0, 0), (1, 0), (1, -1), (0, -1)]))
        self.slowness = 0

        """
        Action Space:
        1) Steer: Discrete 3  - NOOP[0], Left[1], Right[2] - params: min: 0, max: 2
        2) Gas: Discrete 2 - NOOP[0], Go[1] - params: min: 0, max: 1
        3) Brake: Discrete 2  - NOOP[0], Brake[1] - params: min: 0, max: 1

        Observation Space (continuous):
        1) Speed: float32 (0-100)
        2) Sensors: NUM_SENSORS (0 - 20)
        3) Wheel off or not ( for each wheel): 2
        4) Steering: STEER_INTERVALS

        """
        # ###CONT### ###CONSTRUCTOR###
        self.action_space = spaces.MultiDiscrete([3, 2, 2])
        self.observation_space = spaces.Box(low=np.array([MIN_SPEED] + [0.]*NUM_SENSORS + [STEER_MIN]),
                                            high=np.array([MAX_SPEED] + [RAY_CAST_DISTANCE]*NUM_SENSORS + [STEER_MAX]),
                                            dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        self.car.destroy()

    def _create_track(self):
        CHECKPOINTS = 12

        # direction = -1 for right turns, 1 for left turns
        direction = self.track_direction

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            alpha = 2 * math.pi * c / CHECKPOINTS + self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)
            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD
            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))

        # print("\n".join(str(h) for h in checkpoints))
        # self.road_poly = [ (    # uncomment this to see checkpoints
        #    [ (tx,ty) for a,tx,ty in checkpoints ],
        #    (0.7,0.7,0.9) ) ]
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi
            while True:  # Find destination from checkpoints
                failed = True
                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break
                if not failed:
                    break
                alpha -= 2 * math.pi
                continue
            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            proj = r1x * dest_dx + r1y * dest_dy  # destination vector projected on rad
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, direction * x, direction * y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break
        # print "\n".join([str(t) for t in enumerate(track)])

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose == 1:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1:i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2])) +
            np.square(first_perp_y * (track[0][3] - track[-1][3])))
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (x1 - TRACK_WIDTH * math.cos(beta1), y1 - TRACK_WIDTH * math.sin(beta1))
            road1_r = (x1 + TRACK_WIDTH * math.cos(beta1), y1 + TRACK_WIDTH * math.sin(beta1))
            road2_l = (x2 - TRACK_WIDTH * math.cos(beta2), y2 - TRACK_WIDTH * math.sin(beta2))
            road2_r = (x2 + TRACK_WIDTH * math.cos(beta2), y2 + TRACK_WIDTH * math.sin(beta2))
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3)
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.road_visited = False
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (x1 + side * TRACK_WIDTH * math.cos(beta1), y1 + side * TRACK_WIDTH * math.sin(beta1))
                b1_r = (x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                        y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1))
                b2_l = (x2 + side * TRACK_WIDTH * math.cos(beta2), y2 + side * TRACK_WIDTH * math.sin(beta2))
                b2_r = (x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                        y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2))
                self.road_poly.append(([b1_l, b1_r, b2_r, b2_l], (1, 1, 1) if i % 2 == 0 else (1, 0, 0)))
        self.track = track
        return True

    def reset(self):
        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []
        self.track_direction = random.choice([-1, 1])

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose == 1:
                print("retry to generate track (normal if there are not many of this messages)")
        self.car = Car(self.world, *self.track[0][1:4])

        return self.step(None)[0]

    def step(self, action):
        if action is not None:
            self.car.steer(-STEER_ACTION[action[0]])
            self.car.gas(GAS_ACTION[action[1]])
            self.car.brake(BRAKE_ACTION[action[2]])

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        # Get distance to track tiles
        min_left_distance, min_right_distance, close_tiles = self.get_min_distances()
        min_distances = (min_left_distance, min_right_distance)
        wheel_distance_states = (min(DISTANCE_INTERVALS, int(min_left_distance)),
                                 min(DISTANCE_INTERVALS, int(min_right_distance)))

        speed_state = min(self.car.hull.linearVelocity.length, MAX_SPEED)

        if action is not None:
            # Get raycast distances
            raycast_dist_state = self.get_raycast_points(close_tiles)
        else:
            raycast_dist_state = [RAY_CAST_INTERVALS - 1 for i in range(RAY_CAST_INTERVALS)]
        # print(raycast_dist)

        left_wheel_off = 1 if min_left_distance > DISTANCE_INTERVALS else 0
        right_wheel_off = 1 if min_right_distance > DISTANCE_INTERVALS else 0
        wheel_on_off_states = (left_wheel_off, right_wheel_off)

        # Steer interval states
        # Get wheel joint
        joint = self.car.wheels[0].joint
        joint_angle = np.clip(joint.angle, joint.lowerLimit, joint.upperLimit)
        joint_range = joint.upperLimit - joint.lowerLimit
        steer_state = joint_angle + joint_range / 2
        # ###CONT### ###STATE###
        self.state = (speed_state,) + tuple(raycast_dist_state) + (steer_state,)

        step_reward = 0
        done = False
        reward_list = []

        if action is not None:  # First step without action, called from reset()

            ##REWARDS##

            # Negative reward based on time
            time_reward = -0.1
            reward_list.append(("time", time_reward))

            # positive speed reward
            speed_reward = (speed_state ** 0.5) / 50
            reward_list.append(("speed", speed_reward))

            # Increasingly bad reward for not moving
            if speed_state < 2:
                self.slowness += 1
            else:
                self.slowness = 0
            slow_reward = -0.01 * self.slowness
            if self.slowness > 10:
                # Slow for too long? shut 'er down
                self.slowness = 0
                slow_reward -= 10
                done = True
            reward_list.append(("slowness", slow_reward))

            # negative reward for steering away from away raypoint
            # Get max angle (or average of them if multiple)
            max_angles = []
            max_dist = max(raycast_dist_state)
            for index, ray in enumerate(raycast_dist_state):
                if ray == max_dist:
                    max_angles.append(self.raycast_angles[index])
            avg_angle = sum(max_angles) / len(max_angles)
            # Get steering angle
            abs_steer = joint_angle + self.car.hull.angle + math.pi / 2
            # Get difference between them, and score based on the difference
            angle_diff = abs(avg_angle - abs_steer)
            # Score proportional to sum of distances (more close sensors = bad)
            gas = self.car.wheels[2].gas + 1
            raycast_dist_normalized = 3 - (3 - 1) / (75 - 10) * (sum(raycast_dist_state) - 10) + 1
            angle_diff_normalized = 1 - 1 / 1.2 * angle_diff
            steer_reward = 0.3 * (speed_reward / 2 + .1) * angle_diff_normalized * raycast_dist_normalized
            reward_list.append(("steer", steer_reward))

            # Stop the run if both wheels outside bounds of track
            cnt = 0
            for dist in min_distances:
                if dist > DISTANCE_INTERVALS:
                    cnt += 1
            if cnt == len(min_distances):
                done = True
                step_reward -= 100

            # Positive reward for going toward the next tile.
            def minimum_distance(v, w, p):
                # Return minimum distance between line segment vw and point p
                l2 = np.sum((w - v) ** 2)
                if l2 == 0.0:
                    return np.linalg.norm(p - v)  # v == w
                t = max(0, min(1, np.dot(p - v, w - v) / l2))
                projection = v + t * (w - v)
                return np.linalg.norm(p - projection)

            try:
                next_tile = next(tile for tile in close_tiles if tile.road_visited is False)
                # Get closest vertex of next tile
                x, y = self.car.hull.position
                car_pos = np.array([x, y])
                verts = [np.array([vertex[0], vertex[1]])
                         for vertex in next_tile.fixtures[0].shape.vertices]
                i, min_vertex = min([(index, vertex) for index, vertex in enumerate(
                    verts)], key=lambda vert: np.linalg.norm(car_pos - vert[1]))
                opp_vertex = None
                if np.linalg.norm(verts[i] - verts[(i + 1) % 4]) > np.linalg.norm(verts[i] - verts[(i - 1) % 4]):
                    opp_vertex = verts[(i + 1) % 4]
                else:
                    opp_vertex = verts[(i - 1) % 4]

                cur_tile_dist = minimum_distance(min_vertex, opp_vertex, car_pos)
                if hasattr(self, "tile_dist"):
                    dtd = cur_tile_dist - self.tile_dist
                    if dtd <= 1:  # Filter out "jumps" by only updating when decreasing
                        self.dtile_dist = dtd
                    tile_dist_score = 0.1 * (abs(self.dtile_dist) ** 0.5)
                    reward_list.append(("tile_dist_score", tile_dist_score))
                self.tile_dist = cur_tile_dist

            except StopIteration:
                pass

            # Small negative reward based on raycast distance to wall
            raycast_reward = -.5 + 0.5 / (75 - 10) * (sum(raycast_dist_state) - 10) / (speed_state / 10 + 1)
            reward_list.append(("raycast_reward", raycast_reward))

            # Reduce score if wheel off track.
            wheel_reward = -1 * (0.2 if 1 in wheel_on_off_states else 0)
            reward_list.append(("wheel", wheel_reward))

            step_reward += sum([r for n, r in reward_list])
            # print(*[f"{n}:{r:.2f}" for n, r in reward_list])

            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            self.reward += step_reward

            self.prev_reward = self.reward

            # If agent has driven around track completely, game is over
            if self.tile_visited_count == len(self.track):
                done = True

            # If agent has driven out of bounds, game is over
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                done = True
                step_reward = -100

        return self.state, step_reward, done, self.tile_visited_count

    def render(self, mode='human'):
        assert mode in ['human', 'state_pixels', 'rgb_array']
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label('0000', font_size=36,
                                                 x=20, y=WINDOW_H * 2.5 / 40.00, anchor_x='left', anchor_y='center',
                                                 color=(255, 255, 255, 255))
            self.transform = rendering.Transform()

        if "t" not in self.__dict__: return  # reset() not called yet

        zoom = 0.3 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)  # Animate zoom first second
        zoom_state = ZOOM * SCALE * STATE_W / WINDOW_W
        zoom_video = ZOOM * SCALE * VIDEO_W / WINDOW_W
        scroll_x = self.car.hull.position[0]
        scroll_y = self.car.hull.position[1]
        angle = -self.car.hull.angle
        vel = self.car.hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(
            WINDOW_W / 2 - (scroll_x * zoom * math.cos(angle) - scroll_y * zoom * math.sin(angle)),
            WINDOW_H / 2 - (scroll_x * zoom * math.sin(angle) + scroll_y * zoom * math.cos(angle)))
        self.transform.set_rotation(angle)
        # self.transform.set_scale(2, 2)
        # self.transform.set_translation(WINDOW_W/2, WINDOW_H/2)
        self.car.draw(self.viewer, mode != "state_pixels")

        arr = None
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        t = self.transform
        if mode == 'rgb_array':
            VP_W = VIDEO_W
            VP_H = VIDEO_H
        elif mode == 'state_pixels':
            VP_W = STATE_W
            VP_H = STATE_H
        else:
            pixel_scale = 1
            if hasattr(win.context, '_nscontext'):
                pixel_scale = win.context._nscontext.view().backingScaleFactor()  # pylint: disable=protected-access
            VP_W = int(pixel_scale * WINDOW_W)
            VP_H = int(pixel_scale * WINDOW_H)

        gl.glViewport(0, 0, VP_W, VP_H)
        t.enable()
        self.render_road()
        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.onetime_geoms = []
        t.disable()
        self.render_indicators(WINDOW_W, WINDOW_H)
        self.render_raycasts()
        self.render_wall_segments()
        self.render_intersections()

        if mode == 'human':
            win.flip()
            return self.viewer.isopen

        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
        arr = arr.reshape(VP_H, VP_W, 4)
        arr = arr[::-1, :, 0:3]

        return arr

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def render_road(self):
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(0.4, 0.8, 0.4, 1.0)
        gl.glVertex3f(-PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, -PLAYFIELD, 0)
        gl.glVertex3f(-PLAYFIELD, -PLAYFIELD, 0)
        gl.glColor4f(0.4, 0.9, 0.4, 1.0)
        k = PLAYFIELD / 20.0
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                gl.glVertex3f(k * x + k, k * y + 0, 0)
                gl.glVertex3f(k * x + 0, k * y + 0, 0)
                gl.glVertex3f(k * x + 0, k * y + k, 0)
                gl.glVertex3f(k * x + k, k * y + k, 0)
        for poly, color in self.road_poly:
            gl.glColor4f(color[0], color[1], color[2], 1)
            for p in poly:
                gl.glVertex3f(p[0], p[1], 0)
        gl.glEnd()

    def render_indicators(self, W, H):
        gl.glBegin(gl.GL_QUADS)
        s = W / 40.0
        h = H / 40.0
        gl.glColor4f(0, 0, 0, 1)
        gl.glVertex3f(W, 0, 0)
        gl.glVertex3f(W, 5 * h, 0)
        gl.glVertex3f(0, 5 * h, 0)
        gl.glVertex3f(0, 0, 0)

        def vertical_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place + 0) * s, h + h * val, 0)
            gl.glVertex3f((place + 1) * s, h + h * val, 0)
            gl.glVertex3f((place + 1) * s, h, 0)
            gl.glVertex3f((place + 0) * s, h, 0)

        def horiz_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place + 0) * s, 4 * h, 0)
            gl.glVertex3f((place + val) * s, 4 * h, 0)
            gl.glVertex3f((place + val) * s, 2 * h, 0)
            gl.glVertex3f((place + 0) * s, 2 * h, 0)

        true_speed = np.sqrt(np.square(self.car.hull.linearVelocity[0]) + np.square(self.car.hull.linearVelocity[1]))
        vertical_ind(5, 0.02 * true_speed, (1, 1, 1))
        vertical_ind(7, 0.01 * self.car.wheels[0].omega, (0.0, 0, 1))  # ABS sensors
        vertical_ind(8, 0.01 * self.car.wheels[1].omega, (0.0, 0, 1))
        vertical_ind(9, 0.01 * self.car.wheels[2].omega, (0.2, 0, 1))
        vertical_ind(10, 0.01 * self.car.wheels[3].omega, (0.2, 0, 1))
        horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle, (0, 1, 0))
        horiz_ind(30, -0.8 * self.car.hull.angularVelocity, (1, 0, 0))
        gl.glEnd()
        self.score_label.text = "%04i" % self.reward
        self.score_label.draw()

    def render_raycasts(self):
        if hasattr(self, "raycasts"):
            for raycast in self.raycasts:
                path = [(raycast[0][0], raycast[0][1]), (raycast[1][0], raycast[1][1])]
                self.viewer.draw_line(start=path[0], end=path[1], color=(1, 0.0, 0.0), linewidth=3)

    def render_wall_segments(self):
        if hasattr(self, "wall_segments"):
            for path in self.wall_segments:
                self.viewer.draw_line(start=path[0], end=path[1], color=(0.0, 0.0, 1), linewidth=3)

    def render_intersections(self):
        if hasattr(self, "intersections"):
            for point in self.intersections:
                self.viewer.draw_circle(point, color=(0.0, 1, 0.0), radius=1)

    def get_min_distances(self):
        # Retrieves the distance to the nearest track tile centroid. Returns distance from left and right wheels, and close tiles
        wheels = self.car.wheels
        (front_left_wheel, front_right_wheel) = (wheels[0].position, wheels[1].position)
        min_left_distance = 9999
        min_right_distance = 9999
        close_tiles = []
        for road_tile in self.road:
            road_tile_position = road_tile.fixtures[0].shape.centroid
            lt_distance = math.sqrt(abs(road_tile_position.x - front_left_wheel.x) ** 2 + abs(
                road_tile_position.y - front_left_wheel.y) ** 2)
            if lt_distance < min_left_distance:
                min_left_distance = lt_distance
            rt_distance = math.sqrt(abs(road_tile_position.x - front_right_wheel.x) ** 2 + abs(
                road_tile_position.y - front_right_wheel.y) ** 2)
            if rt_distance < min_right_distance:
                min_right_distance = rt_distance

            if lt_distance < RAY_CAST_DISTANCE or rt_distance < RAY_CAST_DISTANCE:
                close_tiles.append((road_tile, lt_distance))
        close_tiles.sort(key=lambda x: x[1])
        close_tiles = [x[0] for x in close_tiles]

        return min_left_distance, min_right_distance, close_tiles

    def get_raycast_points(self, tiles):
        # Loop through my raycast sensors and find intersection distances for each sensor with the given tiles.
        # Angles are arc from -90deg to +90deg
        start_angle = -math.pi / 4
        end_angle = math.pi / 4
        interval = abs(start_angle - end_angle)
        rotation = math.pi / 2  # Correction factor
        angles = np.arange(start_angle, end_angle + interval / (NUM_SENSORS - 1), interval / (NUM_SENSORS - 1))
        # Add current orientation of car
        angles = [i + self.car.hull.angle + rotation for i in angles]
        # Get relative endpoints of raycast
        rel_endpts = [(math.cos(a) * RAY_CAST_DISTANCE, math.sin(a) * RAY_CAST_DISTANCE) for a in angles]
        # Get global enpoints of raycast
        endpts = [(x + self.car.hull.position.x, y + self.car.hull.position.y) for x, y in rel_endpts]
        # Get line segments from car to end of raycast
        raycasts = [((self.car.hull.position.x, self.car.hull.position.y), endpoint) for endpoint in endpts]

        self.raycasts = raycasts
        self.raycast_angles = angles

        # Get wall segments
        wall_segments = []
        for tile in tiles:
            verts = tile.fixtures[0].shape.vertices
            if len(verts) < 4:
                continue
            dist1 = math.sqrt((verts[0][0] - verts[3][0]) ** 2 +
                              (verts[0][1] - verts[3][1]) ** 2)
            dist2 = math.sqrt((verts[0][0] - verts[1][0]) ** 2 +
                              (verts[2][1] - verts[3][1]) ** 2)
            if dist1 < dist2:
                wall_segments.append((verts[0], verts[3]))
                wall_segments.append((verts[1], verts[2]))
            else:
                wall_segments.append((verts[0], verts[1]))
                wall_segments.append((verts[2], verts[3]))

        # Add them to  be drawn later
        self.wall_segments = []
        for wall_segment in wall_segments:
            path = [(wall_segment[0][0], wall_segment[0][1]), (wall_segment[1][0], wall_segment[1][1])]
            self.wall_segments.append(path)

        def intersection(seg1, seg2):
            # Based on this formula http://www-cs.ccny.cuny.edu/~wolberg/capstone/intersection/Intersection%20point%20of%20two%20lines.html
            x1, y1 = seg1[0]
            x2, y2 = seg1[1]
            x3, y3 = seg2[0]
            x4, y4 = seg2[1]

            denom = (x2 - x1) * (y4 - y3) - (x4 - x3) * (y2 - y1)
            if math.isclose(denom, 0):
                # Denominator close to 0 means lines parallel
                return None

            t_num = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            t = t_num / denom
            u_num = (x2 - x1) * (y1 - y3) - (x1 - x3) * (y2 - y1)
            u = u_num / denom

            if t >= 0.0 and t <= 1.0 and u >= 0.0 and u <= 1.0:
                return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

        # Loop through points, get the intersection of the closest wall
        int_dist = []
        self.intersections = []
        for raycast in raycasts:
            ray_int_points = []
            for wall in wall_segments:
                int_point = intersection(raycast, wall)
                if int_point is not None:
                    dist = math.sqrt(
                        (self.car.hull.position.x - int_point[0]) ** 2 + (self.car.hull.position.y - int_point[1]) ** 2)
                    ray_int_points.append((dist, [int_point[0], int_point[1]]))
            if ray_int_points:
                ray_int_points.sort()
                dist = ray_int_points[0][0]
                point = ray_int_points[0][1]
                self.intersections.append(point)
                int_dist.append(dist)
            else:
                int_dist.append(RAY_CAST_DISTANCE - 1)  # Max range

        return int_dist
