# -------------------------------------------------------------------Built-ins
import signal
# -------------------------------------------------------------------3rd Party
import numpy as np
import zmq
import pygame
import quaternion as qt
# -------------------------------------------------------------------------GPB
from vision_messages_pb2 import VisionMessageWrapperPBMsg
from vision_messages_pb2 import TrackHeadOrientationPBMsg
# ----------------------------------------------------------------------Custom
import measured_structs as ms
import head_model_3d as hm

# ****************************************************************************
#                             Settings
# ****************************************************************************
connection_string = "tcp://127.0.0.1:5556"
count_max = 10

select_type = ""

# ****************************************************************************
# ****************************************************************************

keep_running = True


def shandle(signal, frame):
    """ For gracefully handling Ctrl+c

    @param signal: don't worry about it
    @param frame: don't worry about it
    @return: don't worry about it
    """
    global keep_running
    print("gracefully exiting")
    keep_running = False


signal.signal(signal.SIGINT, shandle)

# ****************************************************************************
#                       Reusable structures and GPBs
# ****************************************************************************
_msg_gpb = VisionMessageWrapperPBMsg()
_head_gpb = TrackHeadOrientationPBMsg()
_head = ms.VisionData()
orientation = qt.quaternion(1, 0, 0, 0)
# ****************************************************************************


# ****************************************************************************
#                       Setup PyGame for rendering box
# ****************************************************************************
pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.init()
pygame.display.update()
clock = pygame.time.Clock()
head_model = hm.HeadModel(screen.get_width(), screen.get_height())
font = pygame.font.Font(None, 30)
# ****************************************************************************

# ****************************************************************************
#                      Setup ZMQ subscription socket
# ****************************************************************************
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect(connection_string)
socket.setsockopt(zmq.SUBSCRIBE, "".encode('ascii'))

# ****************************************************************************
# ****************************************************************************
#                           MAIN LOOP!!!
# ****************************************************************************
# ****************************************************************************
count = 0
aa = 1/np.sqrt(2)
q1 = qt.quaternion(aa, aa, 0, 0)
q2 = qt.quaternion(aa, 0, aa, 0)
while keep_running:
    string = socket.recv()
    _msg_gpb.ParseFromString(string)

    if _msg_gpb.MsgName == "TrackHeadOrientationMsg":
        _head_gpb.ParseFromString(_msg_gpb.MsgData)
        _head.from_gpb(_head_gpb)
        orientation = _head.q

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                keep_running = False

        if count > count_max:
            count = 0;
            screen.fill((0, 32, 0))
            dt = clock.tick()
            fps = font.render(str(int(clock.get_fps())), True, pygame.Color('green'))

            ov = head_model.oriented_vertices(q2*q1*orientation)
            for points in ov:
                pygame.draw.polygon(screen, points[0], points[1])

            screen.blit(fps, (10, 10))
            pygame.display.flip()
        count += 1


pygame.display.quit()
pygame.quit()
