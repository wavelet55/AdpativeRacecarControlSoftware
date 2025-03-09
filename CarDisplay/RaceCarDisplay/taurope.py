# Randy Direen
# 8/31/2018
#
# Display to be shown within the car.
#
# Displays orientation of head and current state of Videre

# -------------------------------------------------------------------Built-ins
import asyncio
import signal
import math
import os

# -------------------------------------------------------------------3rd Party
import zmq
from zmq.asyncio import Context
import pygame
import quaternion as qt
# -------------------------------------------------------------------------GPB
from vision_messages_pb2 import VisionMessageWrapperPBMsg
from vision_messages_pb2 import TrackHeadOrientationPBMsg
from vision_messages_pb2 import SipAndPuffPBMsg
from vision_messages_pb2 import VidereSystemControlPBMsg
# ----------------------------------------------------------------------Custom
import measured_structs as ms
import head_model_3d as hm

# ****************************************************************************
#                          Handle Ctrl+c
# ****************************************************************************
keep_running = True

def shandler(sig, frame):
    global keep_running
    print("Shutting Down:")
    # loop.call_soon_threadsafe(loop.stop)
    keep_running = False


signal.signal(signal.SIGINT, shandler)

# ****************************************************************************
#                          Widgets
# ****************************************************************************
class Background(pygame.sprite.Sprite):
    def __init__(self, image_file, location):
        pygame.sprite.Sprite.__init__(self)  #call Sprite initializer
        self.image = pygame.image.load(image_file)
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location

class HeadIndicator(pygame.sprite.Sprite):
    def __init__(self, image_file, location):
        pygame.sprite.Sprite.__init__(self)  #call Sprite initializer
        self.image = pygame.image.load(image_file)
        self.size = self.image.get_size()
        self.image = pygame.transform.scale(self.image,
                                            (int(self.size[0] * 1),
                                             int(self.size[1] * 1)))
        # draw bigger image to screen at x=100 y=100 position
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location

    def rot(self, angle):
        if angle > 70:
            angle = 70
        elif angle < -70:
            angle = -70

        orig_rect = self.image.get_rect()
        rot_image = pygame.transform.rotate(self.image, angle)
        rot_rect = orig_rect.copy()
        rot_rect.center = rot_image.get_rect().center
        rot_image = rot_image.subsurface(rot_rect).copy()
        return rot_image

    def draw(self, screen, angle):
        head_sprite = self.rot(angle*180.0/math.pi )
        size = head_sprite.get_size()
        head_sprite = pygame.transform.scale(head_sprite,
                                             (int(size[0] * .75),
                                              int(size[1] * 1)))
        screen.blit(head_sprite, self.rect)

class UncertaintyIndicator(pygame.sprite.Sprite):
    def __init__(self, location):
        self.uloc = location
        uw = 12
        uh = 12
        self.upoints = [(-uw + self.uloc[0], uh + self.uloc[1] + 1),
                        (uw + self.uloc[0], uh + self.uloc[1] + 1),
                        (uw + self.uloc[0], -uh + self.uloc[1]),
                        (-uw + self.uloc[0], -uh + self.uloc[1])]
        self.rd = (196, 39, 39)
        self.yl = (244, 244, 46)
        self.gn = (53, 244, 46)

    def draw(self, screen, font, uncertainty):
        if uncertainty < 150:
            pygame.draw.polygon(screen, self.gn, self.upoints)
        elif uncertainty < 200:
            pygame.draw.polygon(screen, self.yl, self.upoints)
        else:
            pygame.draw.polygon(screen, self.rd, self.upoints)

        uncert = font.render("{0:5}".format(uncertainty), True, (240, 240, 240))
        screen.blit(uncert, (self.uloc[0] - 75 , self.uloc[1] - 8))

class SipPuffIndicator(pygame.sprite.Sprite):
    def __init__(self, image_file, location):
        pygame.sprite.Sprite.__init__(self)  #call Sprite initializer
        self.image = pygame.image.load(image_file)
        self.size = self.image.get_size()
        self.image = pygame.transform.scale(self.image,
                                             (int(self.size[0] * 1),
                                              int(self.size[1] * 1)))
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location

    def draw(self, screen, val):
        indicator_location = 0
        if val >= 0:
            indicator_location = -147*float(val)/100.0
        if val < 0:
            indicator_location = -162*float(val)/100.0

        screen.blit(self.image, (self.rect.left, self.rect.top + indicator_location))

class StateTextIndicator:
    def __init__(self, location):
        self.uloc = location
        uw = 11
        self.upoints = [(-uw + self.uloc[0], 21 + self.uloc[1]),
                        (139 + self.uloc[0], 21 + self.uloc[1]),
                        (139 + self.uloc[0], -2 + self.uloc[1]),
                        (-uw + self.uloc[0], -2 + self.uloc[1])]
        self.blue = (0x3e, 0xb9, 0xcc)
        self.gray = (100, 100, 100)
        self.gn = (53, 244, 46)

    def draw(self, screen, font, text):
        if text == "BCI ON":
            pygame.draw.polygon(screen, self.gn, self.upoints)
        elif text == "BCI OFF":
            pygame.draw.polygon(screen, self.blue, self.upoints)
        else:
            pygame.draw.polygon(screen, self.gray, self.upoints)

        uncert = font.render(text, True, (240, 240, 240))
        screen.blit(uncert, (self.uloc[0], self.uloc[1]))

class StateIndicator(pygame.sprite.Sprite):
    def __init__(self, image_file, location):
        pygame.sprite.Sprite.__init__(self)  #call Sprite initializer
        self.image = pygame.image.load(image_file)
        self.size = self.image.get_size()
        self.image = pygame.transform.scale(self.image,
                                            (int(self.size[0] * 1.0),
                                             int(self.size[1] * 1.0)))
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location

    def draw(self, screen, val):
        if val == False:
             screen.blit(self.image, self.rect)

# ****************************************************************************
#                      Get info from Videre and display
# ****************************************************************************
class NascarDisplay:
    def __init__(self, ip_address_videre, mode):

        self.mode = mode

        self.loop = asyncio.get_event_loop()

        # Setup ZeroMQ for Asyncio
        self.ctx = Context.instance()
        self.socket = None
        self.socket2 = None
        self.address = ip_address_videre + ':5556'
        self.address2 = ip_address_videre + ':5559'

        self.msg_gpb = VisionMessageWrapperPBMsg()
        self.head_gpb = TrackHeadOrientationPBMsg()
        self.sippuff = SipAndPuffPBMsg()
        self.systemc = VidereSystemControlPBMsg()

        # Parameters to be displayed
        self.head = ms.VisionData()
        aa = 1/math.sqrt(2)
        q1 = qt.quaternion(aa, aa, 0, 0)
        q2 = qt.quaternion(aa, 0, aa, 0)
        self.new_coordinates_q = q2*q1
        self.orientation = qt.quaternion(1, 0, 0, 0)
        self.uncertainty = 0
        self.sippuffp = 0
        self.sippuffi = 0
        self.system_state = 99
        self.states = {99: "None", 1: "RC", 3: "Manual", 5: "Engaged"}

        # Graphics, widgets and things to be drawn on the screen
        pygame.init()
        pygame.display.set_caption('DireenTech Car Display')
        self.size = (640, 480)

        if self.mode == "FullScreen":
            self.screen = pygame.display.set_mode(self.size, pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(self.size)

        self.currentWorkingDir = os.getcwd()
        self.imagesDir = self.currentWorkingDir
        cdispIdx = self.currentWorkingDir.rfind('CarDisplay')
        lastSlashIdx = self.currentWorkingDir.rfind('/')
        if lastSlashIdx > cdispIdx:
            self.imagesDir = self.currentWorkingDir[0:lastSlashIdx]
        self.imagesDir = self.imagesDir + '/images/'
        imgfilename = self.imagesDir + 'NascarDisplay_BACKGROUND.png'
        self.background = Background(imgfilename, [0, 0])

        imgfilename = self.imagesDir + 'HeadDirection_INDICATOR.png'
        self.head_indicator = HeadIndicator(imgfilename, [4, 70])
                                          
        imgfilename = self.imagesDir + 'Throttle_INDICATOR.png'
        self.sip_puff_indicator = SipPuffIndicator(imgfilename, [340, 211])
                                            
        self.state_text_indicator = StateTextIndicator([475, 285])

        imgfilename = self.imagesDir + 'Disarmed_BUTTON.png'
        self.state_indicator = StateIndicator(imgfilename, [2, 258])

        self.uncertainty_indicator = UncertaintyIndicator([603, 254])

        pygame.display.init()
        pygame.display.update()
        self.clock = pygame.time.Clock()
        self.head_model = hm.HeadModel(self.screen.get_width(),
                                       self.screen.get_height(),
                                       .75,
                                       (197, -107))
        self.font = pygame.font.Font(None, 30)

    def run(self):

        group = asyncio.gather(self.recv(),
                               self.recv2(),
                               self.draw())
        try:
            self.loop.run_until_complete(group)
        except asyncio.CancelledError:
            print('Cancelled Asyncio Process')

    # ------------------------------------------------------------------------
    # Receive messages from Videre and handle them
    # ------------------------------------------------------------------------
    async def recv(self):
        global keep_running
        self.socket = self.ctx.socket(zmq.SUB)
        self.socket.connect(self.address)
        self.socket.subscribe(b'')
        while keep_running:
            msg = await self.socket.recv_multipart()
            self.msg_gpb.ParseFromString(msg[0])
            self._handle_messages(self.msg_gpb)

        self.socket.close()
        print("Exiting: recv_orientation")

    def _handle_messages(self, pmsg):
        if self.msg_gpb.MsgName == "TrackHeadOrientationMsg":
            self._handle_track_head_orientation_msg(pmsg)

    def _handle_track_head_orientation_msg(self, pmsg):
        self.head_gpb.ParseFromString(pmsg.MsgData)
        self.head.from_gpb(self.head_gpb)
        self.orientation = self.head.q
        self.uncertainty = int(self.head.cov_norm*1e10)

    async def recv2(self):
        global keep_running
        self.socket2 = self.ctx.socket(zmq.SUB)
        self.socket2.connect(self.address2)
        self.socket2.subscribe(b'')
        while keep_running:
            msg = await self.socket2.recv_multipart()
            self.msg_gpb.ParseFromString(msg[0])
            self._handle_messages2(self.msg_gpb)

        self.socket2.close()
        print("Exiting: recv_orientation")

    def _handle_messages2(self, pmsg):
        if self.msg_gpb.MsgName == "SipAndPuffStatusMsg":
            self._handle_sip_and_puff_msg(pmsg)
        if self.msg_gpb.MsgName == "VidereSystemStatusMsg":
            self._handle_system_control(pmsg)


    def _handle_sip_and_puff_msg(self, pmsg):
        self.sippuff.ParseFromString(pmsg.MsgData)
        self.sippuffp = self.sippuff.SipAndPuffPecent
        self.sippuffi = self.sippuff.SipAndPuffIntegralPercent

    def _handle_system_control(self, pmsg):
        self.systemc.ParseFromString(pmsg.MsgData)
        self.system_state = self.systemc.SystemState
        self.NexusBCIControlEnabled = self.systemc.NexusBCIControlEnabled
        self.BCIThrottleEnable = self.systemc.NexusBCIThrottleEnable

    # ------------------------------------------------------------------------
    # Draw the UI using the received messages
    # ------------------------------------------------------------------------
    async def draw(self):
        global keep_running
        while keep_running:

            self._handle_display_events()

            self._draw_background(self.screen)

            self._draw_fps(self.screen)

            self._draw_steering_orientation(self.screen)

            self._draw_head_orientation(self.screen)

            self._draw_uncertainty_widget(self.screen)

            self._draw_sip_puff_indicator(self.screen)

            self._draw_state_text_indicator(self.screen)
            self._draw_state_indicator(self.screen)

            pygame.display.flip()

            await asyncio.sleep(0.01)

        print("Exiting: draw stopped")

    def _handle_display_events(self):
        global keep_running
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                keep_running = False
                self.socket.close()
                self.socket2.close()
            if (event.type is pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                keep_running = False
                self.socket.close()
                self.socket2.close()
            if (event.type is pygame.KEYDOWN and event.key == pygame.K_f):
                pygame.display.set_mode(self.size, pygame.FULLSCREEN)
            if (event.type is pygame.KEYDOWN and event.key == pygame.K_w):
                pygame.display.set_mode(self.size)

    def _draw_background(self, screen):
        screen.fill([255, 255, 255])
        screen.blit(self.background.image, self.background.rect)

    def _draw_fps(self, screen):
        dt = self.clock.tick()
        fps = self.font.render(str(int(self.clock.get_fps())), True,
                               (100, 100, 100))
        screen.blit(fps, (600, 440))

    def _draw_steering_orientation(self, screen):
        eu =  qt.as_euler_angles(self.new_coordinates_q*self.orientation)
        self.head_indicator.draw(screen, eu[2])

    def _draw_head_orientation(self, screen):
        q = self.new_coordinates_q*self.orientation
        ov = self.head_model.oriented_vertices(q)
        for points in ov:
            pygame.draw.polygon(screen, points[0], points[1])

    def _draw_uncertainty_widget(self, screen):
        self.uncertainty_indicator.draw(self.screen, self.font,
                                        self.uncertainty)

    def _draw_sip_puff_indicator(self, screen):
        if self.system_state == 5:
            self.sip_puff_indicator.draw(screen, self.sippuffi)
        else:
            self.sip_puff_indicator.draw(screen, self.sippuffp)

    def _draw_state_text_indicator(self, screen):
        try:
            if self.NexusBCIControlEnabled:
                if self.BCIThrottleEnable:
                    self.state_text_indicator.draw(screen, self.font, "BCI ON")
                else:
                    self.state_text_indicator.draw(screen, self.font, "BCI OFF")
            else:
                self.state_text_indicator.draw(screen, self.font, self.states[self.system_state])
        except:
            self.state_text_indicator.draw(screen, self.font, "UNKNOWN")

    def _draw_state_indicator(self, screen):
        if self.system_state == 5:
            self.state_indicator.draw(screen, True)
        else:
            self.state_indicator.draw(screen, False)

if __name__ == "__main__":
    import sys
    mode = "Normal"
    ipaddr = "127.0.0.1"
    if len(sys.argv[1:]) > 0:
        if sys.argv[1:][0] == '-F':
            print("Full")
            mode = "FullScreen"
    if len(sys.argv) > 2:
        ipaddr = sys.argv[2:][0]
    
    print("IP Address: ", ipaddr)
    ipaddr = "tcp://" + ipaddr
    display = NascarDisplay(ipaddr, mode)
    display.run()
    loop = asyncio.get_event_loop()
    loop.close()
    print("#End Of Program#")

