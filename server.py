import wx
from time import sleep
import cv2
import socket
import numpy as np
import _pickle as pickle
from threading import Thread
from wx.lib.pubsub import pub as Publisher
import ctypes

import cv2
import numpy as np
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses, get_similarity, get_similarity_score, get_max_human
from val import normalize, pad_width

user32 = ctypes.windll.user32


class SocketThread(Thread):
    """Test Worker Thread Class."""

    def __init__(self):
        """Init Worker Thread Class."""
        Thread.__init__(self)
        self.addr = ("localhost", 6000)

    def run(self):
        """Run Worker Thread."""
        # This is the code executing in the new thread.
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        server.bind(self.addr)
        server.listen(10)

        while True:
            conn, addr = server.accept()
            clientMessage = str(conn.recv(1024), encoding='utf-8')

            print('Client message is:', clientMessage)

            serverMessage = 'I\'m here!'
            conn.sendall(serverMessage.encode())

            Publisher.sendMessage, "update", clientMessage

        conn.close()


class ShowCapture(wx.Panel):
    def __init__(self, parent, capture, teachermod=0, fps=30):
        wx.Panel.__init__(self, parent)
        self.capture = capture
        self.Linewidth = 3
        self.moveerror = 50
        self.teachermod = teachermod
        if not teachermod:
            self.Backpanel = wx.Panel(parent)
        ret, frame = self.capture.read()
        mag_X = self.Size[0]/frame.shape[1]
        mag_Y = self.Size[1]/frame.shape[0]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, None, fx=mag_X, fy=mag_Y)
        self.bmp = wx.Bitmap.FromBuffer(
            frame.shape[1], frame.shape[0], frame)

        self.timer = wx.Timer(self)
        self.timer.Start(1000./fps)

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_TIMER, self.NextFrame)
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnErase)

        self.Bind(wx.EVT_CLOSE, self.OnClose)
        if not teachermod:
            self.changebackcolor(1)
        Publisher.subscribe(lambda self, msg: print(msg), "update")

    # def ResizeCapture(self, width, height):
    #     self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    #     self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # def scale_bitmap(self, bitmap, width, height):
    #     image = wx.ImageFromBitmap(bitmap)
    #     image = image.Scale(width, height, wx.IMAGE_QUALITY_HIGH)
    #     result = wx.BitmapFromImage(image)
    #     return result
    def changebackcolor(self, mode):
        if mode == 1:
            self.Backpanel.SetBackgroundColour("green")
        else:
            self.Backpanel.SetBackgroundColour("red")

    def setsize(self, size):
        if not self.teachermod:
            self.SetSize(size)
            self.Backpanel.SetSize(
                wx.Size(size[0]+self.Linewidth*2, size[1]+self.Linewidth*2))
        else:
            self.SetSize(size)

    def setposition(self, position):
        if not self.teachermod:
            self.SetPosition(
                wx.Point(position[0]-self.Linewidth, position[1]-self.Linewidth))
            self.Backpanel.SetPosition(
                wx.Point(position[0]-self.Linewidth*2, position[1]-self.Linewidth*2))
        else:
            self.SetPosition(position)

    def OnClose(self, event):
        self.thread.terminate()
        self.capture.release()
        self.Destroy()

    def OnErase(self, event):
        # Do nothing, reduces flicker by removing
        # unneeded background erasures and redraws
        pass

    def OnPaint(self, evt):
        dc = wx.BufferedPaintDC(self)

        dc.DrawBitmap(self.bmp, 0, 0)
        self.Layout()

    def Gettailposition(self):
        return wx.Point(self.Position[0]+self.Size[0]+self.Linewidth*2, 0)

    def NextFrame(self, event):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mag_X = self.Size[0]/frame.shape[1]
            mag_Y = self.Size[1]/frame.shape[0]
            frame = cv2.resize(frame, None, fx=mag_X, fy=mag_Y)
            self.bmp = wx.Bitmap.FromBuffer(
                frame.shape[1], frame.shape[0], frame)
            # self.bmp.CopyFromBuffer(frame)
            self.Refresh()


if __name__ == '__main__':

    capture = None
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)

    # capture2 = cv2.VideoCapture(2)
    # capture2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # capture2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # capture3 = cv2.VideoCapture(3)
    # capture3.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # capture3.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    app = wx.App()
    frame = wx.Frame(None)
    frame.SetBackgroundColour((255, 250, 240))
    frame.Maximize(True)
    teachercap = ShowCapture(frame, capture, teachermod=1)
    mainstudentcap = ShowCapture(frame, capture)
    secondstudentcap = ShowCapture(frame, capture)
    thirdstudentcap = ShowCapture(frame, capture)

    frame.Show()
    socketthread = SocketThread()
    socketthread.setDaemon(True)
    socketthread.start()
    teachercap.setsize(wx.Size((1280//2.3, 720//2.4)))
    teachercap.setposition(
        wx.Point(screen_width-teachercap.Size[0], screen_height-teachercap.Size[1]-20))
    mainstudentcap.setsize(wx.Size((1280//1.8, 720//1.9)))
    mainstudentcap.setposition(
        wx.Point(0+mainstudentcap.Linewidth*2, screen_height-mainstudentcap.Size[1]-22))
    secondstudentcap.setsize(wx.Size((1280//3.65, 720//2.7)))
    secondstudentcap.setposition(
        wx.Point(0+mainstudentcap.Linewidth*2, 0+mainstudentcap.Linewidth*2+15))
    thirdstudentcap.setsize(wx.Size((1280//3.65, 720//2.7)))
    thirdstudentcap.setposition(
        wx.Point(0+mainstudentcap.Linewidth*2+secondstudentcap.Gettailposition()[0], 0+mainstudentcap.Linewidth*2+secondstudentcap.Gettailposition()[1]+15))
    thirdstudentcap.changebackcolor(0)
app.MainLoop()
