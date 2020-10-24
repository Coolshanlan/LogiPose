import wx 
from time import sleep
import cv2 
import socket
import numpy as np
import _pickle as pickle
import threading

class ShowCapture(wx.Panel): 
    def __init__(self, parent, capture, fps=30): 
        wx.Panel.__init__(self, parent) 

        if capture:
            self.capture = capture 
            ret, frame = self.capture.read() 

            height, width = frame.shape[:2] 
            parent.SetSize((width, height)) 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

            self.bmp = wx.Bitmap.FromBuffer(width, height, frame) 

            self.timer = wx.Timer(self) 
            self.timer.Start(1000./fps)

            self.Bind(wx.EVT_PAINT, self.OnPaint) 
            self.Bind(wx.EVT_TIMER, self.NextFrame) 
            self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnErase)

        self.Bind(wx.EVT_CLOSE,self.OnClose)

        self.addr = ("localhost", 6000)
        self.thread = threading.Thread(target=self.Socket)
        self.thread.start()

    def OnClose(self,event):
        self.thread.terminate()
        self.Destroy()

    def OnErase(self, event):
        # Do nothing, reduces flicker by removing
        # unneeded background erasures and redraws
        pass

    def OnPaint(self, evt):
        dc = wx.BufferedPaintDC(self) 
        dc.DrawBitmap(self.bmp, 0, 0) 

    def NextFrame(self, event): 
        ret, frame = self.capture.read() 
        if ret: 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            self.bmp.CopyFromBuffer(frame) 
            self.Refresh()

    def Socket(self):
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
            conn.close()

if __name__ == '__main__':
    capture = None
    # capture = cv2.VideoCapture(2) 
    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 

    app = wx.App() 
    frame = wx.Frame(None) 
    cap = ShowCapture(frame, capture) 
    frame.Show() 
    app.MainLoop() 