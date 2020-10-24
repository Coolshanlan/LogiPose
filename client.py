import wx 
from time import sleep
import cv2 
import socket
import numpy as np
import _pickle as pickle
from threading import Thread 
from wx.lib.pubsub import pub as Publisher

class SocketThread(Thread): 
    """Test Worker Thread Class."""
        
    def __init__(self): 
        """Init Worker Thread Class."""
        Thread.__init__(self)     
        self.addr = ("localhost", 6000)
        
    def run(self): 
        """Run Worker Thread."""
        # This is the code executing in the new thread. 
        clientMessage = "ABCDEFG"

        while clientMessage:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            client.connect(self.addr)
            client.sendall(clientMessage.encode())        

            serverMessage = str(client.recv(1024), encoding='utf-8')
            print('Server:', serverMessage)
            client.close()
            
            Publisher.sendMessage, "update", serverMessage

        

class ShowCapture(wx.Panel): 
    def __init__(self, parent, capture, fps=30): 
        wx.Panel.__init__(self, parent) 

        if capture:
            self.capture = capture 
            self.frame = None
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

        self.thread = SocketThread()
        self.thread.setDaemon(True)
        self.thread.start()
        
        Publisher.subscribe(lambda self, msg: print(msg), "update")

    def OnErase(self, event):
        # Do nothing, reduces flicker by removing
        # unneeded background erasures and redraws
        pass

    def OnPaint(self, evt):
        dc = wx.BufferedPaintDC(self) 
        dc.DrawBitmap(self.bmp, 0, 0) 

    def NextFrame(self, event): 
        ret, self.frame = self.capture.read() 
        if ret: 
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB) 
            self.bmp.CopyFromBuffer(self.frame) 
            self.Refresh()
        

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