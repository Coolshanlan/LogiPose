import wx 
from time import sleep
import cv, cv2 

class ShowCapture(wx.Panel): 
    def __init__(self, parent, capture, fps=30): 
        wx.Panel.__init__(self, parent) 

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

if __name__ == '__main__':
    capture = cv2.VideoCapture(2) 
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 

    app = wx.App() 
    frame = wx.Frame(None) 
    cap = ShowCapture(frame, capture) 
    frame.Show() 
    app.MainLoop() 