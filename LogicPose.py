import wx


class Frame(wx.Frame):
    def __init__(self, title_name="defult", childframe=None):
        super(Frame, self).__init__(parent=None,
                                    title=title_name, size=(200, 200))
        self.Center()
        self.size = (500,500)
        self.child = childframe
        self.panel = wx.Panel(self)
        self.btn = wx.Button(pos=(0, 0), size=(100, 100), parent=self.panel)
        self.Bind(wx.EVT_BUTTON, self.button_click, self.btn)

    def button_click(self, event):
        if self.child:
            self.child.Close()
        print("I'm Winner")


app = wx.App()
frame1 = Frame()
frame1.Show()
app.MainLoop()
