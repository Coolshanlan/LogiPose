import wx


class ImageBox(wx.Panel):
    def __init__(self, parent, filepath, id):
        wx.Panel.__init__(self, parent, id)
        imageFile = filepath
        jpg1 = wx.Image(imageFile, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.SetSize(jpg1.GetWidth(), jpg1.GetHeight())
        wx.StaticBitmap(self, -1, jpg1)

        # wx.StaticBitmap(self, -1, jpg1, (10 + jpg1.GetWidth(), 5),
        #                 (jpg1.GetWidth(), jpg1.GetHeight()))
    def changeImage(self, filepath):
        jpg1 = wx.Image(filepath, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        wx.StaticBitmap(self, -1, jpg1)


class Frame(wx.Frame):
    def __init__(self, title_name="defult", childframe=None):
        super(Frame, self).__init__(parent=None,
                                    title=title_name, size=(200, 200))
        # self.Center()
        self.SetSize(1000, 1000)
        self.Maximize(True)
        self.SetBackgroundColour('white')
        self.child = childframe

        self.createImageBox()

    def createImageBox(self):
        self.imagebox1 = ImageBox(self, "data\\sport.jpg", -1)
        self.imagebox1.SetPosition((100, 100))
        self.imagebox1.SetSize((100, 100))

    def button_click(self, event):
        self.imagebox1.changeImage("data\\test.jpg")


app = wx.App()
frame1 = Frame()

# frame1.SetSize(1000,1000)
frame1.Show()
app.MainLoop()
