import wx
import cv2
from threading import Thread
import ctypes
import threading
import cv2
import numpy as np
import torch
from wx.core import Position

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses, get_similarity, get_similarity_score, get_max_human
from val import normalize, pad_width

user32 = ctypes.windll.user32


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale,
                            fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(
        scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(
        2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(
        stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio,
                          fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio,
                      fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad

class ShowSetting(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.SetBackgroundColour((11, 11, 11))
        self.text1 = wx.StaticText(self, label='Nature', pos=(20,0+20))
        self.text2 = wx.StaticText(self, label='Nature', pos=(20,100+20))
        self.text3 = wx.StaticText(self, label='Nature', pos=(20,200+20))
        font = wx.Font(28, wx.FONTFAMILY_ROMAN, wx.FONTSTYLE_ITALIC, wx.FONTWEIGHT_BOLD, False, 'Arial')
        self.text1.SetFont(font)
        self.text1.SetForegroundColour(wx.Colour(255, 0, 0))
        self.text2.SetFont(font)
        self.text2.SetForegroundColour(wx.Colour(255, 0, 0))
        self.text3.SetFont(font)
        self.text3.SetForegroundColour(wx.Colour(255, 0, 0))

class ShowCapture(wx.Panel):
    def __init__(self, parent, capture, teachermod=0, fps=30):
        wx.Panel.__init__(self, parent)
        self.capture = capture
        self.Linewidth = 3
        self.moveerror = 50
        self.teachermod = teachermod
        if not teachermod:
            self.Backpanelred = wx.Panel(parent)
            self.Backpanel = wx.Panel(parent)
            self.Backpanelred.SetBackgroundColour('red')
        ret, frame = self.capture.read()
        mag_X = self.Size[0]/frame.shape[1]
        mag_Y = self.Size[1]/frame.shape[0]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, None, fx=mag_X, fy=mag_Y)
        self.current_pose = None
        self.frametmp = frame
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
            self.Refresh()
            self.Backpanel.Show()
            self.Backpanel.Layout()
        else:
            self.Backpanel.SetBackgroundColour("red")
            self.Backpanel.Hide()
            self.Backpanel.Layout()
        # self.Backpanel.Refresh()

        # self.Layout()
        # self.Backpanel.Layout()
        # self.Refresh()
        # self.Backpanel.Show()
        # self.Backpanel.Layout()
        # self.Layout()

    def getPose(self, height_size=256, stride=8, upsample_ratio=4, num_keypoints=18):
        heatmaps, pafs, scale, pad = infer_fast(
            net, self.frametmp, height_size, stride, upsample_ratio, 0)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(
                heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(
            all_keypoints_by_type, pafs, demo=True)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (
                all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (
                all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(
                        all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(
                        all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)
        if current_poses != [] and len(current_poses) != 0:
            self.current_pose = get_max_human(current_poses)
        else:
            self.current_pose = None

    def setsize(self, size):
        if not self.teachermod:
            self.SetSize(size)
            self.Backpanel.SetSize(
                wx.Size(size[0]+self.Linewidth*2, size[1]+self.Linewidth*2))
            self.Backpanelred.SetSize(
                wx.Size(size[0]+self.Linewidth*2, size[1]+self.Linewidth*2))
        else:
            self.SetSize(size)

    def setposition(self, position):
        if not self.teachermod:
            self.SetPosition(
                wx.Point(position[0]-self.Linewidth, position[1]-self.Linewidth))
            self.Backpanel.SetPosition(
                wx.Point(position[0]-self.Linewidth*2, position[1]-self.Linewidth*2))
            self.Backpanelred.SetPosition(
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

    def Gettailposition(self):
        return wx.Point(self.Position[0]+self.Size[0]+self.Linewidth*2, 0)

    def NextFrame(self, event):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mag_X = self.Size[0]/frame.shape[1]
            mag_Y = self.Size[1]/frame.shape[0]
            self.frametmp = frame
            frame = cv2.resize(frame, None, fx=mag_X, fy=mag_Y)
            self.bmp = wx.Bitmap.FromBuffer(
                frame.shape[1], frame.shape[0], frame)
            # self.bmp.CopyFromBuffer(frame)
            # self.getPose()
            self.Refresh()


def caculate_pose(threshold=25):
    while(True):
        mainstudentcap.getPose()
        secondstudentcap.getPose()
        thirdstudentcap.getPose()
        teachercap.getPose()
        if(teachercap.current_pose == None):
            continue
        if(mainstudentcap.current_pose != None):
            score = get_similarity_score(
                teachercap.current_pose, mainstudentcap.current_pose)[2]
            print("first {}".format(score))
            if score < threshold:
                mainstudentcap.changebackcolor(0)
            else:
                mainstudentcap.changebackcolor(1)
        else:
            mainstudentcap.changebackcolor(0)
        if(secondstudentcap.current_pose != None):
            score = get_similarity_score(
                teachercap.current_pose, secondstudentcap.current_pose)[2]
            print("second {}".format(score))
            if score < threshold:
                secondstudentcap.changebackcolor(0)
            else:
                secondstudentcap.changebackcolor(1)
        else:
            secondstudentcap.changebackcolor(0)
        if(thirdstudentcap.current_pose != None):
            score = get_similarity_score(
                teachercap.current_pose, thirdstudentcap.current_pose)[2]
            print("thrid {}".format(score))
            if score < threshold:
                thirdstudentcap.changebackcolor(0)
            else:
                thirdstudentcap.changebackcolor(1)
        else:
            thirdstudentcap.changebackcolor(0)


if __name__ == '__main__':
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(
        "checkpoint\\checkpoint_iter_370000.pth", map_location='cpu')
    load_state(net, checkpoint)
    net = net.cuda()
    net = net.eval()
    capture = None
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)

    capture2 = cv2.VideoCapture(2)
    capture2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # capture3 = cv2.VideoCapture(3)
    # capture3.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # capture3.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    app = wx.App()
    frame = wx.Frame(None)
    frame.SetBackgroundColour((255, 250, 240))
    frame.Maximize(True)
    teachercap = ShowCapture(frame, capture, teachermod=1)
    mainstudentcap = ShowCapture(frame, capture2)
    secondstudentcap = ShowCapture(frame, capture2)
    thirdstudentcap = ShowCapture(frame, capture)
    setting = ShowSetting(frame)

    frame.Show()
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
    setting.SetSize(wx.Size((1280//2.3, screen_height-720//2.4)))
    setting.SetPosition(
        wx.Point(screen_width-setting.Size[0], 0 + 15))
    cp = threading.Thread(target=caculate_pose)
    cp.start()

    app.MainLoop()
