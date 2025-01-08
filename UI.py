import wx
import cv2
import os
import numpy as np
from PIL import Image
import io

class MyFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="基于隐显视图关联的光场语义分割界面展示", size=(1950, 1300))
        panel = wx.Panel(self)

        # 模型选择按钮
        self.model_choices=["hr48 on UrbanLF_Real", "hr48 on UrbanLF_Syn", "r50 on UrbanLF_Real","r50 on UrbanLF_Syn"]
        self.model_choice_label = wx.StaticText(panel, label="Model Selection:")
        self.model_choice = wx.Choice(panel, choices=self.model_choices)
        self.selected_model = None

        # 图片选择按钮
        self.select_image_label = wx.StaticText(panel,label="Image Selection:")
        self.select_image_button = wx.Button(panel, label="Select Image")

        # 图片显示区域
        self.label1 = wx.StaticText(panel,label="before processing: ")
        self.label2 = wx.StaticText(panel,label="after processing: ")
        self.image_display = wx.StaticBitmap(panel)
        self.image_res = wx.StaticBitmap(panel)

        # 设置窗口中的组件布局
        sizer_widgets = wx.BoxSizer(wx.VERTICAL)
        sizer_widgets.Add(self.model_choice_label, 0, wx.ALL, 5)
        sizer_widgets.Add(self.model_choice, 0, wx.ALL, 5)
        sizer_widgets.Add(self.select_image_label, 0, wx.ALL, 5)
        sizer_widgets.Add(self.select_image_button, 0, wx.ALL, 5)
        #panel.SetSizer(sizer_widgets)

        # 提示布局
        sizer_label = wx.BoxSizer(wx.HORIZONTAL)
        sizer_label.Add(self.label1, 0, wx.ALL | wx.EXPAND, 10)
        sizer_label.Add((620, 5), 0, wx.ALL | wx.EXPAND, 5)
        sizer_label.Add(self.label2, 0, wx.ALL | wx.EXPAND, 10)

        # 设置图片布局
        sizer_images = wx.BoxSizer(wx.HORIZONTAL)
        sizer_images.Add(self.image_display, 1, wx.LEFT | wx.CENTER, 5)
        sizer_images.Add(self.image_res, 1, wx.LEFT | wx.CENTER, 5)
        #panel.SetSizer(sizer_images)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(sizer_widgets,0, wx.EXPAND | wx.ALL, 10)
        sizer.Add(sizer_label,0, wx.EXPAND | wx.ALL, 10)
        sizer.Add(sizer_images,0, wx.EXPAND | wx.LEFT, 20)

        panel.SetSizer(sizer)

        # 绑定选择图片按钮事件
        self.select_image_button.Bind(wx.EVT_BUTTON, self.on_select_image)
        # 绑定模型选择事件
        self.model_choice.Bind(wx.EVT_CHOICE,self.on_model_choice)

    def on_model_choice(self, event):
        self.selected_model = self.model_choices[self.model_choice.GetSelection()]
        wx.MessageBox(f"选择了模型: {self.selected_model}", "模型选择")

    def on_select_image(self, event):

        if self.selected_model == "hr48 on UrbanLF_Real":
            defaultDirectory = "D:\\LF-IENet\\data\\UrbanLF\\UrbanLF_Real\\test"
        elif self.selected_model == "hr48 on UrbanLF_Syn":
            defaultDirectory = "D:\\LF-IENet\\data\\UrbanLF\\UrbanLF_Syn\\test"
        elif self.selected_model == "r50 on UrbanLF_Real":
            defaultDirectory = "D:\\LF-IENet\\data\\UrbanLF\\UrbanLF_Real\\test"
        elif self.selected_model == "r50 on UrbanLF_Syn":
            defaultDirectory = "D:\\LF-IENet\\data\\UrbanLF\\UrbanLF_Syn\\test"
        elif self.selected_model == None:
            wx.MessageBox('请选择模型', '提示', wx.OK | wx.ICON_INFORMATION)
            return


        # 打开文件对话框，选择图片
        wildcard = "Image files (*.jpg;*.png)|*.jpg;*.png"
        dialog = wx.FileDialog(self, "Choose an image file", wildcard=wildcard, style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST, defaultDir=defaultDirectory)
        if dialog.ShowModal() == wx.ID_OK:
            filepath = dialog.GetPath()
            dialog.Destroy()

            # 调用模型进行语义分割预测
            # 这里只是一个示例，您需要根据实际情况替换为您的模型预测代码
            original_image = self.predict_segmentation(filepath)
            segmented_image = None
            root_path = None

            if self.selected_model == "hr48 on UrbanLF_Real":
                root_path = "D:\\LF-IENet\\img_res_hr48_Real\\"
            elif self.selected_model == "hr48 on UrbanLF_Syn":
                root_path = "D:\\LF-IENet\\img_res_hr48_Syn\\"
            elif self.selected_model == "r50 on UrbanLF_Real":
                root_path = "D:\\LF-IENet\\img_res_r50_Real\\"
            elif self.selected_model == "r50 on UrbanLF_Syn":
                root_path = "D:\\LF-IENet\\img_res_r50_Syn\\"


            path_parts = filepath.split('\\')
            image_name = path_parts[-2] + ".png"
            res_path = root_path + image_name
            segmented_image = self.predict_segmentation(res_path)

            # 在窗口中显示预测结果
            if segmented_image is not None:
                self.display_image(original_image)
                self.display_image_res(segmented_image)

    def predict_segmentation(self, image_path):
        # 这里只是一个示例，实际上需要根据您的模型进行预测
        # 以下代码仅用于演示
        image = cv2.imread(image_path)
        if image is not None:
            # 在这里进行语义分割预测
            # 这里只是一个示例，您需要将预测结果返回
            # 以下代码仅用于演示，返回原始图片
            return image
        else:
            return None

    def display_image(self, image):
        # 将OpenCV图像转换为wxPython图像格式
        height, width = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = wx.Bitmap.FromBuffer(width, height, image)

        # 将图像调整为适应窗口大小
        self.image_display.SetBitmap(image)
        self.Layout()

    def display_image_res(self, image):
        # 将OpenCV图像转换为wxPython图像格式
        height, width = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = wx.Bitmap.FromBuffer(width, height, image)

        # 将图像调整为适应窗口大小
        self.image_res.SetBitmap(image)
        self.Layout()

if __name__ == "__main__":
    app = wx.App()
    frame = MyFrame()
    frame.Show()
    app.MainLoop()
