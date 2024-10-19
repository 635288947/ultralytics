import tkinter as tk
from tkinter import filedialog, messagebox
import os
from ultralytics import YOLO

# 创建主窗口
root = tk.Tk()
root.title("YOLO Image Object Detection")
root.geometry("400x200")

# 变量存储上传文件路径和保存路径
uploaded_image_path = tk.StringVar()
save_location = tk.StringVar()


# 选择上传的图片文件
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        uploaded_image_path.set(file_path)
    else:
        messagebox.showwarning("Warning", "No image selected!")


# 选择保存结果的路径
def select_save_location():
    folder_path = filedialog.askdirectory()
    if folder_path:
        save_location.set(folder_path)
    else:
        messagebox.showwarning("Warning", "No save location selected!")


# 运行 YOLO 模型进行目标检测
def run_yolo_model():
    if not uploaded_image_path.get():
        messagebox.showerror("Error", "Please upload an image first!")
        return
    if not save_location.get():
        messagebox.showerror("Error", "Please select a save location!")
        return

    try:
        # 加载 YOLO 模型
        model = YOLO("yolo11n.pt")
        # 运行 YOLO 模型检测物体
        results = model(uploaded_image_path.get())

        # 保存裁剪的结果到指定文件夹
        for i, result in enumerate(results):
            save_path = os.path.join(save_location.get(), f"result_{i}.jpg")
            result.save_crop(save_path)

        messagebox.showinfo("Success", "Detection completed and saved successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


# 创建上传图片按钮
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=10)

# 显示上传图片路径的标签
upload_label = tk.Label(root, textvariable=uploaded_image_path)
upload_label.pack()

# 创建选择保存路径按钮
save_button = tk.Button(root, text="Select Save Location", command=select_save_location)
save_button.pack(pady=10)

# 显示保存路径的标签
save_label = tk.Label(root, textvariable=save_location)
save_label.pack()

# 创建运行模型的按钮
run_button = tk.Button(root, text="Run YOLO Detection", command=run_yolo_model)
run_button.pack(pady=10)

# 启动主循环
root.mainloop()
