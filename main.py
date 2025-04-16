import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2 as cv
import imageio
import threading
import sys
from scipy.fft import fft2, ifft2
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageTk
gamma = 2.2
FAST_SCALE = 0.16
DBL_MIN = sys.float_info.min
RATIO_base = 4940


def filtering_kernel(arr, M):
    arr_out = arr.copy()
    max_value = np.max(arr_out)
    lower_bound = max_value / M

    # Create a mask for values smaller than max but larger than max/10000
    mask = (arr_out < max_value) & (arr_out > lower_bound)

    # Set those values to zero
    arr_out[mask] = 0
    return arr_out


def to_linear(img):  # for linear workflow
    """ to_linear
            Args:
                img (uint8 ndarray, shape(height, width, ch)): image with gamma profile
            return: return a image to linear domain 
    """
    return np.power(img/255, gamma)


def to_non_linear(img):  # for display
    """ to_linear
            Args:
                img (float ndarray, shape(height, width, ch)): image with gamma profile
            return: return a image to nonlinear domain (uint8 image)
    """
    return np.uint8(np.round(np.power(np.clip(img, 0, 1), 1/gamma)*255))


def generate_gaussian_kernel(height, width, sigma=1.0):
    # Create an (x, y) coordinate grid of the kernel
    x = np.linspace(- (width // 2), width // 2, width)
    y = np.linspace(- (height // 2), height // 2, height)
    x, y = np.meshgrid(x, y)

    # Calculate the Gaussian function
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # Normalize the kernel
    kernel = kernel / np.sum(kernel)

    return kernel


def kernal_preprocess(img_in, k_in):
    """ kernal_preprocess
            Args:
                img_in (uint8 ndarray, shape(ch, height, width)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): Blur kernel

            Returns:
                k_pad (uint8 ndarray, shape(height, width)): Blur kernel after preprocessing
    """
    # zero padding
    # k_in = np.where(k_in != 0, 1, 0)
    k_pad = np.pad(k_in, [(0, img_in.shape[0] - k_in.shape[0]),
                   (0, img_in.shape[1] - k_in.shape[1])], 'constant')
    k_pad = np.roll(k_pad, -(k_in.shape[1]//2), axis=1)
    k_pad = np.roll(k_pad, -(k_in.shape[0]//2), axis=0)

    return k_pad


def edgetaper(img, gamma=10, beta=0.2):
    width, height = img.shape[:2]
    dy = 2 * np.pi / width
    dx = 2 * np.pi / height
    # subtract dx and dy to match original function's range
    y = np.linspace(-np.pi, np.pi-dy, width)
    x = np.linspace(-np.pi, np.pi-dx, height)
    w1 = 0.5 * (np.tanh((x + gamma / 2) / beta) -
                np.tanh((x - gamma / 2) / beta))
    w2 = 0.5 * (np.tanh((y + gamma / 2) / beta) -
                np.tanh((y - gamma / 2) / beta))
    w = np.dot(w2.reshape(-1, 1), w1.reshape(1, -1))
    if img.ndim > 2:
        w = w[:, :, np.newaxis].repeat(img.shape[2], axis=2)
    return cv.multiply(img.astype(np.float32), w.astype(np.float32))


def padding(img, pad_num=10, num_channel=1, mode='constant'):
    if pad_num == 0:
        return img
    elif num_channel == 3:
        return np.pad(img, [(pad_num, pad_num), (pad_num, pad_num), (0, 0)], mode)
    else:
        return np.pad(img, [(pad_num, pad_num), (pad_num, pad_num)], mode)


def fft_convolution(image, kernel):
    pad_num = max(image.shape[0]//10, 100)
    image_fft = fft2(padding(image, pad_num=pad_num, mode='symmetric'))
    kernel_fft = fft2(kernal_preprocess(
        padding(image, pad_num=pad_num), padding(kernel, pad_num=pad_num)))
    conv_fft = image_fft * kernel_fft
    conv_result = np.abs(np.real(ifft2(conv_fft)))
    if pad_num != 0:
        return conv_result[pad_num:-pad_num, pad_num:-pad_num]
    else:
        return conv_result


def preprocess(I_file):
    global I_origin, amp_origin, mask_origin, K_origin,  I_processed, mask, amp, K, K_resize, FAST_SCALE
    try:
        I = to_linear(imageio.imread(I_file))
        ratio_input = min(I.shape[0], I.shape[1])
        if ratio_input > RATIO_base*1.5:
            FAST_SCALE /= 2
        I_origin = I.copy()
        # can change kernel to you prefer
        K_RL = np.load('./kernel/kerenl_HDR_4_minus_background_fined_RL.npy')
        K_RL = np.clip(cv.resize(K_RL, dsize=None, fx=ratio_input/RATIO_base,
                       fy=ratio_input/RATIO_base, interpolation=cv.INTER_CUBIC), 0, 1000)
        K_origin = K_RL.copy()
        gaus_kernel = generate_gaussian_kernel(I.shape[0], I.shape[1], 1000)
        amp_origin = I.copy()
        amp_origin[amp_origin < 1] = 0

        def process_channel(i):
            amp_origin[:, :, i] = fft_convolution(
                amp_origin[:, :, i], gaus_kernel)

        with ThreadPoolExecutor() as executor:
            executor.map(process_channel, range(amp_origin.shape[2]))
        # for i in range(amp_origin.shape[2]):
        #     amp_origin[:, :, i] = fft_convolution(
        #         amp_origin[:, :, i], gaus_kernel)

        mask_origin = I.copy()
        mask_origin[mask_origin < 1] = 0
        # small image for realtime display
        scale = FAST_SCALE
        K_RL = cv.resize(K_RL, dsize=None, fx=scale, fy=scale,
                         interpolation=cv.INTER_NEAREST)
        K_resize = K_RL.copy()
        K = np.zeros_like(K_RL)
        K[:, :, 0] = K_RL[:, :, 0]/np.sum(K_RL[:, :, 0])*(2**(-1/3))
        K[:, :, 1] = K_RL[:, :, 1]/np.sum(K_RL[:, :, 1])*(2**(-1/3))
        K[:, :, 2] = K_RL[:, :, 2]/np.sum(K_RL[:, :, 2])*(2**(-1/3))
        I = cv.resize(I, dsize=None, fx=scale, fy=scale,
                      interpolation=cv.INTER_NEAREST)

        mask = I.copy()
        mask[mask < 1] = 0
        amp = cv.resize(amp_origin, dsize=None, fx=scale,
                        fy=scale, interpolation=cv.INTER_NEAREST)
        I_processed = I.copy()
        return True
    except Exception as e:
        result_label.config(text=f"Preprocessing Error: {e}")
        return False


def process_channel(img, amp, mask, K, k, t, channel, M):
    max_amp = np.max(amp[:, :, channel])
    img_irr_map_approx_channel = img[:, :, channel] + k * np.power(
        amp[:, :, channel] / (max_amp + DBL_MIN), t) * mask[:, :, channel]
    conv_k = filtering_kernel(K[:, :, channel], M)
    conv_k = conv_k/conv_k.sum()*(2**(-1/3))
    return fft_convolution(img_irr_map_approx_channel, conv_k)


def process_channel_small(img, amp, mask, K, k, t, channel, M):
    max_amp = np.max(amp[:, :, channel])
    img_irr_map_approx_channel = img[:, :, channel] + k * np.power(
        amp[:, :, channel] / (max_amp + DBL_MIN), t) * mask[:, :, channel]
    conv_k = filtering_kernel(K[:, :, channel], M)
    scale = FAST_SCALE
    conv_k = cv.resize(conv_k, dsize=None, fx=scale,
                       fy=scale, interpolation=cv.INTER_AREA)
    conv_k = conv_k/conv_k.sum()*(2**(-1/3))
    return fft_convolution(img_irr_map_approx_channel, conv_k)


def post(kr, tr, kg, tg, kb, tb, M=100000):
    global mask, amp, K_resize, I_processed, K_origin
    I = I_processed
    img_irr_map_approx = np.zeros_like(I)
    params = [(kr, tr,  0), (kg, tg, 1), (kb, tb, 2)]
    result = np.zeros_like(img_irr_map_approx)

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_channel_small, I, amp,
                            mask, K_origin, k, t, c, M)
            for k, t, c in params
        ]
        for i, future in enumerate(futures):
            result[:, :, params[i][2]] = future.result()

    result_img = Image.fromarray(to_non_linear(result))
    return result_img


def post_out(kr, tr, kg, tg, kb, tb, M=100000):
    global I_origin, amp_origin, mask_origin, K_origin

    img_clear_amp = amp_origin
    mask = mask_origin

    params = [(kr, tr, 0), (kg, tg, 1), (kb, tb, 2)]

    result = np.zeros_like(I_origin)

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_channel, I_origin,
                            img_clear_amp, mask, K_origin, k, t, c, M)
            for k, t, c in params
        ]
        for i, future in enumerate(futures):
            result[:, :, params[i][2]] = future.result()

    return result


def load_file():
    file_path = filedialog.askopenfilename()
    return file_path


def save_result():
    global I_processed, mask, amp
    if I_processed is not None:
        loading_label.pack()  # show animation of saving
        root.update_idletasks()

        threading.Thread(target=_save_result_thread).start()
    else:
        result_label.config(
            text="Please select and preprocess an I file first.")


def _save_result_thread():
    try:
        kr_value = kr_scale.get()
        tr_value = tr_scale.get()
        kg_value = kg_scale.get()
        tg_value = tg_scale.get()
        kb_value = kb_scale.get()
        tb_value = tb_scale.get()
        M_value = np.power(M_scale.get()*10000, 0.9)
        fig = post_out(kr_value, tr_value, kg_value, tg_value, kb_value, tb_value, M_value)

        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"),
                                                            ("JPG files", "*.jpg"),
                                                            ("All files", "*.*")])
        if file_path:
            output_img = to_non_linear(fig)
            if file_path.lower().endswith(".jpg") or file_path.lower().endswith(".jpeg"):
                # save as jpg
                im = Image.fromarray(output_img).convert("RGB")
                im.save(file_path, quality=95, optimize=True)
            else:
                # save as png
                imageio.imsave(file_path, output_img, compress_level=2)
            result_label.config(text=f"Result saved to {file_path}")
        else:
            result_label.config(text="Save operation cancelled.")
    except Exception as e:
        result_label.config(text=f"Error saving file: {e}")
    finally:
        loading_label.pack_forget()


def update_result(*args):
    try:
        kr_value = kr_scale.get()
        tr_value = tr_scale.get()
        kg_value = kg_scale.get()
        tg_value = tg_scale.get()
        kb_value = kb_scale.get()
        tb_value = tb_scale.get()
        M_value = np.power(M_scale.get()*10000,0.9)
        if I_processed is not None:
            result_img = post(kr_value, tr_value, kg_value,
                              tg_value, kb_value, tb_value, M_value)
            photo = ImageTk.PhotoImage(result_img)
            canvas.create_image(0, 0, image=photo, anchor="nw")
            canvas.image = photo
        else:
            result_label.config(
                text="Please select and preprocess an I file first.")
    except Exception as e:
        result_label.config(text="Please select and preprocess an I file first.")


def select_I_file():
    global I_file, I_processed
    I_file = load_file()
    I_label.config(text=f"I File: {I_file}")
    if preprocess(I_file):
        update_result()


if __name__ == '__main__':
    root = tk.Tk()
    root.title(
        "Novel Computational Photography for Soft-Focus Effect in Automatic Post Production")

    I_button = tk.Button(root, text="Select I File", command=select_I_file)
    I_button.pack()

    I_label = tk.Label(root, text="I File: Not Selected")
    I_label.pack()

    canvas = tk.Canvas(root, width=1000, height=1000)
    canvas.pack(side=tk.LEFT)

    slider_length = 500
    slider_resolution = 0.01

    frame = tk.Frame(root)
    frame.pack(side=tk.RIGHT)

    M_label = tk.Label(frame, text="Select M Value")
    M_label.pack()
    M_scale = tk.Scale(frame, from_=1, to=100, orient="horizontal",
                       resolution=slider_resolution*100, command=update_result, length=slider_length)
    M_scale.pack()

    kr_label = tk.Label(frame, text="Select k Value in Red channel")
    kr_label.pack()
    kr_scale = tk.Scale(frame, from_=0, to=150, orient="horizontal",
                        resolution=slider_resolution, command=update_result, length=slider_length)
    kr_scale.pack()

    tr_label = tk.Label(frame, text="Select t Value in Red channel")
    tr_label.pack()
    tr_scale = tk.Scale(frame, from_=0, to=15, orient="horizontal",
                        resolution=slider_resolution, command=update_result, length=slider_length)
    tr_scale.pack()

    kg_label = tk.Label(frame, text="Select k Value in Green channel")
    kg_label.pack()
    kg_scale = tk.Scale(frame, from_=0, to=150, orient="horizontal",
                        resolution=slider_resolution, command=update_result, length=slider_length)
    kg_scale.pack()

    tg_label = tk.Label(frame, text="Select t Value in Red channel")
    tg_label.pack()
    tg_scale = tk.Scale(frame, from_=0, to=15, orient="horizontal",
                        resolution=slider_resolution, command=update_result, length=slider_length)
    tg_scale.pack()

    kb_label = tk.Label(frame, text="Select k Value in Blue channel")
    kb_label.pack()
    kb_scale = tk.Scale(frame, from_=0, to=150, orient="horizontal",
                        resolution=slider_resolution, command=update_result, length=slider_length)
    kb_scale.pack()

    tb_label = tk.Label(frame, text="Select t Value in Red channel")
    tb_label.pack()
    tb_scale = tk.Scale(frame, from_=0, to=15, orient="horizontal",
                        resolution=slider_resolution, command=update_result, length=slider_length)
    tb_scale.pack()
    kr_scale.set(1)
    tr_scale.set(1)
    kg_scale.set(1)
    tg_scale.set(1)
    kb_scale.set(1)
    tb_scale.set(1)

    result_label = tk.Label(frame, text="Result: ")
    result_label.pack()
    save_button = tk.Button(frame, text="Save Result", command=save_result)
    save_button.pack()

    loading_label = tk.Label(frame, text="Saving result, please wait...")
    loading_label.pack_forget()

    root.mainloop()
