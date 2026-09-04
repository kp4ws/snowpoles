"""
Original author: Catherine Breen (July 1, 2024)
Updated by: Kent Pawson (2026) Adapted for multi-pole keypoint configuration and custom dataset pipelines.

Utility script, containing various helper functions used by different scripts in this project.
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2 

def valid_keypoints_plot(args, image, outputs, orig_keypoints, epoch):
    """
    This function plots the regressed (predicted) keypoints and the actual 
    keypoints after each validation epoch for one image in the batch.
    """
    # detach the image, keypoints, and output tensors from GPU to CPU
    image = image.detach().cpu()
    outputs = outputs.detach().cpu().numpy()
    orig_keypoints = orig_keypoints.detach().cpu().numpy()
    # just get a single datapoint from each batch
    img = image[0]  ## something snow in it ## halfway throught the dataset
    output_keypoint = outputs[0]
    orig_keypoint = orig_keypoints[0]
    img = np.array(img, dtype='float32')
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    
    output_keypoint = output_keypoint.reshape(-1, 2)
    orig_keypoint = orig_keypoint.reshape(-1, 2)
    for p in range(output_keypoint.shape[0]):
        #filter out -999 padding
        if orig_keypoint[p, 0] == -999.0 or orig_keypoint[p, 1] == -999.0:
            continue

        if p == 0: 
            plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'r.') ## top
            plt.plot(orig_keypoint[p, 0], orig_keypoint[p, 1], 'b.')
        else:
            plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'r.') ## bottom
            plt.plot(orig_keypoint[p, 0], orig_keypoint[p, 1], 'b.')
    plt.savefig(f"{args.models_output}/training/val_epoch_{epoch}.png")
    plt.close()


def dataset_keypoints_plot(data):
    '''  
    #  This function shows the image faces and keypoint plots that the model
    # will actually see. This is a good way to validate that our dataset is in
    # fact corrent and the faces align wiht the keypoint features. The plot 
    # will be show just before training starts. Press `q` to quit the plot and
    # start training.
    '''
    plt.figure(figsize=(10, 10))
    for i in range(9):
        sample = data[i]
        img = sample['image']
        img = np.array(img, dtype='float32') #/255
        #IPython.embed()
        img = np.transpose(img, (1, 2, 0))
        plt.subplot(3, 3, i+1)
        plt.imshow(img)

        keypoints = sample['keypoints'].reshape(-1, 2)

        for j in range(len(keypoints)):
            if keypoints[j, 0] == -999.0 or keypoints[j, 1] == -999.0:
                continue

            plt.plot(keypoints[j, 0], keypoints[j, 1], 'b.')

    plt.show()
    plt.close()


def eval_keypoints_plot(args, file, image, outputs, eval, orig_keypoints): 
    """
    This function plots the regressed (predicted) keypoints and the actual 
    keypoints after each validation epoch for one image in the batch.
    'eval' is the method to check the model, whether is the valid data (eval) or test data (test)
    """
    # detach the image, keypoints, and output tensors from GPU to CPU
    #IPython.embed()
    image = image.detach().cpu()
    image = image.squeeze(0) ## drop the dimension because no longer need it for model 
    outputs = outputs #.detach().cpu().numpy()
    orig_keypoints = orig_keypoints #.detach().cpu().numpy()#orig_keypoints.detach().cpu().numpy()
    # just get a single datapoint from each batch
    output_keypoint = outputs[0] 
    img = np.array(image, dtype='float32')
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    
    output_keypoint = output_keypoint.reshape(-1, 2)
    orig_keypoints = orig_keypoints.reshape(-1, 2)

    for p in range(output_keypoint.shape[0]):
        if orig_keypoints[p, 0] == -999.0 or orig_keypoints[p, 1] == -999.0:
            continue

        plt.plot(orig_keypoints[p, 0], orig_keypoints[p, 1], 'b.',  markersize=20)
        plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'r.', markersize=20)

    plt.savefig(f"{args.models_output}/{eval}/{eval}_{file}.png")
    plt.close()

def vis_keypoints(image, keypoints, color=(0,255,0), diameter=15):
    image = image.copy()

    for (x, y) in keypoints:
        if x == -999.0 or y == -999.0:
            continue

        print(x, y)
        cv2.circle(image, (int(x), int(y)), diameter, (0, 255, 0), -1)

    plt.imshow(image)
    plt.show()
    plt.close()


def vis_predicted_keypoints(args, file, image, keypoints, color=(0,255,0), diameter=15):
    output_keypoint = keypoints.reshape(-1, 2)

    plt.imshow(image)
    for p in range(output_keypoint.shape[0]):
        if p == 0: 
            plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'r.') ## top
        else:
            plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'r.') ## bottom
    plt.savefig(f"{args.output_path}/predictions/image_{file}.png")
    plt.close()

def MAPE(Y_actual,Y_Predicted):
    mape = ((np.abs(Y_actual - Y_Predicted)/Y_actual)*100)
    return mape

def enable_scroll_zoom_and_pan(ax, base_scale=1.2):
    """Enables mouse-wheel zooming and right-click panning for a matplotlib axis"""
    pan_state = {'is_panning': False, 'start_x': None, 'start_y': None, 'start_xlim': None, 'start_ylim': None}

    def zoom(event):
        if event.inaxes != ax: return
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata = event.xdata 
        ydata = event.ydata 
        
        if event.button == 'up':
            scale_factor = 1 / base_scale # zoom in
        elif event.button == 'down':
            scale_factor = base_scale     # zoom out
        else:
            return

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * (relx)])
        ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])
        ax.figure.canvas.draw_idle()

    def press(event):
        # Button 3 is the RIGHT mouse button
        if event.button == 3 and event.inaxes == ax:
            pan_state['is_panning'] = True
            pan_state['start_x'] = event.x
            pan_state['start_y'] = event.y
            pan_state['start_xlim'] = ax.get_xlim()
            pan_state['start_ylim'] = ax.get_ylim()

    def release(event):
        if event.button == 3:
            pan_state['is_panning'] = False

    def motion(event):
        if pan_state['is_panning'] and pan_state['start_x'] is not None:
            dx_pixels = event.x - pan_state['start_x']
            dy_pixels = event.y - pan_state['start_y']
            bbox = ax.get_window_extent()
            dx_data = dx_pixels * (pan_state['start_xlim'][1] - pan_state['start_xlim'][0]) / bbox.width
            dy_data = dy_pixels * (pan_state['start_ylim'][1] - pan_state['start_ylim'][0]) / bbox.height
            
            ax.set_xlim(pan_state['start_xlim'][0] - dx_data, pan_state['start_xlim'][1] - dx_data)
            ax.set_ylim(pan_state['start_ylim'][0] - dy_data, pan_state['start_ylim'][1] - dy_data)
            ax.figure.canvas.draw_idle()

    ax.figure.canvas.mpl_connect('scroll_event', zoom)
    ax.figure.canvas.mpl_connect('button_press_event', press)
    ax.figure.canvas.mpl_connect('button_release_event', release)
    ax.figure.canvas.mpl_connect('motion_notify_event', motion)

def apply_filter(image):
    # width, height, __ = image.shape
    # for y in range(height):
    #     for x in range(width):
    #         pixel = list(colorsys.rgb_to_hsv(*image[x, y]))
    #         if (pixel[0] < 0.833):
    #             image[x, y] = (0, 0, 0)
    #             continue
    #         pixel[1] = 1
    #         pixel[2] = 255
    #         rgb = colorsys.hsv_to_rgb(*pixel)
    #         image[x, y] = (round(rgb[0]), round(rgb[1]), round(rgb[2]))
    image_rgb = image[:, :, ::-1]
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    mask = image_hsv[:, :, 0] < 149
    image_rgb[mask] = [0,0,0]
    image_hsv[~mask, 1] = 255
    image_hsv[~mask, 2] = 255
    valid_pixels = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    image_rgb[~mask] = valid_pixels[~mask]
    #print("filtered applied!")
    return image_rgb[:, :, ::-1]