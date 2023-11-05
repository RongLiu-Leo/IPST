
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import argparse
import torchvision.io as io
from typing import Any, Dict, List, Optional, Tuple, Union

import av

def write_video(
    filename: str,
    video_array: torch.Tensor,
    fps: float,
    bit_rate: int,
    video_codec: str = "libx264",
    options: Optional[Dict[str, Any]] = None,
    audio_array: Optional[torch.Tensor] = None,
    audio_fps: Optional[float] = None,
    audio_codec: Optional[str] = None,
    audio_options: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Writes a 4d tensor in [T, H, W, C] format in a video file

    Args:
        filename (str): path where the video will be saved
        video_array (Tensor[T, H, W, C]): tensor containing the individual frames,
            as a uint8 tensor in [T, H, W, C] format
        fps (Number): video frames per second
        video_codec (str): the name of the video codec, i.e. "libx264", "h264", etc.
        options (Dict): dictionary containing options to be passed into the PyAV video stream
        audio_array (Tensor[C, N]): tensor containing the audio, where C is the number of channels
            and N is the number of samples
        audio_fps (Number): audio sample rate, typically 44100 or 48000
        audio_codec (str): the name of the audio codec, i.e. "mp3", "aac", etc.
        audio_options (Dict): dictionary containing options to be passed into the PyAV audio stream
    """
    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #     _log_api_usage_once(write_video)
    # _check_av_available()
    video_array = torch.as_tensor(video_array, dtype=torch.uint8).numpy()

    # PyAV does not support floating point numbers with decimal point
    # and will throw OverflowException in case this is not the case
    if isinstance(fps, float):
        fps = np.round(fps)

    with av.open(filename, mode="w") as container:
        stream = container.add_stream(video_codec, rate=fps, bit_rate=bit_rate)
        stream.width = video_array.shape[2]
        stream.height = video_array.shape[1]
        stream.pix_fmt = "yuv420p" if video_codec != "libx264rgb" else "rgb24"
        stream.options = options or {}

        if audio_array is not None:
            audio_format_dtypes = {
                "dbl": "<f8",
                "dblp": "<f8",
                "flt": "<f4",
                "fltp": "<f4",
                "s16": "<i2",
                "s16p": "<i2",
                "s32": "<i4",
                "s32p": "<i4",
                "u8": "u1",
                "u8p": "u1",
            }
            a_stream = container.add_stream(audio_codec, rate=audio_fps)
            a_stream.options = audio_options or {}

            num_channels = audio_array.shape[0]
            audio_layout = "stereo" if num_channels > 1 else "mono"
            audio_sample_fmt = container.streams.audio[0].format.name

            format_dtype = np.dtype(audio_format_dtypes[audio_sample_fmt])
            audio_array = torch.as_tensor(audio_array).numpy().astype(format_dtype)

            frame = av.AudioFrame.from_ndarray(audio_array, format=audio_sample_fmt, layout=audio_layout)

            frame.sample_rate = audio_fps

            for packet in a_stream.encode(frame):
                container.mux(packet)

            for packet in a_stream.encode():
                container.mux(packet)

        for img in video_array:
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pict_type = "NONE"
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush stream
        for packet in stream.encode():
            container.mux(packet)


def load_image(img_path):
    
    image = Image.open(img_path).convert('RGB')
    try:
        dpi = image.info['dpi'][0]
    except:
        dpi = 72
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),   # ImageNet 
                        ])   
    # change image's size to (b, 3, h, w)
    image = transform(image)[:3, :, :].unsqueeze(0)
    return image, dpi


def im_convert(tensor):

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze(0) 
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))   # unnormalized image
    image = image.clip(0, 1)    

    return image

# Instant Style Transfer
class IPST(nn.Module):
    def __init__(self, VGG, content, style):
        super(IPST, self).__init__()

        self.VGG = VGG
        self.content = content
        self.style = transforms.functional.resize(style, self.content.shape[2:], antialias=True)
        self.resolution = 480

        self.content_features = self.get_features(transforms.functional.resize(content, self.resolution, antialias=True), self.VGG)
        self.style_features = self.get_features(transforms.functional.resize(style, self.resolution, antialias=True), self.VGG)
        self.style_gram_matrixs = {layer: self.get_grim_matrix(self.style_features[layer]) for layer in self.style_features}
        
        self.style_net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, kernel_size=1)
        )
        
    def forward(self, x):

        downsample_content = transforms.functional.resize(x, self.resolution, antialias=True)
        # print(downsample_content.shape)

        style = self.style_net(downsample_content)

        style = transforms.functional.resize(style, self.content.shape[2:], antialias=True)

        return style + x
    
    def get_grim_matrix(self, tensor):
        b, c, h, w = tensor.size()
        tensor = tensor.view(b * c, h * w)
        gram_matrix = torch.mm(tensor, tensor.t())
        return gram_matrix
    
    def get_features(self, image, model):
        layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '28': 'conv5_1'}
        features = {}
        x = image
        for name, layer in model._modules.items():
            x = layer(x)    
            if name in layers:
                features[layers[name]] = x
        
        return features
    
    def get_loss(self, target):
        target_features = self.get_features(transforms.functional.resize(target, self.resolution, antialias=True), self.VGG)
        content_loss = torch.mean((target_features['conv4_1'] - self.content_features['conv4_1']) ** 2)
        style_loss = 0
        style_weights = {'conv1_1': 1, 'conv2_1': 1, 'conv3_1': 1, 'conv4_1': 1, 'conv5_1': 1}
        for layer in style_weights:
            target_feature = target_features[layer]  
            target_gram_matrix = self.get_grim_matrix(target_feature)
            style_gram_matrix = self.style_gram_matrixs[layer]

            layer_style_loss = style_weights[layer] * torch.mean((target_gram_matrix - style_gram_matrix) ** 2)
            b, c, h, w = target_feature.shape
            style_loss = style_loss + layer_style_loss / (c * h * w)
        return content_loss, style_loss
    
    def transfer(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        best_loss = float('inf')  
        patience = 10 
        early_stop_counter = 0  
        for epoch in tqdm(range(0, 150)):
            target = self.forward(self.content)
            content_loss, style_loss = self.get_loss(target)
            if epoch==0:
                inital_content_loss = content_loss.item()
                loss = content_loss + style_loss
                initial_loss = loss.item()
            loss = torch.exp(content_loss/inital_content_loss - 1) * content_loss + style_loss # maybe leave the selection of content to style ratio here·····
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            normalized_loss = loss.item()/initial_loss
            # plt.imsave(f'outputs/{epoch}.png', im_convert(target))

            if normalized_loss < best_loss - 0.01:
                best_loss = normalized_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    break
        return target
        

def main():
    seed = 0
    torch.manual_seed(seed)
    parser = argparse.ArgumentParser(description='Image Style Transfer')

    parser.add_argument('--content-image', type=str, help='Path to the content image')
    parser.add_argument('--content-video', type=str, help='Path to the content video')
    parser.add_argument('--style-image', type=str, help='Path to the style image')
    parser.add_argument('--output-folder',type=str, help='Path to output folder', default='./outputs/')
    parser.add_argument('--frame-by-frame',type=bool, help='Transfer videos frame by frame', default=False)

    args = parser.parse_args()

    # Check if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    np.random.seed(seed)
    print(torch.__version__, device)    
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    # Transfer image
    if args.content_image and args.style_image:
        print('Loading VGG model')
        VGG = models.vgg19(weights='DEFAULT').features
        VGG.to(device)
        for parameter in VGG.parameters():
            parameter.requires_grad_(False)
        print('Input:', args.content_image, args.style_image)
        print('Loading input data')
        content_image, dpi = load_image(args.content_image)
        style_image, _ = load_image(args.style_image)
        content_image = content_image.to(device)
        style_image = style_image.to(device)

        ipst = IPST(VGG, content_image, style_image)
        ipst.to(device)
        print('Transfering')
        result = im_convert(ipst.transfer())

        print('Saving')
        plt.imsave(os.path.join(output_folder, os.path.basename(args.content_image)), result, dpi=dpi)

    
    elif args.content_video and args.style_image:
        print('Loading VGG model')
        VGG = models.vgg19(weights='DEFAULT').features
        VGG.to(device)
        for parameter in VGG.parameters():
            parameter.requires_grad_(False)
        print('Input:', args.content_video, args.style_image)
        
        reader = io.VideoReader(args.content_video)
        fps = reader.get_metadata()['video']['fps'][0]
        bit_rate = av.open(args.content_video).streams.video[0].bit_rate

        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

        if args.frame_by_frame:
            total_frame = av.open(args.content_video).streams.video[0].frames
            frames = []
            print('Transfer video frame by frame')
            for i in tqdm(range((total_frame))):
                frame = next(reader)['data'].unsqueeze(0)
                frame = (frame/255. - mean[None, :, None, None]) / std[None, :, None, None]
                frame = frame.to(device)
                if i == 0:
                    content_image = frame
                    style_image,_ = load_image(args.style_image)
                    style_image = style_image.to(device)

                    ipst = IPST(VGG, content_image, style_image)
                    ipst.to(device)
                    frame = ipst.transfer()
                    
                else:
                    with torch.no_grad():
                        frame = ipst.forward(frame)
                frame = frame.cpu().detach()
                frame = (frame * std[None, :, None, None]) + mean[None, :, None, None]
                frame = frame.permute(0, 2, 3, 1)
                frame = frame.clip(0, 1)        
                frame = frame*255
                frame = frame.squeeze(0)
                frames.append(frame.to(torch.uint8))
            frames = torch.stack(frames, 0)
        else:
            print('Loading input data')
            frames = []
            for frame in tqdm(reader):            
                frames.append(frame['data'])
            
            frames = torch.stack(frames, 0).float() / 255
            del frame
            frames = (frames - mean[None, :, None, None]) / std[None, :, None, None]
            frames = frames.to(device)
            print('Transfer the whole video')
            for i in tqdm(range(len(frames))):
                frame = frames[i].unsqueeze(0)
                if i == 0:
                    content_image = frame
                    style_image,_ = load_image(args.style_image)
                    style_image = style_image.to(device)

                    ipst = IPST(VGG, content_image, style_image)
                    ipst.to(device)
                    frame = ipst.transfer()
                    
                else:
                    with torch.no_grad():
                        frame = ipst.forward(frame)
                frames[i] = frame
            frames = frames.cpu().detach()
            frames = (frames * std[None, :, None, None]) + mean[None, :, None, None]
            frames = frames.permute(0, 2, 3, 1)
            frames = frames.clip(0, 1)        
            frames = frames*255
        print('Saving')
        write_video(os.path.join(output_folder, os.path.basename(args.content_video)), frames, fps, bit_rate=bit_rate)

    else:
        print('Please provide --content-image/--content-video and --style-image paths')

if __name__ == '__main__':
    main()