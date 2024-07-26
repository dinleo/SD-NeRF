import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler


def flatten_array(array):
    flattened = np.concatenate([arr.flatten() for arr in array])
    return flattened


@torch.no_grad()
def initialize_weights(m):
    if hasattr(m, 'weight'):
        print(m)
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class ImageDataset(Dataset):
    def __init__(self, data_path, dtype, h, w):
        self.image1 = Image.open(data_path + 'img1.png').convert("RGB")
        self.image2 = Image.open(data_path + 'img2.png').convert("RGB")
        self.dtype = dtype
        self.transform = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.ToTensor()
        ])
        self.images = [self.image1, self.image2]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        if self.dtype == torch.float16:
            image = image.half()

        return image


class NerfDataset(Dataset):
    def __init__(self, data_path, save_path, need_scale=True, only_main=False):
        self.data_path = data_path
        self.save_path = save_path
        self.need_scale = need_scale
        self.only_main = only_main
        self.scaler1 = MinMaxScaler()
        self.scaler2 = MinMaxScaler()
        self.enc3_1 = self.encode('model_200000.npy', self.scaler1)
        self.enc3_2 = self.encode('model_fine_200000.npy', self.scaler2)
        self.npy = self.make_input()

    def __len__(self):
        return len(self.npy)

    def __getitem__(self, idx):
        if self.only_main:
            return self.npy[0]
        else:
            return self.npy[idx]


    def encode(self, path, scaler):
        npy_org = np.load(self.data_path + path, allow_pickle=True)
        enc10 = encode_org_10(npy_org)
        enc3 = encode_10_3(enc10)
        ret = {
            'uns_org': enc3,
            'scl_org': self.scaling(enc3, scaler)
        }

        return ret

    def make_input(self):
        if self.need_scale:
            return torch.tensor(np.array([self.enc3_1['scl_org'], self.enc3_2['scl_org']]), device='cuda', dtype=torch.float32)
        else:
            return torch.tensor(np.array([self.enc3_1['uns_org'], self.enc3_2['uns_org']]), device='cuda', dtype=torch.float32)



    def decode_(self, enc3_rec, scaler):
        enc3_rec = enc3_rec.detach().to('cpu', dtype=torch.float64).numpy()
        ret = {
            'uns_rec': enc3_rec,
            'scl_rec': None,
            'npy_rec': None
        }
        if self.need_scale:
            ret['scl_rec'] = enc3_rec
            enc3_rec = self.inverse_scaling(enc3_rec, scaler)
            ret['uns_rec'] = enc3_rec
        dec10 = decode_3_10(enc3_rec)
        npy_org = decode_10_org(dec10)
        ret['npy_rec'] = npy_org

        return ret

    def decode(self, enc3_recs):
        ret1 = self.decode_(enc3_recs[0], self.scaler1)
        ret2 = self.decode_(enc3_recs[1], self.scaler2)

        return [ret1, ret2]


    def save_dec_npy(self, res):
        path = 'model_200000.npy'
        np.save(self.save_path + path, res[0]['npy_rec'])
        print(f'save to {self.save_path + path}')
        path = 'model_fine_200000.npy'
        np.save(self.save_path + path, res[1]['npy_rec'])
        print(f'save to {self.save_path + path}')


    def scaling(self, org, scaler):
        shape = org.shape
        org = org.reshape(-1, shape[-1])
        org = scaler.fit_transform(org)
        org = org.reshape(shape)

        return org


    def inverse_scaling(self, org, scaler):
        shape = org.shape
        org = org.reshape(-1, shape[-1])
        org = scaler.inverse_transform(org)
        org = org.reshape(shape)

        return org

    def compare_npy(self, original, decoded):
        for i, (orig_param, dec_param) in enumerate(zip(original, decoded)):
            if not np.array_equal(orig_param, dec_param):
                print(f"Mismatch found at parameter {i}")
                print(f"Original shape: {orig_param.shape}, Decoded shape: {dec_param.shape}")
                print(f"Original parameter: \n{orig_param}")
                print(f"Decoded parameter: \n{dec_param}")
                return False
        print("All parameters match.")
        return True


def plt_img(img):
    img = process_img(img)
    plt.imshow(img)
    plt.show()


def process_img(img):
    img = img.clone().detach()
    if len(img.shape) == 4:
        img = img[0]
    img = img.permute(1, 2, 0).to('cpu').detach().numpy()
    img = np.clip(img, 0, 1).astype(np.float32)
    return img


def compare_img(title, img1, img2, criterion):
    if not torch.is_tensor(img1):
        img1 = torch.tensor(img1)
        img2 = torch.tensor(img2)
    loss = criterion(img1, img2)
    img1 = process_img(img1.to('cpu').detach())
    img2 = process_img(img2.to('cpu').detach())
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(title)
    axes[0].imshow(img1)
    axes[0].set_title('Reconstruction')
    axes[0].axis('off')  # 축을 끔

    # 두 번째 이미지 출력
    axes[1].imshow(img2)
    axes[1].set_title('Original')
    axes[1].axis('off')  # 축을 끔

    # 레이아웃 조정 및 출력
    plt.tight_layout()
    plt.show()

    return loss


def make_image(pth, dtype, h, w):
    tr = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor()
    ])
    im = Image.open(pth).convert("RGB")
    im = tr(im)
    im = im.unsqueeze(0).to('cuda')
    if dtype == torch.float16:
        im = im.half()

    return im


def encode_org_10(nerf_npy):
    new_np = np.zeros((10, 256, 256))
    input_layer = np.array([], dtype=np.float32)
    bias_layer = np.array([], dtype=np.float32)

    cn_i = 0

    for i in range(0, len(nerf_npy), 2):
        weight_shape = nerf_npy[i].shape
        if weight_shape == (256, 256):
            new_np[cn_i, :, :] = nerf_npy[i]
            bias_layer = np.append(bias_layer, nerf_npy[i + 1].flatten())
            cn_i += 1
        elif weight_shape == (319, 256):
            new_np[cn_i, :, :] = nerf_npy[i][63:, :]
            input_layer = np.append(input_layer, nerf_npy[i][:63, :].flatten())
            bias_layer = np.append(bias_layer, nerf_npy[i + 1].flatten())
            cn_i += 1
        elif weight_shape == (283, 128):
            input_layer = np.append(input_layer, nerf_npy[i][:27, :].flatten())
            bias_layer = np.append(bias_layer, nerf_npy[i][27:, :].flatten())
            bias_layer = np.append(bias_layer, nerf_npy[i + 1].flatten())
        else:
            input_layer = np.append(input_layer, nerf_npy[i].flatten())
            bias_layer = np.append(bias_layer, nerf_npy[i + 1].flatten())
    # print(input_layer.shape[0]/256)
    # print(bias_layer.shape[0]/256)
    # print((input_layer.shape[0] + bias_layer.shape[0])/256)

    padding = np.zeros((256 * 256) - input_layer.size)
    input_layer = np.concatenate((input_layer, padding))

    padding = np.zeros((256 * 256) - bias_layer.size)
    bias_layer = np.concatenate((bias_layer, padding))

    new_np[8, :, :] = input_layer.reshape(256, 256)
    new_np[9, :, :] = bias_layer.reshape(256, 256)

    return new_np


def decode_10_org(new_np):
    original_params = []

    # input_layer와 bias_layer 복원
    input_layer_flat = new_np[8].flatten()
    bias_layer_flat = new_np[9].flatten()

    input_i = 0
    bias_i = 0

    # Layer 0 복원
    first_weight = input_layer_flat[input_i:input_i + 63 * 256].reshape(63, 256)
    original_params.append(first_weight)
    input_i += 63 * 256

    first_bias = bias_layer_flat[bias_i:bias_i + 256]
    original_params.append(first_bias)
    bias_i += 256

    # Layer 1~8 복원
    for cn_i in range(0, 8):
        weight = new_np[cn_i]
        if cn_i == 4:
            # (319, 256) 레이어 복원
            w5_2 = input_layer_flat[input_i:input_i + 63 * 256].reshape(63, 256)
            input_i += 63 * 256
            original_params.append(np.vstack((w5_2, weight)))
        else:
            original_params.append(weight)

        bias = bias_layer_flat[bias_i:bias_i + 256]
        bias_i += 256
        original_params.append(bias)

    # Layer 9 복원
    w9_1 = input_layer_flat[input_i:input_i + 27 * 128].reshape(27, 128)
    input_i += 27 * 128
    w9_2 = bias_layer_flat[bias_i:bias_i + 256 * 128].reshape(256, 128)
    bias_i += 256 * 128
    original_params.append(np.vstack((w9_1, w9_2)))

    b9 = bias_layer_flat[bias_i:bias_i + 128]
    bias_i += 128
    original_params.append(b9)

    # color Layer 복원
    wc = input_layer_flat[input_i:input_i + 128 * 3].reshape(128, 3)
    input_i += 128 * 3
    original_params.append(wc)

    bc = bias_layer_flat[bias_i:bias_i + 3]
    bias_i += 3
    original_params.append(bc)

    # density Layer 복원
    wd = input_layer_flat[input_i:input_i + 256].reshape(256, 1)
    original_params.append(wd)
    bd = bias_layer_flat[bias_i:bias_i + 1]
    original_params.append(bd)

    return np.array(original_params, dtype=object)


def encode_10_3(org_npy):
    t_shape = (512, 392)
    t_size = 512 * 392
    extra_size = 2048

    layers_6_9 = [org_npy[6], org_npy[7], org_npy[8][:142, :], org_npy[9][:138, :]]
    flattened_6_9 = np.concatenate([layer.flatten() for layer in layers_6_9])

    channel_3 = flattened_6_9[:t_size - extra_size]
    channel_3 = np.pad(channel_3, (0, extra_size)).reshape(t_shape)

    remain_6_9 = flattened_6_9[t_size - extra_size:]
    remain1 = remain_6_9[:extra_size]
    remain2 = remain_6_9[extra_size:]

    # 첫 번째 채널에 0, 1, 2 레이어
    channel_1 = np.concatenate([layer.flatten() for layer in org_npy[0:3]])
    channel_1 = np.concatenate([channel_1, remain1])
    channel_1 = np.pad(channel_1, (0, 2048)).reshape(t_shape)

    # 두 번째 채널에 3, 4, 5 레이어
    channel_2 = np.concatenate([layer.flatten() for layer in org_npy[3:6]])
    channel_2 = np.concatenate([channel_2, remain2])
    channel_2 = np.pad(channel_2, (0, 2048)).reshape(t_shape)

    # 결과 배열 생성
    encoded_npy = np.stack((channel_1, channel_2, channel_3), axis=0)

    return encoded_npy


def decode_3_10(encoded_npy):
    t_shape = (256, 256)
    t_size = 256 * 256

    def unflatten_and_extract(layers, target_shape):
        flattened = layers.flatten()
        extracted_layers = []
        remain_idx = 0
        for i in range(0, len(flattened), t_size):
            if i + t_size < len(flattened):
                layer = flattened[i:i + t_size].reshape(target_shape)
                extracted_layers.append(layer)
            else:
                remain_idx = i
        return extracted_layers, flattened[remain_idx:]

    # 첫 번째 채널에서 0, 1, 2 레이어 복원
    layers_0_2, remain1 = unflatten_and_extract(encoded_npy[0], t_shape)
    remain1 = remain1[:2048]

    # 두 번째 채널에서 3, 4, 5 레이어 복원
    layers_3_5, remain2 = unflatten_and_extract(encoded_npy[1], t_shape)
    remain2 = remain2[:2048]

    # 세 번째 채널에서 6, 7, 8, 9 레이어 복원
    flattened_6_9 = encoded_npy[2].flatten()

    layer_6 = flattened_6_9[:t_size].reshape(t_shape)
    layer_7 = flattened_6_9[t_size:2 * t_size].reshape(t_shape)

    layer_8 = flattened_6_9[2 * t_size:2 * t_size + 256 * 142]
    layer_8 = np.pad(layer_8, (0, 256 * 114), 'constant', constant_values=0).reshape(t_shape)

    layer_9 = flattened_6_9[2 * t_size + 256 * 142:-2048]
    layer_9 = np.concatenate([layer_9, remain1, remain2])

    layer_9 = np.pad(layer_9, (0, 256 * 118), 'constant', constant_values=0).reshape(t_shape)

    layers_0_2.extend(layers_3_5)
    layers_0_2.extend([layer_6, layer_7, layer_8, layer_9])

    return np.array(layers_0_2)
