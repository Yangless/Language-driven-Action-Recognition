import math
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer


# 跟踪输出
# import sys
# import traceback
#
# old_f = sys.stdout
# class F:
#     def write(self, x):
#         old_f.write(x.replace("\n", " [%s]\n" % str(traceback.extract_stack())))
# sys.stdout = F()



class Slice(nn.Module):
    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index:]


class AddReadout(nn.Module):
    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index:] + readout.unsqueeze(1)


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


# class Unflatten(nn.Module):
#     def __init__(self, start_axis, shape):
#         super(Unflatten, self).__init__()
#         self.start_axis = start_axis
#         self.shape = shape
#
#     def forward(self, x):
#         return x.view(*x.shape[:self.start_axis], *self.shape)


class Unflatten(nn.Module):
    def __init__(self, start_axis, shape):
        super(Unflatten, self).__init__()
        self.start_axis = start_axis
        self.shape = shape

    def forward(self, x):
        # 打印输入张量的形状
        print(f"Input shape before unflatten: {x.shape}")
        print(f"start_axis: {self.start_axis}")
        print(f"shape: {self.shape}")

        input_elements = x.numel()  # 输入张量的元素总数
        output_elements = np.prod(x.shape[:self.start_axis]) * np.prod(self.shape)  # 目标形状的元素总数

        print(f"Input elements: {input_elements}")
        print(f"Output elements: {output_elements}")

        if input_elements != output_elements:
            raise ValueError("Total number of elements must be the same for input and target shapes")

        return x.view(*x.shape[:self.start_axis], *self.shape)



class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index:])
        features = torch.cat((x[:, self.start_index:], readout), dim=-1)

        return self.project(features)


class ViT(VisionTransformer):
    def __init__(self, img_size=384, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
                 qk_scale=None, drop_rate=0, attn_drop_rate=0, drop_path_rate=0,
                 norm_layer=nn.LayerNorm, **kwargs):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                         drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                         norm_layer=norm_layer, **kwargs)
        self.patch_size = patch_size
        self.start_index = 1
        features = [256, 512, 1024, 1024]
        readout_oper = [
            ProjectReadout(embed_dim, self.start_index) for _ in features
        ]
        self.act_postprocess1 = nn.Sequential(
            readout_oper[0],
            Transpose(1, 2),
            Unflatten(2, [img_size // 16, img_size // 16]),
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=features[0],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=features[0],
                out_channels=features[0],
                kernel_size=4,
                stride=4,
                padding=0,
                dilation=1,
                groups=1,
            ),
        )

        self.act_postprocess2 = nn.Sequential(
            readout_oper[1],
            Transpose(1, 2),
            Unflatten(2, [img_size // 16, img_size // 16]),
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=features[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=features[1],
                out_channels=features[1],
                kernel_size=2,
                stride=2,
                padding=0,
                dilation=1,
                groups=1,
            ),
        )

        self.act_postprocess3 = nn.Sequential(
            readout_oper[2],
            Transpose(1, 2),

            Unflatten(2, [img_size // 16, img_size // 16]),
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=features[2],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        self.act_postprocess4 = nn.Sequential(
            readout_oper[3],
            Transpose(1, 2),
            Unflatten(2, [img_size // 16, img_size // 16]),
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=features[3],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Conv2d(
                in_channels=features[3],
                out_channels=features[3],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

        self.norm = nn.Identity()
        self.head = nn.Identity()

    def _resize_pos_embed(self, posemb, gs_h, gs_w):
        posemb_tok, posemb_grid = (
            posemb[:, : self.start_index],
            posemb[0, self.start_index:],
        )

        gs_old = int(math.sqrt(len(posemb_grid)))

        posemb_grid = posemb_grid.reshape(
            (1, gs_old, gs_old, -1)).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(
            posemb_grid, size=(gs_h, gs_w), mode="bilinear")
        posemb_grid = posemb_grid.permute(
            0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

        return posemb

    def forward(self, x):
        b, c, h, w = x.shape

        pos_embed = self._resize_pos_embed(
            self.pos_embed, h // self.patch_size, w // self.patch_size
        )
        x = self.patch_embed.proj(x).flatten(2).permute(0, 2, 1)

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + pos_embed
        x = self.pos_drop(x)

        outputs = []
        for index, blk in enumerate(self.blocks):
            x = blk(x)
            if index in [5, 11, 17, 23]:
                outputs.append(x)

        layer_1 = self.act_postprocess1[0:2](outputs[0])
        layer_2 = self.act_postprocess2[0:2](outputs[1])
        layer_3 = self.act_postprocess3[0:2](outputs[2])
        layer_4 = self.act_postprocess4[0:2](outputs[3])

        shape = (-1, 1024, h // self.patch_size, w // self.patch_size)
        layer_1 = layer_1.view(shape)
        layer_2 = layer_2.view(shape)
        layer_3 = layer_3.view(shape)
        layer_4 = layer_4.view(shape)
        # print(layer_1.shape)
        # print(layer_2.shape)
        # print(layer_3.shape)
        # print(layer_4.shape)
        layer_1 = self.act_postprocess1[3:](layer_1)
        layer_2 = self.act_postprocess2[3:](layer_2)
        layer_3 = self.act_postprocess3[3:](layer_3)   #2
        layer_4 = self.act_postprocess4[3:](layer_4)   #2

        return layer_1, layer_2, layer_3, layer_4
import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer

# class CLIPText(nn.Module):
#     def __init__(
#             self,
#             max_text_length: int = 77,
#             vocab_size: int = 49408,
#             text_embed_dim: int = 512,
#             text_heads: int = 8,
#             text_layers: int = 12,
#             text_hidden_act: str = "gelu",
#             projection_dim: int = 512,
#             local_model_path: str = "./clip-vit-base-patch32/"):
#         super().__init__()
#
#         # Load the CLIP text model from local directory
#         self.text_model = CLIPTextModel.from_pretrained(local_model_path)
#         self.text_projection = nn.Parameter(torch.randn(text_embed_dim, projection_dim))

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPConfig


class CLIPText(nn.Module):
    def __init__(
            self,
            max_text_length: int = 77,
            vocab_size: int = 49408,
            text_embed_dim: int = 512,
            text_heads: int = 8,
            text_layers: int = 12,
            text_hidden_act: str = "gelu",
            projection_dim: int = 512):
        super().__init__()

        # Define custom configuration
        config = CLIPConfig(
            vocab_size=vocab_size,
            hidden_size=text_embed_dim,
            num_attention_heads=text_heads,
            num_hidden_layers=text_layers,
            intermediate_size=text_embed_dim * 4,
            hidden_act=text_hidden_act,
            max_position_embeddings=max_text_length,
            attention_dropout = 0.0,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            torchscript=False,
            use_cache=True,
            tie_word_embeddings=False,
            projection_dim=projection_dim,
            eos_token_id=49407

        )

        # Instantiate text model with custom configuration
        self.text_model = CLIPTextModel(config)

        # Projection layer
        self.text_projection = nn.Parameter(torch.randn(text_embed_dim, projection_dim))



    def get_text_features(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        # print(text_outputs,'text_outpuyts')
        # pooled_output = text_outputs[1] if return_dict else text_outputs.last_hidden_state[:, 0, :]
        # text_features = torch.matmul(pooled_output, self.text_projection)
        # return text_features

        pooled_output = text_outputs[1]
        text_features = torch.matmul(pooled_output, self.text_projection)
        return text_features





import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """
        x = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]
        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)
        return output


class ResidualConvUnitCustom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()
        self.bn = bn
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=not self.bn)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=not self.bn)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.activation = activation

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        return out + x


class FeatureFusionBlockCustom(nn.Module):
    """Feature fusion block."""

    def __init__(self, features, activation=nn.ReLU(), deconv=False, bn=False, expand=False, align_corners=True):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlockCustom, self).__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.expand = expand
        out_features = features if not expand else features // 2
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True)
        self.resConfUnit1 = ResidualConvUnitCustom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnitCustom(features, activation, bn)

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]
        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)
        return output


class DifferentiableArgmax(nn.Module):
    def __init__(self, temperature=0.1):
        super(DifferentiableArgmax, self).__init__()
        self.temperature = temperature

    def forward(self, logits):
        return F.softmax(logits / self.temperature, dim=-1)



class Scratch(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024, 1024], out_channels=256):
        super().__init__()
        self.out_c = 512
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))
        self.layer1_rn = nn.Conv2d(in_channels[0], out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(in_channels[1], out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(in_channels[2], out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(in_channels[3], out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.refinenet1 = FeatureFusionBlockCustom(out_channels, bn=True)
        self.refinenet2 = FeatureFusionBlockCustom(out_channels, bn=True)
        self.refinenet3 = FeatureFusionBlockCustom(out_channels, bn=True)
        self.refinenet4 = FeatureFusionBlockCustom(out_channels, bn=True)
        self.head1 = nn.Conv2d(out_channels, self.out_c, kernel_size=1)
        self.output_conv = nn.Sequential(Interpolate(scale_factor=2, mode="bilinear", align_corners=True))
        # 添加一个线性层，用于将 logits_per_image 转换为 [5, 512] 的形状

        self.fc = nn.Linear(102400, 512)
        # 添加两个全连接层
        self.fc1 = nn.Linear(102400, 10240)
        self.fc2 = nn.Linear(10240, 1024)
        self.fc3 = nn.Linear(1024, 512)
        # 添加一个 GumbelSoftmax 层


    def forward(self, layer_1, layer_2, layer_3, layer_4, text_features):
        layer_1_rn = self.layer1_rn(layer_1)
        layer_2_rn = self.layer2_rn(layer_2)
        layer_3_rn = self.layer3_rn(layer_3)
        layer_4_rn = self.layer4_rn(layer_4)
        path_4 = self.refinenet4(layer_4_rn)
        path_3 = self.refinenet3(path_4, layer_3_rn)
        path_2 = self.refinenet2(path_3, layer_2_rn)
        path_1 = self.refinenet1(path_2, layer_1_rn)
        image_features = self.head1(path_1)
        imshape = image_features.shape
        image_features = image_features.permute(0, 2, 3, 1).reshape(-1, self.out_c)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # print("image_features", image_features.shape)
        # print("text_features",text_features.shape)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_per_image = self.logit_scale.exp() * torch.matmul(image_features, text_features.t())  #相似度差异
        # print("logits_per_ori_image", logits_per_image.shape)
        # 转置 logits_per_image 的形状，从 [102400, 5] 到 [5, 102400]
        out2 = logits_per_image.t()
        out2 = self.fc(out2)
        out2 = out2.sum(dim=-1, keepdim=True)
        # 使用线性层转换 logits_per_image 的形状，从 [5, 102400] 到 [5, 512]
        # out2 = self.fc1(out2)
        # out2 = self.fc2(out2)
        # out2 = self.fc3(out2)

        #
        # out = logits_per_image.reshape(imshape[0], imshape[2], imshape[3], -1).permute(0, 3, 1, 2)
        # # print("out_ori", out.shape)
        # out = self.output_conv(out)    #插值恢复原始大小
        # # print(f'out type: {out.dtype}, ori_permute_shape: {out.shape}')
        return out2


class LSeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPText()
        self.vit = ViT()
        self.scratch = Scratch()

    def forward(self, images, texts):
        layer_1, layer_2, layer_3, layer_4 = self.vit.forward(images)
        # print(f'input_ids type: {texts.dtype}, shape: {texts.shape}')
        text_features = self.clip.get_text_features(texts)
        # print(f'text_features type: {text_features.dtype}, shape: {text_features.shape}')
        return self.scratch.forward(layer_1, layer_2, layer_3, layer_4, text_features)




import torchvision.transforms as transforms
import numpy as np
from transformers import CLIPTokenizer

# 使用 CLIPTokenizer 对标签进行编码
tokenizer =  CLIPTokenizer.from_pretrained("./clip-vit-base-patch32/")
#text_model = CLIPTextModel.from_pretrained("./clip-vit-base-patch32/")
eos_token_id = tokenizer.eos_token_id

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import xml.etree.ElementTree as ET

# 定义动作类别
ACTIONS = [
    "applauding",
    "blowing_bubbles",
    "brushing_teeth",
    "cleaning_the_floor",
    "climbing",
    "cooking",
    "cutting_trees",
    "cutting_vegetables",
    "drinking",
    "feeding_a_horse",
    "fishing",
    "fixing_a_bike",
    "fixing_a_car",
    "gardening",
    "holding_an_umbrella",
    "jumping",
    "looking_through_a_microscope",
    "looking_through_a_telescope",
    "playing_guitar",
    "playing_violin",
    "pouring_liquid",
    "pushing_a_cart",
    "reading",
    "phoning",
    "riding_a_bike",
    "riding_a_horse",
    "rowing_a_boat",
    "running",
    "shooting_an_arrow",
    "smoking",
    "taking_photos",
    "texting_message",
    "throwing_frisby",
    "using_a_computer",
    "walking_the_dog",
    "washing_dishes",
    "watching_TV",
    "waving_hands",
    "writing_on_a_board",
    "writing_on_a_book"
]


# 自定义数据集类
class Stanford40Dataset(Dataset):
    def __init__(self, root_dir, annotations_dir, transform=None):
        self.root_dir = root_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.image_paths, self.labels = self.load_annotations()

    def load_annotations(self):
        image_paths = []
        labels = []
        for filename in os.listdir(self.annotations_dir):
            if not filename.endswith('.xml'):
                continue
            xml_path = os.path.join(self.annotations_dir, filename)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            image_file = root.find('filename').text
            action = root.find('.//action').text

            image_paths.append(os.path.join(self.root_dir, image_file))
            labels.append(ACTIONS.index(action))

        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, ACTIONS[label],label

# 定义数据变换
transform = transforms.Compose([
    transforms.Resize((320, 320)),   #960, 640
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 自定义损失函数
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, model_output, text_embedding):
        return nn.functional.mse_loss(model_output, text_embedding)



# 定义训练函数
def test_model(model, dataloader, criterion, optimizer, device):
    model.to(device)
    with torch.no_grad():
       # model.train()
       # for images, texts, labels in dataloader:
        for images, texts , label in dataloader:



            images = images.to(device)
            #images  reshape
            #归一化  images = transform(image).unsqueeze(0)  # 假设 transform 已经定义
            # RGBA 转换  image = Image.fromarray(image).convert("RGBA")

            # texts = texts.to(device)
            # 每次输入几个相似texts，最大化labels
            texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")['input_ids']
            texts = texts.to(dtype=torch.int32)
            texts = texts.to(device)

            label = label.to(device)
            optimizer.zero_grad()

            # 前向传播
            output = model(images, texts)

            output = output.float()
            label = label.float()
           # print(output,label,"output,label")
            #print("images_path",img_path)
            print("output",output,"label", label)
            loss = criterion(output,label)
            print("loss",loss)



# 初始化模型、损失函数和优化器

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
model = LSeg()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 创建数据集和数据加载器
root_dir = 'D:/Y/Stanford40/JPEGImages'
annotations_dir = 'D:/Y/Stanford40/XMLAnnotations'



dataset = Stanford40Dataset(root_dir=root_dir, annotations_dir=annotations_dir, transform=transform)

# 随机选择一部分数据进行加载
num_samples = 1000  # 要加载的数据样本数
indices = np.random.choice(len(dataset), num_samples, replace=False)
from torch.utils.data.sampler import SubsetRandomSampler
sampler = SubsetRandomSampler(indices)

dataloader = DataLoader(dataset, batch_size=4, sampler=sampler)
weights_path = "Lang—ARmodel_epoch_2_loss_1736.764264240265.pth"
#dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
model.load_state_dict(torch.load(weights_path))


print("len_dataset",len(dataset),"len_dataloader",len(dataloader))
# 示例：读取一个批次的数据
# for images, labels ,label in dataloader:
#     # print(images.shape, labels.shape)
#     print(labels)
#     break

# 训练模型
test_model(model, dataloader, criterion, optimizer, device)




