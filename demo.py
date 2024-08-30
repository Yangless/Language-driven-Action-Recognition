import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer


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
        self.fc = nn.Linear(153600, 512)

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
        print("image_features", image_features.shape)
        print("text_features",text_features.shape)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_per_image = self.logit_scale.exp() * torch.matmul(image_features, text_features.t())  #相似度差异
        print("logits_per_ori_image", logits_per_image.shape)

        # 转置 logits_per_image 的形状，从 [153600, 5] 到 [5, 153600]
        out2 = logits_per_image.t()
        # 使用线性层转换 logits_per_image 的形状，从 [5, 153600] 到 [5, 512]
        out2 = self.fc(out2)
        print("out2", out2.shape)

        out = logits_per_image.reshape(imshape[0], imshape[2], imshape[3], -1).permute(0, 3, 1, 2)
        print("out_ori", out.shape)
        out = self.output_conv(out)    #插值恢复原始大小
        print(f'out type: {out.dtype}, ori_permute_shape: {out.shape}')
        return out2


class LSeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPText()
        self.vit = ViT()
        self.scratch = Scratch()

    def forward(self, images, texts):
        layer_1, layer_2, layer_3, layer_4 = self.vit.forward(images)
        #print(texts)
        print(f'input_ids type: {texts.dtype}, shape: {texts.shape}')
        text_features = self.clip.get_text_features(texts)
        print(f'text_features type: {text_features.dtype}, shape: {text_features.shape}')
        return self.scratch.forward(layer_1, layer_2, layer_3, layer_4, text_features)



from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# def get_new_pallete(num_cls):
#     n = num_cls
#     pallete = [0]*(n*3)
#     for j in range(0,n):
#             lab = j
#             pallete[j*3+0] = 0
#             pallete[j*3+1] = 0
#             pallete[j*3+2] = 0
#             i = 0
#             while (lab > 0):
#                     pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
#                     pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
#                     pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
#                     i = i + 1
#                     lab >>= 3
#     return pallete

# def get_new_mask_pallete(npimg, new_palette, out_label_flag=False, labels=None):
#     """Get image color pallete for visualizing masks"""
#     # put colormap
#     out_img = Image.fromarray(npimg.squeeze().astype('uint8'))
#     out_img.putpalette(new_palette)
#
#     if out_label_flag:
#         assert labels is not None
#         u_index = np.unique(npimg)
#         patches = []
#         for i, index in enumerate(u_index):
#             label = labels[index]
#             cur_color = [new_palette[index * 3] / 255.0, new_palette[index * 3 + 1] / 255.0, new_palette[index * 3 + 2] / 255.0]
#             red_patch = mpatches.Patch(color=cur_color, label=label)
#             patches.append(red_patch)
#     return out_img, patches


import torch
import torchvision.transforms as transforms
from transformers import CLIPTokenizer

# Assuming the LSeg model is defined and imported correctly as in the previous code snippets
model = LSeg()
#state_dict = torch.load('data/data169501/LSeg.pth', map_location='cpu')
#model.load_state_dict(state_dict)
model.eval()

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        ),
    ]
)
#local_model_path: str = "./clip-vit-base-patch32/"
#tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
#tokenizer =  CLIPTextModel.from_pretrained("./clip-vit-base-patch32/")



import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPTokenizer

# 指定图像路径
# img_path = 'images/cat.jpeg'
img_path = 'images/dog.jpg'    #640*966


# 加载图像并进行预处理
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w = image.shape[:2]
image = image[:-(h%32) if h%32 else None, :-(w%32) if w%32 else None]
images = transform(image).unsqueeze(0)  # 假设 transform 已经定义
image = Image.fromarray(image).convert("RGBA")

# 使用 CLIPTokenizer 对标签进行编码
tokenizer =  CLIPTokenizer.from_pretrained("./clip-vit-base-patch32/")
#text_model = CLIPTextModel.from_pretrained("./clip-vit-base-patch32/")
eos_token_id = tokenizer.eos_token_id
print(f'EOS token ID: {eos_token_id}')
# 定义一些文本

# 指定类别标签
labels = ['plant', 'cat', 'dog', 'stone', 'other']
# 使用分词器对文本进行编码
inputs = tokenizer(labels, padding=True, truncation=True, return_tensors="pt")['input_ids']
inputs = inputs.to(dtype=torch.int32)
#(inputs.shape)
#print(inputs)

# # 模型前向传播
with torch.no_grad():
    results = model.forward(images, inputs)
    results = torch.argmax(results, dim=1)
    results = results.numpy()
    print("results", results)






# 使用文本模型获取嵌入
# with torch.no_grad():
#texts = text_model(**inputs).last_hidden_state

#print(images)

# # 模型前向传播
# with torch.no_grad():
#     results = model.forward(images, inputs)
#     results = torch.argmax(results, dim=1)
#     results = results.numpy()




#
# # 获取调色板并生成遮罩
# new_palette = get_new_pallete(len(labels))  # 假设该函数已定义
# mask, patches = get_new_mask_pallete(results, new_palette, out_label_flag=True, labels=labels)  # 假设该函数已定义
#
# # 将遮罩转换为图像并进行混合
# seg = mask.convert("RGBA")
# out = Image.blend(image, seg, alpha=0.5)
#
# # 可视化
# plt.axis('off')
# plt.imshow(image)
# plt.figure()
# plt.axis('off')
# plt.imshow(out)
# plt.figure()
# plt.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.5, 1), prop={'size': 20})
# plt.axis('off')
# plt.imshow(seg)
# plt.show()
