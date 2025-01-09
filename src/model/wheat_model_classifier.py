from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
from torchvision import transforms
import os
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = BasicConv2d(192, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(192, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(192, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1),
            BasicConv2d(192, 64, kernel_size=1, stride=1, padding=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(128, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(128, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(128, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(96, 128, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block351(nn.Module):

    def __init__(self, scale=1.0):
        super(Block351, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(384, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(384, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(96, 384, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out

class Mixed_6a(nn.Module):

    def __init__(self):
        super(Mixed_6a, self).__init__()

        self.branch1 = nn.Sequential(
            BasicConv2d(128, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 128, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(128, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 128, kernel_size=(1,7), stride=1),
            BasicConv2d(128, 128, kernel_size=(7,1), stride=1),
            BasicConv2d(128, 128, kernel_size=3, stride=2,padding=3)
        )

        self.branch0 = BasicConv2d(128, 128, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out
    


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 160, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(160, 192, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super(Mixed_7a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1),
            BasicConv2d(288, 320, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(2080, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv2d(224, 256, kernel_size=(3,1), stride=1, padding=(1,0))
        )

        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class InceptionResNetV2(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionResNetV2, self).__init__()
        self.drop = nn.Dropout(p=0.1)
        # Special attributs
        self.input_space = None
        self.input_size = (299, 299, 3)
        self.mean = None
        self.std = None
        # Modules
        self.conv2d_1a = BasicConv2d(3, 64, kernel_size=7, stride=1)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 32, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(32, 128, kernel_size=5, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.repeat = nn.Sequential(
            Block35(scale=1),
            Block35(scale=1)
        )
        self.repeat1 = nn.Sequential(
            Block351(scale=1),
            Block351(scale=1)
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10)
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20)
        )
        self.block8 = Block8(noReLU=True)
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        self.avgpool_1a = nn.AvgPool2d(3, stride=3, count_include_pad=False)
        #self.last_linear = nn.Linear(1536, num_classes)
        self.last_linear1 = nn.Linear(9600,1000)
        self.last_linear2 = nn.Linear(1000,100)

        self.last_linear3 = nn.Linear(100,num_classes)
        
    def features(self, input):
        x = self.conv2d_1a(input)
        #x = self.conv2d_2a(x)
        #x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        #x = self.mixed_5b(x)
        #x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        #x = self.mixed_5b(x)
        #x = self.mixed_5b(x)
        x = self.repeat1(x)
        #x = self.mixed_7a(x)
        #x = self.repeat_2(x)
        #x = self.block8(x)
        #x = self.conv2d_7b(x)
        return x

    def logits(self, features):
        x = self.avgpool_1a(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear1(x)
        
        x = self.last_linear2(x)
        x = self.drop(x)
        x = self.last_linear3(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x
    
# num_classes = 3   
# model = InceptionResNetV2(num_classes)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(), lr=0.00001)



# model.load_state_dict(torch.load("model_resnet_single_image.p"))

# #transformations to be applied to the images
# transform = transforms.Compose([
#     transforms.Resize((150,150)),
#     # transforms.RandomHorizontalFlip(),
#     # transforms.RandomVerticalFlip(),
#     #v2.RandomRotation(10),
#     #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
#     # transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])


# #example on how to apply transform to images

# folder_path = r'C:\Users\Stefano\Desktop\allettato_pics'

# # Get a list of all files and directories in the specified folder
# file_names = os.listdir(folder_path)

# # Filter out directories, if needed
# file_names = [f for f in file_names if os.path.isfile(os.path.join(folder_path, f))]

# #prepare the data into tensors using transform
# data = []
# for name in file_names:
    
#     img_path = os.path.join(folder_path, name)
#     img = Image.open(img_path)
#     data.append(transform(img))
    

# #class that prepares the dataset to be used with torch

class DatasetNoTransform(Dataset):

    def __init__(self, image_list):
        super(DatasetNoTransform, self).__init__()
        self.features = [i for i in image_list]
        self.target = [0 for i in image_list]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        
        x = self.features[idx]
        
        y = self.target[idx]

        return x, y  

# dataset = DatasetNoTransform(data)
# #other torch things
# train_loader = DataLoader(dataset, batch_size=64, shuffle=False)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model.to(device)
   
# #calculating predictions for all images, pred1 is a list with the predictions of the classes (0 = lodeged, 1 = non-lodged, 2 = other)
# # prob1 is a list of lists [p1,p2,p3], with  probabilities p1 = probability of lodged, p2 = probability of non-lodged, p3 = probability of other

# pred1 = []
# prob1 = []
# with torch.no_grad():
#     for inputs1, labels1 in train_loader:
#         inputs1 = inputs1.to(device)  
         

        
        
#         outputs = model(inputs1) 
#         pred = torch.argmax(outputs, dim=1)
#         pred = pred.tolist()
#         outputs = torch.nn.functional.softmax(outputs, dim=1)
#         outputs = outputs.tolist()

        
#         pred1.extend(pred)
#         prob1.extend(outputs)
