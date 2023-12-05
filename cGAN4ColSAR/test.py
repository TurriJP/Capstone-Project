import torch
from PIL import Image
import rasterio
import torchvision.transforms as transforms
from networks import define_G

# Load the pretrained model
netG = define_G(input_nc=1, output_nc=3, ngf=64, netG='unet_256', norm='instance', use_dropout=False)

# Load the pretrained weights
model_path = '/home/jp/Documents/IME/TCC/CODE/SAR-Colorization-Benchmarking-Protocol/Models/cgan4colsar_generator_saved_model.pth'
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
netG.load_state_dict(state_dict, strict=False)

# Load the input image
input_image_path = '/home/jp/Documents/IME/TCC/CODE/SAR-Colorization-Benchmarking-Protocol/SEN12MS_CR_SARColorData/sar_test/test.tif'
input_image_path = '/home/jp/Documents/IME/TCC/CODE/SAR-Colorization-Benchmarking-Protocol/SEN12MS_CR_SARColorData/sar_test/ROIs1158_spring_s1_1_p608.tif'
input_image_path = '/home/jp/Downloads/VV_grayscale.tif'
input_image_path = '/home/jp/Downloads/DSPK_test.tif'
input_image_path = '/media/jp/FreeAgent GoFlex Drive/SAR/SLC_raw/AE02/VV.tif'
# input_image = Image.open(input_image_path)
grayscale_src = rasterio.open(input_image_path)
grayscale = grayscale_src.read()
print(grayscale.shape)


# grayscale = grayscale[:1, :, :] 
# grayscale = grayscale[:1, :2048, :2048]
grayscale = grayscale[:1, :1024, :1024]
grayscale = grayscale / 255.0
# grayscale = grayscale.transpose(2, 0, 1) #1,2,0  2,0,1

# Preprocess the input image
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5), (0.5, 0.5))
# ])

input_tensor = torch.from_numpy(grayscale).float()  #image_np.transpose((2, 0, 1))).float()
print(input_tensor.size())
input_tensor = input_tensor.unsqueeze(0) #transform(grayscale).unsqueeze(0)  # Add batch dimension
print(input_tensor.size())


# Colorize the input image using the generator
with torch.no_grad():
    netG.eval()
    colorized_tensor = netG(input_tensor)

# Convert the output tensor to an image
output_image = transforms.ToPILImage()(colorized_tensor.squeeze(0).cpu())
output_image.show() 
