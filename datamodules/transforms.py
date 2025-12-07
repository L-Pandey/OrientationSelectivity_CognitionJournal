from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# Custom crop transformation
class CenterCropLongDimension(object):
    def __init__(self, ):
        pass
    def __call__(self, img):
        width, height = img.size
        if width > height:
            # Crop 80 pixels from each side of the width
            left = 80
            right = width - 80
            top = 0
            bottom = height
        else:
            # Crop 80 pixels from each side of the height
            left = 0
            right = width
            top = 80
            bottom = height - 80
        img = img.crop((left, top, right, bottom))
        return img

    def __repr__(self):
        return "Custom Transform: Crop"
    

# spatial scrambling - 
class PixelScrambleTransform:
    def __init__(self, patch_size:int, dims:tuple):
        self.patch_size = patch_size
        self.h, self.w = dims[0], dims[1] 

        self.grid_h, self.grid_w = self.h // self.patch_size, self.w // self.patch_size

    def __call__(self, image):
        # Convert PIL image to NumPy array if needed (this will be handled by ToTensor)
        image = np.array(image)
        
        # Scramble the pixels using the scramble function
        scrambled_image = self.scramble(image)
        
        # Convert scrambled image back to PIL Image before returning
        return Image.fromarray(scrambled_image)
    
    def __repr__(self):
        return "Custom Transform Applied: Spatial Scrambling with patch size of {}".format(self.patch_size)
        

    def scramble(self, image):
        """Scrambles an image by shuffling its patches."""
        # h, w, c = image.shape  # Get image dimensions
        # grid_h, grid_w = h // self.patch_size, w // self.patch_size
    
        patches = []
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                patch = image[i*self.patch_size:(i+1)*self.patch_size, j*self.patch_size:(j+1)*self.patch_size, :]
                patches.append(patch)
    
        np.random.shuffle(patches)  # Shuffle patches
    
        # Reconstruct scrambled image
        scrambled_image = np.zeros_like(image)
        idx = 0
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                scrambled_image[i*self.patch_size:(i+1)*self.patch_size, j*self.patch_size:(j+1)*self.patch_size, :] = patches[idx]
                idx += 1
        return scrambled_image
    

def get_torch_transforms(args):
    # data transforms
    if args.transforms == 'gb':
        trans = transforms.Compose([
        transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0))
        ], p=0.5),
        transforms.ToTensor()
        ])
    elif args.transforms == 'gs':
        trans = transforms.Compose([
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
        ])
    elif args.transforms == 'rc':
        trans = transforms.Compose([
        transforms.RandomResizedCrop(size=(64, 64), scale=(0.08, 1.0), ratio=(0.75, 1.3333)),
        transforms.ToTensor()
        ])
    elif args.transforms == 'rhf':
        trans = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
        ])
    elif args.transforms == 'rhf_highProb':
        trans = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.9),
        transforms.ToTensor()
        ])
    elif args.transforms == 'rhf_lowProb':
        trans = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ToTensor()
        ])
    elif args.transforms == 'cj':
        trans = transforms.Compose([
        transforms.RandomApply([
        transforms.ColorJitter(brightness=[0.19999999999999996, 1.8], contrast=[0.19999999999999996, 1.8], saturation=[0.19999999999999996, 1.8], hue=[-0.2, 0.2])
        ], p=0.8),
        transforms.ToTensor()
        ])
    elif args.transforms == 'all':
        # gb
        trans = transforms.Compose([
        transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0))
        ], p=0.5),
        # grayscale 
        transforms.RandomGrayscale(p=0.2),
        # crop
        transforms.RandomResizedCrop(size=(64, 64), scale=(0.08, 1.0), ratio=(0.75, 1.3333)),
        # hflip
        transforms.RandomHorizontalFlip(p=0.5),
        # cj
        transforms.RandomApply([
        transforms.ColorJitter(brightness=[0.19999999999999996, 1.8], contrast=[0.19999999999999996, 1.8], saturation=[0.19999999999999996, 1.8], hue=[-0.2, 0.2])
        ], p=0.8),
        # resize to desired resolution
        transforms.Resize((args.resize_dims, args.resize_dims)),
        # final transform
        transforms.ToTensor()
        ])
    elif args.transforms == 'none':
        trans = transforms.Compose([
        transforms.ToTensor()
        ])
    elif args.transforms == 'resize':
        trans = transforms.Compose([
        transforms.Resize((args.resize_dims, args.resize_dims)),
        transforms.ToTensor()
        ])
    elif args.transforms == 'cropped_resize': # custom transform class
        trans = transforms.Compose([
            CenterCropLongDimension(),
            transforms.Resize((args.resize_dims, args.resize_dims)),
            transforms.ToTensor()
        ])
    elif args.transforms == 'spatialScrambling':
        trans = transforms.Compose([
        PixelScrambleTransform(args.spatialTransform_patchsize, (args.image_size, args.image_size)),
        transforms.ToTensor()
        ])
    else:
        raise ValueError("Incorrect transform selected. Transform not found in the list")
    
    return trans
