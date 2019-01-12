## Add Constant value
def AddConstant(image, k = None):
    if k==None:
        k = 0
    image = image + k/255.0
    return image

def MultiplyConstant(image, m = None): 
    if m == None:
        m = 0
    image = image*(k/255.0)
    return image


# Add Noise)
def AddNoisy(image, noise_typ = None):
    if noise_typ == None:
        return image
    
    if noise_typ == "gauss":
        
        row,col,ch= image[0].shape
        mean = 0
        var = 0.05
        sigma = np.sqrt(var)
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image[0].shape
        print('image shape', image.shape)
        s_vs_p = 0.5
        amount = 0.004
        out = image.copy()
        print('out shape', out.shape)

      # Salt mode
        num_salt = np.ceil(amount * image[0].size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image[0].shape[:2]]
        out[:,coords[0], coords[1], :] = 1

      # Pepper mode
        num_pepper = np.ceil(amount* image[0].size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image[0].shape[:2]]
        out[:,coords[0], coords[1], :] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image[0].shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy
    
## Invert Random Pixel
def InvertPixel(image, P = None):
    if P == None:
        return image
    np.random.seed()
    imageC = image.copy()
    randP = np.random.random(imageC[0].shape[:2])
    imageC[:, randP>P] = 1.0 - imageC[:, randP>P]  
    return imageC

## apply blur to images
def BlurImage(image,k = None, BlurType = None):
    if BlurType == None:
        return image
    
    Blur = np.zeros(image.shape)
    if BlurType == 'Guass' and k == None:
        k = np.array([[1, 4, 7, 4, 1],[4, 16, 26, 16, 4],[7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]])
        k = k.reshape((1, k.shape[0], k.shape[1]))
        #k = np.tile(k, (3,1,1))
        k = k*(1.0/273)
        #k = k.reshape(1, k.shape[0], k.shape[1], k.shape[2])
        for i in range(image.shape[3]):
            Blur[:, :, :, i] = ndimage.convolve(image[:, :, : , i], k, mode = 'constant', cval = 0.0)
        return np.clip(Blur, 0, 1)
    if BlurType == 'Average' and k == None:
        k = np.array([[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        k = k.reshape((1, k.shape[0], k.shape[1]))
        #k = np.tile(k, (3,1,1))
        #print('k', k.shape)
        k = k*(1.0/(k.shape[1]*k.shape[2]))
        for i in range(image.shape[3]):
            Blur[:, :, :, i] = ndimage.convolve(image[:, :, : , i], k, mode = 'constant', cval = 0.0)
        return np.clip(Blur, 0, 1)
        
    if BlurType == 'Median' and k == None:
        for i in range(image.shape[3]):
            Blur[:, :, :, i] = ss.medfilt(image[:, :, : , i], kernel_size = (1,5,5))
        return np.clip(Blur, 0, 1)
    if BlurType == 'MotionBlur' and k == None:
        size = 10
        k = np.zeros((size, size))
        k[int((size-1)/2), :] = np.ones(size)
        k = k / size
        k = k.reshape((1, k.shape[0], k.shape[1]))
        for i in range(image.shape[3]):
            Blur[:, :, :, i] = ndimage.convolve(image[:, :, : , i], k, mode = 'constant', cval = 0.0)
        return np.clip(Blur, 0, 1)
    elif k != None:
        k = k.reshape((1, k.shape[0], k.shape[1]))
        for i in range(image.shape[3]):
            Blur[:, :, :, i] = ndimage.convolve(image[:, :, : , i], k, mode = 'constant', cval = 0.0)
        return np.clip(Blur, 0, 1)

## Image Constrast Correction
def ImageContrastCorrect(image, cor = None):
    if cor == None:
        return image
    if cor == 'Gamma':
        corrected = exposure.adjust_gamma(image, 2)
    elif cor == 'Log':
        corrected = exposure.adjust_log(image, 1)
    return corrected

# Convolutions
def ImageConv(image, masks, convType = None, k = None):
    if convType == None:
        return image
    Convo = np.zeros(image.shape)
    ConvoM = np.zeros(masks.shape)
    if convType == 'Sharp' and k == None:
        k = np.array([[-0.00391, -0.01563, -0.02344, -0.01563, -0.00391],[-0.01563, -0.06250, -0.09375, -0.06250, -0.01563],[-0.02344, -0.09375, 1.85980, -0.09375, -0.02344], [-0.01563, -0.06250, -0.09375, -0.06250, -0.01563], [-0.00391, -0.01563, -0.02344, -0.01563, -0.00391]])
        k = k.reshape((1, k.shape[0], k.shape[1]))
        #k = np.tile(k, (3,1,1)
        #k = k*(1.0/273)
        #k = k.reshape(1, k.shape[0], k.shape[1], k.shape[2])
        for i in range(image.shape[3]):
            Convo[:, :, :, i] = ndimage.convolve(image[:, :, : , i], k, mode = 'constant', cval = 0.0)
            
        ConvoM[:, :, :, 0] = ndimage.convolve(image[:, :, : , 0], k, mode = 'constant', cval = 0.0)
        return np.clip(Convo, 0, 1), np.clip(ConvoM, 0, 1)
    if convType == 'Emboss' and k == None:
        k = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        k = k.reshape((1, k.shape[0], k.shape[1]))
        #k = np.tile(k, (3,1,1)
        #k = k*(1.0/273)
        #k = k.reshape(1, k.shape[0], k.shape[1], k.shape[2])
        for i in range(image.shape[3]):
            Convo[:, :, :, i] = ndimage.convolve(image[:, :, : , i], k, mode = 'constant', cval = 0.0)
        ConvoM[:, :, :, 0] = ndimage.convolve(image[:, :, : , 0], k, mode = 'constant', cval = 0.0)
        return np.clip(Convo, 0, 1), np.clip(ConvoM, 0, 1)
    if convType == 'Edge' and k == None:
        k = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        k = k.reshape((1, k.shape[0], k.shape[1]))
        #k = np.tile(k, (3,1,1)
        #k = k*(1.0/273)
        #k = k.reshape(1, k.shape[0], k.shape[1], k.shape[2])
        for i in range(image.shape[3]):
            Convo[:, :, :, i] = ndimage.convolve(image[:, :, : , i], k, mode = 'constant', cval = 0.0)
        ConvoM[:, :, :, 0] = ndimage.convolve(image[:, :, : , 0], k, mode = 'constant', cval = 0.0)
        return np.clip(Convo, 0, 1), np.clip(ConvoM, 0, 1)

## flip images
def imageFlip(images, masks, flipType = None):
    if flipType == None:
        return images, masks
    
    if flipType == 'lr':
        images = images[:, :, ::-1, :]
        masks = masks[:, :, ::-1]
    if flipType == 'ud':
        images = images[:, ::-1, :, :]
        masks = masks[:, ::-1, :, :]
    return images, masks


## apply transform
def imageTransform(images, masks, transType = None):
    if transType == None:
        return images, masks
    zerI = np.zeros(images.shape)
    zerM = np.zeros(masks.shape)
    if transType == 'Affine':
        tform = AffineTransform(scale=(1.3, 1.1), rotation=1, shear=0.7,
                        translation=(210, 50))
        xx, yy = np.meshgrid(np.arange(0, images[0].shape[1] - 1), np.arange(0, images[0].shape[0] - 1))
        coords = np.stack((xx.flatten(), yy.flatten()), axis = -1).transpose(1,0)
        coords = np.vstack((coords, np.ones(coords.shape[1]))).reshape(1, 3, -1)
        coords = coords.transpose(2,1,0)
        tformI = tform.inverse
        tformP = tform.params
        InvC = np.matmul(tformP, coords)

        X = np.clip(InvC[:, 0, :].astype(int), 0, 255)
        Y = np.clip(InvC[:, 1, :].astype(int), 0, 255)
        img1 = img[0]

        zerI = np.zeros(images.shape)
        zerI[:,xx.flatten(), yy.flatten(), :] = images[:,X.flatten(), Y.flatten(), :]
        zerM[:,xx.flatten(), yy.flatten(), :] = masks[:,X.flatten(), Y.flatten(), :]

#         for i in range(images.shape[0]):
#             zerI[i] = warp(images[i], tform.inverse, output_shape=(images.shape[1], images.shape[2]))
#             zerM[i] = warp(masks[i], tform.inverse, output_shape=(masks.shape[1], masks.shape[2]))
            
        return zerI, zerM
            
#Image resize and Crop
def imageResize(images, masks, HW = None):

    if HW == None:
        H, W = 256, 256
    if HW != None:
        (H, W) = HW
        for i in range(len(images)):
            
            zerI = resize(images, (len(images), H, W), anti_aliasing=True)
            zerM = resize(masks, (len(images), H, W), anti_aliasing=True)
    return zerI, zerM
    
    