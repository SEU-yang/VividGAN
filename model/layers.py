#wrappers for convenience
import torch.nn as nn
from torch.nn.init import xavier_normal , kaiming_normal


def get_facial_landmarks(img):
    rects = detector(img, 0)
    # print(rects.shape)
    rect = rects[0]
    # ~ print(len(rects))
    # ~ print("22222")
    shape = predictor(img, rect)
    a = np.array([[pt.x, pt.y] for pt in shape.parts()])
    b = a.astype('float')  # .reshape(-1,136)
    return b


def _putGaussianMap(center, visible_flag, crop_size_y, crop_size_x, sigma):
    """
    根据一个中心点,生成一个heatmap
    :param center:
    :return:
    """
    grid_y = crop_size_y
    grid_x = crop_size_x
    if visible_flag == False:
        return np.zeros((grid_y, grid_x))
    # start = stride / 2.0 - 0.5
    y_range = [i for i in range(grid_y)]
    x_range = [i for i in range(grid_x)]
    xx, yy = np.meshgrid(x_range, y_range)
    xx = xx
    yy = yy
    d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    exponent = d2 / 2.0 / sigma / sigma
    heatmap = np.exp(-exponent)
    return heatmap


def _putGaussianMaps(keypoints, crop_size_y, crop_size_x, sigma):
    """

    :param keypoints: (15,2)
    :param crop_size_y: int
    :param crop_size_x: int
    :param stride: int
    :param sigma: float
    :return:
    """
    all_keypoints = keypoints
    point_num = all_keypoints.shape[0]
    heatmaps_this_img = []
    for k in range(point_num):
        flag = ~np.isnan(all_keypoints[k, 0])
        heatmap = _putGaussianMap(all_keypoints[k], flag, crop_size_y, crop_size_x, sigma)
        heatmap = heatmap[np.newaxis, ...]
        heatmaps_this_img.append(heatmap)
    heatmaps_this_img = np.concatenate(heatmaps_this_img, axis=0)  # (num_joint,crop_size_y/stride,crop_size_x/stride)
    return heatmaps_this_img


def adjust_learning_rate(optimizer, epoch, lrr):
    ##Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lrr * (0.99 ** (epoch // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_peak_points(heatmaps):
    """

    :param heatmaps: numpy array (N,15,96,96)
    :return:numpy array (N,15,2)
    """
    N, C, H, W = heatmaps.shape
    all_peak_points = []
    for i in range(N):
        peak_points = []
        for j in range(C):
            yy, xx = np.where(heatmaps[i, j] == heatmaps[i, j].max())
            y = yy[0]
            x = xx[0]
            peak_points.append([x, y])
        all_peak_points.append(peak_points)
    all_peak_points = np.array(all_peak_points)
    return all_peak_points


def process(img, landmarks_5pts):
    batch = {}
    name = ['left_eye', 'right_eye', 'nose', 'mouth']
    patch_size = {
        'left_eye': (40, 40),
        'right_eye': (40, 40),
        'nose': (40, 32),
        'mouth': (48, 32),
    }
    landmarks_5pts[3, 0] = (landmarks_5pts[3, 0] + landmarks_5pts[4, 0]) / 2.0
    landmarks_5pts[3, 1] = (landmarks_5pts[3, 1] + landmarks_5pts[4, 1]) / 2.0

    # crops
    for i in range(4):
        x = floor(landmarks_5pts[i, 0])
        y = floor(landmarks_5pts[i, 1])
        batch[name[i]] = img.crop((x - patch_size[name[i]][0] // 2 + 1, y - patch_size[name[i]][1] // 2 + 1,
                                   x + patch_size[name[i]][0] // 2 + 1, y + patch_size[name[i]][1] // 2 + 1))

    return batch


five_pts_idx = [[36, 41], [42, 47], [27, 35], [48, 48], [54, 54]]


def landmarks_68_to_5(x):
    y = []
    for j in range(5):
        y.append(np.mean(x[five_pts_idx[j][0]:five_pts_idx[j][1] + 1], axis=0))
    return np.array(y, np.float32)


def sequential(*kargs ):
    seq = nn.Sequential(*kargs)
    for layer in reversed(kargs):
        if hasattr( layer , 'out_channels'):
            seq.out_channels = layer.out_channels
            break
        if hasattr( layer , 'out_features'):
            seq.out_channels = layer.out_features
            break
    return seq



def weight_initialization( weight , init , activation):
    if init is None:
        return
    if init == "kaiming":
        assert not activation is None
        if hasattr(activation,"negative_slope"):
            kaiming_normal( weight , a = activation.negative_slope )
        else:
            kaiming_normal( weight , a = 0 )
    elif init == "xavier":
        xavier_normal( weight )
    return

def conv( in_channels , out_channels , kernel_size , stride = 1  , padding  = 0 , init = "kaiming" , activation = nn.ReLU() , use_batchnorm = False ):
    convs = []
    if type(padding) == type(list()) :
        assert len(padding) != 3 
        if len(padding)==4:
            convs.append( nn.ReflectionPad2d( padding ) )
            padding = 0

    #print(padding)
    convs.append( nn.Conv2d( in_channels , out_channels , kernel_size , stride , padding ) )
    #weight init
    weight_initialization( convs[-1].weight , init , activation )
    #activation
    if not activation is None:
        convs.append( activation )
    #bn
    if use_batchnorm:
        convs.append( nn.BatchNorm2d( out_channels ) )
    seq = nn.Sequential( *convs )
    seq.out_channels = out_channels
    return seq

def deconv( in_channels , out_channels , kernel_size , stride = 1  , padding  = 0 ,  output_padding = 0 , init = "kaiming" , activation = nn.ReLU() , use_batchnorm = False):
    convs = []
    convs.append( nn.ConvTranspose2d( in_channels , out_channels , kernel_size , stride ,  padding , output_padding ) )
    #weight init
    weight_initialization( convs[0].weight , init , activation )
    #activation
    if not activation is None:
        convs.append( activation )
    #bn
    if use_batchnorm:
        convs.append( nn.BatchNorm2d( out_channels ) )
    seq = nn.Sequential( *convs )
    seq.out_channels = out_channels
    return seq

class ResidualBlock(nn.Module):
    def __init__(self, in_channels , 
                out_channels = None, 
                kernel_size = 3, 
                stride = 1,
                padding = None , 
                weight_init = "kaiming" , 
                activation = nn.ReLU() ,
                is_bottleneck = False ,
                use_projection = False,
                scaling_factor = 1.0
                ):
        super(type(self),self).__init__()
        if out_channels is None:
            out_channels = in_channels // stride
        self.out_channels = out_channels
        self.use_projection = use_projection
        self.scaling_factor = scaling_factor
        self.activation = activation

        convs = []
        assert stride in [1,2]
        if stride == 1 :
            self.shorcut = nn.Sequential()
        else:
            self.shorcut = conv( in_channels , out_channels , 1 , stride , 0 , None , None , False )
        if is_bottleneck:
            convs.append( conv( in_channels     , in_channels//2  , 1 , 1 , 0   , weight_init , activation , False))
            convs.append( conv( in_channels//2  , out_channels//2 , kernel_size , stride , (kernel_size - 1)//2 , weight_init , activation , False))
            convs.append( conv( out_channels//2 , out_channels    , 1 , 1 , 0 , None , None       , False))
        else:
            convs.append( conv( in_channels , in_channels  , kernel_size , 1 , padding if padding is not None else (kernel_size - 1)//2 , weight_init , activation , False))
            convs.append( conv( in_channels , out_channels , kernel_size , 1 , padding if padding is not None else (kernel_size - 1)//2 , None , None       , False))
        
        self.layers = nn.Sequential( *convs )
    def forward(self,x):
        return self.activation( self.layers(x) + self.scaling_factor * self.shorcut(x) )



