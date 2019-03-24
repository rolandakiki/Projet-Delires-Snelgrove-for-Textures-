# Copyright 2017, Xavier Snelgrove
import os
import sys
import numpy as np
from scipy import ndimage
import gram
from gram import JoinMode
from keras.preprocessing.image import load_img

#    parser = argparse.ArgumentParser(description="Synthesize image from texture", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
if os.getcwd()!='/cal/homes/rakiki/subjective-functions':
    os.chdir('subjective-functions')
#    parser.add_argument('--output-width', '-ow', default=512, type=int,
#            help="Pixel width of generated image")

use_spec=False
    
output_width=512
#    parser.add_argument('--output-height', '-oh', type=int,
#            help="Pixel height of generated image. If not specified, equal to output-width.")
output_height=None
#    parser.add_argument('--octaves', '-o',  type=int, default=4,
#           help="Number of octaves (where 1 means \"Consider only 1 scale\")")
octaves=4
#    parser.add_argument('--layers', '-l',  type=int, nargs='+', default=[2, 7],
#            help="Which layers to match gram matrices on")
layers=[2,7]
#    parser.add_argument('--max-iter', '-m', type=int, default=500,
#            help="Maximum iterations for the L-BFGS-B optimizer")
max_iter=1501
#    parser.add_argument("--output-prefix", "-op", default='out',
#            help="Prefix to append to output directory")
output_prefix='out'
#    parser.add_argument("--save-every", "-n", default=10, type=int,
#            help="Save an in-progress optimization image every SAVE_EVERY iterations")
save_every=30
#    parser.add_argument("--source-scale", "-ss", type=float,
#            help="How much to scale the source image by")
source_scale=None
#    parser.add_argument("--source-width", "-sw", type=int,
#            help="Scale source to this width. Mutually exclusive with source-scale")
source_width=None
#    parser.add_argument("--padding-mode", "-p", type=str, choices = ['valid', 'same'], default='valid',
#            help="What boundary condition to use for convolutions")
padding_mode='valid'
#    parser.add_argument("--join-mode", "-j", type=JoinMode,
#            choices = list(JoinMode),
#            default=JoinMode.AVERAGE,
#            help="How to combine gram matrices when multiple sources given")
join_mode=JoinMode.AVERAGE
#
#    parser.add_argument("--count", "-c", type=int, default=1,
#            help="How many images to generate simultaneously")
count=1
#    parser.add_argument("--mul", type=float, default=1.0, help="Multiply target grams by this amount")
mul=1
#    parser.add_argument("--if-weight", type=float, default=1., help="Inter-frame loss weight")
if_weight=1    
#    parser.add_argument("--if-shift", type=float, default=5., help="How many pixel-shift should inter-frame loss approximate?")
if_shift=5
#    parser.add_argument("--if-order", type=int, default=2, help="How many frames should we 'tie' together?")
if_order=2
#    parser.add_argument("--if-distance-type", type=str, choices = ['l2', 'lap1'], default="l2", help="How should we measure the distance between frames?")
if_distance_type='l2'
#    parser.add_argument("--if-octaves", type=int, default=1, help="At how many scales should the distance function operate?")
if_octaves=1
#    parser.add_argument("--seed", type=str, choices = ['random', 'symmetric'], default='random', help="How to seed the optimization")
seed='random'
#    parser.add_argument("--data-dir", "-d", type=str, default="model_data", help="Where to find the VGG weight files")
data_dir='model_data' 
#    parser.add_argument("--output-dir", type=str, default="outputs", help="Where to save the generated outputs")
output_dir='outputs'
#    parser.add_argument("--tol", type=float, default=1e-9, help="Gradient scale at which to terminate optimization")
tol=1e-9
#    parser.add_argument("--source", "-s", required=True, nargs='+',
#            help="List of file to use as source textures")
#
source_im=[
#'bark.jpg'        
'BrickRound0122_1_seamless_S.png'

#'bubble_1024.png'
#'BubbleMarbel.png'
#'CRW_3241_1024.png'
#'CRW_3444_1024.png'
#'fabric_white_blue_1024.png'
#'glass_1024.png'
#'lego_1024.png'
#'marbre_1024.png'
#'metal_ground_1024.png'
#'Pierzga_2006_1024.png'
#'rouille_1024.png'
#'Scrapyard0093_1_seamless_S.png'
#'TexturesCom_BrickSmallBrown0473_1_M_1024.png'
#'TexturesCom_FloorsCheckerboard0046_4_seamless_S_1024.png'
#'TexturesCom_TilesOrnate0085_1_seamless_S.png'
#'TexturesCom_TilesOrnate0158_1_seamless_S.png'
]
#    args = parser.parse_args()

# Any necessary validation here?
if if_octaves > octaves:
    print("Error: if_octaves must be less than octaves, but %d > %d" % (if_octaves, octaves))
    sys.exit(1)

output_size = (output_width, output_height if output_height is not None else output_width)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

_output_dir = "{}.L{}.o{}".format(output_prefix, ",".join(str(l) for l in layers), octaves)
output_dir = os.path.join(output_dir, _output_dir)
if source_scale:
    output_dir += ".w{:.2}".format(source_scale)
if source_width:
    output_dir += ".w{}".format(source_width)
if count > 1:
    output_dir += ".c{}.ifs{}".format(count, if_shift)
if mul != 1.0:
    output_dir += ".m{}".format(mul)
if join_mode != JoinMode.AVERAGE:
    output_dir += ".j{}".format(join_mode.value)
if if_octaves != 1:
    output_dir += ".ifo%d" % if_octaves

output_dir += ".{}x{}".format(*output_size)

suffix = 0
base_output_dir = output_dir
while os.path.exists(output_dir):
    output_dir = base_output_dir + ".{}".format(suffix)
    suffix += 1
    if suffix > 100:
        print("Hmm, maybe in an infinite loop trying to create output directory")
        sys.exit(1)

try:
    os.mkdir(output_dir)
except OSError:
    print("Hmm, failed to make output directory... race condition?")
    sys.exit(1)

# Save the command for good measure
with open(os.path.join(output_dir, "Acommand.txt"), 'w') as f:
    f.write(' '.join(sys.argv))

width = output_width
height = output_height or width

print("About to generate a {}x{} image, matching the Gram matrices for layers {} at {} distinct scales".format(width, height, layers, octaves))

pyramid_model = gram.make_pyramid_model(octaves, padding_mode)

pyramid_model_modified = gram.modify_pyramid(octaves, padding_mode)

pyramid_gram_model = gram.make_pyramid_gram_model(pyramid_model, layers, data_dir=data_dir)

target_pyramid = gram.get_pyramid_for_images( pyramid_model_modified, source_im,
       source_width = source_width, source_scale = source_scale)

target_grams = gram.get_gram_matrices_for_images(pyramid_gram_model, source_im,
        source_width = source_width, source_scale = source_scale, join_mode = join_mode)
target_grams = [t*mul for t in target_grams]
#target_grams = [np.max(t) - t for t in target_grams]

x0 = np.random.randn(count, height, width, 3)

if seed == 'symmetric':
    x0 = x0 + x0[:,::-1,    :, :]
    x0 = x0 + x0[:,   :, ::-1, :]
    blur_radius = 30
    for i in range(3):
        x0[...,i] = blur_radius*50*ndimage.gaussian_filter(x0[...,i], blur_radius)
    x0 += np.random.randn(*(x0.shape)) * 2
else:
    # Shift the whole thing to be near zero
    x0 += 10 - gram.colour_offsets

#x0 = preprocess(load_img('../sources/smokeb768.jpg'))

interframe_distances = []
if count > 1:
    for im in gram.get_images(source_im, source_scale = source_scale, source_width=source_width):
        interframe_distances.append(gram.interframe_distance(pyramid_model, im,
            shift=if_shift,
            interframe_distance_type = if_distance_type,
            interframe_octaves = if_octaves))

    print("Raw interframe distances: ")
    print(interframe_distances)

    #target_distances = np.mean(interframe_distances, axis=1)
    target_distances = interframe_distances[0]
    print("Shifting the source images by {} gives a {} interframe distance of approx {}".format(if_shift, if_distance_type, target_distances))
else:
    target_distances=None
    
    
continue_iter=False
x0_path='outputs/pier_contnd/I1450_F0000.png' 
if continue_iter:
    x0=load_img(x0_path)
    x0=gram.preprocess(x0)
gram.synthesize_animation(pyramid_model, pyramid_gram_model, target_grams, target_pyramid[0], use_spec,
        width = width, height = height, frame_count=count,
        x0 = x0,
        interframe_loss_weight=if_weight,
        interframe_order=if_order,
        target_interframe_distances = target_distances,
        interframe_distance_type = if_distance_type,
        interframe_octaves = if_octaves,
        output_directory = output_dir, max_iter=max_iter, save_every=save_every, tol=tol
        )

print("DONE: ")


