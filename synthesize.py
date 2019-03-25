# Copyright 2017, Xavier Snelgrove
import argparse
import os
import sys
import numpy as np
from scipy import ndimage
import gram
from gram import JoinMode
from keras.preprocessing.image import load_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthesize image from texture", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--output-width', '-ow', default=512, type=int,
            help="Pixel width of generated image")
    parser.add_argument('--output-height', '-oh', type=int,
            help="Pixel height of generated image. If not specified, equal to output-width.")
    parser.add_argument('--octaves', '-o',  type=int, default=4,
            help="Number of octaves (where 1 means \"Consider only 1 scale\")")
    parser.add_argument('--layers', '-l',  type=int, nargs='+', default=[2, 7],
            help="Which layers to match gram matrices on")
    parser.add_argument('--max-iter', '-m', type=int, default=501,
            help="Maximum iterations for the L-BFGS-B optimizer")
    parser.add_argument("--output-prefix", "-op", default='out',
            help="Prefix to append to output directory")
    parser.add_argument("--save-every", "-n", default=10, type=int,
            help="Save an in-progress optimization image every SAVE_EVERY iterations")
    parser.add_argument("--source-scale", "-ss", type=float,
            help="How much to scale the source image by")
    parser.add_argument("--source-width", "-sw", type=int,
            help="Scale source to this width. Mutually exclusive with source-scale")
    parser.add_argument("--padding-mode", "-p", type=str, choices = ['valid', 'same'], default='valid',
            help="What boundary condition to use for convolutions")
    parser.add_argument("--join-mode", "-j", type=JoinMode,
            choices = list(JoinMode),
            default=JoinMode.AVERAGE,
            help="How to combine gram matrices when multiple sources given")

    parser.add_argument("--count", "-c", type=int, default=1,
            help="How many images to generate simultaneously")
    parser.add_argument("--mul", type=float, default=1.0, help="Multiply target grams by this amount")
    parser.add_argument("--if-weight", type=float, default=1., help="Inter-frame loss weight")
    parser.add_argument("--if-shift", type=float, default=5., help="How many pixel-shift should inter-frame loss approximate?")
    parser.add_argument("--if-order", type=int, default=2, help="How many frames should we 'tie' together?")
    parser.add_argument("--if-distance-type", type=str, choices = ['l2', 'lap1'], default="l2", help="How should we measure the distance between frames?")
    parser.add_argument("--if-octaves", type=int, default=1, help="At how many scales should the distance function operate?")
    parser.add_argument("--seed", type=str, choices = ['random', 'symmetric'], default='random', help="How to seed the optimization")
    parser.add_argument("--data-dir", "-d", type=str, default="model_data", help="Where to find the VGG weight files")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Where to save the generated outputs")
    parser.add_argument("--tol", type=float, default=1e-9, help="Gradient scale at which to terminate optimization")
    parser.add_argument("--source", "-s", required=True, nargs='+',
            help="List of file to use as source textures")
    parser.add_argument("--use-spectrum", type=int, choices=[0,1], default=0, help= "use (1) or not (0) the spectrum loss")
    parser.add_argument("--spectrum-mul", type=float, default=1e-10, help= "multiply the spectrum loss by this number")

    args = parser.parse_args()

    # Any necessary validation here?
    if args.if_octaves > args.octaves:
        print("Error: if_octaves must be less than octaves, but %d > %d" % (args.if_octaves, args.octaves))
        sys.exit(1)
        
    if args.use_spectrum:
        args.source_width = args.output_width

    output_size = (args.output_width, args.output_height if args.output_height is not None else args.output_width)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    output_dir = "{}.L{}.o{}".format(args.output_prefix, ",".join(str(l) for l in args.layers), args.octaves)
    output_dir = os.path.join(args.output_dir, output_dir)
    if args.source_scale:
        output_dir += ".w{:.2}".format(args.source_scale)
    if args.source_width:
        output_dir += ".w{}".format(args.source_width)
    if args.count > 1:
        output_dir += ".c{}.ifs{}".format(args.count, args.if_shift)
    if args.mul != 1.0:
        output_dir += ".m{}".format(args.mul)
    if args.join_mode != JoinMode.AVERAGE:
        output_dir += ".j{}".format(args.join_mode.value)
    if args.if_octaves != 1:
        output_dir += ".ifo%d" % args.if_octaves

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

    width = args.output_width
    height = args.output_height or width

    print("About to generate a {}x{} image, matching the Gram matrices for layers {} at {} distinct scales".format(width, height, args.layers, args.octaves))

    
    pyramid_model = gram.make_pyramid_model(args.octaves, args.padding_mode)
    
    
    pyramid_gram_model = gram.make_pyramid_gram_model(pyramid_model, args.layers, data_dir=args.data_dir)
    

    target_pyramid = [0]
    if args.use_spectrum:
        
        pyramid_model_modified = gram.modify_pyramid(args.octaves, args.padding_mode)
        
        target_pyramid = gram.get_pyramid_for_images( pyramid_model_modified, args.source,
               source_width = args.source_width, source_scale = args.source_scale)
        
    target_grams = gram.get_gram_matrices_for_images(pyramid_gram_model, args.source,
            source_width = args.source_width, source_scale = args.source_scale, join_mode = args.join_mode)
    target_grams = [t*args.mul for t in target_grams]
    #target_grams = [np.max(t) - t for t in target_grams]
    
    x0 = np.random.randn(args.count, height, width, 3)

    if args.seed == 'symmetric':
        x0 = x0 + x0[:,::-1,    :, :]
        x0 = x0 + x0[:,   :, ::-1, :]
        blur_radius = 30
        for i in range(3):
            x0[...,i] = blur_radius*50*ndimage.gaussian_filter(x0[...,i], blur_radius)
        x0 += np.random.randn(*(x0.shape)) * 2
    else:
        # Shift the whole thing to be near zero
        x0 += 10 - gram.colour_offsets


    continue_iter=0
    x0_path='outputs/greenfractal1/I0500_F0000.png' 
    if continue_iter:
        x0=load_img(x0_path)
        x0=gram.preprocess(x0)
    #x0 = preprocess(load_img('../sources/smokeb768.jpg'))
    
    interframe_distances = []
    if args.count > 1:
        for im in gram.get_images(args.source, source_scale = args.source_scale, source_width=args.source_width):
            interframe_distances.append(gram.interframe_distance(pyramid_model, im,
                shift=args.if_shift,
                interframe_distance_type = args.if_distance_type,
                interframe_octaves = args.if_octaves))

        print("Raw interframe distances: ")
        print(interframe_distances)

        #target_distances = np.mean(interframe_distances, axis=1)
        target_distances = interframe_distances[0]
        print("Shifting the source images by {} gives a {} interframe distance of approx {}".format(args.if_shift, args.if_distance_type, target_distances))
    else:
        target_distances=None

    gram.synthesize_animation(pyramid_model, pyramid_gram_model, target_grams, target_pyramid[0], args.use_spectrum, args.spectrum_mul,
            width = width, height = height, frame_count=args.count,
            x0 = x0,
            interframe_loss_weight=args.if_weight,
            interframe_order=args.if_order,
            target_interframe_distances = target_distances,
            interframe_distance_type = args.if_distance_type,
            interframe_octaves = args.if_octaves,
            output_directory = output_dir, max_iter=args.max_iter, save_every=args.save_every, tol=args.tol
            )

    print("DONE: ")


