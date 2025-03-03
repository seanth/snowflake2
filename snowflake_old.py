#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/gist/vishnubob/fd4eb04d21716c2c180c2f3b72202d03/snowflake-generator-2-0-sean-hammond.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Snowflake Generator 2.0
# 
# This is a snowflake simulator. It generates realistic looking snowflakes by modeling the cellular diffusion of water vapor along with the phase transitions from gas to frozen ice at a mesoscopic scale. In other words, each cell in the simulation represents millions of water molecules.
# 
# The underlying math driving the model is adopted from the research paper "[MODELING SNOW CRYSTAL GROWTH II: A mesoscopic lattice map with plausible dynamics](https://www.sciencedirect.com/science/article/abs/pii/S0167278907003387)" by Janko Gravner and David Griffeath.
# 
# The first revision of this generator ran entirely on the CPU and required over an hour of computation for each snowflake.  This rewrite introduces a vectorized version of the engine, and is capable of running on CPUs or GPUs.  On modern GPUs, a single snowflake takes just seconds to render.
# 
# This notebook takes a name as a seed for generating a unique snowflake.  The output creates a few plots and an SVG optimized for laser cutting.
# 
# 
# 


salt = 29 #@param {type:"slider", min:0, max:1024, step:1}
mass_ratio_cutoff = 0.25 #@param {type:"slider", min:0.01, max:0.99, step:0.01}

import argparse
import hashlib
import os
from pprint import pprint
# import json
from math import sqrt
from xml.dom import minidom
import sys #only for debug sys.exit() calls


import configparser                         #pip install configparser
import numpy as np                          #pip install numpy
from scipy.interpolate import CubicSpline   #pip install scipy
import scipy.ndimage as ndi
import jax.numpy as jnp                     #pip install jax
import jax
import jax.scipy as jsp
from PIL import Image                       #pip install pillow
import matplotlib.pyplot as plt             #pip install matplotlib
from sklearn.cluster import KMeans          #?pip install sklearn?
from svgpathtools import parse_path         #pip install svgpathtools
import svgwrite                             #pip install svgwrite


##########################################
#Import the options#
try:
    theConfig=configparser.RawConfigParser()
    theConfig.optionxform = str 
    theConfig.read('snowflake.ini')
    theConfigSection='Snowflake Options'
except configparser.MissingSectionHeaderError:
    print("Warning: Invalid config file, no [%s] section.") % (theConfigSection)
    raise

SNOWFLAKE_DEFAULTS={}
for i in theConfig.items(theConfigSection):
    theItem=i[0]
    try:
        theValue=theConfig.getint(theConfigSection, theItem)
    except:
        try:
            theValue=theConfig.getboolean(theConfigSection, theItem)
        except:
            try:
                theValue=theConfig.getfloat(theConfigSection, theItem)
            except:
                try:
                    theValue=theConfig.get(theConfigSection, theItem)
                    if theValue=="None": theValue=None
                except:
                    print("what the...?")
    SNOWFLAKE_DEFAULTS[theItem]=theValue



DefaultCurves = {
    "beta": (1.3, 2),
    "theta": (0.01, 0.04),
    "alpha": (0.02, 0.1),
    "kappa": (0.001, 0.01),
    "mu": (0.01, 0.1),
    "upsilon": (0.00001, 0.0001),
    "sigma": (0.00001, 0.000001),
    "gamma": (0.45, 0.85),
}
ParamOrder = ('beta', 'theta', 'alpha', 'kappa', 'mu', 'upsilon', 'sigma')

conv2d = jsp.signal.convolve

fileNameList = []

def get_gamma_and_params(name, max_steps=SNOWFLAKE_DEFAULTS['max_steps'], salt=0, curves=DefaultCurves):
    t_value = name.encode('utf8')
    hs = hashlib.sha256(t_value)
    hs.hexdigest()
    seed = (int(hs.hexdigest(), base=16) + salt) % (2 ** 32 - 1)
    np.random.seed(seed)
    n_knots = np.random.randint(2, 6, 1)[0] + 1
    df = curves.copy()
    gamma_minmax = df.pop('gamma')
    n_params = len(df)
    minmax = np.array([np.array(df[key]) for key in ParamOrder]).T
    spread = minmax[1] - minmax[0]
    gamma = np.random.uniform(*gamma_minmax, 1)[0]
    knots = np.random.rand(n_params, n_knots) * spread[:, np.newaxis] + minmax[0][:, np.newaxis]
    knot_steps = np.sort(np.round((np.random.rand(n_knots - 1) * max_steps)).astype(int))
    knot_steps = np.array([0] + list(knot_steps))
    ncs = CubicSpline(knot_steps, knots, axis=1)
    params = ncs(np.arange(max_steps))
    return (seed, gamma, params.T)
    
def prep_image(nd, bbox=None, return_bbox=False, zoom_factor=2):
    nd = ndi.rotate(nd, 45)
    nd = ndi.zoom(nd, (1, 1 / sqrt(3)))
    nd = ndi.rotate(nd, -45)
    if bbox is None:
        bbox = ndi.find_objects(nd, max_label=1)[0]
    nd = nd[bbox]
    if zoom_factor:
        nd = ndi.zoom(nd, (zoom_factor, zoom_factor))
    if return_bbox:
        return (nd, bbox)
    return nd

#def potrace(input_img, spot_threshold=None, size=None, margin=None, angle=None, dpi=96, tight=False, theLayer=''):
def potrace(img, spot_threshold=None, size=None, margin=None, angle=None, dpi=96, tight=False, theLayer=''):
    ###### start building out the command
    #-i, --invert. invert the input bitmap before processing.
    cmd = ['potrace -i -b svg --flat']
    if margin != None:
        #-M dim, --margin dim
        #set all four margins. The effect and default value of this option 
        #depend on the backend. For variable-sized backends, the margins will 
        #simply be added around the output image (or subtracted, in case of 
        #negative margins). The default margin for these backends is 0. For 
        #fixed-size backends, the margin settings can be used to control the 
        #placement of the image on the page. If only one of the left and right 
        #margin is given, the image will be placed this distance from the respective 
        #edge of the page, and similarly for top and bottom. If margins are given on 
        #opposite sides, the image is scaled to fit between these margins, unless 
        #the scaling is already determined explicitly by one or more of the -W, 
        #-H, -r, or -x options. By default, fixed-size backends use a non-zero 
        #margin whose width depends on the page size.
        cmd.append(f'-M {margin}')
    if spot_threshold != None:
        #-t n, --turdsize n. suppress speckles of up to this many pixels.
        cmd.append(f'-t {spot_threshold}')
    if size != None:
        #-W dim, --width dim
        #set the width of output image (before any rotation and margins). 
        #If only one of width and height is specified, the other is adjusted 
        #accordingly so that the aspect ratio is preserved.
        #-H dim, --height dim
        #set the height of output image. See -W for details.
        cmd.append(f'-W {size[0]} -H {size[1]}')
    if angle != None:
        #-A angle, --rotate angle
        #set the rotation angle (in degrees). The output will be rotated 
        #counterclockwise by this angle. This is useful for compensating 
        #for images that were scanned not quite upright.
        cmd.append(f'-A {angle}')
    if dpi != None:
        #-r n[xn], --resolution n[xn]
        #for dimension-based backends, set the resolution (in dpi). One inch 
        #in the output image corresponds to this many pixels in the input. 
        #Note that a larger value results in a smaller output image. It is 
        #possible to specify separate resolutions in the x and y directions 
        #by giving an argument of the form nxn. For variable-sized backends, 
        #the default resolution is 72dpi. For fixed-size backends, there is no 
        #default resolution; the image is by default scaled to fit on the page. 
        #This option has no effect for pixel-based backends. If -W or -H are 
        #specified, they take precedence.
        cmd.append(f'-r {dpi}')
    if tight:
        #--tight
        #remove whitespace around the image before scaling and margins are 
        #applied. If this option is given, calculations of the width, height, 
        #and margins are based on the actual vector outline, rather than on 
        #the outer dimensions of the input pixmap, which is the default. In 
        #particular, the --tight option can be used to remove any existing 
        #margins from the input image. 
        cmd.append('--tight')
    input_img = os.path.join(f'{root}/{args.name}_{sn}/', f'input_image{theLayer}.bmp')
    output_svg = os.path.join(f'{root}/{args.name}_{sn}/', f'output_trace{theLayer}.svg')
    print("   ***Generating BMP %s..." % os.path.basename(input_img))
    img.save(input_img)
    #-o filename, --output filename
    #write output to this file. All output is directed to the specified file. If 
    #this option is used, then multiple input filenames are only allowed for multi-page 
    #backends (see BACKEND TYPES below). In this case, each input file may contain 
    #one or more bitmaps, and all the bitmaps from all the input files are processed 
    #and the output concatenated into a single file. A filename of "-" may be given to 
    #specify writing to standard output.
    cmd.append(f'-o {output_svg} {input_img}')
    cmd = str.join(' ', cmd)
    print("   ***Generating SVG %s..." % os.path.basename(output_svg))
    msg = "         Running '%s'" % cmd
    print(msg)
    os.system(cmd) #!{cmd}
    doc = minidom.parse(output_svg)
    width = doc.getElementsByTagName('svg')[0].getAttribute('width')
    height = doc.getElementsByTagName('svg')[0].getAttribute('height')
    paths = [path.getAttribute('d') for path in doc.getElementsByTagName('path')]
    return (width, height, paths, input_img)

def plot_snowflake(crystal_mass=None, attached=None, diffusive_mass=None):
    thePath = f'{root}/{args.name}_{sn}/{args.name}_{sn}.png'
    sz = 7
    fig = plt.figure(figsize=(sz, sz))
    if args.show_axis==False:
        plt.axis("off")
    plt.imshow(prep_image(crystal_mass), interpolation='nearest', cmap=args.color_map)
    #plt.show()
    fig.savefig(thePath)
    #sys.exit()
    
    # fig = plt.figure(figsize=(sz, sz))
    # plt.imshow(prep_image(attached), interpolation='nearest')
    # plt.show()
    
    # fig = plt.figure(figsize=(sz, sz))
    # plt.imshow(prep_image(diffusive_mass), interpolation='nearest')
    # plt.show()

def render_svg(crystal_mass=None, attached=None, size=3.5, margin=.1, random_state=1, svg_fn='output.svg'):
    cm = crystal_mass.reshape(-1, 1)
    sample_weight = np.ravel(attached)
    bands = KMeans(n_clusters=args.numb_layers, random_state=random_state, n_init=10).fit_predict(cm, sample_weight=sample_weight)
    unique, counts = np.unique(bands, return_counts=1)
    bands = bands.reshape(crystal_mass.shape)
    
    (img_ary_, bbox) = prep_image(attached, return_bbox=True, zoom_factor=5)
    img_ary = prep_image(attached, zoom_factor=5)
    img_ary = (img_ary[:, :, np.newaxis] * [0xFF, 0xFF, 0xFF]).astype(np.uint8)
    img = Image.fromarray(img_ary, mode='RGB')
    img_size = img.size
    act_size = (size - 2 * margin)
    act_unit = "in"
    img_size = (f'{act_size}{act_unit}', f'{act_size * (img_size[1] / img_size[0]):.02f}{act_unit}')
    (svg_width, svg_height, paths, bmpFileName) = potrace(img, size=img_size, spot_threshold=args.spot_threshold)
    cut_path = paths[0]
    cut_bbox = np.array(parse_path(cut_path).bbox()).reshape(2, 2).T
    lengths = cut_bbox[1,:] - cut_bbox[0,:]
    cut_bbox = parse_path(cut_path).bbox()
    
    oneten = margin * lengths / act_size
    tru_size = (f'{size}in', f'{size}in')
    dwg = svgwrite.Drawing(svg_fn, size=tru_size, debug=True)
    lengths += (2 * oneten)
    dwg.viewbox(minx=0, width=lengths[0], miny=0, height=lengths[1])
    #dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), rx=None, ry=None, fill='white'))
    root_grp = dwg.g()
    grp = dwg.g()
    root_grp.add(grp)
    grp.translate(oneten)

    global fileNameList
    fileNameList = []
    colors = ['blue', 'cyan', 'green', 'purple', 'orange', 'red']
    for band_idx in range(len(unique)):
        img_ary = prep_image(attached * (bands == band_idx), bbox=bbox, zoom_factor=5)
        img_ary = (img_ary[:, :, np.newaxis] * [0xFF, 0xFF, 0xFF]).astype(np.uint8)
        img = Image.fromarray(img_ary, mode='RGB')
        (svg_width, svg_height, paths, bmpFileName) = potrace(img, size=img_size, theLayer=band_idx)
        fileNameList.append(bmpFileName)
        for path in paths:
            color = colors[band_idx]
            grp.add(dwg.path(d=path, stroke=color, fill='none', stroke_width='.2in'))
    del fileNameList[0]

    grp.add(dwg.path(d=cut_path, stroke='white', fill='none', stroke_width='.2in'))
    root_grp.add(dwg.rect(insert=(0, 0), size=(lengths[0], lengths[1]), rx=100, ry=100, fill='none', stroke='white', stroke_width='.2in'))
    dwg.add(root_grp)
    dwg.save()
        
def do_step(diffusive_mass=None, boundary_mass=None, crystal_mass=None, attached=None, rngkey=None, gamma=None, params=None):
    (beta,  theta, alpha, kappa, mu, upsilon, sigma) = params

    attached = attached.astype(bool)
    #
    # calculate boundary
    n_attached = conv2d(attached, neighbor_mask, mode="same") * (1 - attached)
    boundary = (n_attached >= 1)
    #
    # diffusion step
    conv_diffusive_mass = conv2d(jnp.pad(diffusive_mass, 1, constant_values=(gamma, )), self_and_neighbor_mask, mode="valid")
    diffusive_mass = (n_attached * diffusive_mass + conv_diffusive_mass) / 7.0 * (1 - attached)
    #
    # freeze step
    boundary_mass += ((1 - kappa) * diffusive_mass) * boundary
    crystal_mass += (kappa * diffusive_mass) * boundary
    diffusive_mass *= (1 - boundary)
    #
    # attachment step
    summed_diffusion = conv2d(jnp.pad(diffusive_mass, 1), self_and_neighbor_mask, mode="valid")
    attachments = (n_attached >= 4)
    attachments += \
        jnp.logical_and(n_attached == 3, jnp.logical_or(boundary_mass >= 1, 
            jnp.logical_and(summed_diffusion < theta, boundary_mass >= alpha)))
    attachments += jnp.logical_and(n_attached <= 2, boundary_mass >= beta)
    attachments *= boundary
    attachments = (attachments > 0)
    #
    # process attachments
    crystal_mass += (attachments * boundary_mass)
    boundary_mass = ((1 - attachments) * boundary_mass)
    attached = ((attached + attachments) > 0)
    #
    # re-calculate boundary
    n_attached = conv2d(attached, neighbor_mask, mode="same") * (1 - attached)
    boundary = (n_attached >= 1)
    #
    # melt step
    diffusive_mass += (mu * boundary_mass + upsilon * crystal_mass) * boundary
    boundary_mass = ((1 - mu) * boundary * boundary_mass) + ((1 - boundary) * boundary_mass)
    crystal_mass = ((1 - upsilon) * boundary * crystal_mass) + ((1 - boundary) * crystal_mass)
    #
    # add noise to diffusive mass
    noise = jax.random.uniform(rngkey, diffusive_mass.shape)
    noise = ((noise < .5) * -1) + (noise > .5)
    diffusive_mass = (1 + (noise * sigma)) * diffusive_mass
    #
    attached = attached.astype(jnp.uint8)
    return (diffusive_mass, boundary_mass, crystal_mass, attached)

def run_simulation(random_seed=None, mass_ratio_cutoff=.25, max_steps=SNOWFLAKE_DEFAULTS['max_steps'], gamma=None, params=None):
    random_seed = random_seed if random_seed is not None else 1

    # initialize
    diffusive_mass = jnp.ones(args.shape) * gamma
    boundary_mass = jnp.zeros(args.shape)
    crystal_mass = jnp.zeros(args.shape)
    attached = jnp.zeros(args.shape).astype(np.uint8)
    next_key = jax.random.PRNGKey(random_seed)
    cur_step = 0
    # attach seed crystal
    seed_idx = (args.width // 2, args.height // 2)
    attached = attached.at[seed_idx].set(1)
    crystal_mass = crystal_mass.at[seed_idx].set(diffusive_mass[seed_idx])
    diffusive_mass = diffusive_mass.at[seed_idx].set(0)
    # run simulation
    total_diff_mass = jnp.sum(diffusive_mass)
    for cur_step in range(max_steps):
        mass_ratio = jnp.sum(boundary_mass + crystal_mass) / total_diff_mass
        ###feedback ported from v1
        ###nice to have the masses show in a classroom setting
        if cur_step % 50 == 0:
            msg = "Step #:%d/%d (%d%%), dM: %.2f, bM:%.2f, cM:%.2f, totM %.2f, ratio[(b+c)/d]M: %.2f" % (cur_step, max_steps, ((cur_step/max_steps)*100), jnp.sum(diffusive_mass), jnp.sum(boundary_mass), jnp.sum(crystal_mass), jnp.sum(diffusive_mass) + jnp.sum(crystal_mass) + jnp.sum(boundary_mass), mass_ratio)
            print(msg)
        if mass_ratio >=  mass_ratio_cutoff:
            return (diffusive_mass, boundary_mass, crystal_mass, attached, cur_step)
        (next_key, step_key) = jax.random.split(next_key)
        p_step = tuple(params[cur_step])
        (diffusive_mass, boundary_mass, crystal_mass, attached) = do_step(diffusive_mass, boundary_mass, crystal_mass, attached, step_key, gamma, p_step)

    return (diffusive_mass, boundary_mass, crystal_mass, attached, cur_step)

def get_cli():
    parser = argparse.ArgumentParser(description='Snowflake Generator.')
    parser.add_argument('-n', '--name', dest="name", type=str, help="[str] The name of the snowflake.")
    parser.add_argument('-s', '--size', dest="size", type=int, help="[int] The size of the snowflake.")
    #parser.add_argument('-e', '--env', dest='env', help='Comma seperated key=val env overrides')
    #parser.add_argument('-b', '--bw', dest='bw', action='store_true', help='Write out the image in black and white.')
    #parser.add_argument('-r', '--randomize', dest='randomize', action='store_true', help='Randomize environment.')
    parser.add_argument('-X', '--extrude', dest='pipeline_3d', action='store_true', help='Enable 3d pipeline. False when absent')
    parser.add_argument('-L', '--laser', dest='pipeline_lasercutter', action='store_true', help='Enable Laser Cutter pipeline.')
    parser.add_argument('-m', '--max-steps', dest='max_steps', type=int, help='[int] Maximum number of iterations.')
    parser.add_argument('-M', '--margin', dest='margin', type=float, help='When to stop snowflake growth (between 0 and 1).')
    #parser.add_argument('-c', '--curves', dest='curves', action='store_true', help='Enable use of name to generate environment curves.')
    parser.add_argument('-c', '--color_map', dest='color_map', type=str, help='[str] What colour map should be used for flake image')
    parser.add_argument('-a', '--show_axis', dest='show_axis', type=bool, help='[bool] Show an axis on the flake image[True/False]')
    parser.add_argument('-l', '--layers', dest='numb_layers', type=int, help='[int] Number of layers to produce for 3d printing. Max 5')
    #parser.add_argument('-L', '--datalog', dest='datalog', action='store_true', help='Enable step wise data logging.')
    #parser.add_argument('-D', '--debug', dest='debug', action='store_true', help='Show every step.')
    #parser.add_argument('-v', '--movie', dest='movie', action='store_true', help='Render a movie.')
    parser.add_argument('-W', '--width', dest='width', type=float, help="[int] Width of target render.")
    parser.add_argument('-H', '--height', dest='height', type=float, help="[int] Height of target render.")

    parser.set_defaults(**SNOWFLAKE_DEFAULTS)
    args = parser.parse_args()

    args.name = str.join('', map(str.lower, args.name))
    ####CubicSpline is unhappy with arithmatic characters in the seed name
    args.name = ''.join(filter(str.isalpha,args.name))

    ###Can't be greater than 5 because of the colours
    ###defined in render_svg()
    if args.numb_layers>5: args.numb_layers=5

    #args.target_size = None
    if args.width and args.height:
        args.shape = (args.width, args.height)

    return args

if __name__ == "__main__":
    args = get_cli()
    print("Simulation arguments:")
    print("    Name: ",args.name)
    print("    Size: ", args.size)
    print("    Width: ", args.width)
    print("    Height: ", args.height)
    print("    Number of layers: ", args.numb_layers)
    print("    Prep for 3d printing: ", args.pipeline_3d)
    print("    Prep for lasercutting: ", args.pipeline_lasercutter)


    print("Initializing parameters...")
    (random_seed, gamma, params) = get_gamma_and_params(name=args.name, max_steps=args.max_steps, salt=salt)
    print("Random seed: ", random_seed)
    #sys.exit()

    print("Making necessary folders...")
    sn = hex(random_seed)[2:]
    root = 'content/SnowflakeDesigns'
    theDir = (f'{root}/{args.name}_{sn}')
    os.makedirs(theDir, exist_ok=True)

    ####################################################
    self_mask = np.zeros((3,3))
    self_mask[1,1] = 1
    neighbor_mask = np.fliplr(1 - np.eye(3))
    self_and_neighbor_mask = self_mask + neighbor_mask

    # convert numpy arrays over to JAX
    self_mask = jnp.array(self_mask)
    neighbor_mask = jnp.array(neighbor_mask)
    self_and_neighbor_mask = jnp.array(self_and_neighbor_mask)
    ####################################################



    print("Starting simulation...")
    (diffusive_mass, boundary_mass, crystal_mass, attached, cur_step) =  run_simulation(random_seed=random_seed, mass_ratio_cutoff=mass_ratio_cutoff, max_steps=args.max_steps, gamma=gamma, params=params)
    print("End simulation")
    print()
    print("***Generating colourized PNG...")
    plot_snowflake(crystal_mass=crystal_mass, attached=attached, diffusive_mass=diffusive_mass)

    ####################################################
    svg_fn = f'{root}/{args.name}_{sn}/{args.name}_{sn}.svg'
    print("***Generating BMP and SVG files...")
    render_svg(crystal_mass=crystal_mass, attached=attached, margin=args.margin, random_state=random_seed, svg_fn=svg_fn)
    #sys.exit()


    if args.pipeline_3d==True:
        ###The flow is 1.)bmp-->2.)eps-->3.)dxf-->4.)stl
        ###using 1.)potrace-->2.)potrace-->3.)pstoedit-->4.)openscad
        print(fileNameList)
        sys.exit()

        # print("***Attempting to turn BMP layers into EPS files...")
        # ###can't use potrace to go straight to dxf
        # ###openSCAD can't understand POLYLINE or VERTEX
        # epsFileList = []
        # #########
        # theNewDir = (f'{theDir}/EPS_folder')
        # os.makedirs(theNewDir, exist_ok=True)
        # #########
        # for fileName in fileNameList:
        #     epsFileName = os.path.split(fileName)[1]
        #     epsFileName = os.path.splitext(epsFileName)[0]
        #     epsFileName = "%s_bmp.eps" % epsFileName
        #     epsFileName = os.path.join(theNewDir, epsFileName)
        #     #-i, --invert. invert the input bitmap before processing.
        #     cmd = ['potrace -i -b eps']
        #     if args.margin != None:
        #         #-M dim, --margin dim
        #         #set all four margins. The effect and default value of this option 
        #         #depend on the backend. For variable-sized backends, the margins will 
        #         #simply be added around the output image (or subtracted, in case of 
        #         #negative margins). The default margin for these backends is 0. For 
        #         #fixed-size backends, the margin settings can be used to control the 
        #         #placement of the image on the page. If only one of the left and right 
        #         #margin is given, the image will be placed this distance from the respective 
        #         #edge of the page, and similarly for top and bottom. If margins are given on 
        #         #opposite sides, the image is scaled to fit between these margins, unless 
        #         #the scaling is already determined explicitly by one or more of the -W, 
        #         #-H, -r, or -x options. By default, fixed-size backends use a non-zero 
        #         #margin whose width depends on the page size.
        #         cmd.append(f'-M {args.margin}')
        #     #do no use --tight. It reduces the size of the file and results in openScad
        #     #not being able to center things correctly
        #     cmd.append(f'-o {epsFileName} {fileName}')
        #     cmd = str.join(' ', cmd)
        #     msg = "Running '%s'" % cmd
        #     print(msg)
        #     os.system(cmd)
        #     epsFileList.append(epsFileName)

        # print("***Attempting to generate DXF using pstoedit...")
        # dfxFileList=[]
        # #########
        # theNewDir = (f'{theDir}/DXF_folder')
        # os.makedirs(theNewDir, exist_ok=True)
        # #########
        # for fileName in epsFileList:
        #     dxfFileName = os.path.split(fileName)[1]
        #     dxfFileName = os.path.splitext(dxfFileName)[0]
        #     dxfFileName = "%s_eps.dxf" % dxfFileName
        #     dxfFileName = os.path.join(theNewDir, dxfFileName)
        #     #on windows it _needs_ to have the dxf part in double quotes
        #     #STH 2018.0212
        #     #Have not confirmed this is still true in 2023
        #     #STH 2023.0313
        #     ###############################################
        #     #-dt: boolean   : draw text, i.e. convert text to polygons]
        #     #-f: string    : target format identifier
        #     #dxf: Format group. dxf:   CAD exchange format version 9 - only limited features. Consider using dxf_14 instead.
        #     #-mm: boolean   : use mm coordinates instead of points in DXF (mm=pt/72*25.4)
        #     #-polyaslines: boolean   : use LINE instead of POLYLINE in DXF]
        #     cmd = f'pstoedit -dt -f "dxf:-polyaslines -mm" {fileName} {dxfFileName}'
        #     msg = "Running '%s'" % cmd
        #     print(msg)
        #     os.system(cmd)
        #     dfxFileList.append(dxfFileName)

        print("***Attempting to generate a scad file...")
        scadFileName = "%s_3d.scad" % args.name
        scadFileName = os.path.join(theDir, scadFileName)
        f = open(scadFileName, 'w')
        #I could use args.numb_layers, but the number of files _might_ differ
        numLayers=float(len(dfxFileList))
        heightPerLayer=1.0/numLayers
        #for (i, fileName) in enumerate(dfxFileList,1):
        for (i, fileName) in enumerate(dfxFileList,1):
            fileName=os.path.abspath(fileName)
            # scad_txt = 'scale([30, 30, 30]) linear_extrude(height=%f, layer="0") import("%s");\n' % (i*0.18, fileName)
            #scad_txt = 'scale([30, 30, 30]) linear_extrude(height=%f, layer="0") import("%s");\n' % (i*heightPerLayer, fileName)
            scad_txt = 'linear_extrude(height=%f, layer="0") import(file="%s", center = true, dpi = 96);\n' % (i*heightPerLayer, fileName)
            f.write(scad_txt)
        f.close()
        sys.exit()
        print("***Attempting to generate a STL using OpenSCAD...")
        stlFileName = "%s_3d.stl" % args.name
        stlFileName = os.path.join(theDir, stlFileName)
        if sys.platform=="win32":
            cmd = f'openscad.com -o {stlFileName} {scadFileName}'
        else:
            cmd = f'{args.OpenSCADPath}/Contents/MacOS/OpenSCAD -o {stlFileName} {scadFileName} '
        msg = "Running '%s'" % cmd
        print(msg)
        #os.system(cmd)

    ####################################################
    print()
    plot_snowflake(crystal_mass=crystal_mass, attached=attached, diffusive_mass=diffusive_mass)

    info = {
        'name': args.name,
        'salt': salt,
        'n_steps': cur_step,
        'seed': random_seed,
        'sn': sn,
        'mass_ratio_cutoff': mass_ratio_cutoff,
        'spot_threshold': args.spot_threshold
    }

    ###but why? STH 0313-2023##############################
    # jsfn = f'{root}/{args.name}_{sn}/{args.name}_{sn}.json'
    # with open(jsfn, 'w') as fh:
    #     json.dump(info, fh)
    #######################################################

    print(f"\n\n{'-' * 20}\nGenerator details\n{'-' * 20}\n")
    pprint(info)