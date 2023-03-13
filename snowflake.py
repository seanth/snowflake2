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

#spot threshold (potrace)
spot_threshold = 0 #@param {type:"slider", min:0, max:500, step:10}

import argparse
import hashlib
import os
from pprint import pprint
# import json
from sklearn.cluster import KMeans
from math import sqrt

import configparser                         #pip install configparser
import numpy as np                          #pip install numpy
from scipy.interpolate import CubicSpline   #pip install scipy
import scipy.ndimage as ndi
import jax.numpy as jnp                     #pip install jax
import jax
import jax.scipy as jsp
from PIL import Image                       #pip install pillow
import matplotlib.pyplot as plt             #pip install matplotlib

# from PIL import Image, ImageDraw, ImageFont

# import tempfile
# import random
import matplotlib.pyplot as plt

# from enum import IntEnum
# from xml.dom import minidom

# from IPython.core.display import SVG
# import svgwrite
# from svgpathtools import parse_path
# from scipy import interpolate
import sys

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

# def potrace(img, spot_threshold=None, size=None, margin=None, angle=None, dpi=96, tight=False):
#     thePath = (f'{root}/{args.name}_{sn}/')
#     # start building out the command
#     cmd = ['potrace -i -b svg --flat']
#     if margin != None:
#         cmd.append(f'-M {margin}')
#     if spot_threshold != None:
#         cmd.append(f'-t {spot_threshold}')
#     if size != None:
#         cmd.append(f'-W {size[0]} -H {size[1]}')
#     if angle != None:
#         cmd.append(f'-A {angle}')
#     if dpi != None:
#         cmd.append(f'-r {dpi}')
#     if tight:
#         cmd.append('--tight')
#     print(cmd)
#     input_img = os.path.join(f'{root}/{args.name}_{sn}/', 'input_image.bmp')
#     output_svg = os.path.join(f'{root}/{args.name}_{sn}/', 'output_trace.svg')
#     print(input_img)
#     print(output_svg)
#     img.save(input_img)
#     sys.exit()
#     # acquire temp directory
#     # with tempfile.TemporaryDirectory() as tmp_dir:
#     #     input_img = os.path.join(tmp_dir, 'input_image.bmp')
#     #     output_svg = os.path.join(tmp_dir, 'output_trace.svg')
#     #     img.save(input_img)
#     #     #
#     #     cmd.append(f'-o {output_svg} {input_img}')
#     #     cmd = str.join(' ', cmd)
#     #     #!{cmd}
#     #     doc = minidom.parse(output_svg)
#     width = doc.getElementsByTagName('svg')[0].getAttribute('width')
#     height = doc.getElementsByTagName('svg')[0].getAttribute('height')
#     paths = [path.getAttribute('d') for path in doc.getElementsByTagName('path')]
#     return (width, height, paths)

def plot_snowflake(crystal_mass=None, attached=None, diffusive_mass=None):
    thePath = f'{root}/{args.name}_{sn}/{args.name}_{sn}.png'
    sz = 7
    fig = plt.figure(figsize=(sz, sz))
    if args.show_axis==False:
        plt.axis("off")
    plt.imshow(prep_image(crystal_mass), interpolation='nearest', cmap=args.color_map)
    #plt.show()
    fig.savefig(thePath)
    
    # fig = plt.figure(figsize=(sz, sz))
    # plt.imshow(prep_image(attached), interpolation='nearest')
    # plt.show()
    
    # fig = plt.figure(figsize=(sz, sz))
    # plt.imshow(prep_image(diffusive_mass), interpolation='nearest')
    # plt.show()
    

def render_svg(crystal_mass=None, attached=None, size=3.5, margin=.1, random_state=1, svg_fn='output.svg'):
    cm = crystal_mass.reshape(-1, 1)
    sample_weight = np.ravel(attached)
    bands = KMeans(n_clusters=3, random_state=random_state).fit_predict(cm, sample_weight=sample_weight)
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
    (svg_width, svg_height, paths) = potrace(img, size=img_size, spot_threshold=spot_threshold)
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
    
    colors = ['blue', 'cyan', 'green', 'purple', 'orange', 'red']
    for band_idx in range(len(unique)):
        img_ary = prep_image(attached * (bands == band_idx), bbox=bbox, zoom_factor=5)
        img_ary = (img_ary[:, :, np.newaxis] * [0xFF, 0xFF, 0xFF]).astype(np.uint8)
        img = Image.fromarray(img_ary, mode='RGB')
        (svg_width, svg_height, paths) = potrace(img, size=img_size)
        for path in paths:
            color = colors[band_idx]
            grp.add(dwg.path(d=path, stroke=color, fill='none', stroke_width='.2in'))
    
    grp.add(dwg.path(d=cut_path, stroke='white', fill='none', stroke_width='.2in'))
    root_grp.add(dwg.rect(insert=(0, 0), size=(lengths[0], lengths[1]), rx=100, ry=100, fill='none', stroke='white', stroke_width='.2in'))
    dwg.add(root_grp)
    dwg.save()
    svg = SVG(filename=svg_fn)
    display(svg)
        
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
            #msg = "Step #%d/%dp (%.2f%% scl), %d/%d (%.2f%%), %.2f dM, %.2f bM, %.2f cM, tot %.2f M" % (self.iteration, d, (float(d * 2 * X_SCALE_FACTOR) / self.iteration) * 100, acnt, bcnt, (float(bcnt) / acnt) * 100, dm, bm, cm, dm + cm + bm)
            #msg = "Step #:%d/%d (%d%%), dM: %.2f, bM:%.2f, cM:%.2f, totM %.2f, ratioM: %.2f" % (cur_step, max_steps, ((cur_step/max_steps)*100), ,jnp.sum(diffusive_mass), jnp.sum(boundary_mass), jnp.sum(crystal_mass), jnp.sum(diffusive_mass) + jnp.sum(crystal_mass) + jnp.sum(boundary_mass), mass_ratio)
            msg = "Step #:%d/%d (%d%%), dM: %.2f, bM:%.2f, cM:%.2f, totM %.2f, ratio[(b+c)/d]M: %.2f" % (cur_step, max_steps, ((cur_step/max_steps)*100), jnp.sum(diffusive_mass), jnp.sum(boundary_mass), jnp.sum(crystal_mass), jnp.sum(diffusive_mass) + jnp.sum(crystal_mass) + jnp.sum(boundary_mass), mass_ratio)
            print(msg)
        if mass_ratio >=  mass_ratio_cutoff:
            return (diffusive_mass, boundary_mass, crystal_mass, attached, cur_step)
        (next_key, step_key) = jax.random.split(next_key)
        p_step = tuple(params[cur_step])
        (diffusive_mass, boundary_mass, crystal_mass, attached) = do_step(diffusive_mass, boundary_mass, crystal_mass, attached, step_key, gamma, p_step)

    return (diffusive_mass, boundary_mass, crystal_mass, attached, cur_step)

# (random_seed, gamma, params) = get_gamma_and_params(name=name, salt=salt)

def get_cli():
    parser = argparse.ArgumentParser(description='Snowflake Generator.')
    parser.add_argument('-n', '--name', dest="name", type=str, help="The name of the snowflake.")
    parser.add_argument('-s', '--size', dest="size", type=int, help="The size of the snowflake.")
    parser.add_argument('-e', '--env', dest='env', help='Comma seperated key=val env overrides')
    parser.add_argument('-b', '--bw', dest='bw', action='store_true', help='Write out the image in black and white.')
    parser.add_argument('-r', '--randomize', dest='randomize', action='store_true', help='Randomize environment.')
    parser.add_argument('-x', '--extrude', dest='pipeline_3d', action='store_true', help='Enable 3d pipeline.')
    parser.add_argument('-l', '--laser', dest='pipeline_lasercutter', action='store_true', help='Enable Laser Cutter pipeline.')
    parser.add_argument('-m', '--max-steps', dest='max_steps', type=int, help='Maximum number of iterations.')
    parser.add_argument('-M', '--margin', dest='margin', type=float, help='When to stop snowflake growth (between 0 and 1).')
    #parser.add_argument('-c', '--curves', dest='curves', action='store_true', help='Enable use of name to generate environment curves.')
    parser.add_argument('-c', '--color_map', dest='color_map', type=str, help='What colour map should be used for flake image')
    parser.add_argument('-a', '--show_axis', dest='show_axis', type=bool, help='Show an axis on the flake image?')
    parser.add_argument('-L', '--datalog', dest='datalog', action='store_true', help='Enable step wise data logging.')
    parser.add_argument('-D', '--debug', dest='debug', action='store_true', help='Show every step.')
    parser.add_argument('-v', '--movie', dest='movie', action='store_true', help='Render a movie.')
    parser.add_argument('-W', '--width', dest='width', type=float, help="Width of target render.")
    parser.add_argument('-H', '--height', dest='height', type=float, help="Height of target render.")

    parser.set_defaults(**SNOWFLAKE_DEFAULTS)
    args = parser.parse_args()

    args.name = str.join('', map(str.lower, args.name))
    ####CubicSpline is unhappy with arithmatic characters in the seed name
    args.name = ''.join(filter(str.isalpha,args.name))

    #args.target_size = None
    if args.width and args.height:
        args.shape = (args.width, args.height)
        #args.target_size = (args.width, args.height)
    if args.pipeline_3d:
        args.bw = True
    return args

if __name__ == "__main__":
    args = get_cli()
    print(args)
    print("initializing parameters")
    (random_seed, gamma, params) = get_gamma_and_params(name=args.name, max_steps=args.max_steps, salt=salt)
    print("end initializing parameters")

    print("making necessary folders")
    sn = hex(random_seed)[2:]
    root = 'content/SnowflakeDesigns'
    theDir = (f'{root}/{args.name}_{sn}')
    os.makedirs(theDir, exist_ok=True)
    print("end making necessary folders")

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



    print("starting simulation")
    (diffusive_mass, boundary_mass, crystal_mass, attached, cur_step) =  run_simulation(random_seed=random_seed, mass_ratio_cutoff=mass_ratio_cutoff, max_steps=args.max_steps, gamma=gamma, params=params)
    print("ending run simulation")

    ####################################################
    #svg_fn = f'{root}/{args.name}_{sn}/{args.name}_{sn}.svg'
    #render_svg(crystal_mass=crystal_mass, attached=attached, random_state=random_seed, svg_fn=svg_fn)
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
        'spot_threshold': spot_threshold
    }

    ###but why? STH 0313-2023##############################
    # jsfn = f'{root}/{args.name}_{sn}/{args.name}_{sn}.json'
    # with open(jsfn, 'w') as fh:
    #     json.dump(info, fh)
    #######################################################

    print(f"\n\n{'-' * 20}\nGenerator details\n{'-' * 20}\n")
    pprint(info)