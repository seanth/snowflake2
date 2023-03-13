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
#Name: Enter the name as a generative seed for the snowflake simulator
#max_steps: Maximum number of steps in the simulation. Min 1000. Max 10000

#@title Design your Snowflake

#@markdown ## Snowflake seed
#@markdown Enter the name as a generative seed for the snowflake simulator
name = "best" #@param {type:"string"}

#@markdown Permute snowflake seed with an optional salt value. 
salt = 29 #@param {type:"slider", min:0, max:1024, step:1}

#@markdown Stop the simulaton when the ratio between solid and vaporous water exceeds this value
mass_ratio_cutoff = 0.25 #@param {type:"slider", min:0.01, max:0.99, step:0.01}

#@markdown Maximum number of steps in the simulation
max_steps = 10000 #@param {type:"slider", min:1000, max:10000, step:100}

#@markdown Spot threshold (potrace)
spot_threshold = 0 #@param {type:"slider", min:0, max:500, step:10}

#@markdown Upon the first execution, there might be a few minute delay while the notebook provisions the environment. 

# !(if [ ! -f /tmp/sentinel ]; then apt-get update && apt-get install potrace && pip install svgwrite svgpathtools splines && touch /tmp/sentinel; fi) 2>&1 > /dev/null
# !(if [ ! -f "SourceCodePro-Regular.ttf" ]; then wget https://github.com/adobe-fonts/source-code-pro/raw/release/TTF/SourceCodePro-Regular.ttf > /dev/null 2>/dev/null; fi)

# ft = "SourceCodePro-Regular.ttf"

from PIL import Image, ImageDraw, ImageFont

import os
import json
import hashlib
import tempfile
import random
from pprint import pprint
from math import sqrt
import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
import jax

import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from PIL import Image
from enum import IntEnum

from xml.dom import minidom
from sklearn.cluster import KMeans
from IPython.core.display import SVG
import svgwrite
from svgpathtools import parse_path
from scipy.interpolate import CubicSpline
from scipy import interpolate

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

height = 256
width = 256
shape = (width, height)
self_mask = np.zeros((3,3))
self_mask[1,1] = 1
neighbor_mask = np.fliplr(1 - np.eye(3))
self_and_neighbor_mask = self_mask + neighbor_mask

# convert numpy arrays over to JAX
self_mask = jnp.array(self_mask)
neighbor_mask = jnp.array(neighbor_mask)
self_and_neighbor_mask = jnp.array(self_and_neighbor_mask)

def get_gamma_and_params(name, max_steps=max_steps, salt=0, curves=DefaultCurves):
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

def potrace(img, spot_threshold=None, size=None, margin=None, angle=None, dpi=96, tight=False):
    # start building out the command
    cmd = ['potrace -i -b svg --flat']
    if margin != None:
        cmd.append(f'-M {margin}')
    if spot_threshold != None:
        cmd.append(f'-t {spot_threshold}')
    if size != None:
        cmd.append(f'-W {size[0]} -H {size[1]}')
    if angle != None:
        cmd.append(f'-A {angle}')
    if dpi != None:
        cmd.append(f'-r {dpi}')
    if tight:
        cmd.append('--tight')
    # acquire temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_img = os.path.join(tmp_dir, 'input_image.bmp')
        output_svg = os.path.join(tmp_dir, 'output_trace.svg')
        img.save(input_img)
        #
        cmd.append(f'-o {output_svg} {input_img}')
        cmd = str.join(' ', cmd)
        !{cmd}
        doc = minidom.parse(output_svg)
    width = doc.getElementsByTagName('svg')[0].getAttribute('width')
    height = doc.getElementsByTagName('svg')[0].getAttribute('height')
    paths = [path.getAttribute('d') for path in doc.getElementsByTagName('path')]
    return (width, height, paths)

def plot_snowflake(crystal_mass=None, attached=None, diffusive_mass=None):
    sz = 7
    fig = plt.figure(figsize=(sz, sz))
    plt.imshow(prep_image(crystal_mass), interpolation='nearest')
    plt.show()
    
    '''
    fig = plt.figure(figsize=(sz, sz))
    plt.imshow(prep_image(attached), interpolation='nearest')
    plt.show()
    
    fig = plt.figure(figsize=(sz, sz))
    plt.imshow(prep_image(diffusive_mass), interpolation='nearest')
    plt.show()
    '''

def write_serial_number(name, serial, svg_fn):
    sz = (200, 60)
    txt = Image.new("RGB", sz, (0, 0, 0))
    fnt = ImageFont.truetype(ft, 32)
    d = ImageDraw.Draw(txt)
    d.text((0, 0), serial, font=fnt, fill=(255, 255, 255))
    (svg_width, svg_height, paths) = potrace(txt, tight=True)
    bbox = parse_path(paths[0]).bbox()
    
    dwg = svgwrite.Drawing(svg_fn, size=('1in', '.2in'), debug=True)
    dwg.viewbox(minx=bbox[0], width=bbox[1] - bbox[0], miny=bbox[2], height=bbox[3] - bbox[2])
    dwg.add(dwg.path(d=paths[0], fill='white'))
    dwg.save() 

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
        
@jax.jit
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

def run_simulation(random_seed=None, mass_ratio_cutoff=.25, max_steps=max_steps, gamma=None, params=None):
    random_seed = random_seed if random_seed is not None else 1

    # initialize
    diffusive_mass = jnp.ones(shape) * gamma
    boundary_mass = jnp.zeros(shape)
    crystal_mass = jnp.zeros(shape)
    attached = jnp.zeros(shape).astype(np.uint8)
    next_key = jax.random.PRNGKey(random_seed)
    cur_step = 0
    # attach seed crystal
    seed_idx = (width // 2, height // 2)
    attached = attached.at[seed_idx].set(1)
    crystal_mass = crystal_mass.at[seed_idx].set(diffusive_mass[seed_idx])
    diffusive_mass = diffusive_mass.at[seed_idx].set(0)
    # run simulation
    total_diff_mass = jnp.sum(diffusive_mass)
    for cur_step in range(max_steps):
        mass_ratio = jnp.sum(boundary_mass + crystal_mass) / total_diff_mass
        if mass_ratio >=  mass_ratio_cutoff:
            return (diffusive_mass, boundary_mass, crystal_mass, attached, cur_step)
        (next_key, step_key) = jax.random.split(next_key)
        p_step = tuple(params[cur_step])
        (diffusive_mass, boundary_mass, crystal_mass, attached) = do_step(diffusive_mass, boundary_mass, crystal_mass, attached, step_key, gamma, p_step)

    return (diffusive_mass, boundary_mass, crystal_mass, attached, cur_step)

(random_seed, gamma, params) = get_gamma_and_params(name=name, salt=salt)

sn = hex(random_seed)[2:]
root = '/content/SnowflakeDesigns'
os.makedirs(f'{root}/{name}_{sn}', exist_ok=True)
ser_svg_fn = f'{root}/{name}_{sn}/serial_{name}_{sn}.svg'
svg_fn = f'{root}/{name}_{sn}/{name}_{sn}.svg'
jsfn = f'{root}/{name}_{sn}/{name}_{sn}.json'

(diffusive_mass, boundary_mass, crystal_mass, attached, cur_step) =  run_simulation(random_seed=random_seed, mass_ratio_cutoff=mass_ratio_cutoff, gamma=gamma, params=params)
render_svg(crystal_mass=crystal_mass, attached=attached, random_state=random_seed, svg_fn=svg_fn)
print()
plot_snowflake(crystal_mass=crystal_mass, attached=attached, diffusive_mass=diffusive_mass)
write_serial_number(name, sn, ser_svg_fn)

info = {
    'name': name,
    'salt': salt,
    'n_steps': cur_step,
    'seed': random_seed,
    'sn': sn,
    'mass_ratio_cutoff': mass_ratio_cutoff,
    'spot_threshold': spot_threshold
}

with open(jsfn, 'w') as fh:
    json.dump(info, fh)

print(f"\n\n{'-' * 20}\nGenerator details\n{'-' * 20}\n")
pprint(info)