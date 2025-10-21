#Requirments gdstk; matplotlib; phidl; pya
#Can be installed with the following commands
#pip install gdstk matplotlib
#pip install -U phidl
#pip install pya

#Typically this would be run in a google collab: https://colab.research.google.com/drive/13XbAF1wuioBRHmCExJhXMqmwwqibiN60?usp=sharing

#Based on GDS to PNG from the EDA team in the S25
#This would be run following the EDA team's SOP for mask generation
#with this version of gds2png being run instead of the EDA team's and with the
#parse_whole_exposure being run instead of parse_exposure

import os
import gdspy
import matplotlib.pyplot as plt
import gdstk
import numpy as np


layers_dict = {
        "via": (1, 0),
        "metal": (2, 0),
        "poly": (3, 0),
        "contact": (4, 0),
        "n+": (5, 0),
        "p+": (6, 0),
        "nwell": (7, 0),
        "psub": (8, 0)
    }

def gds_to_png(gds_filename, output_prefix,
                                   layers=["metal"],
                                   layers_dict=None,
                                   template_layer=(8,0),
                                   bbox_layer=(99,0),
                                   template_resolution=(3840, 2160),
                                   dpi=300,
                                   fill_color='black',
                                   bg_color='white'):

    if layers_dict is None:
        raise ValueError("Please provide a layers_dict mapping layer names to (layer, datatype)")

    # Load GDS library
    lib = gdspy.GdsLibrary(infile=gds_filename)
    top_cells = lib.top_level()

    # --- Step 1: Get single template size ---
    template_polygons = []
    for cell in top_cells:
        polys = cell.get_polygons(by_spec=True)
        if isinstance(polys, dict):
            template_polygons.extend(polys.get(template_layer, []))
        else:
            template_polygons.extend([poly for poly, lyr, dt in polys if (lyr, dt) == template_layer])

    if not template_polygons:
        raise RuntimeError(f"No template frames found on layer {template_layer}")

    t_poly = template_polygons[0]
    t_min_x, t_max_x = np.min(t_poly[:,0]), np.max(t_poly[:,0])
    t_min_y, t_max_y = np.min(t_poly[:,1]), np.max(t_poly[:,1])
    t_width = t_max_x - t_min_x
    t_height = t_max_y - t_min_y

    # --- Step 2: Compute merged design bounding box ---
    bbox_polygons = []
    for cell in top_cells:
        polys = cell.get_polygons(by_spec=True)
        if isinstance(polys, dict):
            bbox_polygons.extend(polys.get(bbox_layer, []))
        else:
            bbox_polygons.extend([poly for poly, lyr, dt in polys if (lyr, dt) == bbox_layer])

    if not bbox_polygons:
        raise RuntimeError(f"No polygons found on bounding box layer {bbox_layer}")

    merged_min_x = min([np.min(p[:,0]) for p in bbox_polygons])
    merged_max_x = max([np.max(p[:,0]) for p in bbox_polygons])
    merged_min_y = min([np.min(p[:,1]) for p in bbox_polygons])
    merged_max_y = max([np.max(p[:,1]) for p in bbox_polygons])

    design_width  = merged_max_x - merged_min_x
    design_height = merged_max_y - merged_min_y

    # --- Step 3: Compute scale factor so each template becomes template_resolution ---
    scale_x = template_resolution[0] / t_width
    scale_y = template_resolution[1] / t_height
    scale = min(scale_x, scale_y)

    # --- Step 4: Compute final PNG size ---
    png_width  = int(np.ceil(design_width * scale))
    png_height = int(np.ceil(design_height * scale))

    print(f"Design bounding box: ({merged_min_x},{merged_min_y})-({merged_max_x},{merged_max_y})")
    print(f"Template size (layout units): {t_width}x{t_height}")
    print(f"Scale factor: {scale}")
    print(f"Output PNG size: {png_width}x{png_height}")

    # --- Step 5: Plot each requested layer ---
    for layer_name in layers:
        layer = layers_dict[layer_name]

        fig, ax = plt.subplots(figsize=(png_width/dpi, png_height/dpi))
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        ax.set_position([0, 0, 1, 1])  # fill entire figure

        for cell in top_cells:
            polys = cell.get_polygons(by_spec=True)
            if isinstance(polys, dict):
                layer_polys = polys.get(layer, [])
            else:
                layer_polys = [poly for poly, lyr, dt in polys if (lyr, dt) == layer]

            for poly in layer_polys:
                # Shift origin and scale to pixels
                shifted_scaled = [((x - merged_min_x)*scale, (y - merged_min_y)*scale) for x,y in poly]
                ax.fill(*zip(*shifted_scaled), color=fill_color)

        ax.set_xlim(0, design_width*scale)
        ax.set_ylim(0, design_height*scale)
        ax.set_aspect('equal')
        ax.axis('off')

        png_filename = f"{output_prefix}_layer_{layer_name}.png"
        plt.savefig(png_filename, dpi=dpi, bbox_inches=None, pad_inches=0, facecolor=bg_color)
        plt.close()
        print(f"Saved {png_filename}")


def main():
  layers_dark = ["metal", "poly"] # List layers that blocks light where the pattern is
  layers_clear = ["n+"] # List layers that lets light through where the pattern is

  gds_to_png(f"full_design.gds", f"red", layers = layers_dark,
            layers_dict = layers_dict,
            bg_color = "black", fill_color = "red")
  gds_to_png(f"full_design.gds", f"uv", layers = layers_dark,
            layers_dict = layers_dict,
            bg_color = "black", fill_color = "blue")
  gds_to_png(f"full_design.gds", f"uv", layers = layers_clear,
            layers_dict = layers_dict,
            bg_color = "blue", fill_color = "black")
  gds_to_png(f"full_design.gds", f"red", layers = layers_clear,
            layers_dict = layers_dict,
            bg_color = "red", fill_color = "black")
  
if __name__ == "__main__":
    main()