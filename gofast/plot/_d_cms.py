# -*- coding: utf-8 -*-

D_COLORS = [
    'g', 'gray', 'y', 'blue', 'orange', 'purple', 'lime', 'k', 'cyan',
    (.6, .6, .6), (0, .6, .3), (.9, 0, .8), (.8, .2, .8), (.0, .9, .4),
    'red', 'brown', 'pink', 'olive', 'navy', 'magenta', 'gold', 'beige',
    (.2, .4, .7), (0.1, 0.2, 0.5), (0.7, 0.3, 0.2),
    'maroon', 'teal', 'coral', 'darkorange', 'indigo', 'darkslategray', 
    'midnightblue','fuchsia', 'deeppink', '#FF5733', '#C70039', '#900C3F', 
    (1, 0.2, 0.3, 0.4)
]

D_CMAPS= [ 
    'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn',
    'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r',
    'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd',
    'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r',
    'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 
    'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r',
    'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu',
    'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 
    'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r',
    'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 
    'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 
    'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 
    'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r',
    'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest',
    'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r',
    'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 
    'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r',
    'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2',
    'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 
    'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 
    'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean',
    'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 
    'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 
    'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20',
    'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 
    'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 
    'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r',
    'vlag', 'vlag_r', 'winter', 'winter_r'
    ]
D_MARKERS = [
    'o', '^', 'x', 'D', '8', '*', 'h', 'p', '>', 'o', 'd', 'H',
    's', 'v', '<', '1', '2', '3', '4', '|', '_', '+', 'P', 'X', '$\u2665$',
    '$\u263A$', 'v', '8', 'H', '>', '<', 'p', '*', '+', 'x', 'D', '$\u25EF$',
    '1', '2', '3', '4', '|', '_', 'P', 'X', 'd', '$\u25C6$', '$\u2660$', 
    '$\u2663$','$\u2764$', 'o', 'd', 'h'
]

D_STYLES = [
    '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted',
    (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)), (0, (1, 10)),
    (0, (5, 10)), 'loosely dashed', 'densely dashed', 'loosely dotted', 'densely dotted',
    (0, (1, 1)), (0, (5, 1)), (0, (3, 5, 1, 5, 1, 5)), (0, (3, 10, 1, 10, 1, 10)),
    'loosely dashdotted', 'densely dashdotted', (0, (3, 1, 1, 1, 1, 1)),
    'loosely dashdotdotted', 'densely dashdotdotted'
]


TDG_DIRECTIONS= { 
    "-1":{ 
        "CORR_POS": {
                "NE": [
                    (1., 0.5),
                    {"rotation_mode": "anchor", "rotation": 90},
                ],
                "SW":[
                    (0., 0.5),
                    {"rotation_mode": "anchor", "rotation": 90},
                ],
                
                "SE": [
                    (0.5, 0.05), {"rotation": "horizontal"},
                ],
                
                "NW": [
                    (0.5, 1.), {"rotation": "horizontal"}
                ],
    
                "W": [
                    (0., 0.75),
                    {"rotation_mode": "anchor", "rotation": 60},
                ],
                "E": [
                    (1.0, 0.5),
                    {"rotation_mode": "anchor", "rotation": 60},
                ],
                "N": [
                    (1.0, 0.75),
                    {"rotation_mode": "anchor", "rotation": -60},
                ],
                "S": [
                    (0.0, 0.5),
                    {"rotation_mode": "anchor", "rotation": -60},
                ],
        }, 
        "STD_POS" : {
                "NE":  [
                    (0.25, 0.75),
                    {"rotation_mode": "anchor", "rotation": 45},
                    ], 
                "SW":[
                    (0.75, 0.25),
                   {"rotation_mode": "anchor", "rotation": 45},
                    ], 
                
                "SE": [
                    (0.75, 0.70), 
                    {"rotation_mode": "anchor", "rotation": -45}
                    ], 
                
                "NW":[
                    (0.15, .35),
                    {"rotation_mode": "anchor", "rotation": -45},
                    ], 
                "W": [
                    (0.5, -0.1),{"rotation": "horizontal"},
                ],
                "E": [
                    (0.5, 1.05),{"rotation": "horizontal"},
                ],
                "N": [
                    (-0.1, 0.5),{"rotation": "vertical"},
                ],
                "S": [
                    (1.1, 0.5),{"rotation": "vertical"},
                ],
            }
    
        },
    
    "1":{ 
        "CORR_POS" : {
              "NW":[
                (0., .5),
                {"rotation_mode": "anchor", "rotation": 90},
                ], 
            ###
            "NE":  [
                (0.5, 0.95), {"rotation": "horizontal"}, 
                ], 
            ##
            "SW":[
                (0.5, 0.05), {"rotation": "horizontal"},
                ], 
            
            "SE": [
                (1.1, 0.5), {"rotation": "vertical"}
                ], 
            
    
            "W": [
                (0.20, 0.25),
                {"rotation_mode": "anchor", "rotation": -45},
            ],
            "E": [
                (0.8, .75),
                {"rotation_mode": "anchor", "rotation": -45},
            ##            
            ],
            "N": [
                (.1, 0.7),
                {"rotation_mode": "anchor", "rotation": 45},
                
               
            ],
            ##
            "S": [
                (.85, 0.3),
                {"rotation_mode": "anchor", "rotation": 45},
            ],
        },
    
    "STD_POS" : {
            #
            "NW": [
                (0.65, 0.85), 
                {"rotation_mode": "anchor", "rotation": -45},
            ],
    
            "NE": [
                (0.90, 0.40),
                {"rotation_mode": "anchor", "rotation": 45},
            ],
            
            "SW":[
                (0.15, 0.65),
                {"rotation_mode": "anchor", "rotation": 45},
            ],
            "SE": [
                (0.25, 0.25), 
                {"rotation_mode": "anchor", "rotation": -45},
            ],
            "W": [
                (0.5, 1.05),{"rotation": "horizontal"},
            ##
            ],
            "E": [
                (.5, -0.1),{"rotation": "horizontal"},
            ],
            ##
            "N": [
                (1.1, 0.5),{"rotation": "vertical"},
                
            ],
            
            "S": [
                (-.1, 0.5),{"rotation": "vertical"},
            ],
        }
    
   } 
    
} 

