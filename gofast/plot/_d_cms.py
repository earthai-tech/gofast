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

