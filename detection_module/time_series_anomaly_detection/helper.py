"""
Helper module containing color space definitions and utility functions.
"""

# Color palettes
COLOR_PALETTES = {
    'default': ['#003f5c', '#2f4b7c', '#665191', '#a05195',
                '#d45087', '#f95d6a', '#ff7c43', '#ffa600'],
    
    'pastel': ['#a1c9f4', '#ffb482', '#8de5a1', '#ff9f9b',
               '#d0bbff', '#debb9b', '#fab0e4', '#cfcfcf'],
    
    'vibrant': ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2',
                '#59a14f', '#edc948', '#b07aa1', '#ff9da7'],
    
    'nature': ['#2e8b57', '#3cb371', '#66cdaa', '#98fb98',
               '#90ee90', '#32cd32', '#228b22', '#006400'],
    
    'sunset': ['#ff7f50', '#ff6347', '#ff4500', '#ff8c00',
               '#ffa500', '#ffd700', '#daa520', '#b8860b'],
    
    'ocean': ['#1e90ff', '#4169e1', '#0000cd', '#000080',
              '#00bfff', '#87cefa', '#87ceeb', '#add8e6'],
    
    'autumn': ['#8b4513', '#a0522d', '#cd853f', '#deb887',
               '#d2691e', '#b8860b', '#daa520', '#cd853f'],
    
    'winter': ['#b0c4de', '#a9a9a9', '#808080', '#696969',
               '#778899', '#708090', '#4682b4', '#6495ed'],
    
    'spring': ['#ff69b4', '#ff1493', '#db7093', '#ffb6c1',
               '#ffc0cb', '#ff69b4', '#ff1493', '#c71585'],
    
    'monochrome': ['#000000', '#1a1a1a', '#333333', '#4d4d4d',
                   '#666666', '#808080', '#999999', '#b3b3b3']
}

def get_color_palette(palette_name='default'):
    """
    Get a specific color palette by name.
    
    Args:
        palette_name (str): Name of the color palette. Available options:
            - default: Blue to orange gradient
            - pastel: Soft pastel colors
            - vibrant: Bright and bold colors
            - nature: Green shades
            - sunset: Orange to yellow gradient
            - ocean: Blue shades
            - autumn: Brown and gold tones
            - winter: Cool gray and blue tones
            - spring: Pink shades
            - monochrome: Black to gray gradient
        
    Returns:
        list: List of color hex codes
        
    Raises:
        ValueError: If the palette name is not found
    """
    if palette_name not in COLOR_PALETTES:
        raise ValueError(f"Color palette '{palette_name}' not found. Available palettes: {list(COLOR_PALETTES.keys())}")
    return COLOR_PALETTES[palette_name] 