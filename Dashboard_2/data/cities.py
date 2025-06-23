"""
manages data & data acess for the project
"""

import geopandas as gpd
from shapely.ops import polygonize, unary_union
from shapely.geometry import MultiPoint, LineString
import numpy as np
from scipy.spatial import Delaunay
import os
import pandas as pd
from pathlib import Path

current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parents[2]  # geht zwei Ebenen nach oben

names = ["Madrid", "Barcelona", "Valencia", "Malaga", "Alicante", "Bilbao", "Cordoba", "Murcia", "Seville", "Zaragoza"]

description = {
    "Madrid": 
    """
    Madrid is Spain’s largest city and its undisputed retail capital. With a population of over 3 million and strong purchasing power, 
    Madrid offers high-density foot traffic in prime commercial zones like Gran Vía, Salamanca, and Fuencarral. The city's cosmopolitan vibe, 
    fueled by professionals, students, and tourists, makes it a powerful platform for lifestyle brands seeking visibility and volume. 
    Flagship stores, luxury boutiques, and trend-driven pop-ups thrive here.
    """,
    
    "Barcelona": 
    """
    Barcelona is a magnet for international visitors and a trendsetter in lifestyle, fashion, and urban culture. Its walkable layout and iconic
    shopping corridors—like Passeig de Gràcia and Portal de l’Àngel—attract heavy foot traffic and premium retail activity. The city combines 
    a youthful, design-savvy local audience with millions of global tourists, creating ideal conditions for concept stores, flagship locations, 
    and experiential retail.
    """,
    
    "Valencia": 
    """
    Valencia blends a rich cultural heritage with a growing urban buzz. As Spain’s third-largest city, it offers a balanced mix of local shoppers 
    and tourists, especially in areas like Colón and Ruzafa. The city's modernization and lifestyle-oriented development make it attractive for 
    emerging brands and established retailers aiming for a Mediterranean market presence.
    """,
    
    "Malaga": 
    """
    Malaga has transformed into a vibrant cultural and commercial hub on the Costa del Sol. Known for its booming tourism and luxury appeal, 
    especially in the historic center and along Calle Larios, Malaga presents strong retail opportunities year-round. Its mix of international 
    visitors and affluent second-home residents supports a dynamic retail environment.
    """,
    
    "Alicante": 
    """
    Alicante is a coastal city with strong seasonal retail demand driven by tourism and expat communities. The pedestrian-friendly city center, 
    especially around Avenida Maisonnave, hosts a mix of national chains and local boutiques. Alicante’s sunny climate and relaxed vibe favor 
    lifestyle and leisure-oriented retail formats.
    """,
    
    "Bilbao": 
    """
    Bilbao, the economic heart of the Basque Country, offers a distinct retail landscape anchored in quality and culture. The city’s revitalized 
    core—including Gran Vía and the Old Town—attracts a discerning local clientele. With a focus on design, gastronomy, and sustainability, 
    Bilbao is a fertile ground for premium and niche retail concepts.
    """,
    
    "Cordoba": 
    """
    Cordoba combines deep historical significance with a compact urban structure, creating retail potential in high-footfall areas such as 
    the Judería and Tendillas. Though smaller than Spain’s major metros, its strong cultural tourism and loyal local population support 
    steady retail activity, particularly for heritage and artisan-driven brands.
    """,
    
    "Murcia": 
    """
    Murcia serves as a regional retail center in southeastern Spain, with a growing population and expanding commercial infrastructure. 
    Retail activity concentrates in zones like Gran Vía Escultor Salzillo, catering to a mix of urban residents and suburban shoppers. 
    Murcia’s youthful demographics and regional draw make it ideal for fashion, tech, and value-oriented concepts.
    """,
    
    "Seville": 
    """
    Seville is a cultural powerhouse with a large local population and steady tourist traffic. The retail core—around Calle Tetuán and 
    Sierpes—benefits from consistent footfall and historical charm. Seville’s warm climate, university population, and strong local identity 
    make it well-suited for lifestyle, fashion, and experiential retail brands.
    """,
    
    "Zaragoza": 
    """
    Zaragoza’s strategic location between Madrid and Barcelona gives it regional importance as a logistics and retail hub. With high foot 
    traffic in the city center—particularly along Paseo Independencia—it supports a wide variety of retail formats. A stable population and 
    diversified economy make Zaragoza a valuable mid-size city for national and international retailers alike.
    """
}

center = {
    "Madrid": [40.4168, -3.7038],
    "Barcelona": [41.3874, 2.1686],
    "Valencia": [39.4699, -0.3763],
    "Malaga": [36.7213, -4.4214],
    "Alicante": [38.3452, -0.4810],
    "Bilbao": [43.2630, -2.9349],
    "Cordoba": [37.8882, -4.7794],
    "Murcia": [37.9833, -1.1307],
    "Seville": [37.3886, -5.9823],
    "Zaragoza": [41.6483, -0.8891]
}


information = {
    "Madrid"    : {"population" : 3400000,
                   "tourists"   : 9900000},
    "Barcelona" : {"population" : 1700000,
                   "tourists"   : 15600000},
    "Valencia"  : {"population" : 826000,
                   "tourists"   : 2070000},
    "Malaga"    : {"population" : 590000,
                   "tourists"   : 2500000},
    "Alicante"  : {"population" : 358000 ,
                   "tourists"   : 5000000},
    "Bilbao"    : {"population" : 348000,
                   "tourists"   : 1150000},
    "Cordoba"   : {"population" : 323000,
                   "tourists"   : 1000000},
    "Murcia"    : {"population" : 474000,
                   "tourists"   : 1700000},
    "Seville"   : {"population" : 687000,
                   "tourists"   : 3020000},
    "Zaragoza"  : {"population" : 686000,
                   "tourists"   : 1100000}

}  

# alpha shape should be predifined and not be loaded in users browser

def load_hubs(city):
    # path = f"data/{city}_clusters_2.csv"
    path = os.path.join(PROJECT_ROOT,"clustering_results",f"cluster_allFeatures_{city}_labeled.csv")
    df = pd.read_csv(path)
    return df

def load_pois(city):
    # path = f"../clustering_results/{city}_pois_clusterID.csv"
    path = os.path.join(PROJECT_ROOT,"clustering_results",f"{city}_hdbscan_points.csv")
    df = pd.read_csv(path)
    return df

def load_boundaries(city):
    return os.path.join(PROJECT_ROOT,"Dashboard_2","data","boundaries",f"{city}_boundaries.geojson")