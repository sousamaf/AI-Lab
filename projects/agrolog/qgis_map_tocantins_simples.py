from qgis.core import QgsVectorLayer, QgsProject

shapefile_path_map = "/Users/m/Downloads/TO_Municipios_2024/TO_Municipios_2024.shp"
layer_map = QgsVectorLayer(shapefile_path_map, "my_shapefile", "ogr")

shapefile_path_road = "/Users/m/Downloads/TO_Rodovias/SNV_202504A.shp"
layer_road = QgsVectorLayer(shapefile_path_road, "my_shapefile", "ogr")

if not layer_map.isValid():
    print("Layer failed to load!")
else:
    QgsProject.instance().addMapLayer(layer_map)
    print("Layer loaded and added to project.")

if not layer_road.isValid():
    print("Layer failed to load!")
else:
    QgsProject.instance().addMapLayer(layer_road)
    print("Layer loaded and added to project.")

# Funcionando

from qgis.core import QgsVectorLayer, QgsProject, QgsSingleSymbolRenderer, QgsSymbol
from qgis.PyQt.QtGui import QColor

# 1. Carrega e adiciona as camadas
shapefile_path_map  = "/Users/m/Downloads/TO_Municipios_2024/TO_Municipios_2024.shp"
layer_muni = QgsVectorLayer(shapefile_path_map, "TO_Municipios", "ogr")
QgsProject.instance().addMapLayer(layer_muni)

shapefile_path_road = "/Users/m/Downloads/TO_Rodovias/SNV_202504A.shp"
layer_road = QgsVectorLayer(shapefile_path_road, "TO_Rodovias", "ogr")
QgsProject.instance().addMapLayer(layer_road)

# 2. Estiliza municípios (contorno cinza, sem preenchimento)
symb_muni = QgsSymbol.defaultSymbol(layer_muni.geometryType())
symb_muni.setColor(QColor(0,0,0,0))  # transparente
symb_muni.symbolLayer(0).setStrokeColor(QColor('#444444'))
#symb_muni.symbolLayer(0).setStrokeWidth(0.8)
layer_muni.setRenderer(QgsSingleSymbolRenderer(symb_muni))

# 3. Estiliza rodovias (linha branca fixa)
symb_road = QgsSymbol.defaultSymbol(layer_road.geometryType())
symb_road.symbolLayer(0).setStrokeColor(QColor('#ffffff'))
#symb_road.symbolLayer(0).setStrokeWidth(1.4)
layer_road.setRenderer(QgsSingleSymbolRenderer(symb_road))
