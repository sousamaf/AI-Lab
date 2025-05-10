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

