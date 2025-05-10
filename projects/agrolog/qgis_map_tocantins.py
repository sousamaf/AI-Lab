from qgis.core import QgsVectorLayer, QgsProject

shapefile_path = "/Users/m/Downloads/TO_Municipios_2024/TO_Municipios_2024.shp"
layer = QgsVectorLayer(shapefile_path, "my_shapefile", "ogr")

if not layer.isValid():
    print("Layer failed to load!")
else:
    QgsProject.instance().addMapLayer(layer)
    print("Layer loaded and added to project.")
