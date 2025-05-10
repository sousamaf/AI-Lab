from qgis.core import QgsGraduatedSymbolRenderer, QgsSymbol
# 1. Carregar a malha municipal
shp_path = "/Users/m/Downloads/TO_Municipios_2024/TO_Municipios_2024.shp"
layer_mun = iface.addVectorLayer(shp_path, "Municípios Tocantins", "ogr")

# 2. Carregar a tabela CSV com dados de área plantada
csv_path = "/Users/m/Downloads/TO_Producao/producao_tocantins_2021_2023.csv"
uri = f"file://{csv_path}?delimiter=,&xField=longitude&yField=latitude"
layer_csv = QgsVectorLayer(csv_path, "Produtividade CSV", "ogr")

# 3. Criar o join pelo código do município
join_info = QgsVectorLayerJoinInfo()
join_info.setJoinFieldName("Código")             # do CSV
join_info.setTargetFieldName("CD_GEOCMU")        # do SHP (ajuste conforme o campo existente)
join_info.setJoinLayer(layer_csv)
join_info.setUsingMemoryCache(True)
join_info.setJoinLayerId(layer_csv.id())
layer_mun.addJoin(join_info)

# 4. Aplicar filtro para cultura e ano (Soja - 2023)
expr = '"Cultura" = \'Soja\' AND "Ano" = \'2023\''
layer_mun.setSubsetString(expr)

# 5. Criar simbologia graduada baseada na área plantada
campo_area = "ProducaoEmTonelada"

values = [f[campo_area] for f in layer_mun.getFeatures() if f[campo_area] is not None]

if values:
    symbol = QgsSymbol.defaultSymbol(layer_mun.geometryType())
    renderer = QgsGraduatedSymbolRenderer.createRenderer(
        layer_mun,
        campo_area,
        5,
        QgsGraduatedSymbolRenderer.Quantile,
        symbol
    )
    layer_mun.setRenderer(renderer)
    layer_mun.triggerRepaint()
else:
    iface.messageBar().pushWarning("Aviso", f"Nenhum valor encontrado para '{campo_area}'")
    
# 6. Zoom para a camada
iface.mapCanvas().setExtent(layer_mun.extent())
iface.mapCanvas().refresh()