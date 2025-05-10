## Novo:
from qgis.PyQt.QtGui import QColor
from qgis.core import (
    QgsVectorLayer,
    QgsProject,
    QgsSingleSymbolRenderer,
    QgsSymbol,
    QgsLineSymbol,
    QgsSimpleLineSymbolLayer,
    QgsRendererCategory,
    QgsCategorizedSymbolRenderer,
)


def style_layer_municipios(layer):
    """
    Aplica estilo a municípios: preenchimento transparente e contorno cinza.
    """
    # Cria símbolo padrão e ajusta preenchimento transparente
    symbol = QgsSymbol.defaultSymbol(layer.geometryType())
    symbol.setColor(QColor(0, 0, 0, 0))

    # Ajusta contorno
    sl = symbol.symbolLayer(0)
    sl.setStrokeColor(QColor('#444444'))
    sl.setStrokeWidth(0.8)

    # Aplica renderer
    layer.setRenderer(QgsSingleSymbolRenderer(symbol))
    layer.triggerRepaint()


def style_layer_rodovias(layer, campo_tipo='TIPO'):
    """
    Aplica estilo "topo road" duplo para rodovias federais e estaduais:
      - Federais: cor #fff462 (externa) + #ffffff (interna)
      - Estaduais: cor #ff6f61 (externa) + #ffffff (interna)
    """
    categories = []

    # Categoria: Rodovias Federais
    sym_fed = QgsLineSymbol()
    sym_fed.deleteSymbolLayer(0)
    # Camada externa
    outer_fed = QgsSimpleLineSymbolLayer()
    outer_fed.setColor(QColor('#fff462'))
    outer_fed.setWidth(1.8)
    sym_fed.appendSymbolLayer(outer_fed)
    # Camada interna
    inner_fed = QgsSimpleLineSymbolLayer()
    inner_fed.setColor(QColor('#ffffff'))
    inner_fed.setWidth(0.8)
    sym_fed.appendSymbolLayer(inner_fed)
    cat_fed = QgsRendererCategory('Federal', sym_fed, 'Rodovias Federais')
    categories.append(cat_fed)

    # Categoria: Rodovias Estaduais
    sym_est = QgsLineSymbol()
    sym_est.deleteSymbolLayer(0)
    outer_est = QgsSimpleLineSymbolLayer()
    outer_est.setColor(QColor('#ff6f61'))
    outer_est.setWidth(1.8)
    sym_est.appendSymbolLayer(outer_est)
    inner_est = QgsSimpleLineSymbolLayer()
    inner_est.setColor(QColor('#ffffff'))
    inner_est.setWidth(0.8)
    sym_est.appendSymbolLayer(inner_est)
    cat_est = QgsRendererCategory('Estadual', sym_est, 'Rodovias Estaduais')
    categories.append(cat_est)

    # Renderer categorizado com símbolo padrão para outras categorias
    default_symbol = QgsSymbol.defaultSymbol(layer.geometryType())
    renderer = QgsCategorizedSymbolRenderer(campo_tipo, categories, default_symbol)
    layer.setRenderer(renderer)
    layer.triggerRepaint()


# ------------------------------
# Carregamento e aplicação de estilos
# ------------------------------

# 1. Municípios
municipios_path = "/Users/m/Downloads/TO_Municipios_2024/TO_Municipios_2024.shp"
layer_municipios = QgsVectorLayer(municipios_path, "Municipios_2024", "ogr")
if not layer_municipios.isValid():
    print("Falha ao carregar camada de municípios!")
else:
    QgsProject.instance().addMapLayer(layer_municipios)
    style_layer_municipios(layer_municipios)
    print("Camada de municípios carregada e estilizada.")

# 2. Rodovias
rodovias_path = "/Users/m/Downloads/TO_Rodovias/SNV_202504A.shp"
layer_rodovias = QgsVectorLayer(rodovias_path, "Rodovias_TO", "ogr")
if not layer_rodovias.isValid():
    print("Falha ao carregar camada de rodovias!")
else:
    QgsProject.instance().addMapLayer(layer_rodovias)
    # Ajuste 'campo_tipo' para o nome do campo que diferencia Federal/Estadual
    style_layer_rodovias(layer_rodovias, campo_tipo='TIPO')
    print("Camada de rodovias carregada e estilizada (Federal x Estadual).")
