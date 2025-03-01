import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

def rotate_point(x, y, angle, cx, cy):
    """
    Rotates a point (x,y) around a pivot (cx,cy) by angle (in degrees).
    """
    rad = math.radians(angle)
    # Translate point to origin
    x -= cx
    y -= cy
    # Apply rotation
    x_new = x * math.cos(rad) - y * math.sin(rad)
    y_new = x * math.sin(rad) + y * math.cos(rad)
    # Translate back
    return x_new + cx, y_new + cy

class LayoutDisplayMixin:
    def display_layout(self, layout, title="Layout"):
        """
        Displays the layout of elements on the cutting sheet.
        Each element must have 'tipo', 'x', 'y', and for rotatable objects,
        a 'rotacao' field representing the rotation angle in degrees.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_xlim(0, self.sheet_width)
        ax.set_ylim(0, self.sheet_height)
        ax.set_xlabel("Sheet Width")
        ax.set_ylabel("Sheet Height")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.5)
        
        for recorte in layout:
            # Get rotation if provided; default to 0
            angle = recorte.get("rotacao", 0)
            
            if recorte["tipo"] == "circular":
                # Circles are rotation invariant.
                circ = patches.Circle(
                    (recorte["x"] + recorte["r"], recorte["y"] + recorte["r"]),
                    recorte["r"],
                    edgecolor='red', facecolor='none', lw=2
                )
                ax.add_patch(circ)
            
            elif recorte["tipo"] == "triangular":
                # Define original vertices for the triangle.
                vertices = [
                    [recorte["x"], recorte["y"]],
                    [recorte["x"] + recorte["b"], recorte["y"]],
                    [recorte["x"] + recorte["b"] / 2, recorte["y"] + recorte["h"]]
                ]
                # Define the pivot as the center of the triangle's bounding box.
                pivot = (recorte["x"] + recorte["b"] / 2, recorte["y"] + recorte["h"] / 2)
                # Rotate each vertex.
                rotated_vertices = [rotate_point(v[0], v[1], angle, pivot[0], pivot[1]) for v in vertices]
                triangle = patches.Polygon(rotated_vertices, edgecolor='green', facecolor='none', lw=2)
                ax.add_patch(triangle)
            
            elif recorte["tipo"] == "diamante":
                # Definimos o diamante como um quadrilátero cujos vértices estão
                # nos pontos médio das bordas do bounding box (largura x altura).
                # Exemplo: 
                #    (x + w/2, y)       <- topo
                #    (x + w,   y + h/2) <- direita
                #    (x + w/2, y + h)   <- base
                #    (x,       y + h/2) <- esquerda
                w = recorte["largura"]
                h = recorte["altura"]
                x0, y0 = recorte["x"], recorte["y"]
                
                vertices = [
                    [x0 + w/2, y0],         # Top
                    [x0 + w,   y0 + h/2],   # Right
                    [x0 + w/2, y0 + h],     # Bottom
                    [x0,       y0 + h/2]    # Left
                ]
                # Pivô para rotação: centro do bounding box
                pivot = (x0 + w/2, y0 + h/2)
                
                # Rotaciona cada vértice
                rotated_vertices = [rotate_point(v[0], v[1], angle, pivot[0], pivot[1]) for v in vertices]
                diamond = patches.Polygon(rotated_vertices, edgecolor='magenta', facecolor='none', lw=2)
                ax.add_patch(diamond)
            
            else:  # Assume "retangular"
                rect = patches.Rectangle(
                    (recorte["x"], recorte["y"]),
                    recorte["largura"], recorte["altura"],
                    angle=angle, edgecolor='blue', facecolor='none', lw=2
                )
                ax.add_patch(rect)
        
        plt.show()