import matplotlib.pyplot as plt
import networkx as nx

# Crear un gráfico dirigido
G = nx.DiGraph()

# Añadir nodos con sus respectivas posiciones
positions = {
    "Input Patch": (0, 0),
    "Dropout 1": (1, 1),
    "Downsample 1": (2, 1),
    "Spatial Transfer": (3, 1),
    "Upsample": (4, 1),
    "Dropout 2": (5, 1),
    "Concatenate": (6, 1),
    "Dropout 3": (7, 1),
    "Prediction": (8, 1),
    "DWT": (1, -1),
    "Frequency Transfer": (3, -1),
    "IDWT": (5, -1),
    "Spatial/Frequency Transfer": (6, -1)
}

for node, pos in positions.items():
    G.add_node(node, pos=pos)

# Añadir aristas (conexiones) entre los nodos
edges = [
    ("Input Patch", "Dropout 1"),
    ("Dropout 1", "Downsample 1"),
    ("Downsample 1", "Spatial Transfer"),
    ("Spatial Transfer", "Upsample"),
    ("Upsample", "Dropout 2"),
    ("Dropout 2", "Concatenate"),
    ("Concatenate", "Dropout 3"),
    ("Dropout 3", "Prediction"),
    ("Dropout 1", "DWT"),
    ("DWT", "Frequency Transfer"),
    ("Frequency Transfer", "IDWT"),
    ("IDWT", "Concatenate")
]

G.add_edges_from(edges)

# Obtener posiciones para los nodos
pos = nx.get_node_attributes(G, 'pos')

# Dibujar el gráfico
plt.figure(figsize=(12, 6))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", arrows=True)
plt.title("Network Architecture of the Metamorphic Generator")
plt.show()
