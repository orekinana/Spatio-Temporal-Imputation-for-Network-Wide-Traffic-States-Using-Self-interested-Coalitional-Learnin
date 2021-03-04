import re
import json


if __name__ == "__main__":

    HTML_FILENAME = '../data/road_network/marker_amap.html'
    ROAD_FILENAME = '../data/road_network/JN_edge_lines.json'
    NODE_FILENAME = '../data/road_network/JN_edge_nodes.json'

    with open(HTML_FILENAME, 'r', encoding='utf-8') as f:
        html = f.readlines()

    with open(ROAD_FILENAME, 'r') as road_f:
        road = json.load(road_f)

    with open(NODE_FILENAME, 'r') as node_f:
        node = json.load(node_f)

    html[40] = re.sub(r'^\s*nodes\s=\s(.+?);$', 'nodes = ' + str(node) + ';', html[40])
    html[42] = re.sub(r'^\s*lines\s=\s(.+?);$', 'lines = ' + str(road) + ';', html[42])

    with open(HTML_FILENAME, 'w', encoding='utf-8') as f:
        f.write(''.join(html))
